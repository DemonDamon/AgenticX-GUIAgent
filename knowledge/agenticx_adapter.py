#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent AgenticX Adapter
基于AgenticX框架的知识管理适配器：整合AgenticX的storage和retrieval组件

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# AgenticX Storage Components
from agenticx.storage import (
    StorageManager,
    StorageConfig,
    StorageType,
    BaseVectorStorage,
    VectorRecord,
    VectorDBQuery,
    VectorDBQueryResult,
    BaseKeyValueStorage,
    MilvusStorage,
    PostgresStorage,
    RedisStorage
)

# AgenticX Retrieval Components
from agenticx.retrieval import (
    BaseRetriever,
    VectorRetriever,
    HybridRetriever,
    BM25Retriever,
    RetrievalQuery,
    RetrievalResult,
    RetrievalType
)

from .knowledge_types import (
    KnowledgeItem,
    QueryRequest,
    QueryResult,
    KnowledgeType
)
from utils import setup_logger, get_iso_timestamp
from .embedding_factory import EmbeddingFactory, CachedEmbeddingProvider
from .embedding_config import EmbeddingConfig, EmbeddingStrategy, EmbeddingRequest, ContentType


@dataclass
class AgenticXConfig:
    """AgenticX知识管理配置"""
    storage_type: str = "milvus"  # milvus, postgres, memory
    
    # 存储配置
    milvus_config: Dict[str, Any] = None
    postgres_config: Dict[str, Any] = None
    redis_config: Dict[str, Any] = None
    
    # 向量化配置
    vectorization_enabled: bool = True
    embedding_dimension: int = 1536
    
    # Embedding配置
    embedding_provider: str = "bailian"
    embedding_config: Dict[str, Any] = None
    
    # 检索配置
    retrieval_type: str = "hybrid"  # vector, hybrid, auto
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgenticXConfig':
        """从配置字典创建配置对象"""
        return cls(
            storage_type=config_dict.get("storage_type", "milvus"),
            milvus_config=config_dict.get("database", {}).get("milvus", {}),
            postgres_config=config_dict.get("database", {}).get("postgres", {}),
            redis_config=config_dict.get("database", {}).get("redis", {}),
            vectorization_enabled=config_dict.get("vectorization", {}).get("enabled", True),
            embedding_dimension=config_dict.get("vectorization", {}).get("dimension", 1536),
            embedding_provider=config_dict.get("embedding_provider", "bailian"),
            embedding_config=config_dict.get("embedding_config"),
            retrieval_type=config_dict.get("retrieval_type", "hybrid")
        )
    
    def get_embedding_provider(self):
        """获取embedding提供者（支持混合模式）"""
        if not self.embedding_config:
            return None
        
        # 检查是否启用混合模式
        if self.embedding_config.get('multimodal', {}).get('enabled', False):
            return self.get_hybrid_embedding_manager()
        else:
            return self.get_single_embedding_provider()
    
    def get_single_embedding_provider(self):
        """获取单一embedding提供者"""
        # 创建embedding配置
        provider_config = self.embedding_config.get(self.embedding_provider, {})
        embedding_config = EmbeddingConfig(
            provider=self.embedding_provider,
            model=provider_config.get('model', 'text-embedding-v4'),  # 默认使用v4
            api_key=provider_config.get('api_key', ''),
            api_url=provider_config.get('api_url', ''),
            dimension=provider_config.get('dimension', self.embedding_dimension),
            max_tokens=provider_config.get('max_tokens', 8192),
            batch_size=provider_config.get('batch_size', 100),
            timeout=provider_config.get('timeout', 30),
            retry_count=provider_config.get('retry_count', 3),
            retry_delay=provider_config.get('retry_delay', 1.0)
        )
        
        # 创建提供者
        factory = EmbeddingFactory()
        provider = factory.create_provider(embedding_config)
        
        # 添加缓存包装
        cache_config = self.embedding_config.get('cache', {})
        if cache_config.get('enabled', True):
            provider = CachedEmbeddingProvider(
                provider=provider,
                cache_enabled=True,
                cache_ttl=cache_config.get('ttl', 3600),
                max_cache_entries=cache_config.get('max_entries', 10000)
            )
        
        return provider
    
    def get_hybrid_embedding_manager(self):
        """获取混合embedding管理器"""
        try:
            # 使用工厂方法创建混合管理器
            return EmbeddingFactory.create_hybrid_from_config(self.embedding_config)
        except Exception as e:
            # 降级到单一提供者
            return self.get_single_embedding_provider()


class KnowledgeToVectorAdapter:
    """知识项到向量记录的适配器"""
    
    def __init__(self, embedding_provider=None):
        self.embedding_provider = embedding_provider
        self.logger = logger
    
    def knowledge_to_vector_record(
        self,
        knowledge: KnowledgeItem,
        vector: Optional[List[float]] = None
    ) -> VectorRecord:
        """将知识项转换为向量记录"""
        # 提取文本内容
        text_content = self._extract_text_content(knowledge)
        
        # 构建载荷
        payload = {
            "knowledge_id": knowledge.id,
            "type": knowledge.type.value,
            "title": knowledge.title,
            "description": knowledge.description,
            "content": knowledge.content,
            "keywords": list(knowledge.keywords),
            "domain": knowledge.domain,
            "created_at": knowledge.metadata.created_at,
            "confidence": knowledge.metadata.confidence,
            "importance": getattr(knowledge.metadata, 'importance', 0.5),
            "text_content": text_content
        }
        
        return VectorRecord(
            id=knowledge.id,
            vector=vector or [0.0] * 1536,  # 占位向量
            payload=payload
        )
    
    def vector_result_to_knowledge(
        self,
        vector_result: VectorDBQueryResult
    ) -> Optional[KnowledgeItem]:
        """将向量查询结果转换为知识项"""
        try:
            payload = vector_result.record.payload
            if not payload:
                return None
            
            # 重建知识项
            from .knowledge_types import KnowledgeMetadata
            
            metadata = KnowledgeMetadata(
                created_at=payload.get("created_at", get_iso_timestamp()),
                updated_at=get_iso_timestamp(),
                created_by="system",
                updated_by="system",
                confidence=payload.get("confidence", 0.5)
            )
            
            knowledge = KnowledgeItem(
                id=payload["knowledge_id"],
                type=KnowledgeType(payload["type"]),
                title=payload.get("title", ""),
                description=payload.get("description", ""),
                content=payload.get("content"),
                keywords=set(payload.get("keywords", [])),
                domain=payload.get("domain", "general"),
                metadata=metadata
            )
            
            # 添加相似度信息到元数据
            knowledge.metadata.custom_attributes["similarity_score"] = vector_result.similarity
            
            return knowledge
            
        except Exception as e:
            logger.error(f"转换向量结果到知识项失败: {e}")
            return None
    
    def _extract_text_content(self, knowledge: KnowledgeItem) -> str:
        """从知识项提取文本内容"""
        content_parts = []
        
        if knowledge.title:
            content_parts.append(knowledge.title)
        
        if knowledge.description:
            content_parts.append(knowledge.description)
        
        if knowledge.keywords:
            content_parts.append(" ".join(knowledge.keywords))
        
        if isinstance(knowledge.content, str):
            content_parts.append(knowledge.content)
        elif isinstance(knowledge.content, dict):
            for value in knowledge.content.values():
                if isinstance(value, str):
                    content_parts.append(value)
        
        return " ".join(content_parts)


class AgenticXKnowledgeManager:
    """基于AgenticX的知识管理器"""
    
    def __init__(self, config: AgenticXConfig):
        self.config = config
        self.logger = logger
        
        # 组件
        self.storage_manager: Optional[StorageManager] = None
        self.retriever: Optional[BaseRetriever] = None
        self.embedding_provider = None
        self.adapter = KnowledgeToVectorAdapter()
        
        # 状态
        self._initialized = False
        
        # 初始化embedding提供者
        self._initialize_embedding_provider()

    def get_storage(self, storage_class: type) -> Optional[Any]:
        """从StorageManager中获取特定类型的存储实例"""
        if not self.storage_manager or not self.storage_manager.storages:
            self.logger.warning("Storage manager not initialized or has no storages.")
            return None
        for storage in self.storage_manager.storages:
            if isinstance(storage, storage_class):
                self.logger.info(f"Found storage: {storage.__class__.__name__}")
                return storage
        self.logger.warning(f"Storage of type {storage_class.__name__} not found.")
        return None
    
    def _initialize_embedding_provider(self):
        """初始化embedding提供者"""
        try:
            self.embedding_provider = self.config.get_embedding_provider()
            if self.embedding_provider:
                logger.info(f"Embedding提供者初始化成功: {self.config.embedding_provider}")
            else:
                logger.warning("未配置embedding提供者，将使用MockEmbeddingProvider")
                self.embedding_provider = MockEmbeddingProvider(dimension=self.config.embedding_dimension)
        except Exception as e:
            logger.error(f"Embedding提供者初始化失败: {e}，使用MockEmbeddingProvider")
            self.embedding_provider = MockEmbeddingProvider(dimension=self.config.embedding_dimension)
    
    async def initialize(self) -> bool:
        """初始化知识管理器"""
        if self._initialized:
            return True
        
        try:
            # 创建存储配置
            storage_configs = self._create_storage_configs()
            
            # 初始化存储管理器
            self.storage_manager = StorageManager(storage_configs)
            await self.storage_manager.initialize()
            
            # 初始化检索器
            await self._initialize_retriever()
            
            self._initialized = True
            logger.info("AgenticX知识管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"AgenticX知识管理器初始化失败: {e}")
            return False
    
    def _create_storage_configs(self) -> List[StorageConfig]:
        """创建存储配置"""
        configs = []
        
        # Redis配置（缓存）
        if self.config.redis_config:
            redis_config = StorageConfig(
                storage_type=StorageType.REDIS,
                **self.config.redis_config
            )
            configs.append(redis_config)
        
        # PostgreSQL配置（元数据）
        if self.config.postgres_config:
            postgres_config = StorageConfig(
                storage_type=StorageType.POSTGRES,
                **self.config.postgres_config
            )
            configs.append(postgres_config)
        
        # Milvus配置（向量）
        if self.config.milvus_config and self.config.vectorization_enabled:
            milvus_config = StorageConfig(
                storage_type=StorageType.MILVUS,
                **self.config.milvus_config
            )
            configs.append(milvus_config)
        
        return configs
    
    async def _initialize_retriever(self) -> None:
        """初始化检索器"""
        if not self.config.vectorization_enabled:
            logger.info("向量化未启用，跳过检索器初始化")
            return

        vector_storage = self.get_storage(MilvusStorage)
        if not vector_storage:
            logger.warning("未找到向量存储 (Milvus)，跳过检索器初始化")
            return

        retrieval_type = self.config.retrieval_type
        logger.info(f"初始化检索器，类型: {retrieval_type}")

        if retrieval_type == "vector":
            self.retriever = VectorRetriever(tenant_id="agenticx-guiagent", embedding_provider=self.embedding_provider, vector_storage=vector_storage)
        
        elif retrieval_type == "hybrid":
            vector_retriever = VectorRetriever(tenant_id="agenticx-guiagent", embedding_provider=self.embedding_provider, vector_storage=vector_storage)
            
            # BM25Retriever需要一个文档存储，我们可以使用Postgres或Redis
            doc_storage = self.get_storage(PostgresStorage)
            if not doc_storage:
                doc_storage = self.get_storage(RedisStorage)

            if not doc_storage:
                logger.warning("未找到文档存储 (Postgres/Redis)，混合检索降级为向量检索")
                self.retriever = vector_retriever
            else:
                logger.info(f"为BM25Retriever找到文档存储: {doc_storage.__class__.__name__}")
                bm25_retriever = BM25Retriever(tenant_id="agenticx-guiagent", doc_storage=doc_storage)
                
                self.retriever = HybridRetriever(
                    tenant_id="agenticx-guiagent",
                    vector_retriever=vector_retriever,
                    bm25_retriever=bm25_retriever,
                    # weights=[0.5, 0.5] # 可以选择性添加权重
                )
        else:
            logger.warning(f"不支持的检索类型: {retrieval_type}，将使用默认的向量检索")
            self.retriever = VectorRetriever(tenant_id="agenticx-guiagent", embedding_provider=self.embedding_provider, vector_storage=vector_storage)

        if self.retriever:
            await self.retriever.initialize()
            logger.info(f"检索器 {self.retriever.__class__.__name__} 初始化成功")

    async def store_knowledge(
        self,
        knowledge: KnowledgeItem,
        vector: Optional[List[float]] = None
    ) -> bool:
        """存储知识项"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # 如果没有提供向量，则生成向量
            if vector is None and self.config.vectorization_enabled:
                text_content = self.adapter._extract_text_content(knowledge)
                if text_content and self.embedding_provider:
                    try:
                        # 使用真实的embedding服务
                        if hasattr(self.embedding_provider, 'encode_text'):
                            vector = await self.embedding_provider.encode_text(text_content)
                        elif hasattr(self.embedding_provider, 'aembed'):
                            vectors = await self.embedding_provider.aembed([text_content])
                            if vectors and len(vectors) > 0:
                                vector = vectors[0]
                        else:
                            vectors = self.embedding_provider.embed([text_content])
                            if vectors and len(vectors) > 0:
                                vector = vectors[0]
                        
                        if vector:
                            logger.debug(f"文本向量化成功: {knowledge.id}, 维度: {len(vector)}")
                        else:
                            logger.warning(f"文本向量化返回空结果: {knowledge.id}")
                    except Exception as e:
                        logger.error(f"文本向量化失败: {e}，跳过向量存储")
                        vector = None
            
            # 存储到向量数据库
            if self.config.vectorization_enabled and vector:
                vector_record = self.adapter.knowledge_to_vector_record(knowledge, vector)
                
                # 获取向量存储
                vector_storage = self._get_vector_storage()
                if vector_storage:
                    vector_storage.add([vector_record])
                    logger.debug(f"向量存储成功: {knowledge.id}")
            
            # 存储到键值存储（元数据）
            kv_storage = self._get_kv_storage()
            if kv_storage:
                knowledge_dict = knowledge.to_dict()
                await kv_storage.set(knowledge.id, json.dumps(knowledge_dict, ensure_ascii=False))
                logger.debug(f"元数据存储成功: {knowledge.id}")
            
            logger.info(f"成功存储知识项: {knowledge.id}")
            return True
            
        except Exception as e:
            logger.error(f"存储知识项失败 {knowledge.id}: {e}")
            return False
    
    async def retrieve_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """检索知识项"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # 从键值存储获取
            kv_storage = self._get_kv_storage()
            if kv_storage:
                knowledge_json = await kv_storage.get(knowledge_id)
                if knowledge_json:
                    knowledge_dict = json.loads(knowledge_json)
                    return KnowledgeItem.from_dict(knowledge_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"检索知识项失败 {knowledge_id}: {e}")
            return None
    
    async def query_knowledge(
        self,
        request: QueryRequest,
        query_vector: Optional[List[float]] = None
    ) -> QueryResult:
        """查询知识项"""
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            results = []
            
            # 如果没有提供查询向量但有查询文本，则生成向量
            if query_vector is None and request.query_text and self.config.vectorization_enabled:
                if self.embedding_provider:
                    try:
                        # 使用真实的embedding服务向量化查询文本
                        if hasattr(self.embedding_provider, 'encode_text'):
                            query_vector = await self.embedding_provider.encode_text(request.query_text)
                        elif hasattr(self.embedding_provider, 'aembed'):
                            vectors = await self.embedding_provider.aembed([request.query_text])
                            if vectors and len(vectors) > 0:
                                query_vector = vectors[0]
                        else:
                            vectors = self.embedding_provider.embed([request.query_text])
                            if vectors and len(vectors) > 0:
                                query_vector = vectors[0]
                        
                        if query_vector:
                            logger.debug(f"查询文本向量化成功: 维度 {len(query_vector)}")
                        else:
                            logger.warning("查询文本向量化返回空结果")
                    except Exception as e:
                        logger.error(f"查询文本向量化失败: {e}")
            
            if self.retriever and query_vector and request.query_text:
                # 使用AgenticX检索器
                retrieval_query = RetrievalQuery(
                    text=request.query_text,
                    query_type=RetrievalType.VECTOR if self.config.retrieval_type == "vector" else RetrievalType.HYBRID,
                    limit=request.limit,
                    filters=request.filters or {}
                )
                
                retrieval_results = await self.retriever.retrieve(retrieval_query)
                
                # 转换结果
                for result in retrieval_results:
                    # 从元数据重建知识项
                    knowledge = await self._result_to_knowledge(result)
                    if knowledge:
                        results.append(knowledge)
            
            elif query_vector:
                # 直接使用向量存储查询
                vector_storage = self._get_vector_storage()
                if vector_storage:
                    query = VectorDBQuery(
                        query_vector=query_vector,
                        top_k=request.limit
                    )
                    
                    vector_results = vector_storage.query(query)
                    
                    # 转换结果
                    for vector_result in vector_results:
                        knowledge = self.adapter.vector_result_to_knowledge(vector_result)
                        if knowledge:
                            results.append(knowledge)
            
            # 如果没有向量检索结果，尝试基于文本的简单匹配
            if not results and request.query_text:
                # 这里可以添加基于关键词的检索逻辑
                logger.debug("向量检索无结果，可考虑添加关键词检索")
            
            # 构建查询结果
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                request_id=request.id,
                items=results,
                total_count=len(results),
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"查询知识失败: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                request_id=request.id,
                execution_time=execution_time
            )
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识项"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # 从向量存储删除
            vector_storage = self._get_vector_storage()
            if vector_storage:
                vector_storage.delete([knowledge_id])
            
            # 从键值存储删除
            kv_storage = self._get_kv_storage()
            if kv_storage:
                await kv_storage.delete(knowledge_id)
            
            logger.debug(f"成功删除知识项: {knowledge.id}")
            return True
            
        except Exception as e:
            logger.error(f"删除知识项失败 {knowledge_id}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._initialized:
            await self.initialize()
        
        try:
            stats = {
                "initialized": self._initialized,
                "storage_manager_stats": {},
                "retriever_stats": {}
            }
            
            # 存储管理器统计
            if self.storage_manager:
                stats["storage_manager_stats"] = await self.storage_manager.get_statistics()
            
            # 检索器统计
            if self.retriever:
                stats["retriever_stats"] = await self.retriever.get_stats()
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def _get_vector_storage(self) -> Optional[BaseVectorStorage]:
        """获取向量存储"""
        if not self.storage_manager:
            return None
        
        for storage in self.storage_manager.storages:
            if isinstance(storage, BaseVectorStorage):
                return storage
        return None
    
    def _get_kv_storage(self) -> Optional[BaseKeyValueStorage]:
        """获取键值存储"""
        if not self.storage_manager:
            return None
        
        for storage in self.storage_manager.storages:
            if isinstance(storage, BaseKeyValueStorage):
                return storage
        return None
    
    async def _result_to_knowledge(self, result: RetrievalResult) -> Optional[KnowledgeItem]:
        """将检索结果转换为知识项"""
        try:
            # 如果结果包含chunk_id，尝试从存储获取完整知识项
            if result.chunk_id:
                return await self.retrieve_knowledge(result.chunk_id)
            
            # 否则从元数据重建
            if "knowledge_id" in result.metadata:
                return await self.retrieve_knowledge(result.metadata["knowledge_id"])
            
            return None
            
        except Exception as e:
            logger.error(f"转换检索结果失败: {e}")
            return None
    
    async def close(self) -> None:
        """关闭知识管理器"""
        if self.storage_manager:
            await self.storage_manager.close()
        
        self._initialized = False
        logger.info("AgenticX知识管理器已关闭")


class MockEmbeddingProvider:
    """模拟嵌入提供者（用于测试）"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
    
    async def encode_text(self, text: str) -> List[float]:
        """编码文本为向量"""
        # 基于文本内容生成确定性向量
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        
        np.random.seed(seed)
        vector = np.random.normal(0, 1, self.dimension).astype(np.float32)
        return (vector / np.linalg.norm(vector)).tolist()  # 归一化
    
    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        return [await self.encode_text(text) for text in texts]