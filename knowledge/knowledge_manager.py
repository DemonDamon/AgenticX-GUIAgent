#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Knowledge Manager
基于AgenticX框架的知识管理器：使用AgenticX的storage和retrieval组件

重构说明：
- 使用AgenticX的StorageManager和检索组件
- 集成AgenticX的事件系统进行知识管理
- 提供现代化的知识生命周期管理
- 移除重复的自定义实现

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, UTC
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Callable, Awaitable
)
from uuid import uuid4
from loguru import logger

from .knowledge_types import (
    KnowledgeItem, KnowledgeType, KnowledgeSource, KnowledgeStatus,
    KnowledgeRelation, RelationType, QueryRequest, QueryResult,
    SyncRequest, SyncResult, KnowledgeMetadata
)
# 使用AgenticX核心组件
from agenticx.core.component import Component
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from agenticx.memory.component import MemoryComponent

# 使用AgenticX存储和检索组件
from .agenticx_adapter import AgenticXKnowledgeManager, AgenticXConfig, MockEmbeddingProvider
from .knowledge_store import KnowledgeStoreInterface, KnowledgeStoreFactory
from utils import get_iso_timestamp, setup_logger


class KnowledgeCache:
    """知识缓存"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[KnowledgeItem, datetime]] = {}
        self._access_order = deque()
        self._lock = threading.RLock()
    
    def get(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """获取缓存的知识"""
        with self._lock:
            if knowledge_id in self._cache:
                knowledge, cached_time = self._cache[knowledge_id]
                
                # 检查是否过期
                if datetime.now() - cached_time > timedelta(seconds=self.ttl_seconds):
                    self._remove(knowledge_id)
                    return None
                
                # 更新访问顺序
                if knowledge_id in self._access_order:
                    self._access_order.remove(knowledge_id)
                self._access_order.append(knowledge_id)
                
                return knowledge
            
            return None
    
    def put(self, knowledge: KnowledgeItem) -> None:
        """缓存知识"""
        with self._lock:
            # 如果已存在，先移除
            if knowledge.id in self._cache:
                self._remove(knowledge.id)
            
            # 检查缓存大小
            while len(self._cache) >= self.max_size:
                if self._access_order:
                    oldest_id = self._access_order.popleft()
                    self._remove(oldest_id)
                else:
                    break
            
            # 添加到缓存
            self._cache[knowledge.id] = (knowledge, datetime.now())
            self._access_order.append(knowledge.id)
    
    def remove(self, knowledge_id: str) -> None:
        """移除缓存"""
        with self._lock:
            self._remove(knowledge_id)
    
    def _remove(self, knowledge_id: str) -> None:
        """内部移除方法"""
        if knowledge_id in self._cache:
            del self._cache[knowledge_id]
        if knowledge_id in self._access_order:
            self._access_order.remove(knowledge_id)
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }


class KnowledgeValidator:
    """知识验证器"""
    
    def __init__(self):
        self.logger = logger
    
    async def validate_knowledge(self, knowledge: KnowledgeItem) -> Tuple[bool, float, Dict[str, Any]]:
        """验证知识"""
        try:
            validation_details = {}
            total_score = 0.0
            weight_sum = 0.0
            
            # 基本结构验证
            structure_valid, structure_score = self._validate_structure(knowledge)
            validation_details['structure'] = {
                'valid': structure_valid,
                'score': structure_score,
                'weight': 0.3
            }
            total_score += structure_score * 0.3
            weight_sum += 0.3
            
            # 内容质量验证
            content_valid, content_score = self._validate_content(knowledge)
            validation_details['content'] = {
                'valid': content_valid,
                'score': content_score,
                'weight': 0.4
            }
            total_score += content_score * 0.4
            weight_sum += 0.4
            
            # 元数据验证
            metadata_valid, metadata_score = self._validate_metadata(knowledge)
            validation_details['metadata'] = {
                'valid': metadata_valid,
                'score': metadata_score,
                'weight': 0.3
            }
            total_score += metadata_score * 0.3
            weight_sum += 0.3
            
            # 计算总分
            final_score = total_score / weight_sum if weight_sum > 0 else 0.0
            is_valid = structure_valid and content_valid and metadata_valid and final_score >= 0.6
            
            # 更新知识的验证状态
            knowledge.metadata.validation_status = 'valid' if is_valid else 'invalid'
            knowledge.metadata.validation_score = final_score
            knowledge.metadata.validation_details = validation_details
            
            return is_valid, final_score, validation_details
            
        except Exception as e:
            logger.error(f"Failed to validate knowledge {knowledge.id}: {e}")
            return False, 0.0, {'error': str(e)}
    
    def _validate_structure(self, knowledge: KnowledgeItem) -> Tuple[bool, float]:
        """验证结构"""
        score = 0.0
        
        # 检查必需字段
        if knowledge.id:
            score += 0.2
        if knowledge.title:
            score += 0.2
        if knowledge.content is not None:
            score += 0.3
        if knowledge.type and knowledge.source:
            score += 0.2
        if knowledge.domain:
            score += 0.1
        
        return score >= 0.8, score
    
    def _validate_content(self, knowledge: KnowledgeItem) -> Tuple[bool, float]:
        """验证内容"""
        score = 0.0
        
        if knowledge.content is None:
            return False, 0.0
        
        # 内容长度检查
        if isinstance(knowledge.content, str):
            content_length = len(knowledge.content)
            if content_length > 10:
                score += 0.3
            if content_length > 100:
                score += 0.2
        elif isinstance(knowledge.content, (dict, list)):
            score += 0.5
        
        # 描述质量
        if knowledge.description and len(knowledge.description) > 20:
            score += 0.2
        
        # 关键词质量
        if knowledge.keywords and len(knowledge.keywords) > 0:
            score += 0.3
        
        return score >= 0.6, score
    
    def _validate_metadata(self, knowledge: KnowledgeItem) -> Tuple[bool, float]:
        """验证元数据"""
        score = 0.0
        
        # 时间戳检查
        if knowledge.metadata.created_at:
            score += 0.2
        if knowledge.metadata.updated_at:
            score += 0.2
        
        # 质量指标检查
        if 0 <= knowledge.metadata.confidence <= 1:
            score += 0.2
        if 0 <= knowledge.metadata.reliability <= 1:
            score += 0.2
        if 0 <= knowledge.metadata.accuracy <= 1:
            score += 0.2
        
        return score >= 0.6, score


class KnowledgeLifecycleManager:
    """知识生命周期管理器"""
    
    def __init__(self, store: KnowledgeStoreInterface):
        self.store = store
        self.logger = logger
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """启动生命周期管理"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Knowledge lifecycle manager started")
    
    async def stop(self) -> None:
        """停止生命周期管理"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Knowledge lifecycle manager stopped")
    
    async def _cleanup_loop(self) -> None:
        """清理循环"""
        while self._running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(3600)  # 每小时执行一次清理
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # 出错时等待5分钟
    
    async def _perform_cleanup(self) -> None:
        """执行清理"""
        try:
            # 获取所有知识项
            all_knowledge = await self.store.get_all_knowledge()
            
            expired_count = 0
            obsolete_count = 0
            
            for knowledge in all_knowledge:
                # 检查过期
                if self._is_expired(knowledge):
                    await self.store.delete_knowledge(knowledge.id)
                    expired_count += 1
                    continue
                
                # 检查是否过时
                if self._is_obsolete(knowledge):
                    knowledge.status = KnowledgeStatus.OBSOLETE
                    await self.store.update_knowledge(knowledge)
                    obsolete_count += 1
            
            if expired_count > 0 or obsolete_count > 0:
                logger.info(
                    f"Cleanup completed: {expired_count} expired, {obsolete_count} obsolete"
                )
                
        except Exception as e:
            logger.error(f"Failed to perform cleanup: {e}")
    
    def _is_expired(self, knowledge: KnowledgeItem) -> bool:
        """检查是否过期"""
        if not knowledge.metadata.expiry_date:
            return False
        
        try:
            expiry_date = datetime.fromisoformat(knowledge.metadata.expiry_date.replace('Z', '+00:00'))
            return datetime.now() > expiry_date
        except Exception:
            return False
    
    def _is_obsolete(self, knowledge: KnowledgeItem) -> bool:
        """检查是否过时"""
        if not knowledge.metadata.retention_period:
            return False
        
        try:
            created_date = datetime.fromisoformat(knowledge.metadata.created_at.replace('Z', '+00:00'))
            retention_days = knowledge.metadata.retention_period
            return datetime.now() > created_date + timedelta(days=retention_days)
        except Exception:
            return False


class KnowledgeManager(Component):
    """知识管理器 - 基于AgenticX Component，使用AgenticX存储和检索组件"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None,
        memory: Optional[MemoryComponent] = None,
        embedding_provider: Optional[Any] = None
    ):
        super().__init__(name="knowledge_manager")
        
        self.logger = logger
        self.event_bus = event_bus or EventBus()
        self.memory = memory
        
        # 创建AgenticX配置
        database_config = config.get('database', {})
        self.agenticx_config = AgenticXConfig(
            storage_type=config.get('storage_type', 'milvus'),
            milvus_config=database_config.get('milvus'),
            postgres_config=database_config.get('postgres'),
            redis_config=database_config.get('redis'),
            vectorization_enabled=config.get('vectorization', {}).get('enabled', True),
            embedding_dimension=config.get('vectorization', {}).get('dimension', 1536),
            embedding_provider=config.get('embedding_provider', 'bailian'),
            embedding_config=config.get('embedding_config'),
            retrieval_type=config.get('retrieval_type', 'hybrid')
        )
        
        # 创建AgenticX知识管理器
        self.agenticx_manager = AgenticXKnowledgeManager(self.agenticx_config)
        
        # 设置embedding提供者（优先使用传入的，否则使用配置中的）
        if embedding_provider:
            self.embedding_provider = embedding_provider
        else:
            # 从AgenticX管理器获取embedding提供者
            self.embedding_provider = self.agenticx_manager.embedding_provider
        
        # 统计信息
        self.stats = {
            'total_knowledge': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validations_performed': 0,
            'validations_passed': 0,
            'sync_operations': 0,
            'last_sync': None
        }
        
        # 同步配置
        self.sync_config = {
            'auto_sync': True,
            'sync_interval': 300,  # 5分钟
            'batch_size': 100,
            'max_retries': 3
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # 运行状态
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
    
    async def start(self) -> None:
        """启动知识管理器"""
        if not self._running:
            logger.info(f"🚀 启动知识管理器...")
            self._running = True
            
            # 初始化AgenticX管理器
            logger.info(f"🔧 初始化 AgenticX 管理器...")
            await self.agenticx_manager.initialize()
            logger.info(f"✅ AgenticX 管理器初始化完成")
            
            # 启动自动同步
            if self.sync_config['auto_sync']:
                logger.info(f"⏰ 启动自动同步任务 (间隔: {self.sync_config['sync_interval']}秒)")
                self._sync_task = asyncio.create_task(self._sync_loop())
            else:
                logger.info(f"ℹ️ 自动同步已禁用")
            
            # 发布启动事件
            await self._publish_event('manager_started', {'timestamp': get_iso_timestamp()})
            
            logger.info(f"🎉 知识管理器启动完成")
            logger.info(f"   - 存储类型: {self.agenticx_config.storage_type}")
            logger.info(f"   - 向量化: {'启用' if self.agenticx_config.vectorization_enabled else '禁用'}")
            logger.info(f"   - 检索类型: {self.agenticx_config.retrieval_type}")
    
    async def stop(self) -> None:
        """停止知识管理器"""
        logger.info(f"🛑 停止知识管理器...")
        self._running = False
        
        # 停止同步任务
        if self._sync_task:
            logger.info(f"⏹️ 停止自动同步任务...")
            self._sync_task.cancel()
            try:
                await self._sync_task
                logger.info(f"✅ 自动同步任务已停止")
            except asyncio.CancelledError:
                logger.info(f"✅ 自动同步任务已取消")
        
        # 关闭AgenticX管理器
        logger.info(f"🔧 关闭 AgenticX 管理器...")
        await self.agenticx_manager.close()
        logger.info(f"✅ AgenticX 管理器已关闭")
        
        # 发布停止事件
        await self._publish_event('manager_stopped', {'timestamp': get_iso_timestamp()})
        
        logger.info(f"🏁 知识管理器已停止")
    
    async def store_knowledge(
        self,
        knowledge: KnowledgeItem,
        validate: bool = True,
        cache: bool = True
    ) -> bool:
        """存储知识"""
        try:
            logger.info(f"📚 开始存储知识: {knowledge.id}")
            logger.info(f"   - 标题: {knowledge.title}")
            logger.info(f"   - 知识类型: {knowledge.type.value}")
            logger.info(f"   - 来源: {knowledge.source.value}")
            logger.info(f"   - 领域: {knowledge.domain}")
            
            # 向量化文本内容
            vector = None
            if self.agenticx_config.vectorization_enabled:
                logger.info(f"🔄 开始向量化处理...")
                text_content = self._extract_text_content(knowledge)
                logger.info(f"   - 提取文本内容长度: {len(text_content)} 字符")
                
                if hasattr(self.embedding_provider, 'encode_text'):
                    logger.info(f"   - 使用 encode_text 方法生成向量")
                    vector = await self.embedding_provider.encode_text(text_content)
                elif hasattr(self.embedding_provider, 'embed'):
                    logger.info(f"   - 使用 embed 方法生成向量")
                    from .embedding_config import EmbeddingRequest, ContentType
                    request = EmbeddingRequest(content=text_content, content_type=ContentType.PURE_TEXT)
                    result = await self.embedding_provider.embed(request)
                    vector = result.embeddings[0] if result.embeddings else None
                else:
                    logger.warning(f"   - 向量化提供者不支持文本编码")
                    vector = None
                
                if vector:
                    logger.info(f"✅ 向量生成成功，维度: {len(vector)}")
                else:
                    logger.warning(f"⚠️ 向量生成失败")
            else:
                logger.info(f"ℹ️ 向量化功能已禁用")
            
            # 使用AgenticX管理器存储
            logger.info(f"💾 开始存储到知识库...")
            success = await self.agenticx_manager.store_knowledge(knowledge, vector)
            
            if success:
                # 更新统计
                with self._lock:
                    self.stats['total_knowledge'] += 1
                
                # 发布事件
                await self._publish_event('knowledge_stored', {
                    'knowledge_id': knowledge.id,
                    'type': knowledge.type.value,
                    'timestamp': get_iso_timestamp()
                })
                
                logger.info(f"🎉 知识存储成功: {knowledge.id}")
                logger.info(f"   - 总知识数量: {self.stats['total_knowledge']}")
            else:
                logger.error(f"❌ 知识存储失败: {knowledge.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"💥 存储知识时发生异常 {knowledge.id}: {e}")
            return False
    
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
    
    async def retrieve_knowledge(
        self,
        knowledge_id: str,
        use_cache: bool = True
    ) -> Optional[KnowledgeItem]:
        """检索知识"""
        try:
            logger.info(f"🔍 开始检索知识: {knowledge_id}")
            logger.info(f"   - 使用缓存: {use_cache}")
            
            # 使用AgenticX管理器检索
            logger.info(f"📖 从知识库检索...")
            knowledge = await self.agenticx_manager.retrieve_knowledge(knowledge_id)
            
            if knowledge:
                logger.info(f"✅ 知识检索成功: {knowledge_id}")
                logger.info(f"   - 标题: {knowledge.title}")
                logger.info(f"   - 类型: {knowledge.type.value}")
                logger.info(f"   - 状态: {knowledge.status.value}")
                
                # 发布事件
                await self._publish_event('knowledge_retrieved', {
                    'knowledge_id': knowledge_id,
                    'timestamp': get_iso_timestamp()
                })
            else:
                logger.warning(f"⚠️ 未找到知识: {knowledge_id}")
            
            return knowledge
            
        except Exception as e:
            logger.error(f"💥 检索知识时发生异常 {knowledge_id}: {e}")
            return None
    
    async def update_knowledge(
        self,
        knowledge: KnowledgeItem,
        validate: bool = True
    ) -> bool:
        """更新知识"""
        try:
            # 验证知识
            if validate and self.validator:
                is_valid, score, details = await self.validator.validate_knowledge(knowledge)
                self.stats['validations_performed'] += 1
                
                if is_valid:
                    self.stats['validations_passed'] += 1
                else:
                    logger.warning(f"Knowledge validation failed: {knowledge.id}")
                    return False
            
            # 更新存储
            success = await self.store.update_knowledge(knowledge)
            
            if success:
                # 更新缓存
                self.cache.put(knowledge)
                
                # 发布事件
                await self._publish_event('knowledge_updated', {
                    'knowledge_id': knowledge.id,
                    'version': knowledge.metadata.version,
                    'timestamp': get_iso_timestamp()
                })
                
                logger.debug(f"Updated knowledge: {knowledge.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update knowledge {knowledge.id}: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识"""
        try:
            # 从存储删除
            success = await self.store.delete_knowledge(knowledge_id)
            
            if success:
                # 从缓存删除
                self.cache.remove(knowledge_id)
                
                # 更新统计
                with self._lock:
                    self.stats['total_knowledge'] = max(0, self.stats['total_knowledge'] - 1)
                
                # 发布事件
                await self._publish_event('knowledge_deleted', {
                    'knowledge_id': knowledge_id,
                    'timestamp': get_iso_timestamp()
                })
                
                logger.debug(f"Deleted knowledge: {knowledge_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete knowledge {knowledge_id}: {e}")
            return False
    
    async def query_knowledge(self, request: QueryRequest) -> QueryResult:
        """查询知识"""
        try:
            logger.info(f"🔎 开始查询知识: {request.id}")
            logger.info(f"   - 查询文本: {request.query_text[:100]}..." if request.query_text and len(request.query_text) > 100 else f"   - 查询文本: {request.query_text}")
            logger.info(f"   - 查询类型: {request.query_type}")
            logger.info(f"   - 限制数量: {request.limit}")
            
            # 向量化查询文本（如果启用）
            query_vector = None
            if self.agenticx_config.vectorization_enabled and request.query_text:
                logger.info(f"🔄 开始查询文本向量化...")
                if hasattr(self.embedding_provider, 'encode_text'):
                    logger.info(f"   - 使用 encode_text 方法")
                    query_vector = await self.embedding_provider.encode_text(request.query_text)
                elif hasattr(self.embedding_provider, 'embed'):
                    logger.info(f"   - 使用 embed 方法")
                    from .embedding_config import EmbeddingRequest, ContentType
                    request_obj = EmbeddingRequest(content=request.query_text, content_type=ContentType.PURE_TEXT)
                    result = await self.embedding_provider.embed(request_obj)
                    query_vector = result.embeddings[0] if result.embeddings else None
                else:
                    logger.warning(f"   - 向量化提供者不支持查询向量化")
                    query_vector = None
                
                if query_vector:
                    logger.info(f"✅ 查询向量生成成功，维度: {len(query_vector)}")
                else:
                    logger.warning(f"⚠️ 查询向量生成失败")
            else:
                logger.info(f"ℹ️ 跳过向量化（功能禁用或无查询文本）")
            
            # 使用AgenticX管理器查询
            logger.info(f"🔍 执行知识库查询...")
            result = await self.agenticx_manager.query_knowledge(request, query_vector)
            
            logger.info(f"✅ 查询完成: {request.id}")
            logger.info(f"   - 找到结果: {len(result.items)} 条")
            logger.info(f"   - 执行时间: {result.execution_time:.3f} 秒")
            
            # 记录前几个结果的详细信息
            for i, item in enumerate(result.items[:3]):
                logger.info(f"   - 结果 {i+1}: {item.title} (相似度: {getattr(item, 'similarity_score', 'N/A')})")
            
            # 发布事件
            await self._publish_event('knowledge_queried', {
                'request_id': request.id,
                'result_count': len(result.items),
                'execution_time': result.execution_time,
                'timestamp': get_iso_timestamp()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"💥 查询知识时发生异常: {e}")
            return QueryResult(request_id=request.id)
    
    async def sync_knowledge(self, request: SyncRequest) -> SyncResult:
        """同步知识"""
        start_time = datetime.now()
        
        try:
            with self._lock:
                self.stats['sync_operations'] += 1
            
            # 执行同步逻辑
            synced_count = 0
            failed_count = 0
            conflicts = []
            
            # 这里可以实现具体的同步逻辑
            # 例如：与远程知识库同步、合并冲突等
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = SyncResult(
                request_id=request.id,
                success=failed_count == 0,
                synced_count=synced_count,
                failed_count=failed_count,
                execution_time=execution_time
            )
            
            # 更新统计
            with self._lock:
                self.stats['last_sync'] = get_iso_timestamp()
            
            # 发布事件
            await self._publish_event('knowledge_synced', {
                'request_id': request.id,
                'synced_count': synced_count,
                'failed_count': failed_count,
                'timestamp': get_iso_timestamp()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to sync knowledge: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return SyncResult(
                request_id=request.id,
                success=False,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识统计"""
        try:
            # 获取存储统计
            total_count = await self.store.get_knowledge_count()
            
            # 获取缓存统计
            cache_stats = self.cache.get_stats()
            
            # 合并统计信息
            stats = {
                **self.stats,
                'total_knowledge': total_count,
                'cache_stats': cache_stats,
                'timestamp': get_iso_timestamp()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {}
    
    def register_event_callback(self, event_type: str, callback: Callable) -> None:
        """注册事件回调"""
        self.event_callbacks[event_type].append(callback)
    
    def unregister_event_callback(self, event_type: str, callback: Callable) -> None:
        """取消注册事件回调"""
        if callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """发布事件"""
        try:
            # 调用注册的回调
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_type, data)
                    else:
                        callback(event_type, data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
            
            # 发布到EventBus - 增强健壮性检查
            event_bus = getattr(self, 'event_bus', None)  # 安全获取event_bus
            if event_bus and hasattr(event_bus, 'publish'):
                try:
                    event = Event(
                        type=f'knowledge_{event_type}',
                        source='knowledge_manager',
                        data=data
                    )
                    await event_bus.publish_async(event)
                except Exception as publish_error:
                    # 如果发布失败，记录但不抛出异常
                    logger.debug(f"Event publish failed for {event_type}: {publish_error}")
                
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
    
    async def _sync_loop(self) -> None:
        """同步循环"""
        while self._running:
            try:
                # 创建同步请求
                sync_request = SyncRequest(
                    id=str(uuid4()),
                    source_agent='auto_sync',
                    sync_type='incremental'
                )
                
                # 执行同步
                await self.sync_knowledge(sync_request)
                
                # 等待下次同步
                await asyncio.sleep(self.sync_config['sync_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)  # 出错时等待1分钟
    
    async def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("Knowledge cache cleared")
    
    async def rebuild_indexes(self) -> bool:
        """重建索引"""
        try:
            # 这里可以实现索引重建逻辑
            # 例如：重新构建搜索索引、关系索引等
            
            logger.info("Knowledge indexes rebuilt")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild indexes: {e}")
            return False
    
    async def export_knowledge(
        self,
        filters: Optional[Dict[str, Any]] = None,
        format: str = 'json'
    ) -> Optional[str]:
        """导出知识"""
        try:
            # 查询知识
            query_request = QueryRequest(
                id=str(uuid4()),
                filters=filters or {},
                limit=10000  # 大批量导出
            )
            
            result = await self.query_knowledge(query_request)
            
            if format == 'json':
                # 转换为可序列化的格式
                export_data = []
                for knowledge in result.items:
                    knowledge_dict = {
                        'id': knowledge.id,
                        'type': knowledge.type.value,
                        'source': knowledge.source.value,
                        'status': knowledge.status.value,
                        'title': knowledge.title,
                        'content': knowledge.content,
                        'description': knowledge.description,
                        'keywords': list(knowledge.keywords),
                        'context': knowledge.context,
                        'domain': knowledge.domain,
                        'scope': knowledge.scope,
                        'metadata': knowledge.metadata.__dict__,
                        'parent_id': knowledge.parent_id,
                        'children_ids': list(knowledge.children_ids),
                        'related_ids': list(knowledge.related_ids),
                        'schema_version': knowledge.schema_version,
                        'data_format': knowledge.data_format,
                        'encoding': knowledge.encoding
                    }
                    export_data.append(knowledge_dict)
                
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export knowledge: {e}")
            return None
    
    async def import_knowledge(
        self,
        data: str,
        format: str = 'json',
        validate: bool = True
    ) -> Tuple[int, int]:
        """导入知识"""
        try:
            imported_count = 0
            failed_count = 0
            
            if format == 'json':
                import_data = json.loads(data)
                
                for item_dict in import_data:
                    try:
                        # 重建知识对象
                        metadata = KnowledgeMetadata(**item_dict['metadata'])
                        
                        knowledge = KnowledgeItem(
                            id=item_dict['id'],
                            type=KnowledgeType(item_dict['type']),
                            source=KnowledgeSource(item_dict['source']),
                            status=KnowledgeStatus(item_dict['status']),
                            title=item_dict['title'],
                            content=item_dict['content'],
                            description=item_dict['description'],
                            keywords=set(item_dict['keywords']),
                            context=item_dict['context'],
                            domain=item_dict['domain'],
                            scope=item_dict['scope'],
                            metadata=metadata,
                            parent_id=item_dict.get('parent_id'),
                            children_ids=set(item_dict.get('children_ids', [])),
                            related_ids=set(item_dict.get('related_ids', [])),
                            schema_version=item_dict.get('schema_version', '1.0'),
                            data_format=item_dict.get('data_format', 'json'),
                            encoding=item_dict.get('encoding', 'utf-8')
                        )
                        
                        # 存储知识
                        success = await self.store_knowledge(knowledge, validate=validate)
                        
                        if success:
                            imported_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to import knowledge item: {e}")
                        failed_count += 1
            
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            logger.info(f"Import completed: {imported_count} success, {failed_count} failed")
            return imported_count, failed_count
            
        except Exception as e:
            logger.error(f"Failed to import knowledge: {e}")
            return 0, 0
    
    def update_sync_config(self, config: Dict[str, Any]) -> None:
        """更新同步配置"""
        self.sync_config.update(config)
        logger.info(f"Sync config updated: {config}")
    
    def get_sync_config(self) -> Dict[str, Any]:
        """获取同步配置"""
        return self.sync_config.copy()