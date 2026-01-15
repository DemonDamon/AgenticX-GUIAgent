#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Factory - Embedding服务工厂
基于AgenticX Embeddings的统一embedding服务管理

Author: AgenticX Team
Date: 2025
"""

from loguru import logger
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from agenticx.embeddings import (
        BaseEmbeddingProvider,
        BailianEmbeddingProvider,
        OpenAIEmbeddingProvider,
        SiliconFlowEmbeddingProvider,
        LiteLLMEmbeddingProvider,
        EmbeddingRouter,
        EmbeddingError
    )
except ImportError as e:
    logger.warning(f"AgenticX embeddings模块导入失败: {e}")
    # 降级到本地实现
    BaseEmbeddingProvider = None
    BailianEmbeddingProvider = None
    OpenAIEmbeddingProvider = None
    SiliconFlowEmbeddingProvider = None
    LiteLLMEmbeddingProvider = None
    EmbeddingRouter = None
    EmbeddingError = Exception

from utils import setup_logger
from .embedding_config import EmbeddingConfig, EmbeddingStrategy
# 延迟导入避免循环依赖
# from .hybrid_embedding_manager import HybridEmbeddingManager


class MockEmbeddingProvider:
    """模拟Embedding提供者（降级方案）"""
    
    def __init__(self, dimension: int = 1536, **kwargs):
        self.dimension = dimension
        self.logger = logger
        logger.warning("使用MockEmbeddingProvider，建议配置真实的embedding服务")
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """同步embedding接口"""
        import hashlib
        import numpy as np
        
        embeddings = []
        for text in texts:
            # 基于文本内容生成确定性向量
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            
            np.random.seed(seed)
            vector = np.random.normal(0, 1, self.dimension).astype(np.float32)
            normalized_vector = (vector / np.linalg.norm(vector)).tolist()
            embeddings.append(normalized_vector)
        
        return embeddings
    
    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """异步embedding接口"""
        return self.embed(texts, **kwargs)
    
    async def close(self):
        """关闭连接"""
        pass


class EmbeddingFactory:
    """Embedding服务工厂"""
    
    def __init__(self):
        self.logger = logger
        self._providers = {}
        self._router = None
    
    def create_provider(
        self, 
        config: EmbeddingConfig
    ) -> BaseEmbeddingProvider:
        """创建embedding提供者"""
        provider_type = config.provider.lower()
        
        try:
            if provider_type == "bailian" and BailianEmbeddingProvider:
                return self._create_bailian_provider(config)
            elif provider_type == "openai" and OpenAIEmbeddingProvider:
                return self._create_openai_provider(config)
            elif provider_type == "siliconflow" and SiliconFlowEmbeddingProvider:
                return self._create_siliconflow_provider(config)
            elif provider_type == "litellm" and LiteLLMEmbeddingProvider:
                return self._create_litellm_provider(config)
            else:
                logger.warning(f"不支持的embedding提供者: {provider_type}，使用Mock提供者")
                return MockEmbeddingProvider(dimension=config.dimension)
        
        except Exception as e:
            logger.error(f"创建{provider_type}提供者失败: {e}，使用Mock提供者")
            return MockEmbeddingProvider(dimension=config.dimension)
    
    def _create_bailian_provider(self, config: EmbeddingConfig) -> BailianEmbeddingProvider:
        """创建百炼embedding提供者"""
        return BailianEmbeddingProvider(
            api_key=config.api_key,
            model=config.model,
            api_url=config.api_url,
            dimension=config.dimension,
            max_tokens=config.max_tokens,
            batch_size=config.batch_size,
            timeout=config.timeout,
            retry_count=config.retry_count,
            retry_delay=config.retry_delay
        )
    
    def _create_openai_provider(self, config: EmbeddingConfig) -> OpenAIEmbeddingProvider:
        """创建OpenAI embedding提供者"""
        return OpenAIEmbeddingProvider(
            api_key=config.api_key,
            model=config.model,
            api_base=config.api_url
        )
    
    def _create_siliconflow_provider(self, config: EmbeddingConfig) -> SiliconFlowEmbeddingProvider:
        """创建SiliconFlow embedding提供者"""
        return SiliconFlowEmbeddingProvider(
            api_key=config.api_key,
            model=config.model,
            api_base=config.api_url
        )
    
    def _create_litellm_provider(self, config: EmbeddingConfig) -> LiteLLMEmbeddingProvider:
        """创建LiteLLM embedding提供者"""
        return LiteLLMEmbeddingProvider(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_url
        )
    
    def create_router(
        self, 
        configs: List[EmbeddingConfig],
        fallback_enabled: bool = True
    ) -> Optional[EmbeddingRouter]:
        """创建embedding路由器"""
        if not EmbeddingRouter:
            logger.warning("EmbeddingRouter不可用，无法创建路由器")
            return None
        
        try:
            providers = {}
            for config in configs:
                provider = self.create_provider(config)
                providers[config.provider] = provider
            
            router = EmbeddingRouter(
                providers=providers,
                fallback_enabled=fallback_enabled
            )
            
            logger.info(f"创建embedding路由器成功，包含{len(providers)}个提供者")
            return router
        
        except Exception as e:
            logger.error(f"创建embedding路由器失败: {e}")
            return None
    
    def create_hybrid_manager(
        self,
        text_config: EmbeddingConfig,
        multimodal_config: Optional[EmbeddingConfig] = None,
        strategy: Optional[EmbeddingStrategy] = None
    ):
        """创建混合embedding管理器"""
        try:
            # 延迟导入避免循环依赖
            from .hybrid_embedding_manager import HybridEmbeddingManager
            
            # 创建文本embedding提供者
            text_provider = self.create_provider(text_config)
            
            # 创建多模态embedding提供者
            if multimodal_config:
                multimodal_provider = self.create_provider(multimodal_config)
            else:
                # 为多模态创建独立的provider实例，配置多模态模型
                multimodal_provider = self.create_provider(EmbeddingConfig(
                    provider=text_config.provider,
                    model="multimodal-embedding-v1",  # 使用多模态模型
                    api_key=text_config.api_key,
                    api_url=text_config.api_url,
                    dimension=text_config.dimension,
                    max_tokens=text_config.max_tokens,
                    batch_size=10,  # 多模态批次较小
                    timeout=text_config.timeout,
                    retry_count=text_config.retry_count,
                    retry_delay=text_config.retry_delay
                ))
                # 确保多模态provider使用正确的模型
                if hasattr(multimodal_provider, 'multimodal_model'):
                    multimodal_provider.multimodal_model = "multimodal-embedding-v1"
            
            # 创建策略
            if strategy is None:
                strategy = EmbeddingStrategy()
            
            # 创建混合管理器
            hybrid_manager = HybridEmbeddingManager(
                text_provider=text_provider,
                multimodal_provider=multimodal_provider,
                strategy=strategy
            )
            
            logger.info("混合embedding管理器创建成功")
            return hybrid_manager
            
        except Exception as e:
            logger.error(f"创建混合embedding管理器失败: {e}")
            raise
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'EmbeddingFactory':
        """从配置字典创建工厂"""
        factory = cls()
        
        # 解析配置
        provider = config_dict.get('provider', 'bailian')
        provider_config = config_dict.get(provider, {})
        
        # 创建配置对象
        embedding_config = EmbeddingConfig(
            provider=provider,
            model=provider_config.get('model', 'text-embedding-v4'),  # 默认使用v4
            api_key=provider_config.get('api_key', ''),
            api_url=provider_config.get('api_url', ''),
            dimension=provider_config.get('dimension', 1536),
            max_tokens=provider_config.get('max_tokens', 8192),
            batch_size=provider_config.get('batch_size', 100),
            timeout=provider_config.get('timeout', 30),
            retry_count=provider_config.get('retry_count', 3),
            retry_delay=provider_config.get('retry_delay', 1.0)
        )
        
        return factory
    
    @classmethod
    def create_hybrid_from_config(cls, config_dict: Dict[str, Any]):
        """从配置字典创建混合embedding管理器"""
        # 延迟导入避免循环依赖
        from .hybrid_embedding_manager import HybridEmbeddingManager
        
        factory = cls()
        
        # 解析基础配置
        provider = config_dict.get('provider', 'bailian')
        provider_config = config_dict.get(provider, {})
        
        # 创建文本embedding配置
        text_config = EmbeddingConfig(
            provider=provider,
            model=provider_config.get('model', 'text-embedding-v4'),
            api_key=provider_config.get('api_key', ''),
            api_url=provider_config.get('api_url', ''),
            dimension=provider_config.get('dimension', 1536),
            max_tokens=provider_config.get('max_tokens', 8192),
            batch_size=provider_config.get('batch_size', 100),
            timeout=provider_config.get('timeout', 30),
            retry_count=provider_config.get('retry_count', 3),
            retry_delay=provider_config.get('retry_delay', 1.0)
        )
        
        # 创建多模态embedding配置
        multimodal_config = None
        if 'multimodal' in config_dict:
            multimodal_settings = config_dict['multimodal']
            multimodal_config = EmbeddingConfig(
                provider=provider,
                model=multimodal_settings.get('multimodal_model', 'multimodal-embedding-v1'),
                api_key=provider_config.get('api_key', ''),
                api_url=provider_config.get('api_url', ''),
                dimension=multimodal_settings.get('unified_dimension', 1536),
                max_tokens=provider_config.get('max_tokens', 8192),
                batch_size=multimodal_settings.get('batch_size', 10),  # 多模态批次较小
                timeout=provider_config.get('timeout', 30),
                retry_count=provider_config.get('retry_count', 3),
                retry_delay=provider_config.get('retry_delay', 1.0)
            )
        
        # 创建策略配置
        strategy = EmbeddingStrategy()
        if 'performance' in config_dict:
            perf_config = config_dict['performance']
            strategy.max_batch_size_text = perf_config.get('max_batch_size_text', 100)
            strategy.max_batch_size_multimodal = perf_config.get('max_batch_size_multimodal', 10)
            strategy.timeout_seconds = perf_config.get('request_timeout', 30)
        
        if 'cache' in config_dict:
            cache_config = config_dict['cache']
            strategy.cache_enabled = cache_config.get('enabled', True)
            strategy.cache_ttl_text = cache_config.get('ttl', 3600)
            strategy.cache_ttl_multimodal = cache_config.get('ttl', 3600) * 2  # 多模态缓存更久
            strategy.cache_max_entries = cache_config.get('max_entries', 10000)
        
        if 'fallback' in config_dict:
            fallback_config = config_dict['fallback']
            strategy.fallback_enabled = fallback_config.get('enabled', True)
            strategy.fallback_to_text = True  # 总是降级到文本
        
        # 创建混合管理器
        return factory.create_hybrid_manager(text_config, multimodal_config, strategy)


class CachedEmbeddingProvider:
    """带缓存的Embedding提供者包装器"""
    
    def __init__(
        self, 
        provider: BaseEmbeddingProvider,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        max_cache_entries: int = 10000
    ):
        self.provider = provider
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.max_cache_entries = max_cache_entries
        self.logger = logger
        
        # 简单的内存缓存
        self._cache = {}
        self._cache_timestamps = {}
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._cache_timestamps:
            return False
        
        import time
        timestamp = self._cache_timestamps[cache_key]
        return (time.time() - timestamp) < self.cache_ttl
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        if len(self._cache) <= self.max_cache_entries:
            return
        
        import time
        current_time = time.time()
        
        # 移除过期项
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if (current_time - timestamp) >= self.cache_ttl
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        
        # 如果还是太多，移除最老的项
        if len(self._cache) > self.max_cache_entries:
            sorted_items = sorted(
                self._cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            remove_count = len(self._cache) - self.max_cache_entries
            for key, _ in sorted_items[:remove_count]:
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """带缓存的embedding"""
        if not self.cache_enabled:
            return self.provider.embed(texts, **kwargs)
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if self._is_cache_valid(cache_key):
                embeddings.append(self._cache[cache_key])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 处理未缓存的文本
        if uncached_texts:
            try:
                uncached_embeddings = self.provider.embed(uncached_texts, **kwargs)
                
                # 更新缓存和结果
                import time
                current_time = time.time()
                
                for i, embedding in enumerate(uncached_embeddings):
                    result_index = uncached_indices[i]
                    embeddings[result_index] = embedding
                    
                    # 缓存结果
                    cache_key = self._get_cache_key(uncached_texts[i])
                    self._cache[cache_key] = embedding
                    self._cache_timestamps[cache_key] = current_time
                
                # 清理缓存
                self._cleanup_cache()
                
            except Exception as e:
                logger.error(f"Embedding处理失败: {e}")
                raise
        
        return embeddings
    
    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """异步embedding（简单实现，实际应该异步处理缓存）"""
        return self.embed(texts, **kwargs)
    
    async def close(self):
        """关闭提供者"""
        if hasattr(self.provider, 'close'):
            await self.provider.close()
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Embedding缓存已清空")