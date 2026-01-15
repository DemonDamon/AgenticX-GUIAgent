#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Embedding Manager - 混合embedding策略管理器
智能选择文本embedding或多模态embedding，优化成本和性能

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
import hashlib
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from utils import setup_logger
from .embedding_config import (
    EmbeddingType, ContentType, EmbeddingRequest, EmbeddingResult, EmbeddingStrategy
)


class ContentAnalyzer:
    """内容分析器"""
    
    def __init__(self):
        self.logger = logger
    
    def analyze_content(self, content: Union[str, List[str], List[Dict[str, Any]]]) -> ContentType:
        """分析内容类型"""
        try:
            if isinstance(content, str):
                return ContentType.PURE_TEXT
            
            elif isinstance(content, list):
                if not content:
                    return ContentType.UNKNOWN
                
                # 检查第一个元素的类型
                first_item = content[0]
                
                if isinstance(first_item, str):
                    return ContentType.PURE_TEXT
                
                elif isinstance(first_item, dict):
                    return self._analyze_multimodal_content(content)
            
            return ContentType.UNKNOWN
            
        except Exception as e:
            logger.error(f"内容分析失败: {e}")
            return ContentType.UNKNOWN
    
    def _analyze_multimodal_content(self, content: List[Dict[str, Any]]) -> ContentType:
        """分析多模态内容"""
        text_count = 0
        image_count = 0
        video_count = 0
        
        for item in content:
            if isinstance(item, dict):
                if 'text' in item:
                    text_count += 1
                if 'image' in item or 'image_url' in item:
                    image_count += 1
                if 'video' in item:
                    video_count += 1
            elif isinstance(item, str):
                text_count += 1
        
        # 如果包含图片或视频，就认为是多模态内容
        if image_count > 0 or video_count > 0:
            if text_count == 0:
                return ContentType.IMAGES_ONLY
            else:
                return ContentType.MIXED_MEDIA
        else:
            return ContentType.PURE_TEXT
    
    def estimate_content_complexity(self, content: Union[str, List[str], List[Dict[str, Any]]]) -> float:
        """估算内容复杂度 (0-1)"""
        try:
            if isinstance(content, str):
                # 基于文本长度和特殊字符
                length_score = min(len(content) / 1000, 1.0)
                special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
                special_score = min(special_chars / 50, 1.0)
                return (length_score + special_score) / 2
            
            elif isinstance(content, list):
                if not content:
                    return 0.0
                
                if isinstance(content[0], str):
                    # 文本列表
                    total_length = sum(len(text) for text in content)
                    return min(total_length / 5000, 1.0)
                
                elif isinstance(content[0], dict):
                    # 多模态内容
                    base_complexity = 0.5
                    
                    # 图片和视频增加复杂度
                    for item in content:
                        if 'image' in item:
                            base_complexity += 0.2
                        if 'video' in item:
                            base_complexity += 0.3
                    
                    return min(base_complexity, 1.0)
            
            return 0.5
            
        except Exception:
            return 0.5


class CostEstimator:
    """成本估算器"""
    
    def __init__(self):
        self.logger = logger
        
        # 成本配置（示例价格，实际需要根据API定价调整）
        self.cost_config = {
            'text_embedding': {
                'price_per_1k_tokens': 0.0001,  # $0.0001 per 1K tokens
                'avg_tokens_per_char': 0.25
            },
            'multimodal_embedding': {
                'price_per_image': 0.001,  # $0.001 per image
                'price_per_video': 0.005,  # $0.005 per video
                'price_per_text': 0.0002   # $0.0002 per text in multimodal
            }
        }
    
    def estimate_text_cost(self, content: Union[str, List[str]]) -> float:
        """估算文本embedding成本"""
        try:
            if isinstance(content, str):
                char_count = len(content)
            else:
                char_count = sum(len(text) for text in content)
            
            token_count = char_count * self.cost_config['text_embedding']['avg_tokens_per_char']
            cost = (token_count / 1000) * self.cost_config['text_embedding']['price_per_1k_tokens']
            
            return cost
            
        except Exception as e:
            logger.error(f"文本成本估算失败: {e}")
            return 0.0
    
    def estimate_multimodal_cost(self, content: List[Dict[str, Any]]) -> float:
        """估算多模态embedding成本"""
        try:
            total_cost = 0.0
            
            for item in content:
                if 'text' in item:
                    total_cost += self.cost_config['multimodal_embedding']['price_per_text']
                if 'image' in item:
                    total_cost += self.cost_config['multimodal_embedding']['price_per_image']
                if 'video' in item:
                    total_cost += self.cost_config['multimodal_embedding']['price_per_video']
            
            return total_cost
            
        except Exception as e:
            logger.error(f"多模态成本估算失败: {e}")
            return 0.0


class HybridEmbeddingCache:
    """混合embedding缓存"""
    
    def __init__(self, strategy: EmbeddingStrategy):
        self.strategy = strategy
        self.logger = logger
        
        # 分层缓存
        self._text_cache = {}  # 文本embedding缓存
        self._multimodal_cache = {}  # 多模态embedding缓存
        self._cache_timestamps = {}  # 缓存时间戳
        self._cache_access_count = defaultdict(int)  # 访问计数
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def _generate_cache_key(self, content: Any, embedding_type: EmbeddingType) -> str:
        """生成缓存键"""
        try:
            content_str = str(content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            return f"{embedding_type.value}:{content_hash}"
        except Exception:
            return f"{embedding_type.value}:{hash(str(content))}"
    
    def get(self, content: Any, embedding_type: EmbeddingType) -> Optional[List[List[float]]]:
        """获取缓存的embedding"""
        if not self.strategy.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(content, embedding_type)
        self.stats['total_requests'] += 1
        
        # 选择缓存存储
        cache_store = self._text_cache if embedding_type == EmbeddingType.TEXT else self._multimodal_cache
        ttl = self.strategy.cache_ttl_text if embedding_type == EmbeddingType.TEXT else self.strategy.cache_ttl_multimodal
        
        if cache_key in cache_store:
            # 检查是否过期
            timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < ttl:
                self.stats['hits'] += 1
                self._cache_access_count[cache_key] += 1
                return cache_store[cache_key]
            else:
                # 过期，删除
                self._remove_from_cache(cache_key, embedding_type)
        
        self.stats['misses'] += 1
        return None
    
    def put(self, content: Any, embedding_type: EmbeddingType, embeddings: List[List[float]]) -> None:
        """缓存embedding结果"""
        if not self.strategy.cache_enabled:
            return
        
        cache_key = self._generate_cache_key(content, embedding_type)
        
        # 选择缓存存储
        cache_store = self._text_cache if embedding_type == EmbeddingType.TEXT else self._multimodal_cache
        
        # 检查缓存大小限制
        total_cache_size = len(self._text_cache) + len(self._multimodal_cache)
        if total_cache_size >= self.strategy.cache_max_entries:
            self._evict_lru_entries()
        
        # 存储到缓存
        cache_store[cache_key] = embeddings
        self._cache_timestamps[cache_key] = time.time()
        self._cache_access_count[cache_key] = 1
    
    def _remove_from_cache(self, cache_key: str, embedding_type: EmbeddingType) -> None:
        """从缓存中移除"""
        cache_store = self._text_cache if embedding_type == EmbeddingType.TEXT else self._multimodal_cache
        
        cache_store.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)
        self._cache_access_count.pop(cache_key, None)
    
    def _evict_lru_entries(self) -> None:
        """驱逐最少使用的缓存项"""
        # 按访问次数和时间排序，移除最少使用的项
        all_keys = list(self._cache_access_count.keys())
        
        # 按访问次数和时间戳排序
        sorted_keys = sorted(all_keys, key=lambda k: (
            self._cache_access_count[k],
            self._cache_timestamps.get(k, 0)
        ))
        
        # 移除前10%的项
        evict_count = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:evict_count]:
            # 确定embedding类型
            embedding_type = EmbeddingType.TEXT if key.startswith('text:') else EmbeddingType.MULTIMODAL
            self._remove_from_cache(key, embedding_type)
            self.stats['evictions'] += 1
    
    def clear(self) -> None:
        """清空缓存"""
        self._text_cache.clear()
        self._multimodal_cache.clear()
        self._cache_timestamps.clear()
        self._cache_access_count.clear()
        
        logger.info("混合embedding缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        hit_rate = self.stats['hits'] / max(self.stats['total_requests'], 1)
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'text_cache_size': len(self._text_cache),
            'multimodal_cache_size': len(self._multimodal_cache),
            'total_cache_size': len(self._text_cache) + len(self._multimodal_cache)
        }


class HybridEmbeddingManager:
    """混合embedding管理器"""
    
    def __init__(
        self,
        text_provider: Any,
        multimodal_provider: Any,
        strategy: Optional[EmbeddingStrategy] = None
    ):
        self.text_provider = text_provider
        self.multimodal_provider = multimodal_provider
        self.strategy = strategy or EmbeddingStrategy()
        self.logger = logger
        
        # 组件
        self.content_analyzer = ContentAnalyzer()
        self.cost_estimator = CostEstimator()
        self.cache = HybridEmbeddingCache(self.strategy)
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'text_requests': 0,
            'multimodal_requests': 0,
            'cache_hits': 0,
            'fallback_count': 0,
            'total_cost': 0.0,
            'avg_processing_time': 0.0
        }
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """智能embedding处理"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 详细调试日志
            print(f"\n🔍 混合Embedding处理开始:")
            print(f"原始内容: {request.content}")
            print(f"原始内容类型: {request.content_type}")
            
            # 1. 分析内容类型
            if (request.content_type == ContentType.UNKNOWN or request.content_type == ContentType.AUTO) and self.strategy.auto_detect_content_type:
                request.content_type = self.content_analyzer.analyze_content(request.content)
                print(f"内容分析结果: {request.content_type}")
            
            # 2. 选择embedding策略
            embedding_type = self._select_embedding_type(request)
            print(f"选择的embedding类型: {embedding_type}")
            print(f"策略配置: prefer_multimodal_for_gui={self.strategy.prefer_multimodal_for_gui}")
            print(f"成本阈值: {self.strategy.cost_threshold_multimodal}")
            
            # 3. 检查缓存
            cached_result = self.cache.get(request.content, embedding_type)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                processing_time = time.time() - start_time
                
                return EmbeddingResult(
                    embeddings=cached_result,
                    embedding_type=embedding_type,
                    cache_hit=True,
                    processing_time=processing_time,
                    cost_estimate=0.0
                )
            
            # 4. 执行embedding
            print(f"\n📡 开始执行embedding:")
            print(f"使用的provider类型: {type(self.text_provider).__name__ if embedding_type == EmbeddingType.TEXT else type(self.multimodal_provider).__name__}")
            print(f"text_provider有aembed_multimodal方法: {hasattr(self.text_provider, 'aembed_multimodal')}")
            print(f"multimodal_provider有aembed_multimodal方法: {hasattr(self.multimodal_provider, 'aembed_multimodal')}")
            print(f"text_provider模型: {getattr(self.text_provider, 'model', 'N/A')}")
            print(f"multimodal_provider模型: {getattr(self.multimodal_provider, 'model', 'N/A')}")
            
            embeddings, cost = await self._execute_embedding(request.content, embedding_type)
            
            # 打印API响应详情
            print(f"\n📊 API响应详情:")
            print(f"返回的embedding数量: {len(embeddings)}")
            if embeddings:
                print(f"第一个embedding维度: {len(embeddings[0])}")
                print(f"第一个embedding前5个值: {embeddings[0][:5]}")
                if len(embeddings) > 1:
                    print(f"第二个embedding前5个值: {embeddings[1][:5]}")
            print(f"估算成本: {cost}")
            
            # 5. 缓存结果
            self.cache.put(request.content, embedding_type, embeddings)
            
            # 6. 更新统计
            processing_time = time.time() - start_time
            self.stats['total_cost'] += cost
            self._update_processing_time_stats(processing_time)
            
            if embedding_type == EmbeddingType.TEXT:
                self.stats['text_requests'] += 1
            else:
                self.stats['multimodal_requests'] += 1
            
            return EmbeddingResult(
                embeddings=embeddings,
                embedding_type=embedding_type,
                cache_hit=False,
                processing_time=processing_time,
                cost_estimate=cost
            )
            
        except Exception as e:
            logger.error(f"Embedding处理失败: {e}")
            
            # 尝试降级策略
            if self.strategy.fallback_enabled:
                return await self._fallback_embedding(request, start_time)
            else:
                raise
    
    def _select_embedding_type(self, request: EmbeddingRequest) -> EmbeddingType:
        """选择embedding类型"""
        content_type = request.content_type
        
        # 强制指定类型的情况
        if hasattr(request, 'force_embedding_type'):
            return request.force_embedding_type
        
        # 基于内容类型选择
        if content_type == ContentType.PURE_TEXT:
            return EmbeddingType.TEXT
        elif content_type == ContentType.MULTIMODAL:
            # 强制使用多模态模型
            return EmbeddingType.MULTIMODAL
        elif content_type in [ContentType.IMAGES_ONLY, ContentType.MIXED_MEDIA]:
            return EmbeddingType.MULTIMODAL
        elif content_type == ContentType.TEXT_WITH_IMAGES:
            # 基于策略决定
            if self.strategy.prefer_multimodal_for_gui:
                return EmbeddingType.MULTIMODAL
            else:
                # 基于成本考虑
                text_cost = self.cost_estimator.estimate_text_cost(request.content)
                multimodal_cost = self.cost_estimator.estimate_multimodal_cost(request.content)
                
                if multimodal_cost <= self.strategy.cost_threshold_multimodal:
                    return EmbeddingType.MULTIMODAL
                else:
                    return EmbeddingType.TEXT
        
        # 默认使用文本embedding
        return EmbeddingType.TEXT
    
    async def _execute_embedding(
        self, 
        content: Any, 
        embedding_type: EmbeddingType
    ) -> Tuple[List[List[float]], float]:
        """执行embedding"""
        if embedding_type == EmbeddingType.TEXT:
            # 文本embedding
            if isinstance(content, str):
                embeddings = await self.text_provider.aembed([content])
            elif isinstance(content, list):
                # 检查是否是多模态格式的内容
                if content and isinstance(content[0], dict):
                    # 从多模态内容中提取文本
                    text_content = self._extract_text_from_multimodal(content)
                    embeddings = await self.text_provider.aembed([text_content])
                else:
                    # 纯文本列表
                    embeddings = await self.text_provider.aembed(content)
            else:
                # 其他格式，尝试转换为字符串
                embeddings = await self.text_provider.aembed([str(content)])
            
            cost = self.cost_estimator.estimate_text_cost(content)
            
        else:
            # 多模态embedding
            if hasattr(self.multimodal_provider, 'aembed_multimodal'):
                # 确保内容格式正确
                if isinstance(content, list):
                    if all(isinstance(item, dict) for item in content):
                        embeddings = await self.multimodal_provider.aembed_multimodal(content)
                    else:
                        # 将字符串列表转换为多模态格式
                        formatted_content = [{'text': item} if isinstance(item, str) else item for item in content]
                        embeddings = await self.multimodal_provider.aembed_multimodal(formatted_content)
                else:
                    # 单个内容转换为列表
                    formatted_content = [{'text': content} if isinstance(content, str) else content]
                    embeddings = await self.multimodal_provider.aembed_multimodal(formatted_content)
            else:
                # 降级到文本embedding
                logger.warning("多模态provider不支持aembed_multimodal，降级到文本embedding")
                text_content = self._extract_text_from_multimodal(content if isinstance(content, list) else [content])
                embeddings = await self.text_provider.aembed([text_content])
                embedding_type = EmbeddingType.TEXT
            
            cost = self.cost_estimator.estimate_multimodal_cost(content)
        
        return embeddings, cost
    
    def _extract_text_from_multimodal(self, content: List[Dict[str, Any]]) -> str:
        """从多模态内容中提取文本"""
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if 'text' in item:
                    text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
        return ' '.join(text_parts) if text_parts else ''
    
    async def _fallback_embedding(self, request: EmbeddingRequest, start_time: float) -> EmbeddingResult:
        """降级embedding处理"""
        self.stats['fallback_count'] += 1
        
        try:
            if self.strategy.fallback_to_text:
                # 降级到文本embedding
                if isinstance(request.content, list) and isinstance(request.content[0], dict):
                    # 多模态内容转文本
                    text_content = self._extract_text_from_multimodal(request.content)
                    embeddings = await self.text_provider.aembed([text_content])
                else:
                    # 直接使用文本embedding
                    embeddings = await self.text_provider.aembed(request.content)
                
                processing_time = time.time() - start_time
                cost = self.cost_estimator.estimate_text_cost(request.content)
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    embedding_type=EmbeddingType.TEXT,
                    cache_hit=False,
                    processing_time=processing_time,
                    cost_estimate=cost,
                    metadata={'fallback': True}
                )
            else:
                raise Exception("降级策略未启用")
                
        except Exception as e:
            logger.error(f"降级embedding也失败: {e}")
            raise
    
    def _update_processing_time_stats(self, processing_time: float) -> None:
        """更新处理时间统计"""
        current_avg = self.stats['avg_processing_time']
        total_requests = self.stats['total_requests']
        
        # 计算新的平均处理时间
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.stats['avg_processing_time'] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cache_stats = self.cache.get_stats()
        
        return {
            **self.stats,
            'cache_stats': cache_stats,
            'hit_rate': cache_stats['hit_rate'],
            'cost_per_request': self.stats['total_cost'] / max(self.stats['total_requests'], 1)
        }
    
    def update_strategy(self, strategy: EmbeddingStrategy) -> None:
        """更新策略配置"""
        self.strategy = strategy
        self.cache.strategy = strategy
        logger.info("混合embedding策略已更新")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    async def close(self) -> None:
        """关闭管理器"""
        if hasattr(self.text_provider, 'close'):
            await self.text_provider.close()
        if hasattr(self.multimodal_provider, 'close'):
            await self.multimodal_provider.close()
        
        logger.info("混合embedding管理器已关闭")