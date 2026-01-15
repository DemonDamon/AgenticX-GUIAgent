#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Configuration - Embedding配置模块
独立的配置类，避免循环导入

Author: AgenticX Team
Date: 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union


class EmbeddingType(Enum):
    """Embedding类型"""
    TEXT = "text"
    MULTIMODAL = "multimodal"
    AUTO = "auto"


class ContentType(Enum):
    """内容类型"""
    PURE_TEXT = "pure_text"
    TEXT_WITH_IMAGES = "text_with_images"
    IMAGES_ONLY = "images_only"
    MIXED_MEDIA = "mixed_media"
    AUTO = "auto"  # 自动检测
    UNKNOWN = "unknown"
    MULTIMODAL = "multimodal"  # 强制使用多模态模型


@dataclass
class EmbeddingConfig:
    """Embedding配置"""
    provider: str = "bailian"
    model: str = "text-embedding-v4"  # 默认使用v4，支持维度参数
    api_key: str = ""
    api_url: str = ""
    dimension: int = 1536
    max_tokens: int = 8192
    batch_size: int = 100
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_ttl: int = 3600


@dataclass
class EmbeddingRequest:
    """Embedding请求"""
    content: Union[str, List[str], List[Dict[str, Any]]]
    content_type: ContentType = ContentType.UNKNOWN
    priority: str = "normal"  # high, normal, low
    cache_key: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class EmbeddingResult:
    """Embedding结果"""
    embeddings: List[List[float]]
    embedding_type: EmbeddingType
    cache_hit: bool = False
    processing_time: float = 0.0
    cost_estimate: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class EmbeddingStrategy:
    """Embedding策略配置"""
    # 成本控制
    max_cost_per_request: float = 1.0
    cost_threshold_multimodal: float = 0.5
    
    # 性能控制
    max_batch_size_text: int = 100
    max_batch_size_multimodal: int = 10
    timeout_seconds: int = 30
    
    # 缓存策略
    cache_enabled: bool = True
    cache_ttl_text: int = 3600  # 1小时
    cache_ttl_multimodal: int = 7200  # 2小时
    cache_max_entries: int = 10000
    
    # 降级策略
    fallback_enabled: bool = True
    fallback_to_text: bool = True
    max_retries: int = 3
    
    # 智能选择策略
    auto_detect_content_type: bool = True
    prefer_multimodal_for_gui: bool = True
    text_only_threshold: float = 0.8  # 纯文本比例阈值