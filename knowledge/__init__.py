#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Knowledge Pool Module
基于AgenticX框架的共享知识池模块：提供跨智能体的知识管理和共享机制

重构说明：
- 基于AgenticX框架重构，避免重复实现
- 使用AgenticX的Memory和Storage组件
- 集成AgenticX的事件系统进行知识同步
- 提供现代化的知识管理和检索功能

Author: AgenticX Team
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "AgenticX Team"
__description__ = "AgenticX-GUIAgent共享知识池模块 - 基于AgenticX框架重构"

# 导入核心组件
from .knowledge_pool import KnowledgePool
from .knowledge_manager import KnowledgeManager
from .knowledge_store import (
    KnowledgeStoreInterface,
    InMemoryKnowledgeStore,
    SQLiteKnowledgeStore,
    KnowledgeStoreFactory
)

# 导入AgenticX适配器
from .agenticx_adapter import (
    AgenticXKnowledgeManager,
    AgenticXConfig,
    KnowledgeToVectorAdapter,
    MockEmbeddingProvider
)

# 导入数据结构
from .knowledge_types import (
    KnowledgeItem,
    KnowledgeType,
    KnowledgeSource,
    KnowledgeStatus,
    KnowledgeMetadata,
    KnowledgeRelation,
    KnowledgeGraph,
    QueryRequest,
    QueryResult,
    SyncRequest,
    SyncResult
)

__all__ = [
    # 核心组件
    "KnowledgePool",
    "KnowledgeManager",
    "KnowledgeStoreInterface",
    "InMemoryKnowledgeStore",
    "SQLiteKnowledgeStore",
    "KnowledgeStoreFactory",
    
    # AgenticX适配器
    "AgenticXKnowledgeManager",
    "AgenticXConfig",
    "KnowledgeToVectorAdapter",
    "MockEmbeddingProvider",
    
    # 数据结构
    "KnowledgeItem",
    "KnowledgeType",
    "KnowledgeSource",
    "KnowledgeStatus",
    "KnowledgeMetadata",
    "KnowledgeRelation",
    "KnowledgeGraph",
    "QueryRequest",
    "QueryResult",
    "SyncRequest",
    "SyncResult"
]