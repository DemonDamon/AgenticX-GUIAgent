#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Knowledge Types
基于AgenticX框架的知识池数据类型定义：定义知识项、查询、同步等核心数据结构

重构说明：
- 基于AgenticX的数据模型重构
- 集成AgenticX的事件和消息系统
- 使用现代化的数据结构和类型注解
- 提供与AgenticX框架的无缝集成

Author: AgenticX Team
Date: 2025
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Callable, Awaitable
)
from uuid import uuid4


class KnowledgeType(Enum):
    """知识类型"""
    FACTUAL = "factual"  # 事实性知识
    PROCEDURAL = "procedural"  # 程序性知识
    EXPERIENTIAL = "experiential"  # 经验性知识
    CONTEXTUAL = "contextual"  # 上下文知识
    PATTERN = "pattern"  # 模式知识
    RULE = "rule"  # 规则知识
    STRATEGY = "strategy"  # 策略知识
    METADATA = "metadata"  # 元数据知识
    TEMPORAL = "temporal"  # 时序知识
    CAUSAL = "causal"  # 因果知识
    SEMANTIC = "semantic"  # 语义知识
    STRUCTURAL = "structural"  # 结构知识


class KnowledgeSource(Enum):
    """知识来源"""
    AGENT_EXPERIENCE = "agent_experience"  # 智能体经验
    USER_INPUT = "user_input"  # 用户输入
    EXTERNAL_API = "external_api"  # 外部API
    LEARNING_PROCESS = "learning_process"  # 学习过程
    REFLECTION = "reflection"  # 反思过程
    SYNTHESIS = "synthesis"  # 合成过程
    OPTIMIZATION = "optimization"  # 优化过程
    ERROR_HANDLING = "error_handling"  # 错误处理
    COLLABORATION = "collaboration"  # 协作过程
    MANUAL_INPUT = "manual_input"  # 手动输入
    AUTOMATED_EXTRACTION = "automated_extraction"  # 自动提取
    KNOWLEDGE_FUSION = "knowledge_fusion"  # 知识融合


class KnowledgeStatus(Enum):
    """知识状态"""
    DRAFT = "draft"  # 草稿
    PENDING = "pending"  # 待审核
    VALIDATED = "validated"  # 已验证
    ACTIVE = "active"  # 活跃
    DEPRECATED = "deprecated"  # 已弃用
    ARCHIVED = "archived"  # 已归档
    CONFLICTED = "conflicted"  # 冲突
    MERGED = "merged"  # 已合并
    DELETED = "deleted"  # 已删除


class RelationType(Enum):
    """关系类型"""
    DEPENDS_ON = "depends_on"  # 依赖关系
    CONFLICTS_WITH = "conflicts_with"  # 冲突关系
    SUPPORTS = "supports"  # 支持关系
    EXTENDS = "extends"  # 扩展关系
    REPLACES = "replaces"  # 替换关系
    SIMILAR_TO = "similar_to"  # 相似关系
    PART_OF = "part_of"  # 部分关系
    CONTAINS = "contains"  # 包含关系
    PRECEDES = "precedes"  # 前置关系
    FOLLOWS = "follows"  # 后续关系
    CAUSES = "causes"  # 因果关系
    CORRELATES_WITH = "correlates_with"  # 相关关系


@dataclass
class KnowledgeMetadata:
    """知识元数据"""
    # 基础信息
    created_at: str
    updated_at: str
    created_by: str
    updated_by: str
    version: int = 1
    
    # 质量指标
    confidence: float = 0.5  # 置信度 (0-1)
    reliability: float = 0.5  # 可靠性 (0-1)
    accuracy: float = 0.5  # 准确性 (0-1)
    completeness: float = 0.5  # 完整性 (0-1)
    freshness: float = 1.0  # 新鲜度 (0-1)
    
    # 使用统计
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_accessed: Optional[str] = None
    last_success: Optional[str] = None
    last_failure: Optional[str] = None
    
    # 验证信息
    validation_status: str = "pending"
    validation_score: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    # 标签和分类
    tags: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    priority: int = 5  # 优先级 (1-10)
    
    # 生命周期
    expiry_date: Optional[str] = None
    retention_period: Optional[int] = None  # 保留期（天）
    
    # 扩展属性
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeItem:
    """知识项"""
    # 基础标识
    id: str = field(default_factory=lambda: str(uuid4()))
    type: KnowledgeType = KnowledgeType.FACTUAL
    source: KnowledgeSource = KnowledgeSource.AGENT_EXPERIENCE
    status: KnowledgeStatus = KnowledgeStatus.DRAFT
    
    # 核心内容
    title: str = ""
    content: Any = None
    description: str = ""
    keywords: Set[str] = field(default_factory=set)
    
    # 上下文信息
    context: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    scope: str = "local"  # local, global, shared
    
    # 元数据
    metadata: KnowledgeMetadata = field(default_factory=KnowledgeMetadata)
    
    # 关系信息
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    related_ids: Set[str] = field(default_factory=set)
    
    # 结构化数据
    schema_version: str = "1.0"
    data_format: str = "json"  # json, text, binary, structured
    encoding: str = "utf-8"
    
    def __post_init__(self):
        """后初始化处理"""
        if not self.metadata.created_at:
            from utils import get_iso_timestamp
            self.metadata.created_at = get_iso_timestamp()
            self.metadata.updated_at = self.metadata.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source.value,
            "status": self.status.value,
            "title": self.title,
            "content": self.content,
            "description": self.description,
            "keywords": list(self.keywords),
            "context": self.context,
            "domain": self.domain,
            "scope": self.scope,
            "metadata": {
                "created_at": self.metadata.created_at,
                "updated_at": self.metadata.updated_at,
                "created_by": self.metadata.created_by,
                "updated_by": self.metadata.updated_by,
                "version": self.metadata.version,
                "confidence": self.metadata.confidence,
                "reliability": self.metadata.reliability,
                "accuracy": self.metadata.accuracy,
                "completeness": self.metadata.completeness,
                "freshness": self.metadata.freshness,
                "access_count": self.metadata.access_count,
                "success_count": self.metadata.success_count,
                "failure_count": self.metadata.failure_count,
                "last_accessed": self.metadata.last_accessed,
                "last_success": self.metadata.last_success,
                "last_failure": self.metadata.last_failure,
                "validation_status": self.metadata.validation_status,
                "validation_score": self.metadata.validation_score,
                "validation_details": self.metadata.validation_details,
                "tags": list(self.metadata.tags),
                "categories": list(self.metadata.categories),
                "priority": self.metadata.priority,
                "expiry_date": self.metadata.expiry_date,
                "retention_period": self.metadata.retention_period,
                "custom_attributes": self.metadata.custom_attributes
            },
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "related_ids": list(self.related_ids),
            "schema_version": self.schema_version,
            "data_format": self.data_format,
            "encoding": self.encoding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """从字典创建知识项"""
        metadata_data = data.get("metadata", {})
        metadata = KnowledgeMetadata(
            created_at=metadata_data.get("created_at", ""),
            updated_at=metadata_data.get("updated_at", ""),
            created_by=metadata_data.get("created_by", ""),
            updated_by=metadata_data.get("updated_by", ""),
            version=metadata_data.get("version", 1),
            confidence=metadata_data.get("confidence", 0.5),
            reliability=metadata_data.get("reliability", 0.5),
            accuracy=metadata_data.get("accuracy", 0.5),
            completeness=metadata_data.get("completeness", 0.5),
            freshness=metadata_data.get("freshness", 1.0),
            access_count=metadata_data.get("access_count", 0),
            success_count=metadata_data.get("success_count", 0),
            failure_count=metadata_data.get("failure_count", 0),
            last_accessed=metadata_data.get("last_accessed"),
            last_success=metadata_data.get("last_success"),
            last_failure=metadata_data.get("last_failure"),
            validation_status=metadata_data.get("validation_status", "pending"),
            validation_score=metadata_data.get("validation_score", 0.0),
            validation_details=metadata_data.get("validation_details", {}),
            tags=set(metadata_data.get("tags", [])),
            categories=set(metadata_data.get("categories", [])),
            priority=metadata_data.get("priority", 5),
            expiry_date=metadata_data.get("expiry_date"),
            retention_period=metadata_data.get("retention_period"),
            custom_attributes=metadata_data.get("custom_attributes", {})
        )
        
        return cls(
            id=data.get("id", str(uuid4())),
            type=KnowledgeType(data.get("type", "factual")),
            source=KnowledgeSource(data.get("source", "agent_experience")),
            status=KnowledgeStatus(data.get("status", "draft")),
            title=data.get("title", ""),
            content=data.get("content"),
            description=data.get("description", ""),
            keywords=set(data.get("keywords", [])),
            context=data.get("context", {}),
            domain=data.get("domain", "general"),
            scope=data.get("scope", "local"),
            metadata=metadata,
            parent_id=data.get("parent_id"),
            children_ids=set(data.get("children_ids", [])),
            related_ids=set(data.get("related_ids", [])),
            schema_version=data.get("schema_version", "1.0"),
            data_format=data.get("data_format", "json"),
            encoding=data.get("encoding", "utf-8")
        )
    
    def update_metadata(self, **kwargs) -> None:
        """更新元数据"""
        from utils import get_iso_timestamp
        
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        
        self.metadata.updated_at = get_iso_timestamp()
        self.metadata.version += 1
    
    def add_tag(self, tag: str) -> None:
        """添加标签"""
        self.metadata.tags.add(tag)
        self.update_metadata()
    
    def remove_tag(self, tag: str) -> None:
        """移除标签"""
        self.metadata.tags.discard(tag)
        self.update_metadata()
    
    def add_category(self, category: str) -> None:
        """添加分类"""
        self.metadata.categories.add(category)
        self.update_metadata()
    
    def remove_category(self, category: str) -> None:
        """移除分类"""
        self.metadata.categories.discard(category)
        self.update_metadata()
    
    def calculate_quality_score(self) -> float:
        """计算质量分数"""
        weights = {
            "confidence": 0.25,
            "reliability": 0.25,
            "accuracy": 0.25,
            "completeness": 0.15,
            "freshness": 0.10
        }
        
        score = (
            self.metadata.confidence * weights["confidence"] +
            self.metadata.reliability * weights["reliability"] +
            self.metadata.accuracy * weights["accuracy"] +
            self.metadata.completeness * weights["completeness"] +
            self.metadata.freshness * weights["freshness"]
        )
        
        return min(1.0, max(0.0, score))
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if not self.metadata.expiry_date:
            return False
        
        try:
            from datetime import datetime
            expiry = datetime.fromisoformat(self.metadata.expiry_date.replace('Z', '+00:00'))
            return datetime.now(expiry.tzinfo) > expiry
        except Exception:
            return False


@dataclass
class KnowledgeRelation:
    """知识关系"""
    id: str = field(default_factory=lambda: str(uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.SIMILAR_TO
    strength: float = 0.5  # 关系强度 (0-1)
    confidence: float = 0.5  # 置信度 (0-1)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    created_by: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            from utils import get_iso_timestamp
            self.created_at = get_iso_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "created_by": self.created_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeRelation':
        """从字典创建关系"""
        return cls(
            id=data.get("id", str(uuid4())),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=RelationType(data.get("relation_type", "similar_to")),
            strength=data.get("strength", 0.5),
            confidence=data.get("confidence", 0.5),
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            created_by=data.get("created_by", "")
        )


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    nodes: Dict[str, KnowledgeItem] = field(default_factory=dict)
    edges: Dict[str, KnowledgeRelation] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            from utils import get_iso_timestamp
            self.created_at = get_iso_timestamp()
            self.updated_at = self.created_at
    
    def add_node(self, knowledge_item: KnowledgeItem) -> None:
        """添加节点"""
        self.nodes[knowledge_item.id] = knowledge_item
        self._update_timestamp()
    
    def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            # 移除相关边
            edges_to_remove = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source_id == node_id or edge.target_id == node_id
            ]
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
            self._update_timestamp()
            return True
        return False
    
    def add_edge(self, relation: KnowledgeRelation) -> None:
        """添加边"""
        if relation.source_id in self.nodes and relation.target_id in self.nodes:
            self.edges[relation.id] = relation
            self._update_timestamp()
    
    def remove_edge(self, edge_id: str) -> bool:
        """移除边"""
        if edge_id in self.edges:
            del self.edges[edge_id]
            self._update_timestamp()
            return True
        return False
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """获取邻居节点"""
        neighbors = set()
        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbors.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbors.add(edge.source_id)
        return list(neighbors)
    
    def get_related_nodes(
        self,
        node_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 1
    ) -> List[str]:
        """获取相关节点"""
        if node_id not in self.nodes:
            return []
        
        visited = set()
        queue = [(node_id, 0)]
        related = []
        
        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            if current_id != node_id:
                related.append(current_id)
            
            if depth < max_depth:
                for edge in self.edges.values():
                    if relation_types and edge.relation_type not in relation_types:
                        continue
                    
                    next_id = None
                    if edge.source_id == current_id:
                        next_id = edge.target_id
                    elif edge.target_id == current_id:
                        next_id = edge.source_id
                    
                    if next_id and next_id not in visited:
                        queue.append((next_id, depth + 1))
        
        return related
    
    def _update_timestamp(self) -> None:
        """更新时间戳"""
        from utils import get_iso_timestamp
        self.updated_at = get_iso_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class QueryRequest:
    """查询请求"""
    id: str = field(default_factory=lambda: str(uuid4()))
    query_text: str = ""
    query_type: str = "semantic"  # semantic, keyword, structured, graph
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_by: str = "relevance"  # relevance, date, quality, popularity
    sort_order: str = "desc"  # asc, desc
    limit: int = 10
    offset: int = 0
    include_metadata: bool = True
    include_relations: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    requester_id: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from utils import get_iso_timestamp
            self.timestamp = get_iso_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "query_text": self.query_text,
            "query_type": self.query_type,
            "filters": self.filters,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "limit": self.limit,
            "offset": self.offset,
            "include_metadata": self.include_metadata,
            "include_relations": self.include_relations,
            "context": self.context,
            "requester_id": self.requester_id,
            "timestamp": self.timestamp
        }


@dataclass
class QueryResult:
    """查询结果"""
    request_id: str
    items: List[KnowledgeItem] = field(default_factory=list)
    total_count: int = 0
    execution_time: float = 0.0
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from utils import get_iso_timestamp
            self.timestamp = get_iso_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "items": [item.to_dict() for item in self.items],
            "total_count": self.total_count,
            "execution_time": self.execution_time,
            "relevance_scores": self.relevance_scores,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class SyncRequest:
    """同步请求"""
    id: str = field(default_factory=lambda: str(uuid4()))
    sync_type: str = "incremental"  # full, incremental, selective
    source_agent: str = ""
    target_agents: List[str] = field(default_factory=list)
    knowledge_ids: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from utils import get_iso_timestamp
            self.timestamp = get_iso_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "sync_type": self.sync_type,
            "source_agent": self.source_agent,
            "target_agents": self.target_agents,
            "knowledge_ids": self.knowledge_ids,
            "filters": self.filters,
            "priority": self.priority,
            "context": self.context,
            "timestamp": self.timestamp
        }


@dataclass
class SyncResult:
    """同步结果"""
    request_id: str
    success: bool = False
    synced_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    execution_time: float = 0.0
    synced_items: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from utils import get_iso_timestamp
            self.timestamp = get_iso_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "synced_count": self.synced_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "execution_time": self.execution_time,
            "synced_items": self.synced_items,
            "failed_items": self.failed_items,
            "errors": self.errors,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }