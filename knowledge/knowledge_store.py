#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Knowledge Store
基于AgenticX框架的知识存储：实现知识的持久化存储、索引管理和查询功能

重构说明：
- 基于AgenticX的Storage组件重构
- 使用AgenticX的数据模型和接口
- 集成AgenticX的事件系统进行存储监控
- 提供高性能的知识存储和检索功能

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Callable, Awaitable
)
from uuid import uuid4

from .knowledge_types import (
    KnowledgeItem, KnowledgeType, KnowledgeSource, KnowledgeStatus,
    KnowledgeRelation, RelationType, QueryRequest, QueryResult
)
from utils import get_iso_timestamp, setup_logger


class KnowledgeStoreInterface(ABC):
    """知识存储接口"""
    
    @abstractmethod
    async def store_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """存储知识"""
        pass
    
    @abstractmethod
    async def retrieve_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """检索知识"""
        pass
    
    @abstractmethod
    async def update_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """更新知识"""
        pass
    
    @abstractmethod
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识"""
        pass
    
    @abstractmethod
    async def query_knowledge(self, request: QueryRequest) -> QueryResult:
        """查询知识"""
        pass
    
    @abstractmethod
    async def get_knowledge_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """获取知识数量"""
        pass


class InMemoryKnowledgeStore(KnowledgeStoreInterface):
    """内存知识存储"""
    
    def __init__(self):
        self.logger = logger
        self._knowledge_items: Dict[str, KnowledgeItem] = {}
        self._relations: Dict[str, KnowledgeRelation] = {}
        self._indexes = {
            'type': defaultdict(set),
            'source': defaultdict(set),
            'status': defaultdict(set),
            'domain': defaultdict(set),
            'keywords': defaultdict(set),
            'tags': defaultdict(set),
            'categories': defaultdict(set)
        }
        self._lock = threading.RLock()
    
    async def store_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """存储知识"""
        try:
            with self._lock:
                self._knowledge_items[knowledge.id] = knowledge
                self._update_indexes(knowledge)
                
                logger.debug(f"Stored knowledge: {knowledge.id}")
                return True
        except Exception as e:
            logger.error(f"Failed to store knowledge {knowledge.id}: {e}")
            return False
    
    async def retrieve_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """检索知识"""
        try:
            with self._lock:
                knowledge = self._knowledge_items.get(knowledge_id)
                if knowledge:
                    # 更新访问统计
                    knowledge.metadata.access_count += 1
                    knowledge.metadata.last_accessed = get_iso_timestamp()
                    logger.debug(f"Retrieved knowledge: {knowledge_id}")
                return knowledge
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge {knowledge_id}: {e}")
            return None
    
    async def update_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """更新知识"""
        try:
            with self._lock:
                if knowledge.id in self._knowledge_items:
                    # 移除旧索引
                    old_knowledge = self._knowledge_items[knowledge.id]
                    self._remove_from_indexes(old_knowledge)
                    
                    # 更新知识
                    knowledge.metadata.updated_at = get_iso_timestamp()
                    knowledge.metadata.version += 1
                    self._knowledge_items[knowledge.id] = knowledge
                    
                    # 更新索引
                    self._update_indexes(knowledge)
                    
                    logger.debug(f"Updated knowledge: {knowledge.id}")
                    return True
                else:
                    logger.warning(f"Knowledge not found for update: {knowledge.id}")
                    return False
        except Exception as e:
            logger.error(f"Failed to update knowledge {knowledge.id}: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识"""
        try:
            with self._lock:
                if knowledge_id in self._knowledge_items:
                    knowledge = self._knowledge_items[knowledge_id]
                    self._remove_from_indexes(knowledge)
                    del self._knowledge_items[knowledge_id]
                    
                    # 删除相关关系
                    relations_to_remove = [
                        rel_id for rel_id, rel in self._relations.items()
                        if rel.source_id == knowledge_id or rel.target_id == knowledge_id
                    ]
                    for rel_id in relations_to_remove:
                        del self._relations[rel_id]
                    
                    logger.debug(f"Deleted knowledge: {knowledge_id}")
                    return True
                else:
                    logger.warning(f"Knowledge not found for deletion: {knowledge_id}")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete knowledge {knowledge_id}: {e}")
            return False
    
    async def query_knowledge(self, request: QueryRequest) -> QueryResult:
        """查询知识"""
        start_time = datetime.now()
        
        try:
            with self._lock:
                # 获取候选知识项
                candidates = self._get_candidates(request)
                
                # 计算相关性分数
                scored_items = []
                relevance_scores = {}
                
                for knowledge in candidates:
                    score = self._calculate_relevance_score(knowledge, request)
                    if score > 0:
                        scored_items.append((knowledge, score))
                        relevance_scores[knowledge.id] = score
                
                # 排序
                scored_items.sort(key=lambda x: x[1], reverse=(request.sort_order == "desc"))
                
                # 分页
                total_count = len(scored_items)
                start_idx = request.offset
                end_idx = start_idx + request.limit
                paged_items = scored_items[start_idx:end_idx]
                
                # 构建结果
                result_items = [item[0] for item in paged_items]
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = QueryResult(
                    request_id=request.id,
                    items=result_items,
                    total_count=total_count,
                    execution_time=execution_time,
                    relevance_scores=relevance_scores
                )
                
                logger.debug(f"Query completed: {len(result_items)} items found")
                return result
                
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                request_id=request.id,
                execution_time=execution_time
            )
    
    async def get_knowledge_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """获取知识数量"""
        try:
            with self._lock:
                if not filters:
                    return len(self._knowledge_items)
                
                count = 0
                for knowledge in self._knowledge_items.values():
                    if self._matches_filters(knowledge, filters):
                        count += 1
                
                return count
        except Exception as e:
            logger.error(f"Failed to get knowledge count: {e}")
            return 0
    
    def _update_indexes(self, knowledge: KnowledgeItem) -> None:
        """更新索引"""
        self._indexes['type'][knowledge.type.value].add(knowledge.id)
        self._indexes['source'][knowledge.source.value].add(knowledge.id)
        self._indexes['status'][knowledge.status.value].add(knowledge.id)
        self._indexes['domain'][knowledge.domain].add(knowledge.id)
        
        for keyword in knowledge.keywords:
            self._indexes['keywords'][keyword.lower()].add(knowledge.id)
        
        for tag in knowledge.metadata.tags:
            self._indexes['tags'][tag.lower()].add(knowledge.id)
        
        for category in knowledge.metadata.categories:
            self._indexes['categories'][category.lower()].add(knowledge.id)
    
    def _remove_from_indexes(self, knowledge: KnowledgeItem) -> None:
        """从索引中移除"""
        self._indexes['type'][knowledge.type.value].discard(knowledge.id)
        self._indexes['source'][knowledge.source.value].discard(knowledge.id)
        self._indexes['status'][knowledge.status.value].discard(knowledge.id)
        self._indexes['domain'][knowledge.domain].discard(knowledge.id)
        
        for keyword in knowledge.keywords:
            self._indexes['keywords'][keyword.lower()].discard(knowledge.id)
        
        for tag in knowledge.metadata.tags:
            self._indexes['tags'][tag.lower()].discard(knowledge.id)
        
        for category in knowledge.metadata.categories:
            self._indexes['categories'][category.lower()].discard(knowledge.id)
    
    def _get_candidates(self, request: QueryRequest) -> List[KnowledgeItem]:
        """获取候选知识项"""
        candidate_ids = set()
        
        # 基于查询文本的候选
        if request.query_text:
            query_words = request.query_text.lower().split()
            for word in query_words:
                # 关键词匹配
                if word in self._indexes['keywords']:
                    candidate_ids.update(self._indexes['keywords'][word])
                
                # 标签匹配
                if word in self._indexes['tags']:
                    candidate_ids.update(self._indexes['tags'][word])
                
                # 分类匹配
                if word in self._indexes['categories']:
                    candidate_ids.update(self._indexes['categories'][word])
        
        # 基于过滤器的候选
        if request.filters:
            filter_candidates = self._apply_filters(request.filters)
            if candidate_ids:
                candidate_ids &= filter_candidates
            else:
                candidate_ids = filter_candidates
        
        # 如果没有候选，返回所有知识项
        if not candidate_ids:
            candidate_ids = set(self._knowledge_items.keys())
        
        return [self._knowledge_items[kid] for kid in candidate_ids if kid in self._knowledge_items]
    
    def _apply_filters(self, filters: Dict[str, Any]) -> Set[str]:
        """应用过滤器"""
        candidate_ids = set(self._knowledge_items.keys())
        
        for filter_key, filter_value in filters.items():
            if filter_key in self._indexes:
                if isinstance(filter_value, list):
                    filter_candidates = set()
                    for value in filter_value:
                        filter_candidates.update(self._indexes[filter_key].get(value, set()))
                    candidate_ids &= filter_candidates
                else:
                    candidate_ids &= self._indexes[filter_key].get(filter_value, set())
        
        return candidate_ids
    
    def _calculate_relevance_score(self, knowledge: KnowledgeItem, request: QueryRequest) -> float:
        """计算相关性分数"""
        score = 0.0
        
        if not request.query_text:
            return 1.0  # 无查询文本时返回基础分数
        
        query_text = request.query_text.lower()
        
        # 标题匹配
        if knowledge.title and query_text in knowledge.title.lower():
            score += 0.3
        
        # 描述匹配
        if knowledge.description and query_text in knowledge.description.lower():
            score += 0.2
        
        # 关键词匹配
        query_words = set(query_text.split())
        keyword_matches = len(query_words & {kw.lower() for kw in knowledge.keywords})
        if keyword_matches > 0:
            score += 0.2 * (keyword_matches / len(query_words))
        
        # 标签匹配
        tag_matches = len(query_words & {tag.lower() for tag in knowledge.metadata.tags})
        if tag_matches > 0:
            score += 0.1 * (tag_matches / len(query_words))
        
        # 内容匹配（如果内容是字符串）
        if isinstance(knowledge.content, str) and query_text in knowledge.content.lower():
            score += 0.2
        
        # 质量加权
        quality_score = knowledge.calculate_quality_score()
        score *= (0.5 + 0.5 * quality_score)
        
        return min(1.0, score)
    
    def _matches_filters(self, knowledge: KnowledgeItem, filters: Dict[str, Any]) -> bool:
        """检查是否匹配过滤器"""
        for filter_key, filter_value in filters.items():
            if filter_key == "type":
                if isinstance(filter_value, list):
                    if knowledge.type.value not in filter_value:
                        return False
                else:
                    if knowledge.type.value != filter_value:
                        return False
            elif filter_key == "source":
                if isinstance(filter_value, list):
                    if knowledge.source.value not in filter_value:
                        return False
                else:
                    if knowledge.source.value != filter_value:
                        return False
            elif filter_key == "status":
                if isinstance(filter_value, list):
                    if knowledge.status.value not in filter_value:
                        return False
                else:
                    if knowledge.status.value != filter_value:
                        return False
            elif filter_key == "domain":
                if isinstance(filter_value, list):
                    if knowledge.domain not in filter_value:
                        return False
                else:
                    if knowledge.domain != filter_value:
                        return False
            elif filter_key == "tags":
                if isinstance(filter_value, list):
                    if not any(tag in knowledge.metadata.tags for tag in filter_value):
                        return False
                else:
                    if filter_value not in knowledge.metadata.tags:
                        return False
            elif filter_key == "categories":
                if isinstance(filter_value, list):
                    if not any(cat in knowledge.metadata.categories for cat in filter_value):
                        return False
                else:
                    if filter_value not in knowledge.metadata.categories:
                        return False
        
        return True
    
    async def store_relation(self, relation: KnowledgeRelation) -> bool:
        """存储关系"""
        try:
            with self._lock:
                self._relations[relation.id] = relation
                logger.debug(f"Stored relation: {relation.id}")
                return True
        except Exception as e:
            logger.error(f"Failed to store relation {relation.id}: {e}")
            return False
    
    async def get_relations(
        self,
        knowledge_id: str,
        relation_types: Optional[List[RelationType]] = None
    ) -> List[KnowledgeRelation]:
        """获取关系"""
        try:
            with self._lock:
                relations = []
                for relation in self._relations.values():
                    if (relation.source_id == knowledge_id or relation.target_id == knowledge_id):
                        if not relation_types or relation.relation_type in relation_types:
                            relations.append(relation)
                return relations
        except Exception as e:
            logger.error(f"Failed to get relations for {knowledge_id}: {e}")
            return []
    
    async def get_all_knowledge(self) -> List[KnowledgeItem]:
        """获取所有知识"""
        try:
            with self._lock:
                return list(self._knowledge_items.values())
        except Exception as e:
            logger.error(f"Failed to get all knowledge: {e}")
            return []
    
    async def clear_all(self) -> bool:
        """清空所有数据"""
        try:
            with self._lock:
                self._knowledge_items.clear()
                self._relations.clear()
                for index in self._indexes.values():
                    index.clear()
                logger.info("Cleared all knowledge data")
                return True
        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
            return False


class SQLiteKnowledgeStore(KnowledgeStoreInterface):
    """SQLite知识存储"""
    
    def __init__(self, db_path: str = "knowledge.db"):
        self.logger = logger
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建知识表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_items (
                        id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        status TEXT NOT NULL,
                        title TEXT,
                        content TEXT,
                        description TEXT,
                        keywords TEXT,
                        context TEXT,
                        domain TEXT,
                        scope TEXT,
                        metadata TEXT,
                        parent_id TEXT,
                        children_ids TEXT,
                        related_ids TEXT,
                        schema_version TEXT,
                        data_format TEXT,
                        encoding TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
                
                # 创建关系表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_relations (
                        id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        relation_type TEXT NOT NULL,
                        strength REAL,
                        confidence REAL,
                        context TEXT,
                        metadata TEXT,
                        created_at TEXT,
                        created_by TEXT
                    )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON knowledge_items(type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge_items(source)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON knowledge_items(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON knowledge_items(domain)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_items(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON knowledge_items(updated_at)")
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON knowledge_relations(source_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON knowledge_relations(target_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON knowledge_relations(relation_type)")
                
                conn.commit()
                logger.info(f"Database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def store_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """存储知识"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO knowledge_items (
                            id, type, source, status, title, content, description,
                            keywords, context, domain, scope, metadata,
                            parent_id, children_ids, related_ids,
                            schema_version, data_format, encoding,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        knowledge.id,
                        knowledge.type.value,
                        knowledge.source.value,
                        knowledge.status.value,
                        knowledge.title,
                        json.dumps(knowledge.content) if knowledge.content is not None else None,
                        knowledge.description,
                        json.dumps(list(knowledge.keywords)),
                        json.dumps(knowledge.context),
                        knowledge.domain,
                        knowledge.scope,
                        json.dumps(knowledge.metadata.to_dict() if hasattr(knowledge.metadata, 'to_dict') else knowledge.metadata.__dict__),
                        knowledge.parent_id,
                        json.dumps(list(knowledge.children_ids)),
                        json.dumps(list(knowledge.related_ids)),
                        knowledge.schema_version,
                        knowledge.data_format,
                        knowledge.encoding,
                        knowledge.metadata.created_at,
                        knowledge.metadata.updated_at
                    ))
                    
                    conn.commit()
                    logger.debug(f"Stored knowledge: {knowledge.id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to store knowledge {knowledge.id}: {e}")
            return False
    
    async def retrieve_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """检索知识"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "SELECT * FROM knowledge_items WHERE id = ?",
                        (knowledge_id,)
                    )
                    
                    row = cursor.fetchone()
                    if row:
                        knowledge = self._row_to_knowledge(row)
                        
                        # 更新访问统计
                        knowledge.metadata.access_count += 1
                        knowledge.metadata.last_accessed = get_iso_timestamp()
                        await self.update_knowledge(knowledge)
                        
                        logger.debug(f"Retrieved knowledge: {knowledge_id}")
                        return knowledge
                    
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge {knowledge_id}: {e}")
            return None
    
    async def update_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """更新知识"""
        knowledge.metadata.updated_at = get_iso_timestamp()
        knowledge.metadata.version += 1
        return await self.store_knowledge(knowledge)
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 删除知识项
                    cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (knowledge_id,))
                    
                    # 删除相关关系
                    cursor.execute(
                        "DELETE FROM knowledge_relations WHERE source_id = ? OR target_id = ?",
                        (knowledge_id, knowledge_id)
                    )
                    
                    conn.commit()
                    logger.debug(f"Deleted knowledge: {knowledge_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete knowledge {knowledge_id}: {e}")
            return False
    
    async def query_knowledge(self, request: QueryRequest) -> QueryResult:
        """查询知识"""
        start_time = datetime.now()
        
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 构建查询
                    query = "SELECT * FROM knowledge_items WHERE 1=1"
                    params = []
                    
                    # 应用过滤器
                    if request.filters:
                        for filter_key, filter_value in request.filters.items():
                            if filter_key in ['type', 'source', 'status', 'domain']:
                                if isinstance(filter_value, list):
                                    placeholders = ','.join(['?' for _ in filter_value])
                                    query += f" AND {filter_key} IN ({placeholders})"
                                    params.extend(filter_value)
                                else:
                                    query += f" AND {filter_key} = ?"
                                    params.append(filter_value)
                    
                    # 文本搜索
                    if request.query_text:
                        query += " AND (title LIKE ? OR description LIKE ? OR keywords LIKE ?)"
                        search_term = f"%{request.query_text}%"
                        params.extend([search_term, search_term, search_term])
                    
                    # 排序
                    if request.sort_by == "date":
                        query += f" ORDER BY created_at {request.sort_order.upper()}"
                    else:
                        query += f" ORDER BY updated_at {request.sort_order.upper()}"
                    
                    # 分页
                    query += " LIMIT ? OFFSET ?"
                    params.extend([request.limit, request.offset])
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # 获取总数
                    count_query = query.split(" ORDER BY")[0].replace("SELECT *", "SELECT COUNT(*)")
                    count_params = params[:-2]  # 移除LIMIT和OFFSET参数
                    cursor.execute(count_query, count_params)
                    total_count = cursor.fetchone()[0]
                    
                    # 转换结果
                    items = [self._row_to_knowledge(row) for row in rows]
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    result = QueryResult(
                        request_id=request.id,
                        items=items,
                        total_count=total_count,
                        execution_time=execution_time
                    )
                    
                    logger.debug(f"Query completed: {len(items)} items found")
                    return result
                    
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                request_id=request.id,
                execution_time=execution_time
            )
    
    async def get_knowledge_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """获取知识数量"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    query = "SELECT COUNT(*) FROM knowledge_items WHERE 1=1"
                    params = []
                    
                    if filters:
                        for filter_key, filter_value in filters.items():
                            if filter_key in ['type', 'source', 'status', 'domain']:
                                if isinstance(filter_value, list):
                                    placeholders = ','.join(['?' for _ in filter_value])
                                    query += f" AND {filter_key} IN ({placeholders})"
                                    params.extend(filter_value)
                                else:
                                    query += f" AND {filter_key} = ?"
                                    params.append(filter_value)
                    
                    cursor.execute(query, params)
                    return cursor.fetchone()[0]
                    
        except Exception as e:
            logger.error(f"Failed to get knowledge count: {e}")
            return 0
    
    def _row_to_knowledge(self, row: Tuple) -> KnowledgeItem:
        """将数据库行转换为知识项"""
        from .knowledge_types import KnowledgeMetadata
        
        # 解析元数据
        metadata_dict = json.loads(row[11]) if row[11] else {}
        metadata = KnowledgeMetadata(
            created_at=metadata_dict.get('created_at', row[18]),
            updated_at=metadata_dict.get('updated_at', row[19]),
            created_by=metadata_dict.get('created_by', ''),
            updated_by=metadata_dict.get('updated_by', ''),
            version=metadata_dict.get('version', 1),
            confidence=metadata_dict.get('confidence', 0.5),
            reliability=metadata_dict.get('reliability', 0.5),
            accuracy=metadata_dict.get('accuracy', 0.5),
            completeness=metadata_dict.get('completeness', 0.5),
            freshness=metadata_dict.get('freshness', 1.0),
            access_count=metadata_dict.get('access_count', 0),
            success_count=metadata_dict.get('success_count', 0),
            failure_count=metadata_dict.get('failure_count', 0),
            last_accessed=metadata_dict.get('last_accessed'),
            last_success=metadata_dict.get('last_success'),
            last_failure=metadata_dict.get('last_failure'),
            validation_status=metadata_dict.get('validation_status', 'pending'),
            validation_score=metadata_dict.get('validation_score', 0.0),
            validation_details=metadata_dict.get('validation_details', {}),
            tags=set(metadata_dict.get('tags', [])),
            categories=set(metadata_dict.get('categories', [])),
            priority=metadata_dict.get('priority', 5),
            expiry_date=metadata_dict.get('expiry_date'),
            retention_period=metadata_dict.get('retention_period'),
            custom_attributes=metadata_dict.get('custom_attributes', {})
        )
        
        return KnowledgeItem(
            id=row[0],
            type=KnowledgeType(row[1]),
            source=KnowledgeSource(row[2]),
            status=KnowledgeStatus(row[3]),
            title=row[4] or '',
            content=json.loads(row[5]) if row[5] else None,
            description=row[6] or '',
            keywords=set(json.loads(row[7])) if row[7] else set(),
            context=json.loads(row[8]) if row[8] else {},
            domain=row[9] or 'general',
            scope=row[10] or 'local',
            metadata=metadata,
            parent_id=row[12],
            children_ids=set(json.loads(row[13])) if row[13] else set(),
            related_ids=set(json.loads(row[14])) if row[14] else set(),
            schema_version=row[15] or '1.0',
            data_format=row[16] or 'json',
            encoding=row[17] or 'utf-8'
        )
    
    async def store_relation(self, relation: KnowledgeRelation) -> bool:
        """存储关系"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO knowledge_relations (
                            id, source_id, target_id, relation_type,
                            strength, confidence, context, metadata,
                            created_at, created_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        relation.id,
                        relation.source_id,
                        relation.target_id,
                        relation.relation_type.value,
                        relation.strength,
                        relation.confidence,
                        json.dumps(relation.context),
                        json.dumps(relation.metadata),
                        relation.created_at,
                        relation.created_by
                    ))
                    
                    conn.commit()
                    logger.debug(f"Stored relation: {relation.id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to store relation {relation.id}: {e}")
            return False
    
    async def get_relations(
        self,
        knowledge_id: str,
        relation_types: Optional[List[RelationType]] = None
    ) -> List[KnowledgeRelation]:
        """获取关系"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    query = "SELECT * FROM knowledge_relations WHERE source_id = ? OR target_id = ?"
                    params = [knowledge_id, knowledge_id]
                    
                    if relation_types:
                        type_values = [rt.value for rt in relation_types]
                        placeholders = ','.join(['?' for _ in type_values])
                        query += f" AND relation_type IN ({placeholders})"
                        params.extend(type_values)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    relations = []
                    for row in rows:
                        relation = KnowledgeRelation(
                            id=row[0],
                            source_id=row[1],
                            target_id=row[2],
                            relation_type=RelationType(row[3]),
                            strength=row[4],
                            confidence=row[5],
                            context=json.loads(row[6]) if row[6] else {},
                            metadata=json.loads(row[7]) if row[7] else {},
                            created_at=row[8],
                            created_by=row[9]
                        )
                        relations.append(relation)
                    
                    return relations
                    
        except Exception as e:
            logger.error(f"Failed to get relations for {knowledge_id}: {e}")
            return []
    
    async def clear_all(self) -> bool:
        """清空所有数据"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM knowledge_items")
                    cursor.execute("DELETE FROM knowledge_relations")
                    conn.commit()
                    logger.info("Cleared all knowledge data")
                    return True
        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
            return False


class KnowledgeStoreFactory:
    """知识存储工厂"""
    
    @staticmethod
    def create_store(store_type: str = "memory", **kwargs) -> KnowledgeStoreInterface:
        """创建知识存储"""
        if store_type == "memory":
            return InMemoryKnowledgeStore()
        elif store_type == "sqlite":
            return SQLiteKnowledgeStore(**kwargs)
        else:
            raise ValueError(f"Unsupported store type: {store_type}")