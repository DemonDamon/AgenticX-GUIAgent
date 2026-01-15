#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prior Knowledge Retriever - 先验知识检索器 (基于AgenticX框架重构)

学习引擎第一阶段：从知识库中检索相关的先验知识。

重构说明：
- 基于AgenticX的Component基类重构
- 使用AgenticX的事件系统进行知识检索通知
- 集成AgenticX的存储组件进行知识管理
- 遵循AgenticX的组件生命周期和错误处理
"""

import asyncio
from loguru import logger
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, UTC
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from enum import Enum

from agenticx.core.component import Component

from core.info_pool import InfoPool, InfoType, InfoPriority
from utils import get_iso_timestamp


class KnowledgeType(Enum):
    """知识类型"""
    ACTION_PATTERN = "action_pattern"  # 动作模式
    ERROR_SOLUTION = "error_solution"  # 错误解决方案
    BEST_PRACTICE = "best_practice"  # 最佳实践
    PERFORMANCE_INSIGHT = "performance_insight"  # 性能洞察
    GENERAL = "general"  # 通用知识
    STRATEGY = "strategy"  # 策略知识
    CONTEXT = "context"  # 上下文知识


class KnowledgeSource(Enum):
    """知识来源"""
    LEARNING_SYSTEM = "learning_system"  # 学习系统
    MANUAL_INPUT = "manual_input"  # 手动输入
    EXTERNAL_API = "external_api"  # 外部API
    KNOWLEDGE_BASE = "knowledge_base"  # 知识库
    EXPERIENCE_REPLAY = "experience_replay"  # 经验回放
    AGENT_FEEDBACK = "agent_feedback"  # 智能体反馈


@dataclass
class KnowledgeRetrievalResult:
    """知识检索结果"""
    matches: List['KnowledgeMatch']
    total_found: int
    retrieval_time: float
    context: 'RetrievalContext'
    metadata: Dict[str, Any]
    timestamp: str


@dataclass
class KnowledgeMatch:
    """知识匹配结果"""
    knowledge_id: str
    knowledge_type: str
    title: str
    content: Dict[str, Any]
    relevance_score: float
    confidence_score: float
    source: str
    created_at: str
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class RetrievalContext:
    """检索上下文"""
    task_description: str
    task_type: str
    current_context: Dict[str, Any]
    agent_id: str
    priority_types: List[str]
    max_results: int = 10
    min_relevance: float = 0.3
    include_metadata: bool = True


class PriorKnowledgeRetriever(Component):
    """先验知识检索器
    
    负责：
    1. 任务相关知识检索
    2. 知识相关性评估
    3. 知识质量评分
    4. 检索结果排序
    5. 知识适用性分析
    """
    
    def __init__(self, info_pool: InfoPool, knowledge_base_path: str = "knowledge_base"):
        super().__init__()
        self.info_pool = info_pool
        self.knowledge_base_path = knowledge_base_path
        self.logger = logger
        
        # 检索统计
        self.retrieval_stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "average_relevance": 0.0,
            "knowledge_usage_count": defaultdict(int),
            "retrieval_history": []
        }
        
        # 知识类型权重
        self.knowledge_type_weights = {
            "action_pattern": 1.0,
            "error_solution": 0.9,
            "best_practice": 0.8,
            "performance_insight": 0.7,
            "general": 0.5
        }
        
        # 任务类型映射
        self.task_type_keywords = {
            "click": ["点击", "tap", "click", "按钮", "button"],
            "input": ["输入", "input", "text", "文本", "键盘"],
            "swipe": ["滑动", "swipe", "scroll", "滚动", "拖拽"],
            "wait": ["等待", "wait", "delay", "暂停", "sleep"],
            "screenshot": ["截图", "screenshot", "capture", "图像"],
            "locate": ["定位", "locate", "find", "查找", "元素"]
        }
    
    async def retrieve_knowledge(
        self,
        retrieval_context: RetrievalContext
    ) -> List[KnowledgeMatch]:
        """检索先验知识
        
        Args:
            retrieval_context: 检索上下文
        
        Returns:
            匹配的知识列表
        """
        logger.info(f"开始检索先验知识: {retrieval_context.task_description}")
        
        try:
            # 加载知识库
            knowledge_items = await self._load_knowledge_base()
            
            if not knowledge_items:
                logger.warning("知识库为空")
                return []
            
            # 计算相关性分数
            scored_knowledge = await self._score_knowledge_relevance(
                knowledge_items, retrieval_context
            )
            
            # 过滤和排序
            filtered_knowledge = self._filter_and_sort_knowledge(
                scored_knowledge, retrieval_context
            )
            
            # 转换为KnowledgeMatch对象
            knowledge_matches = self._convert_to_matches(filtered_knowledge)
            
            # 更新统计信息
            self._update_retrieval_stats(retrieval_context, knowledge_matches)
            
            # 发布检索结果
            self.info_pool.publish(
                InfoType.LEARNING_UPDATE,
                {
                    "stage": "prior_knowledge_retrieval",
                    "agent_id": retrieval_context.agent_id,
                    "task_description": retrieval_context.task_description,
                    "retrieved_count": len(knowledge_matches),
                    "matches": [match.__dict__ for match in knowledge_matches[:5]]  # 只发布前5个
                },
                source_agent="PriorKnowledgeRetriever",
                priority=InfoPriority.NORMAL
            )
            
            logger.info(f"检索完成，找到{len(knowledge_matches)}条相关知识")
            return knowledge_matches
            
        except Exception as e:
            logger.error(f"知识检索失败: {e}")
            self.retrieval_stats["total_retrievals"] += 1
            return []
    
    async def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """加载知识库"""
        knowledge_items = []
        
        if not os.path.exists(self.knowledge_base_path):
            return knowledge_items
        
        try:
            for filename in os.listdir(self.knowledge_base_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.knowledge_base_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            knowledge_item = json.load(f)
                            knowledge_items.append(knowledge_item)
                    except Exception as e:
                        logger.warning(f"加载知识文件失败 {filename}: {e}")
        except Exception as e:
            logger.error(f"读取知识库目录失败: {e}")
        
        return knowledge_items
    
    async def _score_knowledge_relevance(
        self,
        knowledge_items: List[Dict[str, Any]],
        context: RetrievalContext
    ) -> List[Tuple[Dict[str, Any], float]]:
        """计算知识相关性分数"""
        scored_items = []
        
        # 提取任务关键词
        task_keywords = self._extract_task_keywords(
            context.task_description, context.task_type
        )
        
        for item in knowledge_items:
            relevance_score = await self._calculate_relevance_score(
                item, task_keywords, context
            )
            
            if relevance_score >= context.min_relevance:
                scored_items.append((item, relevance_score))
        
        return scored_items
    
    def _extract_task_keywords(self, task_description: str, task_type: str) -> List[str]:
        """提取任务关键词"""
        keywords = []
        
        # 添加任务类型关键词
        if task_type in self.task_type_keywords:
            keywords.extend(self.task_type_keywords[task_type])
        
        # 从任务描述中提取关键词
        description_lower = task_description.lower()
        
        # 简单的关键词提取（可以使用更复杂的NLP方法）
        common_words = {"的", "了", "在", "是", "有", "和", "与", "或", "但", "然后", "接着"}
        words = [word.strip() for word in description_lower.replace('，', ' ').replace('。', ' ').split()]
        keywords.extend([word for word in words if len(word) > 1 and word not in common_words])
        
        return list(set(keywords))  # 去重
    
    async def _calculate_relevance_score(
        self,
        knowledge_item: Dict[str, Any],
        task_keywords: List[str],
        context: RetrievalContext
    ) -> float:
        """计算相关性分数"""
        score = 0.0
        
        # 1. 知识类型匹配 (30%)
        knowledge_type = knowledge_item.get("type", "general")
        type_weight = self.knowledge_type_weights.get(knowledge_type, 0.5)
        
        # 优先匹配指定类型
        if context.priority_types and knowledge_type in context.priority_types:
            type_weight += 0.2
        
        score += type_weight * 0.3
        
        # 2. 关键词匹配 (25%)
        keyword_score = self._calculate_keyword_match_score(
            knowledge_item, task_keywords
        )
        score += keyword_score * 0.25
        
        # 3. 任务类型匹配 (20%)
        task_type_score = self._calculate_task_type_match_score(
            knowledge_item, context.task_type
        )
        score += task_type_score * 0.2
        
        # 4. 知识质量 (15%)
        quality_score = self._calculate_quality_score(knowledge_item)
        score += quality_score * 0.15
        
        # 5. 时间新鲜度 (10%)
        freshness_score = self._calculate_freshness_score(knowledge_item)
        score += freshness_score * 0.1
        
        return min(1.0, score)
    
    def _calculate_keyword_match_score(
        self,
        knowledge_item: Dict[str, Any],
        task_keywords: List[str]
    ) -> float:
        """计算关键词匹配分数"""
        if not task_keywords:
            return 0.5
        
        # 检查标题匹配
        title = knowledge_item.get("title", "").lower()
        title_matches = sum(1 for keyword in task_keywords if keyword in title)
        
        # 检查内容匹配
        content_str = str(knowledge_item.get("content", {})).lower()
        content_matches = sum(1 for keyword in task_keywords if keyword in content_str)
        
        # 检查标签匹配
        tags = [tag.lower() for tag in knowledge_item.get("tags", [])]
        tag_matches = sum(1 for keyword in task_keywords if any(keyword in tag for tag in tags))
        
        # 计算匹配分数
        total_matches = title_matches * 2 + content_matches + tag_matches * 1.5
        max_possible_matches = len(task_keywords) * 4.5  # 最大可能匹配数
        
        return min(1.0, total_matches / max_possible_matches) if max_possible_matches > 0 else 0.0
    
    def _calculate_task_type_match_score(
        self,
        knowledge_item: Dict[str, Any],
        task_type: str
    ) -> float:
        """计算任务类型匹配分数"""
        content = knowledge_item.get("content", {})
        
        # 检查内容中的任务类型
        content_task_type = content.get("task_type", "")
        if content_task_type == task_type:
            return 1.0
        
        # 检查相关任务类型
        related_types = {
            "click": ["tap", "button"],
            "input": ["text", "keyboard"],
            "swipe": ["scroll", "drag"],
            "wait": ["delay", "sleep"],
            "screenshot": ["capture", "image"],
            "locate": ["find", "search"]
        }
        
        if task_type in related_types:
            for related_type in related_types[task_type]:
                if related_type in content_task_type:
                    return 0.7
        
        return 0.3  # 基础分数
    
    def _calculate_quality_score(self, knowledge_item: Dict[str, Any]) -> float:
        """计算知识质量分数"""
        score = 0.0
        
        # 重要性分数
        importance = knowledge_item.get("importance", 0.5)
        score += importance * 0.4
        
        # 可靠性分数
        metadata = knowledge_item.get("metadata", {})
        reliability = metadata.get("reliability", 0.5)
        score += reliability * 0.3
        
        # 访问频率分数
        access_count = knowledge_item.get("access_count", 0)
        access_score = min(1.0, access_count / 10.0)  # 访问10次以上得满分
        score += access_score * 0.2
        
        # 内容完整性分数
        content = knowledge_item.get("content", {})
        completeness = 0.5
        if "steps" in content or "procedure" in content:
            completeness += 0.3
        if "success_rate" in content:
            completeness += 0.2
        score += completeness * 0.1
        
        return min(1.0, score)
    
    def _calculate_freshness_score(self, knowledge_item: Dict[str, Any]) -> float:
        """计算时间新鲜度分数"""
        try:
            created_at = datetime.fromisoformat(
                knowledge_item.get("created_at", "").replace('Z', '+00:00')
            )
            now = datetime.now().replace(tzinfo=created_at.tzinfo)
            days_old = (now - created_at).days
            
            # 7天内：满分，30天内：线性递减，30天后：最低分
            if days_old <= 7:
                return 1.0
            elif days_old <= 30:
                return 1.0 - (days_old - 7) / 23 * 0.7  # 从1.0递减到0.3
            else:
                return 0.3
        except:
            return 0.5  # 默认分数
    
    def _filter_and_sort_knowledge(
        self,
        scored_knowledge: List[Tuple[Dict[str, Any], float]],
        context: RetrievalContext
    ) -> List[Tuple[Dict[str, Any], float]]:
        """过滤和排序知识"""
        # 按分数排序
        sorted_knowledge = sorted(scored_knowledge, key=lambda x: x[1], reverse=True)
        
        # 限制结果数量
        return sorted_knowledge[:context.max_results]
    
    def _convert_to_matches(
        self,
        scored_knowledge: List[Tuple[Dict[str, Any], float]]
    ) -> List[KnowledgeMatch]:
        """转换为KnowledgeMatch对象"""
        matches = []
        
        for knowledge_item, relevance_score in scored_knowledge:
            # 计算置信度分数
            confidence_score = self._calculate_confidence_score(
                knowledge_item, relevance_score
            )
            
            match = KnowledgeMatch(
                knowledge_id=knowledge_item.get("id", ""),
                knowledge_type=knowledge_item.get("type", "general"),
                title=knowledge_item.get("title", ""),
                content=knowledge_item.get("content", {}),
                relevance_score=relevance_score,
                confidence_score=confidence_score,
                source=knowledge_item.get("source", "unknown"),
                created_at=knowledge_item.get("created_at", ""),
                tags=knowledge_item.get("tags", []),
                metadata=knowledge_item.get("metadata", {})
            )
            
            matches.append(match)
        
        return matches
    
    def _calculate_confidence_score(
        self,
        knowledge_item: Dict[str, Any],
        relevance_score: float
    ) -> float:
        """计算置信度分数"""
        # 基于相关性、重要性和可靠性计算置信度
        importance = knowledge_item.get("importance", 0.5)
        metadata = knowledge_item.get("metadata", {})
        reliability = metadata.get("reliability", 0.5)
        
        confidence = (relevance_score * 0.5 + importance * 0.3 + reliability * 0.2)
        return min(1.0, confidence)
    
    def _update_retrieval_stats(
        self,
        context: RetrievalContext,
        matches: List[KnowledgeMatch]
    ) -> None:
        """更新检索统计信息"""
        self.retrieval_stats["total_retrievals"] += 1
        
        if matches:
            self.retrieval_stats["successful_retrievals"] += 1
            
            # 更新平均相关性
            avg_relevance = sum(match.relevance_score for match in matches) / len(matches)
            current_avg = self.retrieval_stats["average_relevance"]
            total_successful = self.retrieval_stats["successful_retrievals"]
            
            self.retrieval_stats["average_relevance"] = (
                (current_avg * (total_successful - 1) + avg_relevance) / total_successful
            )
            
            # 更新知识使用统计
            for match in matches:
                self.retrieval_stats["knowledge_usage_count"][match.knowledge_id] += 1
        
        # 记录检索历史
        retrieval_record = {
            "timestamp": get_iso_timestamp(),
            "task_description": context.task_description,
            "task_type": context.task_type,
            "agent_id": context.agent_id,
            "results_count": len(matches),
            "avg_relevance": sum(match.relevance_score for match in matches) / len(matches) if matches else 0.0
        }
        
        self.retrieval_stats["retrieval_history"].append(retrieval_record)
        
        # 保持最近100条记录
        if len(self.retrieval_stats["retrieval_history"]) > 100:
            self.retrieval_stats["retrieval_history"] = self.retrieval_stats["retrieval_history"][-100:]
    
    async def get_knowledge_by_type(
        self,
        knowledge_type: str,
        limit: int = 10
    ) -> List[KnowledgeMatch]:
        """按类型获取知识"""
        context = RetrievalContext(
            task_description="",
            task_type="",
            current_context={},
            agent_id="system",
            priority_types=[knowledge_type],
            max_results=limit,
            min_relevance=0.0
        )
        
        knowledge_items = await self._load_knowledge_base()
        filtered_items = [
            item for item in knowledge_items
            if item.get("type") == knowledge_type
        ]
        
        # 按重要性排序
        sorted_items = sorted(
            filtered_items,
            key=lambda x: x.get("importance", 0.5),
            reverse=True
        )
        
        scored_items = [(item, item.get("importance", 0.5)) for item in sorted_items[:limit]]
        return self._convert_to_matches(scored_items)
    
    async def get_recent_knowledge(
        self,
        days: int = 7,
        limit: int = 10
    ) -> List[KnowledgeMatch]:
        """获取最近的知识"""
        knowledge_items = await self._load_knowledge_base()
        recent_items = []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for item in knowledge_items:
            try:
                created_at = datetime.fromisoformat(
                    item.get("created_at", "").replace('Z', '+00:00')
                )
                if created_at.replace(tzinfo=None) >= cutoff_date:
                    recent_items.append(item)
            except:
                continue
        
        # 按创建时间排序
        sorted_items = sorted(
            recent_items,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        
        scored_items = [(item, 0.8) for item in sorted_items[:limit]]  # 给定默认分数
        return self._convert_to_matches(scored_items)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        stats = self.retrieval_stats.copy()
        
        # 计算成功率
        if stats["total_retrievals"] > 0:
            stats["success_rate"] = stats["successful_retrievals"] / stats["total_retrievals"]
        else:
            stats["success_rate"] = 0.0
        
        # 获取最常用的知识
        usage_count = dict(stats["knowledge_usage_count"])
        stats["most_used_knowledge"] = sorted(
            usage_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return stats
    
    async def clear_cache(self) -> None:
        """清理缓存"""
        # 重置统计信息（保留历史记录）
        self.retrieval_stats["knowledge_usage_count"].clear()
        logger.info("先验知识检索器缓存已清理")