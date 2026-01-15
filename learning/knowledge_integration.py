#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M13: 知识学习协作 - Learning与Knowledge模块协作接口

实现learning模块与knowledge模块的协作机制，包括：
1. 学习洞察到知识的转换
2. 知识到学习上下文的转换
3. 经验与知识的双向同步
4. 协作策略的优化

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
from enum import Enum

import torch
import numpy as np

# 导入learning模块组件
from .rl_core.experience import Experience
from .rl_core.policies import LearningPattern, LearningInsight

# 注意：这里假设knowledge模块的接口，实际使用时需要根据具体实现调整
try:
    # 尝试导入knowledge模块（如果存在）
    from ..knowledge import (
        KnowledgePool, KnowledgeManager, KnowledgeItem, 
        KnowledgeType, KnowledgeSource, KnowledgeStatus
    )
    KNOWLEDGE_MODULE_AVAILABLE = True
except ImportError:
    # 如果knowledge模块不存在，定义基础接口
    KNOWLEDGE_MODULE_AVAILABLE = False
    
    class KnowledgeItem:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class KnowledgePool:
        async def add_knowledge(self, knowledge: KnowledgeItem):
            pass
        
        async def query(self, **kwargs):
            return []
    
    class KnowledgeManager:
        def __init__(self):
            self.knowledge_pool = KnowledgePool()


class IntegrationType(Enum):
    """集成类型"""
    EXPERIENCE_TO_KNOWLEDGE = "experience_to_knowledge"
    KNOWLEDGE_TO_CONTEXT = "knowledge_to_context"
    BIDIRECTIONAL_SYNC = "bidirectional_sync"
    STRATEGY_OPTIMIZATION = "strategy_optimization"


class SyncStrategy(Enum):
    """同步策略"""
    IMMEDIATE = "immediate"  # 立即同步
    BATCH = "batch"  # 批量同步
    SELECTIVE = "selective"  # 选择性同步
    ADAPTIVE = "adaptive"  # 自适应同步


@dataclass
class IntegrationConfig:
    """集成配置"""
    # 基础配置
    enable_integration: bool = True
    sync_strategy: SyncStrategy = SyncStrategy.ADAPTIVE
    
    # 转换配置
    experience_to_knowledge_threshold: float = 0.7
    knowledge_relevance_threshold: float = 0.6
    max_knowledge_items_per_sync: int = 100
    
    # 权重配置
    learning_weight: float = 0.7
    knowledge_weight: float = 0.3
    
    # 同步频率配置
    immediate_sync_threshold: float = 0.9
    batch_sync_interval_minutes: int = 30
    selective_sync_criteria: List[str] = field(default_factory=lambda: [
        'high_reward', 'novel_pattern', 'error_recovery'
    ])
    
    # 质量控制
    min_experience_quality: float = 0.5
    max_sync_failures: int = 3
    enable_quality_filtering: bool = True


@dataclass
class SyncResult:
    """同步结果"""
    sync_id: str
    sync_type: IntegrationType
    success: bool
    items_processed: int
    items_synced: int
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeLearningBridge:
    """知识学习桥接器
    
    负责learning模块和knowledge模块之间的数据转换和同步。
    """
    
    def __init__(self, 
                 knowledge_manager: Optional[Any] = None,
                 config: Optional[IntegrationConfig] = None):
        self.knowledge_manager = knowledge_manager
        self.config = config or IntegrationConfig()
        
        # 同步状态
        self.sync_history = deque(maxlen=1000)
        self.pending_syncs = defaultdict(list)
        self.sync_statistics = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'experiences_converted': 0,
            'knowledge_items_created': 0,
            'knowledge_items_retrieved': 0
        }
        
        # 质量评估器
        self.quality_assessor = ExperienceQualityAssessor()
        self.relevance_calculator = KnowledgeRelevanceCalculator()
        
        # 后台任务
        self.batch_sync_task = None
        self.cleanup_task = None
        
        self.logger = logger
        
        # 启动后台任务
        if self.config.enable_integration:
            self._start_background_tasks()
    
    async def sync_learning_insights_to_knowledge(self, 
                                                 insights: List[LearningInsight], 
                                                 knowledge_pool: Any) -> SyncResult:
        """将学习洞察同步到知识池"""
        sync_id = f"insights_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        result = SyncResult(
            sync_id=sync_id,
            sync_type=IntegrationType.EXPERIENCE_TO_KNOWLEDGE,
            success=False,
            items_processed=len(insights),
            items_synced=0
        )
        
        try:
            synced_count = 0
            
            for insight in insights:
                try:
                    # 评估洞察质量
                    if not self._should_sync_insight(insight):
                        continue
                    
                    # 转换为知识项
                    knowledge_item = await self._convert_insight_to_knowledge(insight)
                    
                    if knowledge_item:
                        # 添加到知识池
                        if hasattr(knowledge_pool, 'add_knowledge'):
                            await knowledge_pool.add_knowledge(knowledge_item)
                        elif hasattr(knowledge_pool, 'add'):
                            await knowledge_pool.add(knowledge_item)
                        
                        synced_count += 1
                        
                except Exception as e:
                    result.errors.append(f"同步洞察失败: {str(e)}")
                    logger.warning(f"同步洞察失败: {e}")
            
            result.items_synced = synced_count
            result.success = synced_count > 0
            result.duration = (datetime.now() - start_time).total_seconds()
            
            # 更新统计
            self.sync_statistics['total_syncs'] += 1
            if result.success:
                self.sync_statistics['successful_syncs'] += 1
                self.sync_statistics['knowledge_items_created'] += synced_count
            else:
                self.sync_statistics['failed_syncs'] += 1
            
            logger.info(f"洞察同步完成: {sync_id}, 成功: {synced_count}/{len(insights)}")
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"洞察同步失败: {e}")
        
        # 记录同步历史
        self.sync_history.append(result)
        
        return result
    
    async def retrieve_relevant_knowledge(self, 
                                        learning_context: Dict, 
                                        knowledge_manager: Any) -> List[Any]:
        """检索相关知识"""
        try:
            if not knowledge_manager or not hasattr(knowledge_manager, 'knowledge_pool'):
                return []
            
            knowledge_pool = knowledge_manager.knowledge_pool
            
            # 构建查询参数
            query_params = self._build_knowledge_query(learning_context)
            
            # 执行查询
            if hasattr(knowledge_pool, 'query'):
                knowledge_items = await knowledge_pool.query(**query_params)
            else:
                knowledge_items = []
            
            # 过滤和排序
            relevant_items = await self._filter_relevant_knowledge(
                knowledge_items, learning_context
            )
            
            # 更新统计
            self.sync_statistics['knowledge_items_retrieved'] += len(relevant_items)
            
            logger.debug(f"检索到相关知识: {len(relevant_items)} 项")
            
            return relevant_items
            
        except Exception as e:
            logger.error(f"检索相关知识失败: {e}")
            return []
    
    async def update_knowledge_from_experience(self, 
                                             experiences: List[Experience], 
                                             knowledge_pool: Any) -> SyncResult:
        """从经验更新知识"""
        sync_id = f"exp_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        result = SyncResult(
            sync_id=sync_id,
            sync_type=IntegrationType.EXPERIENCE_TO_KNOWLEDGE,
            success=False,
            items_processed=len(experiences),
            items_synced=0
        )
        
        try:
            # 根据同步策略处理
            if self.config.sync_strategy == SyncStrategy.IMMEDIATE:
                result = await self._immediate_sync_experiences(experiences, knowledge_pool, result)
            elif self.config.sync_strategy == SyncStrategy.BATCH:
                result = await self._batch_sync_experiences(experiences, knowledge_pool, result)
            elif self.config.sync_strategy == SyncStrategy.SELECTIVE:
                result = await self._selective_sync_experiences(experiences, knowledge_pool, result)
            else:  # ADAPTIVE
                result = await self._adaptive_sync_experiences(experiences, knowledge_pool, result)
            
            result.duration = (datetime.now() - start_time).total_seconds()
            
            # 更新统计
            self.sync_statistics['total_syncs'] += 1
            self.sync_statistics['experiences_converted'] += result.items_synced
            
            if result.success:
                self.sync_statistics['successful_syncs'] += 1
            else:
                self.sync_statistics['failed_syncs'] += 1
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"经验同步失败: {e}")
        
        # 记录同步历史
        self.sync_history.append(result)
        
        return result
    
    def _should_sync_insight(self, insight: LearningInsight) -> bool:
        """判断是否应该同步洞察"""
        # 检查重要性阈值
        if insight.importance < self.config.experience_to_knowledge_threshold:
            return False
        
        # 检查是否可操作
        if not insight.actionable:
            return False
        
        # 检查质量过滤
        if self.config.enable_quality_filtering:
            quality_score = self._assess_insight_quality(insight)
            if quality_score < self.config.min_experience_quality:
                return False
        
        return True
    
    def _assess_insight_quality(self, insight: LearningInsight) -> float:
        """评估洞察质量"""
        quality_score = 0.0
        
        # 重要性权重
        quality_score += insight.importance * 0.4
        
        # 可操作性权重
        quality_score += (1.0 if insight.actionable else 0.0) * 0.3
        
        # 描述完整性权重
        description_score = min(1.0, len(insight.description) / 100.0)
        quality_score += description_score * 0.2
        
        # 类型相关性权重
        type_relevance = 1.0 if insight.insight_type in [
            'performance_improvement', 'error_pattern', 'optimization_opportunity'
        ] else 0.5
        quality_score += type_relevance * 0.1
        
        return quality_score
    
    async def _convert_insight_to_knowledge(self, insight: LearningInsight) -> Optional[Any]:
        """将洞察转换为知识项"""
        try:
            if not KNOWLEDGE_MODULE_AVAILABLE:
                # 简化的知识项创建
                return KnowledgeItem(
                    id=insight.insight_id,
                    type='learning_insight',
                    content=insight.description,
                    importance=insight.importance,
                    actionable=insight.actionable,
                    metadata=insight.metadata or {},
                    timestamp=datetime.now()
                )
            
            # 使用实际的knowledge模块接口
            knowledge_item = KnowledgeItem(
                knowledge_id=insight.insight_id,
                knowledge_type=self._map_insight_type_to_knowledge_type(insight.insight_type),
                title=f"学习洞察: {insight.insight_type}",
                content=insight.description,
                source=KnowledgeSource.LEARNING_SYSTEM,
                status=KnowledgeStatus.ACTIVE,
                importance_score=insight.importance,
                confidence_score=0.8,  # 默认置信度
                metadata={
                    'insight_type': insight.insight_type,
                    'actionable': insight.actionable,
                    'original_insight_id': insight.insight_id,
                    'conversion_timestamp': datetime.now().isoformat(),
                    **(insight.metadata or {})
                },
                tags=['learning', 'insight', insight.insight_type],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return knowledge_item
            
        except Exception as e:
            logger.error(f"转换洞察到知识项失败: {e}")
            return None
    
    def _map_insight_type_to_knowledge_type(self, insight_type: str) -> str:
        """映射洞察类型到知识类型"""
        mapping = {
            'performance_improvement': 'OPTIMIZATION',
            'error_pattern': 'ERROR_HANDLING',
            'user_behavior': 'USER_PATTERN',
            'system_behavior': 'SYSTEM_PATTERN',
            'optimization_opportunity': 'OPTIMIZATION',
            'learning_pattern': 'LEARNING_STRATEGY'
        }
        
        return mapping.get(insight_type, 'GENERAL')
    
    def _build_knowledge_query(self, learning_context: Dict) -> Dict[str, Any]:
        """构建知识查询参数"""
        query_params = {
            'max_results': self.config.max_knowledge_items_per_sync,
            'min_relevance': self.config.knowledge_relevance_threshold
        }
        
        # 添加上下文相关的查询条件
        if 'agent_type' in learning_context:
            query_params['tags'] = [learning_context['agent_type']]
        
        if 'task_type' in learning_context:
            query_params['content_keywords'] = [learning_context['task_type']]
        
        if 'performance_threshold' in learning_context:
            query_params['min_importance'] = learning_context['performance_threshold']
        
        return query_params
    
    async def _filter_relevant_knowledge(self, 
                                        knowledge_items: List[Any], 
                                        learning_context: Dict) -> List[Any]:
        """过滤相关知识"""
        relevant_items = []
        
        for item in knowledge_items:
            relevance_score = self.relevance_calculator.calculate_relevance(
                item, learning_context
            )
            
            if relevance_score >= self.config.knowledge_relevance_threshold:
                # 添加相关性分数到元数据
                if hasattr(item, 'metadata'):
                    item.metadata['relevance_score'] = relevance_score
                
                relevant_items.append(item)
        
        # 按相关性排序
        relevant_items.sort(
            key=lambda x: x.metadata.get('relevance_score', 0.0) if hasattr(x, 'metadata') else 0.0,
            reverse=True
        )
        
        return relevant_items
    
    async def _immediate_sync_experiences(self, 
                                        experiences: List[Experience], 
                                        knowledge_pool: Any, 
                                        result: SyncResult) -> SyncResult:
        """立即同步经验"""
        synced_count = 0
        
        for experience in experiences:
            try:
                if self.quality_assessor.assess_experience_quality(experience) >= self.config.min_experience_quality:
                    knowledge_item = await self._convert_experience_to_knowledge(experience)
                    
                    if knowledge_item and hasattr(knowledge_pool, 'add_knowledge'):
                        await knowledge_pool.add_knowledge(knowledge_item)
                        synced_count += 1
                        
            except Exception as e:
                result.errors.append(f"同步经验失败: {str(e)}")
        
        result.items_synced = synced_count
        result.success = synced_count > 0
        
        return result
    
    async def _batch_sync_experiences(self, 
                                    experiences: List[Experience], 
                                    knowledge_pool: Any, 
                                    result: SyncResult) -> SyncResult:
        """批量同步经验"""
        # 添加到待同步队列
        self.pending_syncs['batch'].extend(experiences)
        
        # 如果达到批量大小，立即处理
        if len(self.pending_syncs['batch']) >= self.config.max_knowledge_items_per_sync:
            batch_experiences = self.pending_syncs['batch'][:self.config.max_knowledge_items_per_sync]
            self.pending_syncs['batch'] = self.pending_syncs['batch'][self.config.max_knowledge_items_per_sync:]
            
            result = await self._immediate_sync_experiences(batch_experiences, knowledge_pool, result)
        
        return result
    
    async def _selective_sync_experiences(self, 
                                        experiences: List[Experience], 
                                        knowledge_pool: Any, 
                                        result: SyncResult) -> SyncResult:
        """选择性同步经验"""
        selected_experiences = []
        
        for experience in experiences:
            if self._meets_selective_criteria(experience):
                selected_experiences.append(experience)
        
        if selected_experiences:
            result = await self._immediate_sync_experiences(selected_experiences, knowledge_pool, result)
        
        return result
    
    async def _adaptive_sync_experiences(self, 
                                       experiences: List[Experience], 
                                       knowledge_pool: Any, 
                                       result: SyncResult) -> SyncResult:
        """自适应同步经验"""
        high_priority = []
        normal_priority = []
        
        for experience in experiences:
            quality = self.quality_assessor.assess_experience_quality(experience)
            
            if quality >= self.config.immediate_sync_threshold:
                high_priority.append(experience)
            elif quality >= self.config.min_experience_quality:
                normal_priority.append(experience)
        
        # 立即同步高优先级经验
        if high_priority:
            result = await self._immediate_sync_experiences(high_priority, knowledge_pool, result)
        
        # 批量处理普通优先级经验
        if normal_priority:
            batch_result = await self._batch_sync_experiences(normal_priority, knowledge_pool, 
                                                            SyncResult(
                                                                sync_id=result.sync_id + "_batch",
                                                                sync_type=result.sync_type,
                                                                success=False,
                                                                items_processed=len(normal_priority),
                                                                items_synced=0
                                                            ))
            
            result.items_synced += batch_result.items_synced
            result.errors.extend(batch_result.errors)
        
        result.success = result.items_synced > 0
        
        return result
    
    def _meets_selective_criteria(self, experience: Experience) -> bool:
        """检查是否满足选择性同步条件"""
        criteria = self.config.selective_sync_criteria
        
        # 高奖励经验
        if 'high_reward' in criteria and experience.reward > 0.8:
            return True
        
        # 新颖模式
        if 'novel_pattern' in criteria and experience.metadata.get('novelty_score', 0) > 0.7:
            return True
        
        # 错误恢复
        if 'error_recovery' in criteria and experience.metadata.get('error_recovery', False):
            return True
        
        return False
    
    async def _convert_experience_to_knowledge(self, experience: Experience) -> Optional[Any]:
        """将经验转换为知识项"""
        try:
            if not KNOWLEDGE_MODULE_AVAILABLE:
                return KnowledgeItem(
                    id=f"exp_{experience.agent_id}_{experience.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    type='experience',
                    content=f"Agent {experience.agent_id} experience with reward {experience.reward}",
                    reward=experience.reward,
                    agent_id=experience.agent_id,
                    metadata=experience.metadata,
                    timestamp=experience.timestamp
                )
            
            # 使用实际的knowledge模块接口
            knowledge_item = KnowledgeItem(
                knowledge_id=f"exp_{experience.agent_id}_{experience.timestamp.strftime('%Y%m%d_%H%M%S')}",
                knowledge_type=self._map_experience_to_knowledge_type(experience),
                title=f"经验记录: {experience.agent_id}",
                content=self._generate_experience_description(experience),
                source=KnowledgeSource.LEARNING_SYSTEM,
                status=KnowledgeStatus.ACTIVE,
                importance_score=self._calculate_experience_importance(experience),
                confidence_score=0.7,
                metadata={
                    'agent_id': experience.agent_id,
                    'reward': experience.reward,
                    'done': experience.done,
                    'episode_id': experience.episode_id,
                    'step_id': experience.step_id,
                    'conversion_timestamp': datetime.now().isoformat(),
                    **(experience.metadata or {})
                },
                tags=['experience', experience.agent_id, 'rl_learning'],
                created_at=experience.timestamp,
                updated_at=datetime.now()
            )
            
            return knowledge_item
            
        except Exception as e:
            logger.error(f"转换经验到知识项失败: {e}")
            return None
    
    def _map_experience_to_knowledge_type(self, experience: Experience) -> str:
        """映射经验到知识类型"""
        if experience.reward > 0.8:
            return 'SUCCESS_PATTERN'
        elif experience.reward < -0.5:
            return 'FAILURE_PATTERN'
        elif experience.done:
            return 'COMPLETION_PATTERN'
        else:
            return 'BEHAVIOR_PATTERN'
    
    def _generate_experience_description(self, experience: Experience) -> str:
        """生成经验描述"""
        description = f"智能体 {experience.agent_id} 在步骤 {experience.step_id} 执行动作获得奖励 {experience.reward:.3f}"
        
        if experience.done:
            description += "，任务完成"
        
        if experience.metadata:
            if 'action_type' in experience.metadata:
                description += f"，动作类型: {experience.metadata['action_type']}"
            
            if 'success' in experience.metadata:
                description += f"，执行{'成功' if experience.metadata['success'] else '失败'}"
        
        return description
    
    def _calculate_experience_importance(self, experience: Experience) -> float:
        """计算经验重要性"""
        importance = 0.5  # 基础重要性
        
        # 奖励影响
        importance += abs(experience.reward) * 0.3
        
        # 完成状态影响
        if experience.done:
            importance += 0.2
        
        # 元数据影响
        if experience.metadata:
            if experience.metadata.get('novelty_score', 0) > 0.7:
                importance += 0.2
            
            if experience.metadata.get('error_recovery', False):
                importance += 0.3
        
        return min(1.0, importance)
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 批量同步任务
        if self.config.sync_strategy == SyncStrategy.BATCH:
            self.batch_sync_task = asyncio.create_task(self._batch_sync_loop())
        
        # 清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _batch_sync_loop(self):
        """批量同步循环"""
        interval = self.config.batch_sync_interval_minutes * 60
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                if self.pending_syncs['batch'] and self.knowledge_manager:
                    batch_experiences = self.pending_syncs['batch'][:self.config.max_knowledge_items_per_sync]
                    self.pending_syncs['batch'] = self.pending_syncs['batch'][self.config.max_knowledge_items_per_sync:]
                    
                    await self.update_knowledge_from_experience(
                        batch_experiences, 
                        self.knowledge_manager.knowledge_pool
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"批量同步循环错误: {e}")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                
                # 清理旧的同步历史
                cutoff_time = datetime.now() - timedelta(days=7)
                self.sync_history = deque(
                    [result for result in self.sync_history if result.timestamp > cutoff_time],
                    maxlen=1000
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环错误: {e}")
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """获取集成统计信息"""
        return {
            'config': {
                'enable_integration': self.config.enable_integration,
                'sync_strategy': self.config.sync_strategy.value,
                'learning_weight': self.config.learning_weight,
                'knowledge_weight': self.config.knowledge_weight
            },
            'statistics': self.sync_statistics.copy(),
            'sync_history_length': len(self.sync_history),
            'pending_syncs': {k: len(v) for k, v in self.pending_syncs.items()},
            'recent_sync_results': [
                {
                    'sync_id': result.sync_id,
                    'success': result.success,
                    'items_synced': result.items_synced,
                    'duration': result.duration,
                    'timestamp': result.timestamp.isoformat()
                }
                for result in list(self.sync_history)[-10:]
            ]
        }
    
    async def shutdown(self):
        """关闭集成服务"""
        try:
            if self.batch_sync_task:
                self.batch_sync_task.cancel()
                await self.batch_sync_task
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                await self.cleanup_task
            
            logger.info("知识学习协作服务已关闭")
            
        except Exception as e:
            logger.error(f"关闭知识学习协作服务失败: {e}")


class LearningKnowledgeAdapter:
    """学习知识适配器
    
    提供learning和knowledge模块之间的数据格式转换。
    """
    
    def __init__(self):
        self.logger = logger
    
    def convert_experience_to_knowledge(self, experience: Experience) -> Optional[Any]:
        """将经验转换为知识项（同步版本）"""
        try:
            return KnowledgeItem(
                knowledge_id=f"exp_{experience.agent_id}_{int(experience.timestamp.timestamp())}",
                title=f"经验: {experience.agent_id}",
                content=f"奖励: {experience.reward}, 完成: {experience.done}",
                metadata={
                    'source': 'rl_experience',
                    'agent_id': experience.agent_id,
                    'reward': experience.reward,
                    'timestamp': experience.timestamp.isoformat()
                }
            )
        except Exception as e:
            logger.error(f"转换经验到知识失败: {e}")
            return None
    
    def convert_knowledge_to_learning_context(self, knowledge: Any) -> Dict[str, Any]:
        """将知识转换为学习上下文"""
        try:
            context = {
                'knowledge_id': getattr(knowledge, 'knowledge_id', 'unknown'),
                'knowledge_type': getattr(knowledge, 'knowledge_type', 'general'),
                'importance': getattr(knowledge, 'importance_score', 0.5),
                'confidence': getattr(knowledge, 'confidence_score', 0.5)
            }
            
            # 添加元数据
            if hasattr(knowledge, 'metadata') and knowledge.metadata:
                context.update(knowledge.metadata)
            
            # 添加内容摘要
            if hasattr(knowledge, 'content'):
                context['content_summary'] = knowledge.content[:200] + '...' if len(knowledge.content) > 200 else knowledge.content
            
            return context
            
        except Exception as e:
            logger.error(f"转换知识到学习上下文失败: {e}")
            return {}
    
    def merge_learning_and_knowledge_insights(self, 
                                            learning_insights: List[LearningInsight], 
                                            knowledge_insights: List[Any]) -> List[Dict[str, Any]]:
        """合并学习和知识洞察"""
        merged_insights = []
        
        # 添加学习洞察
        for insight in learning_insights:
            merged_insights.append({
                'source': 'learning',
                'type': insight.insight_type,
                'content': insight.description,
                'importance': insight.importance,
                'actionable': insight.actionable,
                'metadata': insight.metadata or {}
            })
        
        # 添加知识洞察
        for knowledge in knowledge_insights:
            merged_insights.append({
                'source': 'knowledge',
                'type': getattr(knowledge, 'knowledge_type', 'general'),
                'content': getattr(knowledge, 'content', ''),
                'importance': getattr(knowledge, 'importance_score', 0.5),
                'actionable': True,  # 假设知识都是可操作的
                'metadata': getattr(knowledge, 'metadata', {})
            })
        
        # 按重要性排序
        merged_insights.sort(key=lambda x: x['importance'], reverse=True)
        
        return merged_insights


class ExperienceQualityAssessor:
    """经验质量评估器"""
    
    def assess_experience_quality(self, experience: Experience) -> float:
        """评估经验质量"""
        quality_score = 0.5  # 基础分数
        
        # 奖励质量
        reward_quality = min(1.0, abs(experience.reward))
        quality_score += reward_quality * 0.3
        
        # 完整性
        if experience.done:
            quality_score += 0.2
        
        # 元数据丰富度
        if experience.metadata:
            metadata_score = min(1.0, len(experience.metadata) / 10.0)
            quality_score += metadata_score * 0.2
        
        # 时效性
        time_diff = (datetime.now() - experience.timestamp).total_seconds()
        freshness = max(0.0, 1.0 - time_diff / 3600.0)  # 1小时内为新鲜
        quality_score += freshness * 0.1
        
        return min(1.0, quality_score)


class KnowledgeRelevanceCalculator:
    """知识相关性计算器"""
    
    def calculate_relevance(self, knowledge_item: Any, learning_context: Dict) -> float:
        """计算知识相关性"""
        relevance_score = 0.0
        
        # 类型匹配
        knowledge_type = getattr(knowledge_item, 'knowledge_type', '')
        context_type = learning_context.get('task_type', '')
        
        if knowledge_type and context_type and knowledge_type.lower() in context_type.lower():
            relevance_score += 0.3
        
        # 智能体匹配
        if hasattr(knowledge_item, 'metadata') and knowledge_item.metadata:
            item_agent = knowledge_item.metadata.get('agent_id', '')
            context_agent = learning_context.get('agent_id', '')
            
            if item_agent and context_agent and item_agent == context_agent:
                relevance_score += 0.2
        
        # 重要性权重
        importance = getattr(knowledge_item, 'importance_score', 0.5)
        relevance_score += importance * 0.3
        
        # 时效性
        if hasattr(knowledge_item, 'updated_at'):
            time_diff = (datetime.now() - knowledge_item.updated_at).total_seconds()
            freshness = max(0.0, 1.0 - time_diff / (7 * 24 * 3600))  # 7天内为相关
            relevance_score += freshness * 0.2
        
        return min(1.0, relevance_score)


# 工具函数
def create_knowledge_learning_bridge(
    knowledge_manager: Optional[Any] = None,
    config: Optional[IntegrationConfig] = None
) -> KnowledgeLearningBridge:
    """创建知识学习桥接器"""
    return KnowledgeLearningBridge(
        knowledge_manager=knowledge_manager,
        config=config
    )


def create_integration_config(**kwargs) -> IntegrationConfig:
    """创建集成配置"""
    return IntegrationConfig(**kwargs)