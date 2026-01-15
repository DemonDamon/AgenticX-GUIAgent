#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Knowledge Pool
基于AgenticX框架的知识池：实现统一的知识管理接口和多智能体间的知识共享机制

重构说明：
- 基于AgenticX的Component重构
- 使用AgenticX的事件系统进行知识共享
- 集成AgenticX的Memory和Storage组件
- 提供现代化的多智能体知识协作机制

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
    Callable, Awaitable, AsyncIterator
)
from uuid import uuid4
from enum import Enum

from .knowledge_types import (
    KnowledgeItem, KnowledgeType, KnowledgeSource, KnowledgeStatus,
    KnowledgeRelation, RelationType, QueryRequest, QueryResult,
    SyncRequest, SyncResult, KnowledgeMetadata, KnowledgeGraph
)
# 使用AgenticX核心组件
from agenticx.core.component import Component
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from agenticx.memory.component import MemoryComponent

from .knowledge_store import KnowledgeStoreInterface, KnowledgeStoreFactory
from .knowledge_manager import KnowledgeManager
from .config_loader import load_knowledge_config
from utils import get_iso_timestamp, setup_logger


class AccessLevel(Enum):
    """访问级别"""
    PUBLIC = "public"          # 公开访问
    PROTECTED = "protected"    # 受保护访问
    PRIVATE = "private"        # 私有访问
    RESTRICTED = "restricted"  # 限制访问


class ShareScope(Enum):
    """共享范围"""
    GLOBAL = "global"          # 全局共享
    TEAM = "team"              # 团队共享
    AGENT = "agent"            # 智能体间共享
    LOCAL = "local"            # 本地共享


class KnowledgeSubscription:
    """知识订阅"""
    
    def __init__(
        self,
        subscriber_id: str,
        knowledge_types: Set[KnowledgeType],
        domains: Set[str],
        keywords: Set[str],
        callback: Optional[Callable] = None,
        filters: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.subscriber_id = subscriber_id
        self.knowledge_types = knowledge_types
        self.domains = domains
        self.keywords = keywords
        self.callback = callback
        self.filters = filters or {}
        self.created_at = get_iso_timestamp()
        self.last_notified = None
        self.notification_count = 0
        self.active = True


class KnowledgeAccess:
    """知识访问控制"""
    
    def __init__(
        self,
        knowledge_id: str,
        access_level: AccessLevel = AccessLevel.PUBLIC,
        share_scope: ShareScope = ShareScope.GLOBAL,
        allowed_agents: Optional[Set[str]] = None,
        denied_agents: Optional[Set[str]] = None,
        permissions: Optional[Dict[str, bool]] = None
    ):
        self.knowledge_id = knowledge_id
        self.access_level = access_level
        self.share_scope = share_scope
        self.allowed_agents = allowed_agents or set()
        self.denied_agents = denied_agents or set()
        self.permissions = permissions or {
            'read': True,
            'write': False,
            'delete': False,
            'share': False
        }
        self.created_at = get_iso_timestamp()
        self.updated_at = get_iso_timestamp()


class KnowledgeUsage:
    """知识使用记录"""
    
    def __init__(
        self,
        knowledge_id: str,
        agent_id: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.knowledge_id = knowledge_id
        self.agent_id = agent_id
        self.operation = operation  # read, write, update, delete, share
        self.context = context or {}
        self.timestamp = get_iso_timestamp()
        self.success = True
        self.error_message = None


class KnowledgeRecommendation:
    """知识推荐"""
    
    def __init__(
        self,
        knowledge_id: str,
        target_agent: str,
        relevance_score: float,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.knowledge_id = knowledge_id
        self.target_agent = target_agent
        self.relevance_score = relevance_score
        self.reason = reason
        self.context = context or {}
        self.created_at = get_iso_timestamp()
        self.viewed = False
        self.accepted = False
        self.feedback_score = None


class KnowledgePool(Component):
    """知识池 - 基于AgenticX Component"""
    
    def __init__(
        self,
        knowledge_manager: Optional['KnowledgeManager'] = None,
        event_bus: Optional[EventBus] = None,
        memory: Optional[MemoryComponent] = None,
        enable_access_control: bool = True,
        enable_recommendations: bool = True,
        enable_subscriptions: bool = True
    ):
        super().__init__(name="knowledge_pool")
        
        self.logger = logger
        self.event_bus = event_bus or EventBus()
        
        # 加载知识管理配置并创建KnowledgeManager
        if knowledge_manager:
            self.knowledge_manager = knowledge_manager
        else:
            knowledge_config = load_knowledge_config()
            self.knowledge_manager = KnowledgeManager(
                config=knowledge_config, 
                event_bus=self.event_bus
            )
            
        self.memory = memory
        
        # 功能开关
        self.enable_access_control = enable_access_control
        self.enable_recommendations = enable_recommendations
        self.enable_subscriptions = enable_subscriptions
        
        # 访问控制
        self.access_controls: Dict[str, KnowledgeAccess] = {}
        
        # 订阅管理
        self.subscriptions: Dict[str, KnowledgeSubscription] = {}
        self.subscriber_index: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> subscription_ids
        
        # 使用记录
        self.usage_records: deque = deque(maxlen=10000)
        self.usage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # 推荐系统
        self.recommendations: Dict[str, KnowledgeRecommendation] = {}
        self.agent_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 知识图谱
        self.knowledge_graph = KnowledgeGraph()
        
        # 缓存
        self.query_cache: Dict[str, Tuple[QueryResult, datetime]] = {}
        self.cache_ttl = 300  # 5分钟
        
        # 统计信息
        self.pool_stats = {
            'total_knowledge': 0,
            'active_subscriptions': 0,
            'total_accesses': 0,
            'recommendations_generated': 0,
            'knowledge_shared': 0,
            'last_sync': None
        }
        
        # 运行状态
        self._running = False
        self._recommendation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
    
    async def start(self) -> None:
        """启动知识池"""
        if not self._running:
            self._running = True
            
            # 启动知识管理器
            await self.knowledge_manager.start()
            
            # 启动推荐系统
            if self.enable_recommendations:
                self._recommendation_task = asyncio.create_task(self._recommendation_loop())
            
            # 启动清理任务
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # 注册事件回调
            self.knowledge_manager.register_event_callback(
                'knowledge_stored', self._on_knowledge_stored
            )
            self.knowledge_manager.register_event_callback(
                'knowledge_updated', self._on_knowledge_updated
            )
            
            # 发布启动事件
            await self._publish_event('pool_started', {'timestamp': get_iso_timestamp()})
            
            logger.info("Knowledge pool started")
    
    async def stop(self) -> None:
        """停止知识池"""
        self._running = False
        
        # 停止任务
        for task in [self._recommendation_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 停止知识管理器
        await self.knowledge_manager.stop()
        
        # 发布停止事件
        await self._publish_event('pool_stopped', {'timestamp': get_iso_timestamp()})
        
        logger.info("Knowledge pool stopped")
    
    async def contribute_knowledge(
        self,
        knowledge: KnowledgeItem,
        contributor_id: str,
        access_level: AccessLevel = AccessLevel.PUBLIC,
        share_scope: ShareScope = ShareScope.GLOBAL
    ) -> bool:
        """贡献知识"""
        try:
            # 设置贡献者信息
            knowledge.metadata.contributor_id = contributor_id
            knowledge.metadata.contribution_timestamp = get_iso_timestamp()
            
            # 存储知识
            success = await self.knowledge_manager.store_knowledge(knowledge)
            
            if success:
                # 设置访问控制
                if self.enable_access_control:
                    access_control = KnowledgeAccess(
                        knowledge_id=knowledge.id,
                        access_level=access_level,
                        share_scope=share_scope
                    )
                    self.access_controls[knowledge.id] = access_control
                
                # 更新知识图谱
                await self._update_knowledge_graph(knowledge)
                
                # 记录使用
                await self._record_usage(
                    knowledge.id, contributor_id, 'contribute'
                )
                
                # 触发订阅通知
                if self.enable_subscriptions:
                    await self._notify_subscribers(knowledge)
                
                # 更新统计
                with self._lock:
                    self.pool_stats['total_knowledge'] += 1
                    self.pool_stats['knowledge_shared'] += 1
                
                # 发布事件
                await self._publish_event('knowledge_contributed', {
                    'knowledge_id': knowledge.id,
                    'contributor_id': contributor_id,
                    'type': knowledge.type.value,
                    'timestamp': get_iso_timestamp()
                })
                
                logger.info(f"Knowledge contributed: {knowledge.id} by {contributor_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to contribute knowledge: {e}")
            return False
    
    async def access_knowledge(
        self,
        knowledge_id: str,
        accessor_id: str,
        operation: str = 'read'
    ) -> Optional[KnowledgeItem]:
        """访问知识"""
        try:
            # 检查访问权限
            if self.enable_access_control:
                if not await self._check_access_permission(knowledge_id, accessor_id, operation):
                    logger.warning(
                        f"Access denied: {accessor_id} -> {knowledge_id} ({operation})"
                    )
                    return None
            
            # 获取知识
            knowledge = await self.knowledge_manager.retrieve_knowledge(knowledge_id)
            
            if knowledge:
                # 记录使用
                await self._record_usage(knowledge_id, accessor_id, operation)
                
                # 更新访问统计
                with self._lock:
                    self.pool_stats['total_accesses'] += 1
                
                # 发布事件
                await self._publish_event('knowledge_accessed', {
                    'knowledge_id': knowledge_id,
                    'accessor_id': accessor_id,
                    'operation': operation,
                    'timestamp': get_iso_timestamp()
                })
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Failed to access knowledge {knowledge_id}: {e}")
            return None
    
    async def query_knowledge(
        self,
        request: QueryRequest,
        requester_id: str
    ) -> QueryResult:
        """查询知识"""
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(request, requester_id)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # 执行查询
            result = await self.knowledge_manager.query_knowledge(request)
            
            # 过滤访问权限
            if self.enable_access_control:
                filtered_items = []
                for knowledge in result.items:
                    if await self._check_access_permission(
                        knowledge.id, requester_id, 'read'
                    ):
                        filtered_items.append(knowledge)
                
                result.items = filtered_items
                result.total_count = len(filtered_items)
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            # 记录查询
            await self._record_usage('query', requester_id, 'query', {
                'request_id': request.id,
                'result_count': len(result.items)
            })
            
            # 发布事件
            await self._publish_event('knowledge_queried', {
                'request_id': request.id,
                'requester_id': requester_id,
                'result_count': len(result.items),
                'timestamp': get_iso_timestamp()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            return QueryResult(request_id=request.id)
    
    async def share_knowledge(
        self,
        knowledge_id: str,
        sharer_id: str,
        target_agents: Set[str],
        share_scope: ShareScope = ShareScope.AGENT
    ) -> bool:
        """分享知识"""
        try:
            # 检查分享权限
            if self.enable_access_control:
                if not await self._check_access_permission(knowledge_id, sharer_id, 'share'):
                    logger.warning(
                        f"Share denied: {sharer_id} -> {knowledge_id}"
                    )
                    return False
            
            # 更新访问控制
            if knowledge_id in self.access_controls:
                access_control = self.access_controls[knowledge_id]
                access_control.allowed_agents.update(target_agents)
                access_control.share_scope = share_scope
                access_control.updated_at = get_iso_timestamp()
            
            # 记录分享
            await self._record_usage(knowledge_id, sharer_id, 'share', {
                'target_agents': list(target_agents),
                'share_scope': share_scope.value
            })
            
            # 通知目标智能体
            for target_agent in target_agents:
                await self._notify_agent_knowledge_shared(
                    target_agent, knowledge_id, sharer_id
                )
            
            # 更新统计
            with self._lock:
                self.pool_stats['knowledge_shared'] += 1
            
            # 发布事件
            await self._publish_event('knowledge_shared', {
                'knowledge_id': knowledge_id,
                'sharer_id': sharer_id,
                'target_agents': list(target_agents),
                'share_scope': share_scope.value,
                'timestamp': get_iso_timestamp()
            })
            
            logger.info(
                f"Knowledge shared: {knowledge_id} by {sharer_id} to {len(target_agents)} agents"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to share knowledge {knowledge_id}: {e}")
            return False
    
    async def subscribe_knowledge(
        self,
        subscriber_id: str,
        knowledge_types: Set[KnowledgeType],
        domains: Set[str] = None,
        keywords: Set[str] = None,
        callback: Optional[Callable] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """订阅知识"""
        try:
            subscription = KnowledgeSubscription(
                subscriber_id=subscriber_id,
                knowledge_types=knowledge_types,
                domains=domains or set(),
                keywords=keywords or set(),
                callback=callback,
                filters=filters
            )
            
            # 存储订阅
            self.subscriptions[subscription.id] = subscription
            self.subscriber_index[subscriber_id].add(subscription.id)
            
            # 更新统计
            with self._lock:
                self.pool_stats['active_subscriptions'] += 1
            
            # 发布事件
            await self._publish_event('knowledge_subscribed', {
                'subscription_id': subscription.id,
                'subscriber_id': subscriber_id,
                'knowledge_types': [kt.value for kt in knowledge_types],
                'timestamp': get_iso_timestamp()
            })
            
            logger.info(f"Knowledge subscription created: {subscription.id}")
            
            return subscription.id
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            return ""
    
    async def unsubscribe_knowledge(self, subscription_id: str) -> bool:
        """取消订阅知识"""
        try:
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                subscriber_id = subscription.subscriber_id
                
                # 移除订阅
                del self.subscriptions[subscription_id]
                self.subscriber_index[subscriber_id].discard(subscription_id)
                
                # 更新统计
                with self._lock:
                    self.pool_stats['active_subscriptions'] = max(
                        0, self.pool_stats['active_subscriptions'] - 1
                    )
                
                # 发布事件
                await self._publish_event('knowledge_unsubscribed', {
                    'subscription_id': subscription_id,
                    'subscriber_id': subscriber_id,
                    'timestamp': get_iso_timestamp()
                })
                
                logger.info(f"Knowledge subscription removed: {subscription_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove subscription {subscription_id}: {e}")
            return False
    
    async def get_recommendations(
        self,
        agent_id: str,
        limit: int = 10
    ) -> List[KnowledgeRecommendation]:
        """获取知识推荐"""
        try:
            # 获取该智能体的推荐
            agent_recommendations = [
                rec for rec in self.recommendations.values()
                if rec.target_agent == agent_id and not rec.viewed
            ]
            
            # 按相关性排序
            agent_recommendations.sort(
                key=lambda x: x.relevance_score, reverse=True
            )
            
            # 限制数量
            return agent_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recommendations for {agent_id}: {e}")
            return []
    
    async def provide_feedback(
        self,
        recommendation_id: str,
        feedback_score: float,
        accepted: bool = False
    ) -> bool:
        """提供推荐反馈"""
        try:
            if recommendation_id in self.recommendations:
                recommendation = self.recommendations[recommendation_id]
                recommendation.feedback_score = feedback_score
                recommendation.accepted = accepted
                recommendation.viewed = True
                
                # 更新智能体偏好
                agent_id = recommendation.target_agent
                if agent_id not in self.agent_preferences:
                    self.agent_preferences[agent_id] = {}
                
                # 记录偏好
                knowledge = await self.knowledge_manager.retrieve_knowledge(
                    recommendation.knowledge_id
                )
                if knowledge:
                    # 更新类型偏好
                    type_key = f"type_{knowledge.type.value}"
                    if type_key not in self.agent_preferences[agent_id]:
                        self.agent_preferences[agent_id][type_key] = 0.5
                    
                    # 调整偏好分数
                    current_score = self.agent_preferences[agent_id][type_key]
                    adjustment = (feedback_score - 0.5) * 0.1
                    new_score = max(0.0, min(1.0, current_score + adjustment))
                    self.agent_preferences[agent_id][type_key] = new_score
                
                # 发布事件
                await self._publish_event('recommendation_feedback', {
                    'recommendation_id': recommendation_id,
                    'feedback_score': feedback_score,
                    'accepted': accepted,
                    'timestamp': get_iso_timestamp()
                })
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to provide feedback: {e}")
            return False
    
    async def get_knowledge_graph(self) -> KnowledgeGraph:
        """获取知识图谱"""
        return self.knowledge_graph
    
    async def get_usage_stats(
        self,
        agent_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """获取使用统计"""
        try:
            stats = {}
            
            # 过滤使用记录
            filtered_records = []
            for record in self.usage_records:
                # 按智能体过滤
                if agent_id and record.agent_id != agent_id:
                    continue
                
                # 按时间范围过滤
                if time_range:
                    record_time = datetime.fromisoformat(
                        record.timestamp.replace('Z', '+00:00')
                    )
                    if not (time_range[0] <= record_time <= time_range[1]):
                        continue
                
                filtered_records.append(record)
            
            # 统计操作类型
            operation_counts = defaultdict(int)
            for record in filtered_records:
                operation_counts[record.operation] += 1
            
            stats['operation_counts'] = dict(operation_counts)
            stats['total_operations'] = len(filtered_records)
            
            # 统计知识类型访问
            type_access_counts = defaultdict(int)
            for record in filtered_records:
                if record.operation == 'read':
                    # 这里需要获取知识类型，简化处理
                    type_access_counts['unknown'] += 1
            
            stats['type_access_counts'] = dict(type_access_counts)
            
            # 添加池统计
            stats['pool_stats'] = self.pool_stats.copy()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}
    
    async def sync_with_external(
        self,
        external_pool_url: str,
        sync_type: str = 'bidirectional'
    ) -> SyncResult:
        """与外部知识池同步"""
        try:
            # 创建同步请求
            sync_request = SyncRequest(
                id=str(uuid4()),
                source='external_pool',
                target=external_pool_url,
                sync_type=sync_type
            )
            
            # 执行同步
            result = await self.knowledge_manager.sync_knowledge(sync_request)
            
            # 更新统计
            with self._lock:
                self.pool_stats['last_sync'] = get_iso_timestamp()
            
            # 发布事件
            await self._publish_event('external_sync_completed', {
                'sync_request_id': sync_request.id,
                'external_url': external_pool_url,
                'sync_type': sync_type,
                'success': result.success,
                'synced_count': result.synced_count,
                'timestamp': get_iso_timestamp()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to sync with external pool: {e}")
            return SyncResult(
                request_id=str(uuid4()),
                success=False,
                error_message=str(e)
            )
    
    async def _check_access_permission(
        self,
        knowledge_id: str,
        agent_id: str,
        operation: str
    ) -> bool:
        """检查访问权限"""
        try:
            if knowledge_id not in self.access_controls:
                return True  # 默认允许访问
            
            access_control = self.access_controls[knowledge_id]
            
            # 检查拒绝列表
            if agent_id in access_control.denied_agents:
                return False
            
            # 检查访问级别
            if access_control.access_level == AccessLevel.PRIVATE:
                return agent_id in access_control.allowed_agents
            elif access_control.access_level == AccessLevel.RESTRICTED:
                return agent_id in access_control.allowed_agents
            elif access_control.access_level == AccessLevel.PROTECTED:
                # 受保护访问需要特定权限
                return access_control.permissions.get(operation, False)
            else:  # PUBLIC
                return True
            
        except Exception as e:
            logger.error(f"Failed to check access permission: {e}")
            return False
    
    async def _record_usage(
        self,
        knowledge_id: str,
        agent_id: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录使用"""
        try:
            usage = KnowledgeUsage(
                knowledge_id=knowledge_id,
                agent_id=agent_id,
                operation=operation,
                context=context
            )
            
            self.usage_records.append(usage)
            
            # 更新统计
            with self._lock:
                self.usage_stats[agent_id][operation] += 1
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
    
    async def _notify_subscribers(self, knowledge: KnowledgeItem) -> None:
        """通知订阅者"""
        try:
            for subscription in self.subscriptions.values():
                if not subscription.active:
                    continue
                
                # 检查类型匹配
                if knowledge.type not in subscription.knowledge_types:
                    continue
                
                # 检查域匹配
                if subscription.domains and knowledge.domain not in subscription.domains:
                    continue
                
                # 检查关键词匹配
                if subscription.keywords:
                    knowledge_keywords = knowledge.keywords or set()
                    if not subscription.keywords.intersection(knowledge_keywords):
                        continue
                
                # 检查过滤器
                if subscription.filters:
                    # 这里可以实现更复杂的过滤逻辑
                    pass
                
                # 发送通知
                await self._send_subscription_notification(
                    subscription, knowledge
                )
                
        except Exception as e:
            logger.error(f"Failed to notify subscribers: {e}")
    
    async def _send_subscription_notification(
        self,
        subscription: KnowledgeSubscription,
        knowledge: KnowledgeItem
    ) -> None:
        """发送订阅通知"""
        try:
            # 调用回调函数
            if subscription.callback:
                try:
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(knowledge)
                    else:
                        subscription.callback(knowledge)
                except Exception as e:
                    logger.error(f"Error in subscription callback: {e}")
            
            # 发布到EventBus
            if self.event_bus:
                event = Event(
                    type='knowledge_notification',
                    data={
                        'subscription_id': subscription.id,
                        'subscriber_id': subscription.subscriber_id,
                        'knowledge_id': knowledge.id,
                        'knowledge_type': knowledge.type.value,
                        'timestamp': get_iso_timestamp()
                    },
                    source='knowledge_pool'
                )
                await self.event_bus.publish(event)
            
            # 更新订阅统计
            subscription.last_notified = get_iso_timestamp()
            subscription.notification_count += 1
            
        except Exception as e:
            logger.error(f"Failed to send subscription notification: {e}")
    
    async def _notify_agent_knowledge_shared(
        self,
        agent_id: str,
        knowledge_id: str,
        sharer_id: str
    ) -> None:
        """通知智能体知识被分享"""
        try:
            if self.event_bus:
                event = Event(
                    type='knowledge_shared_notification',
                    data={
                        'target_agent': agent_id,
                        'knowledge_id': knowledge_id,
                        'sharer_id': sharer_id,
                        'timestamp': get_iso_timestamp()
                    },
                    source='knowledge_pool'
                )
                await self.event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Failed to notify agent about shared knowledge: {e}")
    
    async def _update_knowledge_graph(self, knowledge: KnowledgeItem) -> None:
        """更新知识图谱"""
        try:
            # 添加知识节点
            self.knowledge_graph.add_knowledge(knowledge)
            
            # 建立关系
            if knowledge.parent_id:
                relation = KnowledgeRelation(
                    source_id=knowledge.parent_id,
                    target_id=knowledge.id,
                    relation_type=RelationType.PARENT_CHILD,
                    strength=1.0
                )
                self.knowledge_graph.add_relation(relation)
            
            # 建立相关关系
            for related_id in knowledge.related_ids:
                relation = KnowledgeRelation(
                    source_id=knowledge.id,
                    target_id=related_id,
                    relation_type=RelationType.RELATED,
                    strength=0.8
                )
                self.knowledge_graph.add_relation(relation)
            
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")
    
    async def _generate_recommendations(self) -> None:
        """生成推荐"""
        try:
            # 获取所有活跃的智能体
            active_agents = set()
            for subscription in self.subscriptions.values():
                if subscription.active:
                    active_agents.add(subscription.subscriber_id)
            
            # 为每个智能体生成推荐
            for agent_id in active_agents:
                await self._generate_agent_recommendations(agent_id)
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
    
    async def _generate_agent_recommendations(self, agent_id: str) -> None:
        """为智能体生成推荐"""
        try:
            # 获取智能体偏好
            preferences = self.agent_preferences.get(agent_id, {})
            
            # 获取智能体的订阅
            agent_subscriptions = [
                self.subscriptions[sub_id]
                for sub_id in self.subscriber_index.get(agent_id, set())
                if sub_id in self.subscriptions and self.subscriptions[sub_id].active
            ]
            
            if not agent_subscriptions:
                return
            
            # 查询相关知识
            for subscription in agent_subscriptions:
                query_request = QueryRequest(
                    id=str(uuid4()),
                    knowledge_types=list(subscription.knowledge_types),
                    domains=list(subscription.domains) if subscription.domains else None,
                    keywords=list(subscription.keywords) if subscription.keywords else None,
                    limit=20
                )
                
                result = await self.knowledge_manager.query_knowledge(query_request)
                
                # 计算推荐分数
                for knowledge in result.items:
                    relevance_score = self._calculate_relevance_score(
                        knowledge, subscription, preferences
                    )
                    
                    if relevance_score > 0.6:  # 阈值
                        recommendation = KnowledgeRecommendation(
                            knowledge_id=knowledge.id,
                            target_agent=agent_id,
                            relevance_score=relevance_score,
                            reason=f"Matches subscription {subscription.id}",
                            context={
                                'subscription_id': subscription.id,
                                'knowledge_type': knowledge.type.value
                            }
                        )
                        
                        self.recommendations[recommendation.id] = recommendation
                        
                        # 更新统计
                        with self._lock:
                            self.pool_stats['recommendations_generated'] += 1
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations for {agent_id}: {e}")
    
    def _calculate_relevance_score(
        self,
        knowledge: KnowledgeItem,
        subscription: KnowledgeSubscription,
        preferences: Dict[str, Any]
    ) -> float:
        """计算相关性分数"""
        try:
            score = 0.0
            
            # 类型匹配
            if knowledge.type in subscription.knowledge_types:
                score += 0.3
                
                # 考虑偏好
                type_key = f"type_{knowledge.type.value}"
                if type_key in preferences:
                    score += preferences[type_key] * 0.2
            
            # 域匹配
            if subscription.domains and knowledge.domain in subscription.domains:
                score += 0.2
            
            # 关键词匹配
            if subscription.keywords and knowledge.keywords:
                keyword_overlap = len(
                    subscription.keywords.intersection(knowledge.keywords)
                ) / len(subscription.keywords)
                score += keyword_overlap * 0.3
            
            # 质量分数
            score += knowledge.metadata.confidence * 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _generate_cache_key(self, request: QueryRequest, requester_id: str) -> str:
        """生成缓存键"""
        key_data = {
            'requester_id': requester_id,
            'knowledge_types': sorted([kt.value for kt in request.knowledge_types]) if request.knowledge_types else [],
            'domains': sorted(request.domains) if request.domains else [],
            'keywords': sorted(request.keywords) if request.keywords else [],
            'filters': request.filters,
            'limit': request.limit
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """获取缓存结果"""
        if cache_key in self.query_cache:
            result, cached_time = self.query_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                return result
            else:
                del self.query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: QueryResult) -> None:
        """缓存结果"""
        self.query_cache[cache_key] = (result, datetime.now())
        
        # 限制缓存大小
        if len(self.query_cache) > 1000:
            # 移除最旧的缓存项
            oldest_key = min(
                self.query_cache.keys(),
                key=lambda k: self.query_cache[k][1]
            )
            del self.query_cache[oldest_key]
    
    async def _recommendation_loop(self) -> None:
        """推荐循环"""
        while self._running:
            try:
                await self._generate_recommendations()
                await asyncio.sleep(1800)  # 每30分钟生成一次推荐
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recommendation loop: {e}")
                await asyncio.sleep(300)  # 出错时等待5分钟
    
    async def _cleanup_loop(self) -> None:
        """清理循环"""
        while self._running:
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(3600)  # 每小时清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # 出错时等待5分钟
    
    async def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        try:
            current_time = datetime.now()
            
            # 清理过期的推荐
            expired_recommendations = []
            for rec_id, recommendation in self.recommendations.items():
                created_time = datetime.fromisoformat(
                    recommendation.created_at.replace('Z', '+00:00')
                )
                if current_time - created_time > timedelta(days=7):  # 7天过期
                    expired_recommendations.append(rec_id)
            
            for rec_id in expired_recommendations:
                del self.recommendations[rec_id]
            
            # 清理查询缓存
            expired_cache_keys = []
            for cache_key, (result, cached_time) in self.query_cache.items():
                if current_time - cached_time > timedelta(seconds=self.cache_ttl):
                    expired_cache_keys.append(cache_key)
            
            for cache_key in expired_cache_keys:
                del self.query_cache[cache_key]
            
            if expired_recommendations or expired_cache_keys:
                logger.info(
                    f"Cleanup completed: {len(expired_recommendations)} recommendations, "
                    f"{len(expired_cache_keys)} cache entries"
                )
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
    
    async def _on_knowledge_stored(self, event_type: str, data: Dict[str, Any]) -> None:
        """知识存储事件处理"""
        try:
            knowledge_id = data.get('knowledge_id')
            if knowledge_id:
                # 获取知识并更新图谱
                knowledge = await self.knowledge_manager.retrieve_knowledge(knowledge_id)
                if knowledge:
                    await self._update_knowledge_graph(knowledge)
                    
                    # 触发订阅通知
                    if self.enable_subscriptions:
                        await self._notify_subscribers(knowledge)
            
        except Exception as e:
            logger.error(f"Error handling knowledge_stored event: {e}")
    
    async def _on_knowledge_updated(self, event_type: str, data: Dict[str, Any]) -> None:
        """知识更新事件处理"""
        try:
            knowledge_id = data.get('knowledge_id')
            if knowledge_id:
                # 清理相关缓存
                expired_cache_keys = []
                for cache_key in self.query_cache.keys():
                    # 简化处理：清理所有缓存
                    expired_cache_keys.append(cache_key)
                
                for cache_key in expired_cache_keys:
                    del self.query_cache[cache_key]
            
        except Exception as e:
            logger.error(f"Error handling knowledge_updated event: {e}")
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """发布事件"""
        try:
            event_bus = self.event_bus  # 获取引用避免竞态条件
            if event_bus:
                event = Event(
                    type=event_type,
                    data=data,
                    source='knowledge_pool'
                )
                await event_bus.publish(event)
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """获取池统计"""
        try:
            # 获取知识管理器统计
            km_stats = await self.knowledge_manager.get_knowledge_stats()
            
            # 合并统计信息
            stats = {
                **self.pool_stats,
                'knowledge_manager_stats': km_stats,
                'active_subscriptions': len([
                    s for s in self.subscriptions.values() if s.active
                ]),
                'total_recommendations': len(self.recommendations),
                'cache_size': len(self.query_cache),
                'timestamp': get_iso_timestamp()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get pool stats: {e}")
            return {}
    
    async def clear_all_caches(self) -> None:
        """清空所有缓存"""
        try:
            # 清空查询缓存
            self.query_cache.clear()
            
            # 清空知识管理器缓存
            await self.knowledge_manager.clear_cache()
            
            logger.info("All caches cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
    
    async def export_pool_data(self, include_usage: bool = False) -> Dict[str, Any]:
        """导出池数据"""
        try:
            export_data = {
                'knowledge': await self.knowledge_manager.export_knowledge(),
                'access_controls': {
                    k: {
                        'knowledge_id': v.knowledge_id,
                        'access_level': v.access_level.value,
                        'share_scope': v.share_scope.value,
                        'allowed_agents': list(v.allowed_agents),
                        'denied_agents': list(v.denied_agents),
                        'permissions': v.permissions,
                        'created_at': v.created_at,
                        'updated_at': v.updated_at
                    }
                    for k, v in self.access_controls.items()
                },
                'subscriptions': {
                    k: {
                        'subscriber_id': v.subscriber_id,
                        'knowledge_types': [kt.value for kt in v.knowledge_types],
                        'domains': list(v.domains),
                        'keywords': list(v.keywords),
                        'filters': v.filters,
                        'created_at': v.created_at,
                        'active': v.active
                    }
                    for k, v in self.subscriptions.items()
                },
                'stats': await self.get_pool_stats(),
                'export_timestamp': get_iso_timestamp()
            }
            
            if include_usage:
                export_data['usage_records'] = [
                    {
                        'knowledge_id': record.knowledge_id,
                        'agent_id': record.agent_id,
                        'operation': record.operation,
                        'context': record.context,
                        'timestamp': record.timestamp,
                        'success': record.success
                    }
                    for record in list(self.usage_records)
                ]
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export pool data: {e}")
            return {}