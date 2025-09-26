#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InfoPool信息共享池模块

基于AgenticX EventBus实现智能体间的信息共享机制，支持实时状态同步和知识传递。
使用AgenticX的事件系统替代自定义的InfoPool实现。
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from loguru import logger

# 使用AgenticX的事件系统
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from agenticx.core.component import Component

from utils import get_iso_timestamp, safe_json_dumps, safe_json_loads


class InfoType(Enum):
    """信息类型枚举"""
    TASK_STATUS = "task_status"
    AGENT_STATE = "agent_state"
    SCREEN_STATE = "screen_state"
    ACTION_RESULT = "action_result"
    KNOWLEDGE = "knowledge"
    ERROR = "error"
    METRIC = "metric"
    REFLECTION = "reflection"
    LEARNING_UPDATE = "learning_update"
    TASK_COMPLETION = "task_completion"
    REFLECTION_RESULT = "reflection_result"
    PERFORMANCE_METRICS = "performance_metrics"
    AGENT_STATUS = "agent_status"
    ERROR_REPORT = "error_report"
    # 从communication.py融合的类型
    TASK_PLAN = "task_plan"
    STATUS = "status"
    AGENT_EVENT = "agent_event"


class InfoPriority(Enum):
    """信息优先级枚举"""
    LOW = 1
    NORMAL = 2
    MEDIUM = 2  # 从communication.py融合的别名
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgenticXGUIAgentInfoPool:
    """AgenticX-GUIAgent信息池 - 参考Mobile Agent v3设计，基于AgenticX框架"""
    
    # 用户输入和累积知识
    instruction: str = ""
    task_name: str = ""
    additional_knowledge_manager: List[str] = field(default_factory=list)
    additional_knowledge_executor: List[str] = field(default_factory=list)
    add_info_token: str = "[add_info]"
    
    # UI元素信息
    ui_elements_list_before: str = ""  # 操作前的UI元素列表
    ui_elements_list_after: str = ""   # 操作后的UI元素列表
    action_pool: List[Dict[str, Any]] = field(default_factory=list)
    
    # 工作记忆
    summary_history: List[str] = field(default_factory=list)  # 动作描述历史
    action_history: List[str] = field(default_factory=list)   # 动作历史
    action_outcomes: List[str] = field(default_factory=list)  # 动作结果
    error_descriptions: List[str] = field(default_factory=list)
    
    last_summary: str = ""        # 最后一个动作描述
    last_action: str = ""         # 最后一个动作
    last_action_thought: str = "" # 最后一个动作的思考
    important_notes: str = ""
    
    error_flag_plan: bool = False      # 如果执行器多次尝试后错误未解决
    error_description_plan: str = ""   # 用于修改计划的错误说明
    
    # 规划相关
    plan: str = ""
    completed_plan: str = ""
    progress_status: str = ""
    progress_status_history: List[str] = field(default_factory=list)
    finish_thought: str = ""
    current_subgoal: str = ""
    err_to_manager_thresh: int = 2
    
    # 未来任务
    future_tasks: List[Dict[str, Any]] = field(default_factory=list)
    
    # AgenticX扩展字段
    current_screenshot: Optional[str] = None
    llm_analysis_history: List[Dict[str, Any]] = field(default_factory=list)
    multimodal_context: Dict[str, Any] = field(default_factory=dict)
    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_action_result(self, action: str, summary: str, outcome: str, error_desc: str = ""):
        """添加动作结果"""
        self.action_history.append(action)
        self.summary_history.append(summary)
        self.action_outcomes.append(outcome)
        self.error_descriptions.append(error_desc)
        
        self.last_action = action
        self.last_summary = summary
        
        # 保持历史记录在合理范围内
        max_history = 50
        if len(self.action_history) > max_history:
            self.action_history = self.action_history[-max_history:]
            self.summary_history = self.summary_history[-max_history:]
            self.action_outcomes = self.action_outcomes[-max_history:]
            self.error_descriptions = self.error_descriptions[-max_history:]
    
    def update_plan(self, new_plan: str, completed_subgoal: str = ""):
        """更新计划"""
        if completed_subgoal:
            if self.completed_plan:
                self.completed_plan += f"\n{completed_subgoal}"
            else:
                self.completed_plan = completed_subgoal
        
        self.plan = new_plan
    
    def update_progress(self, status: str):
        """更新进度状态"""
        self.progress_status = status
        self.progress_status_history.append(status)
        
        # 保持进度历史在合理范围内
        if len(self.progress_status_history) > 20:
            self.progress_status_history = self.progress_status_history[-20:]
    
    def add_llm_analysis(self, analysis: Dict[str, Any]):
        """添加LLM分析结果"""
        analysis["timestamp"] = get_iso_timestamp()
        self.llm_analysis_history.append(analysis)
        
        # 保持分析历史在合理范围内
        if len(self.llm_analysis_history) > 10:
            self.llm_analysis_history = self.llm_analysis_history[-10:]
    
    def update_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """更新智能体状态"""
        self.agent_states[agent_id] = {
            **state,
            "last_updated": get_iso_timestamp()
        }
    
    def get_recent_failures(self, count: int = 3) -> List[Dict[str, str]]:
        """获取最近的失败记录"""
        failures = []
        for i in range(len(self.action_outcomes)):
            if self.action_outcomes[i] != "A":  # 非成功状态
                failures.append({
                    "action": self.action_history[i],
                    "summary": self.summary_history[i],
                    "outcome": self.action_outcomes[i],
                    "error": self.error_descriptions[i]
                })
        
        return failures[-count:] if failures else []
    
    def should_escalate_to_manager(self) -> bool:
        """判断是否应该上报给管理器"""
        recent_failures = self.get_recent_failures(self.err_to_manager_thresh)
        return len(recent_failures) >= self.err_to_manager_thresh
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "instruction": self.instruction,
            "task_name": self.task_name,
            "plan": self.plan,
            "completed_plan": self.completed_plan,
            "progress_status": self.progress_status,
            "current_subgoal": self.current_subgoal,
            "last_action": self.last_action,
            "last_summary": self.last_summary,
            "important_notes": self.important_notes,
            "current_screenshot": self.current_screenshot,
            "action_history_count": len(self.action_history),
            "recent_failures": self.get_recent_failures(),
            "agent_states": self.agent_states
        }


@dataclass
class InfoEntry:
    """信息条目 - 基于AgenticX Event的扩展"""
    id: str
    type: InfoType
    priority: InfoPriority
    source_agent: str
    target_agents: Set[str] = field(default_factory=set)  # 空集合表示广播
    data: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None  # 新增：会话ID
    reply_to_id: Optional[str] = None      # 新增：回复的消息ID
    timestamp: str = field(default_factory=get_iso_timestamp)
    ttl: Optional[int] = None  # 生存时间（秒）
    processed_by: Set[str] = field(default_factory=set)

    def to_event(self) -> Event:
        """转换为AgenticX Event"""
        return Event(
            type=self.type.value,
            data={
                "info_id": self.id,
                "priority": self.priority.value,
                "source_agent": self.source_agent,
                "target_agents": list(self.target_agents),
                "content": self.data
            },
            agent_id=self.source_agent
        )
    
    @classmethod
    def from_event(cls, event: Event) -> 'InfoEntry':
        """从AgenticX Event创建InfoEntry"""
        data = event.data
        return cls(
            id=data.get("info_id", event.id),
            type=InfoType(event.type),
            priority=InfoPriority(data.get("priority", InfoPriority.NORMAL.value)),
            source_agent=event.agent_id or data.get("source_agent", "unknown"),
            target_agents=set(data.get("target_agents", [])),
            data=data.get("content", {}),
            conversation_id=data.get("conversation_id"),  # 新增
            reply_to_id=data.get("reply_to_id")            # 新增
        )

    def is_expired(self) -> bool:
        """检查信息是否过期"""
        if self.ttl is None:
            return False
        
        from datetime import datetime
        created_time = datetime.fromisoformat(self.timestamp)
        current_time = datetime.now()
        return (current_time - created_time).total_seconds() > self.ttl
    
    def is_target(self, agent_id: str) -> bool:
        """检查是否为目标智能体"""
        return not self.target_agents or agent_id in self.target_agents
    
    def mark_processed(self, agent_id: str) -> None:
        """标记已被处理"""
        self.processed_by.add(agent_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "source_agent": self.source_agent,
            "target_agents": list(self.target_agents),
            "data": self.data,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "processed_by": list(self.processed_by),
            "conversation_id": self.conversation_id,  # 新增
            "reply_to_id": self.reply_to_id            # 新增
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfoEntry":
        """从字典创建"""
        return cls(
            id=data["id"],
            type=InfoType(data["type"]),
            priority=InfoPriority(data["priority"]),
            source_agent=data["source_agent"],
            target_agents=set(data.get("target_agents", [])),
            data=data.get("data", {}),
            timestamp=data.get("timestamp", get_iso_timestamp()),
            ttl=data.get("ttl"),
            processed_by=set(data.get("processed_by", [])),
            conversation_id=data.get("conversation_id"),  # 新增
            reply_to_id=data.get("reply_to_id")            # 新增
        )


class InfoPool(Component):
    """信息共享池
    
    基于AgenticX EventBus实现，提供智能体间的信息共享机制，支持：
    - 实时信息发布和订阅 (通过EventBus)
    - 基于优先级的信息处理
    - 信息过期和清理
    - 状态同步和历史信息查询
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        max_entries: int = 10000,
        cleanup_interval: int = 60,
        sync_interval: int = 5,
        name: Optional[str] = None,
        **kwargs
    ):
        # 初始化Component基类
        super().__init__(name=name or "InfoPool", **kwargs)
        
        if event_bus:
            self.event_bus = event_bus
        else:
            logger.warning("No event bus provided to InfoPool, creating a new one.")
            self.event_bus = EventBus()

        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval
        self.sync_interval = sync_interval
        
        # 信息存储 (用于查询和持久化)
        self._entries: Dict[str, InfoEntry] = {}
        self._type_index: Dict[InfoType, Set[str]] = defaultdict(set)
        self._agent_index: Dict[str, Set[str]] = defaultdict(set)
        self._priority_queue: Dict[InfoPriority, List[str]] = defaultdict(list)
        self._conversation_index: Dict[str, List[str]] = defaultdict(list) # 新增：会话索引
        
        # 状态管理
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # 从communication.py融合的共享状态字段
        self.current_task: Optional[str] = None
        self.execution_plan: Optional[Dict[str, Any]] = None
        self.action_history: List[Dict[str, Any]] = []
        self.error_descriptions: List[str] = []
        self.important_notes: str = ""
        self._lock = asyncio.Lock()
        
        # 日志
        self.logger = logger
    
    async def start(self) -> None:
        """启动信息池"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("InfoPool已启动")
    
    async def stop(self) -> None:
        """停止信息池"""
        if not self._running:
            return
        
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("InfoPool已停止")
    
    async def publish(self, info_type: Union[InfoType, str], data: Any, source_agent: str,
                      priority: Union[InfoPriority, str] = InfoPriority.NORMAL,
                      destination_agents: Optional[Set[str]] = None,
                      ttl: Optional[int] = None,
                      conversation_id: Optional[str] = None,
                      reply_to_id: Optional[str] = None) -> str:
        """发布一条新信息到信息池。

        Args:
            info_type: 信息类型 (InfoType枚举成员或其value)
            data: 信息内容
            source_agent: 信息来源的智能体ID
            priority: 信息优先级 (InfoPriority枚举成员或其value)
            destination_agents: 目标智能体ID集合，如果为None则为广播
            ttl: 生存时间（秒）
            conversation_id: 对话ID，用于将信息分组
            reply_to_id: 回复的信息ID

        Returns:
            创建的信息条目的ID
        """
        if not self._running:
            raise RuntimeError("信息池未运行，无法发布信息。")

        # --- Start of new code ---
        # 规范化输入，确保我们处理的是枚举成员
        if isinstance(info_type, str):
            try:
                info_type = InfoType(info_type)
            except ValueError:
                logger.error(f"无效的信息类型字符串: '{info_type}'")
                raise
        
        if isinstance(priority, str):
            try:
                priority = InfoPriority(priority)
            except ValueError:
                logger.error(f"无效的优先级字符串: '{priority}'")
                raise
        # --- End of new code ---

        async with self._lock:
            entry_id = str(uuid.uuid4())
            
        entry = InfoEntry(
            id=entry_id,
            type=info_type,
            data=data,
            source_agent=source_agent,
            priority=priority,
            target_agents=destination_agents or set(),
            conversation_id=conversation_id,
            reply_to_id=reply_to_id,
        )

        self._entries[entry.id] = entry
        self._type_index[info_type].add(entry.id)
        self._agent_index[source_agent].add(entry.id)
        self._priority_queue[priority].append(entry.id)
        if conversation_id:
            self._conversation_index[conversation_id].append(entry.id)

        if len(self._entries) > self.max_entries:
            self._cleanup_expired()

        event = entry.to_event()
        if self.event_bus:
            await self.event_bus.publish_async(event)

        logger.debug(
            f"Published info: {info_type.value} from {source_agent} "
            f"to {destination_agents or 'ALL'} (Priority: {priority.value})"
        )
        return entry.id

    def subscribe(
        self,
        callback: Callable[[Event], Any],
        info_types: Optional[List[InfoType]] = None
    ) -> str:
        """订阅事件并返回订阅ID"""
        import uuid
        sub_id = str(uuid.uuid4())
        
        # 存储订阅信息以便后续取消订阅
        if not hasattr(self, '_subscriptions'):
            self._subscriptions = {}
        
        self._subscriptions[sub_id] = {
            'callback': callback,
            'info_types': info_types
        }
        
        if info_types:
            for it in info_types:
                self.event_bus.subscribe(it.value, callback)
        else:
            self.event_bus.subscribe(None, callback)
            
        return sub_id

    def unsubscribe(
        self,
        callback: Callable[[Event], Any] = None,
        info_types: Optional[List[InfoType]] = None,
        sub_id: Optional[str] = None
    ) -> None:
        """取消订阅事件，可以通过callback或sub_id"""
        if sub_id and hasattr(self, '_subscriptions') and sub_id in self._subscriptions:
            # 通过订阅ID取消订阅
            sub_info = self._subscriptions[sub_id]
            callback = sub_info['callback']
            info_types = sub_info['info_types']
            del self._subscriptions[sub_id]
        
        if callback:
            if info_types:
                for it in info_types:
                    self.event_bus.unsubscribe(it.value, callback)
            else:
                self.event_bus.unsubscribe(None, callback)

    def get_entries(
        self,
        info_types: Optional[List[InfoType]] = None,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
        priority: Optional[InfoPriority] = None,
        unprocessed_only: bool = False
    ) -> List[InfoEntry]:
        """获取信息条目
        
        Args:
            info_types: 信息类型过滤
            source_agent: 源智能体过滤
            target_agent: 目标智能体过滤
            priority: 优先级过滤
            unprocessed_only: 仅返回未处理的信息
        
        Returns:
            匹配的信息条目列表
        """
        entries = []
        
        for entry in self._entries.values():
            # 检查过期
            if entry.is_expired():
                continue
            
            # 类型过滤
            if info_types and entry.type not in info_types:
                continue
            
            # 源智能体过滤
            if source_agent and entry.source_agent != source_agent:
                continue
            
            # 目标智能体过滤
            if target_agent and not entry.is_target(target_agent):
                continue
            
            # 优先级过滤
            if priority and entry.priority != priority:
                continue
            
            # 未处理过滤
            if unprocessed_only and target_agent and target_agent in entry.processed_by:
                continue
            
            entries.append(entry)
        
        # 按优先级和时间排序
        entries.sort(key=lambda x: (x.priority.value, x.timestamp), reverse=True)
        return entries
    
    def mark_processed(self, entry_id: str, agent_id: str) -> bool:
        """标记信息已处理
        
        Args:
            entry_id: 信息ID
            agent_id: 智能体ID
        
        Returns:
            是否成功标记
        """
        if entry_id in self._entries:
            self._entries[entry_id].mark_processed(agent_id)
            return True
        return False
    
    def update_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """更新智能体状态
        
        Args:
            agent_id: 智能体ID
            state: 状态数据
        """
        self._agent_states[agent_id] = state
        
        # 发布状态更新信息
        self.publish(
            InfoType.AGENT_STATE,
            {"agent_id": agent_id, "state": state},
            agent_id,
            priority=InfoPriority.NORMAL
        )
    
    def get_conversation(self, conversation_id: str) -> List[InfoEntry]:
        """根据对话ID获取整个对话链"""
        conversation_entries = {}
        for entry in self._entries.values():
            if entry.id == conversation_id or entry.conversation_id == conversation_id:
                conversation_entries[entry.id] = entry
        return sorted(list(conversation_entries.values()), key=lambda e: e.timestamp)

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体的最新状态"""
        return self._agent_states.get(agent_id)
    
    def get_all_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有智能体状态"""
        return self._agent_states.copy()
    
    async def reply_to(
        self,
        original_entry_id: str,
        source_agent: str,
        data: Dict[str, Any],
        priority: Optional[InfoPriority] = None,
    ) -> Optional[str]:
        """
        回复一个已经存在的信息，并将其链接到同一个会话中
        """
        original_entry = self._entries.get(original_entry_id)
        if not original_entry:
            logger.error(f"无法找到要回复的原始信息: {original_entry_id}")
            return None

        conversation_id = original_entry.conversation_id or original_entry.id
        destination_agents = {original_entry.source_agent}

        return await self.publish(
            info_type=original_entry.type,
            data=data,
            source_agent=source_agent,
            priority=priority,
            destination_agents=destination_agents,
            conversation_id=conversation_id,
            reply_to_id=original_entry_id
        )

    def get_conversation(self, conversation_id: str) -> List[InfoEntry]:
        """根据对话ID获取整个对话链"""
        conversation_entries = []
        # Find all entries belonging to this conversation
        for entry in self._entries.values():
            # The root entry has the conversation_id as its own id
            # Replies have the conversation_id field set
            if entry.id == conversation_id or entry.conversation_id == conversation_id:
                conversation_entries.append(entry)
        
        # 按时间戳排序并返回
        return sorted(list(conversation_entries), key=lambda e: e.timestamp)

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体的最新状态"""
        return self._agent_states.get(agent_id)

    def get_all_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有智能体状态"""
        return self._agent_states.copy()
    
    async def reply_to(
        self,
        original_entry_id: str,
        source_agent: str,
        data: Dict[str, Any],
        priority: Optional[InfoPriority] = None,
    ) -> Optional[str]:
        """
        回复一个已经存在的信息，并将其链接到同一个会话中
        """
        original_entry = self._entries.get(original_entry_id)
        if not original_entry:
            logger.error(f"无法找到要回复的原始信息: {original_entry_id}")
            return None

        final_priority = priority if priority is not None else original_entry.priority
        if final_priority is None:
            final_priority = InfoPriority.NORMAL

        conversation_id = original_entry.conversation_id or original_entry.id
        destination_agents = {original_entry.source_agent}

        return await self.publish(
            info_type=original_entry.type,
            data=data,
            source_agent=source_agent,
            priority=final_priority,
            destination_agents=destination_agents,
            conversation_id=conversation_id,
            reply_to_id=original_entry_id
        )

    def get_conversation(self, conversation_id: str) -> List[InfoEntry]:
        """根据对话ID获取整个对话链"""
        conversation_entries = []
        # Find all entries belonging to this conversation
        for entry in self._entries.values():
            # The root entry has the conversation_id as its own id
            # Replies have the conversation_id field set
            if entry.id == conversation_id or entry.conversation_id == conversation_id:
                conversation_entries.append(entry)
        
        # Remove duplicates, just in case
        unique_entries = list({entry.id: entry for entry in conversation_entries}.values())
        
        return sorted(unique_entries, key=lambda e: e.timestamp)

    def get_stats(self) -> Dict[str, Any]:
        """获取信息池的统计数据。"""
        stats = {
            "total_entries": len(self._entries),
            "entries_by_type": {t.value: len(ids) for t, ids in self._type_index.items()},
            "entries_by_priority": {p.value: len(ids) for p, ids in self._priority_queue.items()},
            "active_agents": len(self._agent_states),
            "active_conversations": len(self._conversation_index), # 新增
            "subscribers": self.event_bus.get_subscriber_count() if self.event_bus else 0,
            # 从communication.py融合的统计信息
            "current_task": self.current_task,
            "action_history_count": len(self.action_history),
            "error_count": len(self.error_descriptions)
        }
        return stats
    
    # 从communication.py融合的便捷方法
    async def get_shared_state(self) -> Dict[str, Any]:
        """获取当前共享状态
        
        Returns:
            共享状态字典
        """
        return {
            "current_task": self.current_task,
            "execution_plan": self.execution_plan,
            "action_history": self.action_history[-10:],  # 最近10个操作
            "error_descriptions": self.error_descriptions[-5:],  # 最近5个错误
            "important_notes": self.important_notes,
            "total_entries": len(self._entries),
            "last_updated": get_iso_timestamp()
        }
    
    async def update_task(self, task_description: str, source_agent: str):
        """更新当前任务
        
        Args:
            task_description: 任务描述
            source_agent: 来源智能体
        """
        self.current_task = task_description
        await self.publish(
            info_type=InfoType.TASK_PLAN, 
            data={"task": task_description, "status": "started"}, 
            priority=InfoPriority.HIGH,
            source_agent=source_agent
        )
    
    async def update_execution_plan(self, plan: Dict[str, Any], source_agent: str):
        """更新执行计划
        
        Args:
            plan: 执行计划
            source_agent: 来源智能体
        """
        self.execution_plan = plan
        await self.publish(
            info_type=InfoType.TASK_PLAN, 
            data={"plan": plan, "type": "execution_plan"}, 
            priority=InfoPriority.HIGH,
            source_agent=source_agent
        )
    
    async def add_action_result(self, action_result: Dict[str, Any], source_agent: str):
        """添加操作结果
        
        Args:
            action_result: 操作结果
            source_agent: 来源智能体
        """
        self.action_history.append(action_result)
        # 保持历史记录在合理范围内
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-30:]
        
        await self.publish(
            info_type=InfoType.ACTION_RESULT, 
            data=action_result, 
            priority=InfoPriority.NORMAL,
            source_agent=source_agent
        )
    
    async def add_reflection(self, reflection: Dict[str, Any], source_agent: str):
        """添加反思结果
        
        Args:
            reflection: 反思结果
            source_agent: 来源智能体
        """
        await self.publish(
            info_type=InfoType.REFLECTION, 
            data=reflection, 
            priority=InfoPriority.NORMAL,
            source_agent=source_agent
        )
    
    async def add_knowledge(self, knowledge: Dict[str, Any], source_agent: str):
        """添加知识
        
        Args:
            knowledge: 知识内容
            source_agent: 来源智能体
        """
        await self.publish(
            info_type=InfoType.KNOWLEDGE, 
            data=knowledge, 
            priority=InfoPriority.LOW,
            source_agent=source_agent
        )
    
    async def add_error(self, error_description: str, source_agent: str):
        """添加错误描述
        
        Args:
            error_description: 错误描述
            source_agent: 来源智能体
        """
        self.error_descriptions.append(error_description)
        # 保持错误记录在合理范围内
        if len(self.error_descriptions) > 20:
            self.error_descriptions = self.error_descriptions[-10:]
        
        await self.publish(
            info_type=InfoType.ERROR, 
            data={"error": error_description, "timestamp": get_iso_timestamp()}, 
            priority=InfoPriority.HIGH,
            source_agent=source_agent
        )
    
    async def update_notes(self, notes: str, source_agent: str):
        """更新重要笔记
        
        Args:
            notes: 笔记内容
            source_agent: 来源智能体
        """
        self.important_notes = notes
        await self.publish(
             InfoType.KNOWLEDGE, 
             {"notes": notes, "type": "important_notes"}, 
             InfoPriority.NORMAL,
             source_agent
         )
    
    async def add_info(self, info_type: InfoType, content: Any, source_agent: str,
                      priority: InfoPriority = InfoPriority.NORMAL) -> str:
        """添加信息到池中（兼容communication.py接口）
        
        Args:
            info_type: 信息类型
            content: 信息内容
            source_agent: 来源智能体
            priority: 优先级
            
        Returns:
            信息ID
        """
        async with self._lock:
            # 更新共享状态
            await self._update_shared_state(info_type, content)
            
            # 发布信息
            return await self.publish(
                info_type=info_type,
                data=content if isinstance(content, dict) else {"content": content},
                priority=priority,
                source_agent=source_agent
            )
    
    async def _update_shared_state(self, info_type: InfoType, content: Any):
        """根据新信息更新共享状态（从communication.py融合）
        
        Args:
            info_type: 信息类型
            content: 信息内容
        """
        if info_type == InfoType.TASK_PLAN and isinstance(content, dict):
            if "task" in content:
                self.current_task = content["task"]
            if "plan" in content:
                self.execution_plan = content["plan"]
        
        elif info_type == InfoType.ACTION_RESULT:
            if isinstance(content, dict):
                self.action_history.append(content)
                # 保持历史记录在合理范围内
                if len(self.action_history) > 50:
                    self.action_history = self.action_history[-30:]
        
        elif info_type == InfoType.ERROR and isinstance(content, dict):
            if "error" in content:
                self.error_descriptions.append(content["error"])
                # 保持错误记录在合理范围内
                if len(self.error_descriptions) > 20:
                    self.error_descriptions = self.error_descriptions[-10:]
    
    async def clear_old_entries(self, hours: int = 24):
        """清理旧的信息条目（从communication.py融合）
        
        Args:
            hours: 保留最近多少小时的数据
        """
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        async with self._lock:
            old_count = len(self._entries)
            expired_ids = []
            
            for entry_id, entry in self._entries.items():
                entry_time = datetime.fromisoformat(entry.timestamp)
                if entry_time < cutoff_time and entry.priority.value < InfoPriority.HIGH.value:
                    expired_ids.append(entry_id)
            
            # 清理过期条目
            for entry_id in expired_ids:
                if entry_id in self._entries:
                    entry = self._entries[entry_id]
                    # 从索引中移除
                    self._type_index[entry.type].discard(entry_id)
                    self._agent_index[entry.source_agent].discard(entry_id)
                    if entry_id in self._priority_queue[entry.priority]:
                        self._priority_queue[entry.priority].remove(entry_id)
                    # 从主存储中移除
                    del self._entries[entry_id]
            
            cleaned_count = old_count - len(self._entries)
            
            if cleaned_count > 0:
                # 发布清理事件
                await self.publish(
                    InfoType.STATUS,
                    {
                        'event_type': 'entries_cleaned',
                        'cleaned_count': cleaned_count,
                        'remaining_count': len(self._entries),
                        'cutoff_hours': hours
                    },
                    InfoPriority.LOW,
                    "system"
                )
    
    def _cleanup_expired(self) -> int:
        """清理过期信息"""
        expired_ids = []
        
        for entry_id, entry in self._entries.items():
            if entry.is_expired():
                expired_ids.append(entry_id)
        
        # 如果没有过期信息但超过最大条目数，清理最旧的信息
        if not expired_ids and len(self._entries) > self.max_entries:
            # 按时间排序，删除最旧的
            sorted_entries = sorted(
                self._entries.items(),
                key=lambda x: x[1].timestamp
            )
            expired_ids = [entry_id for entry_id, _ in sorted_entries[:len(self._entries) - self.max_entries + 100]]
        
        # 删除过期信息
        for entry_id in expired_ids:
            entry = self._entries.pop(entry_id, None)
            if entry:
                self._type_index[entry.type].discard(entry_id)
                self._agent_index[entry.source_agent].discard(entry_id)
                for priority_list in self._priority_queue.values():
                    if entry_id in priority_list:
                        priority_list.remove(entry_id)
        
        if expired_ids:
            logger.debug(f"清理了 {len(expired_ids)} 条过期信息")
        
        return len(expired_ids)
    
    async def _cleanup_loop(self) -> None:
        """清理循环"""
        while self._running:
            try:
                self._cleanup_expired()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                await asyncio.sleep(self.cleanup_interval)