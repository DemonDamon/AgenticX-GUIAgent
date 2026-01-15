"""上下文和状态管理模块

基于AgenticX框架提供智能体上下文管理和状态同步功能。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
import json
from collections import defaultdict

# 使用AgenticX的上下文组件
from agenticx.core.agent import AgentContext
from agenticx.core.component import Component
from agenticx.memory.component import MemoryComponent


class ContextType(Enum):
    """上下文类型枚举"""
    GLOBAL = "global"
    SESSION = "session"
    TASK = "task"
    AGENT = "agent"
    TEMPORARY = "temporary"


class StateType(Enum):
    """状态类型枚举"""
    PERSISTENT = "persistent"
    TRANSIENT = "transient"
    SHARED = "shared"
    PRIVATE = "private"


@dataclass
class ContextEntry:
    """上下文条目"""
    key: str
    value: Any
    context_type: ContextType = ContextType.TEMPORARY
    state_type: StateType = StateType.TRANSIENT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_value(self, value: Any) -> None:
        """更新值"""
        self.value = value
        self.updated_at = datetime.now()
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class AgentContext:
    """智能体上下文"""
    agent_id: str
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 上下文数据
    global_context: Dict[str, ContextEntry] = field(default_factory=dict)
    session_context: Dict[str, ContextEntry] = field(default_factory=dict)
    task_context: Dict[str, ContextEntry] = field(default_factory=dict)
    agent_context: Dict[str, ContextEntry] = field(default_factory=dict)
    temporary_context: Dict[str, ContextEntry] = field(default_factory=dict)
    
    # 状态信息
    current_task_id: Optional[str] = None
    current_state: Dict[str, Any] = field(default_factory=dict)
    
    def get_context_store(self, context_type: ContextType) -> Dict[str, ContextEntry]:
        """获取指定类型的上下文存储"""
        if context_type == ContextType.GLOBAL:
            return self.global_context
        elif context_type == ContextType.SESSION:
            return self.session_context
        elif context_type == ContextType.TASK:
            return self.task_context
        elif context_type == ContextType.AGENT:
            return self.agent_context
        else:
            return self.temporary_context
    
    def set_value(
        self,
        key: str,
        value: Any,
        context_type: ContextType = ContextType.TEMPORARY,
        state_type: StateType = StateType.TRANSIENT,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """设置上下文值
        
        Args:
            key: 键名
            value: 值
            context_type: 上下文类型
            state_type: 状态类型
            expires_at: 过期时间
            metadata: 元数据
        """
        context_store = self.get_context_store(context_type)
        
        entry = ContextEntry(
            key=key,
            value=value,
            context_type=context_type,
            state_type=state_type,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        context_store[key] = entry
        self.updated_at = datetime.now()
    
    def get_value(self, key: str, context_type: ContextType = ContextType.TEMPORARY) -> Any:
        """获取上下文值
        
        Args:
            key: 键名
            context_type: 上下文类型
            
        Returns:
            值或None
        """
        context_store = self.get_context_store(context_type)
        entry = context_store.get(key)
        
        if entry and not entry.is_expired():
            return entry.value
        
        return None