#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent基础智能体模块

提供所有AgenticX-GUIAgent智能体的基类和核心功能。
基于AgenticX框架构建，继承其核心Agent类和Component。
"""

import asyncio
from loguru import logger
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

# 导入AgenticX核心组件
from agenticx.core.agent import Agent, AgentContext, AgentResult
from agenticx.core.component import Component
from agenticx.core.event import Event, TaskStartEvent, TaskEndEvent, ErrorEvent
from agenticx.core.tool import BaseTool
from agenticx.llms.base import BaseLLMProvider
from agenticx.memory.component import MemoryComponent

from core.info_pool import InfoPool, InfoType
from config import AgentConfig
from utils import get_iso_timestamp


@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    status: str = "idle"  # idle, busy, error, offline
    current_task: Optional[str] = None
    last_action: Optional[str] = None
    last_update: str = field(default_factory=get_iso_timestamp)
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs) -> None:
        """更新状态"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = get_iso_timestamp()


class BaseAgenticXGUIAgentAgent(Component, ABC):
    """AgenticX-GUIAgent智能体基类
    
    基于AgenticX框架构建，提供所有AgenticX-GUIAgent智能体的基础功能：
    - 继承AgenticX的Agent模型
    - 使用AgenticX事件系统进行通信
    - 状态管理和同步
    - 异步任务执行
    - 错误处理和恢复
    - 学习能力集成
    """
    
    def __init__(
        self,
        agent_config: AgentConfig,
        llm: Optional[BaseLLMProvider] = None,
        memory: Optional[MemoryComponent] = None,
        tools: Optional[List[BaseTool]] = None,
        info_pool: Optional[InfoPool] = None
    ):
        # 初始化Component基类
        super().__init__(name=f"agent_{agent_config.id}")
        
        # 创建AgenticX Agent实例
        self.agent = Agent(
            id=agent_config.id,
            name=agent_config.name,
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=getattr(agent_config, 'backstory', None),
            organization_id=getattr(agent_config, 'organization_id', 'default'),
            llm=llm,
            tool_names=[tool.name for tool in (tools or [])]
        )
        
        self.config = agent_config
        self.info_pool = info_pool or InfoPool()
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.memory = memory
        self.llm_provider = llm  # 添加LLM提供者属性
        
        # 状态管理
        self.state = AgentState(agent_id=agent_config.id)
        self._running = False
        self._task_queue = asyncio.Queue()
        self._current_task: Optional[asyncio.Task] = None
        
        # 学习相关
        self.learning_enabled = getattr(agent_config, 'learning_enabled', False)
        self._learning_components: List[Component] = []
        
        # 日志
        self.logger = logger
        
        # 设置事件监听
        self._setup_event_listeners()
    
    async def start(self) -> None:
        """启动智能体"""
        if self._running:
            return
        
        self._running = True
        self.state.update(status="idle")
        
        # 初始化组件
        await self.initialize()
        
        # 启动任务处理循环
        asyncio.create_task(self._task_processing_loop())
        
        # 初始化学习组件
        if self.learning_enabled:
            await self._initialize_learning_components()
        
        # 发布启动事件
        start_event = Event(
            type="agent_start",
            data={"agent_id": self.config.id, "state": self.state.__dict__},
            agent_id=self.config.id
        )
        await self._publish_event(start_event)
        
        logger.info(f"智能体 {self.config.id} 已启动")
    
    async def stop(self) -> None:
        """停止智能体"""
        if not self._running:
            return
        
        self._running = False
        self.state.update(status="offline")
        
        # 取消当前任务
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        
        # 清理组件资源
        await self.cleanup()
        
        # 发布停止事件
        stop_event = Event(
            type="agent_stop",
            data={"agent_id": self.config.id, "state": self.state.__dict__},
            agent_id=self.config.id
        )
        await self._publish_event(stop_event)
        
        logger.info(f"智能体 {self.config.id} 已停止")
    
    async def execute_task(self, task_context: Dict[str, Any]) -> AgentResult:
        """执行任务
        
        Args:
            task_context: 任务上下文
            
        Returns:
            AgentResult: 任务执行结果
        """
        task_id = task_context.get('task_id', str(uuid4()))
        
        # 创建AgenticX任务上下文
        context = AgentContext(
            agent_id=self.agent.id,
            task_id=task_id,
            variables=task_context
        )
        
        # 发布任务开始事件
        start_event = TaskStartEvent(
            task_description=task_context.get('description', 'Unknown task'),
            agent_id=self.agent.id,
            task_id=task_id
        )
        await self._publish_event(start_event)
        
        try:
            self.state.update(status="busy", current_task=task_id)
            
            # 执行具体任务实现
            result_data = await self._execute_task_impl(task_context)
            
            # 创建成功结果
            result = AgentResult(
                agent_id=self.agent.id,
                task_id=task_id,
                success=True,
                output=result_data
            )
            
            # 发布任务完成事件
            end_event = TaskEndEvent(
                success=True,
                result=result_data,
                agent_id=self.agent.id,
                task_id=task_id
            )
            await self._publish_event(end_event)
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            
            # 创建失败结果
            result = AgentResult(
                agent_id=self.agent.id,
                task_id=task_id,
                success=False,
                error=str(e)
            )
            
            # 发布错误事件
            error_event = ErrorEvent(
                error_type="task_execution_error",
                error_message=str(e),
                agent_id=self.agent.id,
                task_id=task_id
            )
            await self._publish_event(error_event)
            
        finally:
            self.state.update(status="idle", current_task=None)
            
        return result
    
    @abstractmethod
    async def _execute_task_impl(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """具体任务执行实现，由子类重写
        
        Args:
            task_context: 任务上下文
            
        Returns:
            任务执行结果
        """
        raise NotImplementedError("子类必须实现_execute_task_impl方法")
    
    async def _publish_event(self, event: Event) -> None:
        """发布事件
        
        Args:
            event: 要发布的事件
        """
        # 使用InfoPool发布事件
        await self.info_pool.publish(
            info_type=InfoType.AGENT_EVENT,
            data=event.model_dump(),
            source_agent=self.config.id
        )
        self.logger.debug(f"发布事件: {event.type} - {event.data}")
    
    def _setup_event_listeners(self) -> None:
        """设置事件监听器"""
        # 子类可以重写此方法来设置特定的事件监听
        pass
    
    async def _initialize_learning_components(self) -> None:
        """初始化学习组件"""
        if self.learning_enabled:
            # 这里可以初始化具体的学习组件
            logger.info(f"智能体 {self.config.id} 学习组件已初始化")
    
    async def _task_processing_loop(self) -> None:
        """任务处理循环"""
        while self._running:
            try:
                # 从队列获取任务（带超时）
                task_context = await asyncio.wait_for(
                    self._task_queue.get(), 
                    timeout=1.0
                )
                
                # 执行任务
                self._current_task = asyncio.create_task(
                    self.execute_task(task_context)
                )
                await self._current_task
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                logger.error(f"任务处理循环错误: {e}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具实例或None
        """
        return self.tools.get(tool_name)
    
    async def run(self, task_context: Dict[str, Any]) -> AgentResult:
        """运行智能体任务（兼容性方法）
        
        Args:
            task_context: 任务上下文
            
        Returns:
            AgentResult: 任务执行结果
        """
        return await self.execute_task(task_context)