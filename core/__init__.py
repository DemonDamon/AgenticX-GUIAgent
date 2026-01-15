#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent核心模块

基于AgenticX框架提供AgenticX-GUIAgent系统的核心功能和组件。
"""

__version__ = "0.1.0"
__author__ = "AgenticX Team"
__description__ = "AgenticX-GUIAgent核心模块 - 基于AgenticX框架"

# 导入核心组件
from .system import AgenticXGUIAgentSystem
from .base_agent import BaseAgenticXGUIAgentAgent, AgentState
from .info_pool import InfoPool, InfoType, InfoPriority, InfoEntry
from .coordinator import AgentCoordinator
from .task import TaskManager, Task, TaskStatus, TaskPriority
from .context import AgentContext, ContextType, StateType
# from .agents import ManagerAgent, ExecutorAgent, ActionReflectorAgent, NotetakerAgent

__all__ = [
    "AgenticXGUIAgentSystem",
    "BaseAgenticXGUIAgentAgent",
    "AgentState",
    "InfoPool",
    "InfoType",
    "InfoPriority", 
    "InfoEntry",
    "AgentCoordinator",
    "TaskManager",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "AgentContext",
    "ContextType",
    "StateType"
    # "ManagerAgent",
    # "ExecutorAgent",
    # "ActionReflectorAgent",
    # "NotetakerAgent"
]