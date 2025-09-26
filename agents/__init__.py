#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Agents Module

基于AgenticX框架的AgenticX-GUIAgent系统智能体实现。
包含四个核心智能体：任务管理、动作执行、反思分析和知识记录。
"""

from .manager_agent import ManagerAgent
from .executor_agent import ExecutorAgent
from .action_reflector_agent import ActionReflectorAgent
from .notetaker_agent import NotetakerAgent

__all__ = [
    "ManagerAgent",
    "ExecutorAgent", 
    "ActionReflectorAgent",
    "NotetakerAgent"
]

__version__ = "1.0.0"
__author__ = "AgenticX Team"
__description__ = "AgenticX-GUIAgent智能体模块 - 基于AgenticX框架的四个核心智能体实现"