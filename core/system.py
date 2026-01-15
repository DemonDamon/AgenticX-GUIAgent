#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent系统主类模块

提供AgenticX-GUIAgent系统的核心功能和生命周期管理。
"""

import asyncio
from loguru import logger
from typing import Dict, List, Any, Optional, Type
from pathlib import Path

# 使用AgenticX核心组件
from agenticx.core.agent import Agent
from agenticx.core.tool import BaseTool
from agenticx.core.event_bus import EventBus
from agenticx.core.component import Component
from agenticx.memory.component import MemoryComponent
from agenticx.llms.base import BaseLLMProvider

from .base_agent import BaseAgenticXGUIAgentAgent
from config import AgenticXGUIAgentConfig, AgentConfig, WorkflowConfig
from utils import setup_logger, ensure_directory


class AgenticXGUIAgentSystem(Component):
    """AgenticX-GUIAgent系统主类
    
    基于AgenticX Platform构建，负责整个AgenticX-GUIAgent系统的初始化、配置、启动和管理。
    提供统一的系统接口和生命周期管理。
    """
    
    def __init__(self, config: AgenticXGUIAgentConfig):
        super().__init__(name="agenticx_guiagent_system")
        
        self.config = config
        self.logger = logger
        
        # 使用AgenticX EventBus作为核心
        self.event_bus = EventBus()
        
        # 智能体管理
        self.agents: Dict[str, BaseAgenticXGUIAgentAgent] = {}
        self.agent_classes: Dict[str, Type[BaseAgenticXGUIAgentAgent]] = {}
        
        # 系统状态
        self._initialized = False
        self._running = False
        
        # 确保必要目录存在
        self._ensure_directories()
    
    async def initialize(self) -> None:
        """初始化系统"""
        if self._initialized:
            return
        
        logger.info("开始初始化AgenticX-GUIAgent系统...")
        
        try:
            # 初始化InfoPool
            await self._initialize_info_pool()
            
            # 初始化协调器
            await self._initialize_coordinator()
            
            # 注册智能体类
            self._register_agent_classes()
            
            # 创建智能体实例
            await self._create_agents()
            
            # 注册工作流
            await self._register_workflows()
            
            self._initialized = True
            logger.info("AgenticX-GUIAgent系统初始化完成")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    async def start(self) -> None:
        """启动系统"""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            return
        
        logger.info("启动AgenticX-GUIAgent系统...")
        
        try:
            # 启动InfoPool
            await self.info_pool.start()
            
            # 启动协调器
            await self.coordinator.start()
            
            # 启动所有智能体
            for agent in self.agents.values():
                await agent.start()
            
            self._running = True
            logger.info("AgenticX-GUIAgent系统启动完成")
            
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """停止系统"""
        if not self._running:
            return
        
        logger.info("停止AgenticX-GUIAgent系统...")
        
        try:
            # 停止所有智能体
            for agent in self.agents.values():
                try:
                    await agent.stop()
                except Exception as e:
                    logger.error(f"停止智能体失败: {agent.config.id}, 错误: {e}")
            
            # 停止协调器
            if self.coordinator:
                await self.coordinator.stop()
            
            # 停止InfoPool
            if self.info_pool:
                await self.info_pool.stop()
            
            self._running = False
            logger.info("AgenticX-GUIAgent系统已停止")
            
        except Exception as e:
            logger.error(f"系统停止失败: {e}")
    
    async def execute_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """执行移动GUI任务
        
        Args:
            task_description: 任务描述
            **kwargs: 额外参数
        
        Returns:
            任务执行结果
        """
        if not self._running:
            raise RuntimeError("系统未启动")
        
        task_context = {
            "task_id": f"task_{asyncio.get_event_loop().time()}",
            "description": task_description,
            "timestamp": asyncio.get_event_loop().time(),
            **kwargs
        }
        
        logger.info(f"开始执行任务: {task_description}")
        
        try:
            # 通过协调器执行工作流
            result = await self.coordinator.execute_workflow(
                "mobile_gui_task_execution",
                task_context
            )
            
            logger.info(f"任务执行完成: {task_context['task_id']}")
            return result
            
        except Exception as e:
            logger.error(f"任务执行失败: {task_description}, 错误: {e}")
            raise
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgenticXGUIAgentAgent]:
        """获取智能体
        
        Args:
            agent_id: 智能体ID
        
        Returns:
            智能体实例或None
        """
        return self.agents.get(agent_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态
        
        Returns:
            系统状态信息
        """
        agent_states = {}
        for agent_id, agent in self.agents.items():
            agent_states[agent_id] = agent.get_state().__dict__
        
        return {
            "initialized": self._initialized,
            "running": self._running,
            "agents": agent_states,
            "info_pool_stats": self.info_pool.get_stats() if self.info_pool else {},
            "coordinator_status": self.coordinator.get_status() if self.coordinator else {}
        }
    
    def register_agent_class(
        self,
        agent_type: str,
        agent_class: Type[BaseAgenticXGUIAgentAgent]
    ) -> None:
        """注册智能体类
        
        Args:
            agent_type: 智能体类型
            agent_class: 智能体类
        """
        self.agent_classes[agent_type] = agent_class
        logger.info(f"注册智能体类: {agent_type} -> {agent_class.__name__}")
    
    async def add_agent(
        self,
        agent_config: AgentConfig,
        agent_class: Optional[Type[BaseAgenticXGUIAgentAgent]] = None
    ) -> BaseAgenticXGUIAgentAgent:
        """添加智能体
        
        Args:
            agent_config: 智能体配置
            agent_class: 智能体类（可选）
        
        Returns:
            创建的智能体实例
        """
        if agent_config.id in self.agents:
            raise ValueError(f"智能体已存在: {agent_config.id}")
        
        # 确定智能体类
        if agent_class is None:
            agent_class = self.agent_classes.get(agent_config.type)
            if agent_class is None:
                raise ValueError(f"未知的智能体类型: {agent_config.type}")
        
        # 创建智能体实例
        agent = agent_class(
            agent_config=agent_config,
            info_pool=self.info_pool
        )
        
        # 注册到协调器
        self.coordinator.register_agent(agent_config.id, agent)
        
        # 如果系统已运行，启动智能体
        if self._running:
            await agent.start()
        
        self.agents[agent_config.id] = agent
        logger.info(f"添加智能体: {agent_config.id}")
        
        return agent
    
    async def remove_agent(self, agent_id: str) -> None:
        """移除智能体
        
        Args:
            agent_id: 智能体ID
        """
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # 停止智能体
        if self._running:
            await agent.stop()
        
        # 从协调器注销
        self.coordinator.unregister_agent(agent_id)
        
        # 移除
        del self.agents[agent_id]
        logger.info(f"移除智能体: {agent_id}")
    
    async def _initialize_info_pool(self) -> None:
        """初始化InfoPool"""
        self.info_pool = InfoPool(self.config.info_pool)
        await self.info_pool.initialize()
        logger.info("InfoPool初始化完成")
    
    async def _initialize_coordinator(self) -> None:
        """初始化协调器"""
        self.coordinator = AgentCoordinator(
            self.config.workflow,
            self.info_pool
        )
        await self.coordinator.initialize()
        logger.info("协调器初始化完成")
    
    def _register_agent_classes(self) -> None:
        """注册智能体类"""
        # 这里会在后续实现具体的智能体类时进行注册
        # 目前先预留接口
        pass
    
    async def _create_agents(self) -> None:
        """创建智能体实例"""
        for agent_config in self.config.agents.values():
            try:
                # 获取智能体类
                agent_class = self.agent_classes.get(agent_config.type)
                if agent_class is None:
                    logger.warning(f"未找到智能体类: {agent_config.type}，跳过创建")
                    continue
                
                # 创建智能体
                agent = agent_class(
                    agent_config=agent_config,
                    info_pool=self.info_pool
                )
                
                # 注册到协调器
                self.coordinator.register_agent(agent_config.id, agent)
                
                self.agents[agent_config.id] = agent
                logger.info(f"创建智能体: {agent_config.id}")
                
            except Exception as e:
                logger.error(f"创建智能体失败: {agent_config.id}, 错误: {e}")
    
    async def _register_workflows(self) -> None:
        """注册工作流"""
        for workflow_config in self.config.workflow.workflows.values():
            try:
                await self.coordinator.register_workflow(
                    workflow_config.id,
                    workflow_config
                )
                logger.info(f"注册工作流: {workflow_config.id}")
                
            except Exception as e:
                logger.error(f"注册工作流失败: {workflow_config.id}, 错误: {e}")
    
    def _ensure_directories(self) -> None:
        """确保必要目录存在"""
        directories = [
            "logs",
            "data",
            "knowledge",
            "models",
            "screenshots",
            "recordings"
        ]
        
        for directory in directories:
            ensure_directory(directory)
        
        logger.info("目录结构检查完成")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()