#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能体协调器模块

基于AgenticX框架负责管理多智能体的协作和工作流执行。
"""

import asyncio
from loguru import logger
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum

# 使用AgenticX核心组件
from agenticx.core.component import Component
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from agenticx.core.workflow import Workflow
from agenticx.collaboration.manager import CollaborationManager

from config import WorkflowConfig, WorkflowNodeConfig, WorkflowEdgeConfig
from utils import get_iso_timestamp
from core.info_pool import InfoType, InfoPriority


class NodeStatus(Enum):
    """节点状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """工作流状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeExecution:
    """节点执行状态"""
    node_id: str
    status: NodeStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class WorkflowExecution:
    """工作流执行状态"""
    workflow_id: str
    status: WorkflowStatus
    start_time: str
    end_time: Optional[str] = None
    nodes: Dict[str, NodeExecution] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = {}
        if self.context is None:
            self.context = {}


class AgentCoordinator(Component):
    """智能体协调器 - 基于AgenticX框架
    
    负责：
    - 工作流执行管理
    - 智能体任务分配
    - 状态同步和监控
    - 错误处理和恢复
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        max_concurrent_workflows: int = 10,
        max_node_retries: int = 3
    ):
        super().__init__(name="agent_coordinator")
        
        self.event_bus = event_bus or EventBus()
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_node_retries = max_node_retries
        
        # 使用AgenticX的协作管理器
        self.collaboration_manager = CollaborationManager()
        
        # 工作流管理
        self._workflows: Dict[str, WorkflowConfig] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._agent_registry: Dict[str, Any] = {}  # agent_id -> agent_instance
        
        # 执行控制
        self._running = False
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
        # 回调管理
        self._node_callbacks: Dict[str, List[Callable]] = {}
        self._workflow_callbacks: Dict[str, List[Callable]] = {}
        
        # 日志
        self.logger = logger
    
    async def start(self) -> None:
        """启动协调器"""
        if self._running:
            return
        
        self._running = True
        
        # 订阅InfoPool事件
        self.info_pool.subscribe(
            self._handle_agent_state_update,
            [InfoType.AGENT_STATE.value]
        )
        
        logger.info("AgentCoordinator已启动")
    
    async def stop(self) -> None:
        """停止协调器"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消所有执行任务
        for task in self._execution_tasks.values():
            task.cancel()
        
        # 等待任务完成
        if self._execution_tasks:
            await asyncio.gather(
                *self._execution_tasks.values(),
                return_exceptions=True
            )
        
        logger.info("AgentCoordinator已停止")
    
    def register_workflow(self, workflow: WorkflowConfig) -> None:
        """注册工作流
        
        Args:
            workflow: 工作流配置
        """
        self._workflows[workflow.id] = workflow
        logger.info(f"注册工作流: {workflow.id} ({workflow.name})")
    
    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """注册智能体
        
        Args:
            agent_id: 智能体ID
            agent_instance: 智能体实例
        """
        self._agent_registry[agent_id] = agent_instance
        logger.info(f"注册智能体: {agent_id}")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """执行工作流
        
        Args:
            workflow_id: 工作流ID
            context: 执行上下文
        
        Returns:
            执行ID
        
        Raises:
            ValueError: 工作流不存在或已达到最大并发数
        """
        if workflow_id not in self._workflows:
            raise ValueError(f"工作流不存在: {workflow_id}")
        
        if len(self._execution_tasks) >= self.max_concurrent_workflows:
            raise ValueError("已达到最大并发工作流数量")
        
        # 创建执行实例
        import uuid
        execution_id = str(uuid.uuid4())
        
        workflow = self._workflows[workflow_id]
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=get_iso_timestamp(),
            context=context or {}
        )
        
        # 初始化节点执行状态
        for node in workflow.nodes:
            execution.nodes[node.id] = NodeExecution(
                node_id=node.id,
                status=NodeStatus.PENDING
            )
        
        self._executions[execution_id] = execution
        
        # 启动执行任务
        task = asyncio.create_task(
            self._execute_workflow_task(execution_id, workflow, execution)
        )
        self._execution_tasks[execution_id] = task
        
        logger.info(f"开始执行工作流: {workflow_id} (执行ID: {execution_id})")
        
        # 发布工作流开始事件
        self.info_pool.publish(
            InfoType.TASK_STATUS.value,
            {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "started",
                "context": context
            },
            "coordinator",
            priority=InfoPriority.HIGH.value
        )
        
        return execution_id
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """取消工作流执行
        
        Args:
            execution_id: 执行ID
        
        Returns:
            是否成功取消
        """
        if execution_id not in self._execution_tasks:
            return False
        
        # 取消任务
        task = self._execution_tasks[execution_id]
        task.cancel()
        
        # 更新状态
        if execution_id in self._executions:
            execution = self._executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = get_iso_timestamp()
        
        logger.info(f"取消工作流执行: {execution_id}")
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行状态
        
        Args:
            execution_id: 执行ID
        
        Returns:
            执行状态或None
        """
        return self._executions.get(execution_id)
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """获取活跃的执行"""
        return [
            execution for execution in self._executions.values()
            if execution.status == WorkflowStatus.RUNNING
        ]
    
    def add_node_callback(
        self,
        node_id: str,
        callback: Callable[[str, NodeExecution], None]
    ) -> None:
        """添加节点回调
        
        Args:
            node_id: 节点ID
            callback: 回调函数
        """
        if node_id not in self._node_callbacks:
            self._node_callbacks[node_id] = []
        self._node_callbacks[node_id].append(callback)
    
    def add_workflow_callback(
        self,
        workflow_id: str,
        callback: Callable[[str, WorkflowExecution], None]
    ) -> None:
        """添加工作流回调
        
        Args:
            workflow_id: 工作流ID
            callback: 回调函数
        """
        if workflow_id not in self._workflow_callbacks:
            self._workflow_callbacks[workflow_id] = []
        self._workflow_callbacks[workflow_id].append(callback)
    
    async def _execute_workflow_task(
        self,
        execution_id: str,
        workflow: WorkflowConfig,
        execution: WorkflowExecution
    ) -> None:
        """执行工作流任务"""
        try:
            # 构建依赖图
            dependencies = self._build_dependency_graph(workflow)
            
            # 执行节点
            await self._execute_nodes(execution_id, workflow, execution, dependencies)
            
            # 更新完成状态
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = get_iso_timestamp()
            
            logger.info(f"工作流执行完成: {execution_id}")
            
        except asyncio.CancelledError:
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = get_iso_timestamp()
            logger.info(f"工作流执行被取消: {execution_id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = get_iso_timestamp()
            logger.error(f"工作流执行失败: {execution_id}, 错误: {e}")
            
        finally:
            # 清理任务
            self._execution_tasks.pop(execution_id, None)
            
            # 发布完成事件
            self.info_pool.publish(
                InfoType.TASK_STATUS.value,
                {
                    "execution_id": execution_id,
                    "workflow_id": workflow.id,
                    "status": execution.status.value,
                    "result": self._get_execution_result(execution)
                },
                "coordinator",
                priority=InfoPriority.HIGH.value
            )
            
            # 调用工作流回调
            for callback in self._workflow_callbacks.get(workflow.id, []):
                try:
                    callback(execution_id, execution)
                except Exception as e:
                    logger.error(f"工作流回调错误: {e}")
    
    def _build_dependency_graph(self, workflow: WorkflowConfig) -> Dict[str, Set[str]]:
        """构建依赖图
        
        Returns:
            节点ID -> 依赖节点ID集合的映射
        """
        dependencies = {node.id: set() for node in workflow.nodes}
        
        for edge in workflow.edges:
            dependencies[edge.to_node].add(edge.from_node)
        
        return dependencies
    
    async def _execute_nodes(
        self,
        execution_id: str,
        workflow: WorkflowConfig,
        execution: WorkflowExecution,
        dependencies: Dict[str, Set[str]]
    ) -> None:
        """执行节点"""
        completed_nodes = set()
        
        while len(completed_nodes) < len(workflow.nodes):
            # 找到可执行的节点
            ready_nodes = []
            for node in workflow.nodes:
                if (node.id not in completed_nodes and
                    execution.nodes[node.id].status == NodeStatus.PENDING and
                    dependencies[node.id].issubset(completed_nodes)):
                    ready_nodes.append(node)
            
            if not ready_nodes:
                # 检查是否有失败的节点
                failed_nodes = [
                    node_id for node_id, node_exec in execution.nodes.items()
                    if node_exec.status == NodeStatus.FAILED
                ]
                if failed_nodes:
                    raise Exception(f"节点执行失败: {failed_nodes}")
                
                # 没有可执行的节点，可能存在循环依赖
                pending_nodes = [
                    node_id for node_id, node_exec in execution.nodes.items()
                    if node_exec.status == NodeStatus.PENDING
                ]
                if pending_nodes:
                    raise Exception(f"检测到循环依赖或无法满足的依赖: {pending_nodes}")
                
                break
            
            # 并行执行准备好的节点
            tasks = []
            for node in ready_nodes:
                task = asyncio.create_task(
                    self._execute_node(execution_id, node, execution)
                )
                tasks.append((node.id, task))
            
            # 等待节点完成
            for node_id, task in tasks:
                try:
                    await task
                    completed_nodes.add(node_id)
                except Exception as e:
                    logger.error(f"节点执行失败: {node_id}, 错误: {e}")
                    execution.nodes[node_id].status = NodeStatus.FAILED
                    execution.nodes[node_id].error = str(e)
                    execution.nodes[node_id].end_time = get_iso_timestamp()
                    
                    # 如果是关键节点失败，停止执行
                    raise
    
    async def _execute_node(
        self,
        execution_id: str,
        node: WorkflowNodeConfig,
        execution: WorkflowExecution
    ) -> None:
        """执行单个节点"""
        node_execution = execution.nodes[node.id]
        
        # 更新状态
        node_execution.status = NodeStatus.RUNNING
        node_execution.start_time = get_iso_timestamp()
        
        logger.info(f"开始执行节点: {node.id} (类型: {node.type})")
        
        try:
            # 获取智能体实例
            if node.agent_id not in self._agent_registry:
                raise ValueError(f"智能体未注册: {node.agent_id}")
            
            agent = self._agent_registry[node.agent_id]
            
            # 准备任务上下文
            task_context = {
                "execution_id": execution_id,
                "node_id": node.id,
                "workflow_context": execution.context,
                "previous_results": self._get_previous_results(execution, node.id)
            }
            
            # 执行智能体任务
            if hasattr(agent, 'execute_task'):
                result = await agent.execute_task(task_context)
            else:
                # 兼容性处理
                result = await agent.run(task_context)
            
            # 更新结果
            node_execution.status = NodeStatus.COMPLETED
            node_execution.result = result
            node_execution.end_time = get_iso_timestamp()
            
            logger.info(f"节点执行完成: {node.id}")
            
            # 调用节点回调
            for callback in self._node_callbacks.get(node.id, []):
                try:
                    callback(execution_id, node_execution)
                except Exception as e:
                    logger.error(f"节点回调错误: {e}")
            
        except Exception as e:
            node_execution.status = NodeStatus.FAILED
            node_execution.error = str(e)
            node_execution.end_time = get_iso_timestamp()
            
            logger.error(f"节点执行失败: {node.id}, 错误: {e}")
            
            # 重试逻辑
            if node_execution.retry_count < self.max_node_retries:
                node_execution.retry_count += 1
                node_execution.status = NodeStatus.PENDING
                logger.info(f"重试节点: {node.id} (第{node_execution.retry_count}次)")
                await asyncio.sleep(2 ** node_execution.retry_count)  # 指数退避
                await self._execute_node(execution_id, node, execution)
            else:
                raise
    
    def _get_previous_results(self, execution: WorkflowExecution, node_id: str) -> Dict[str, Any]:
        """获取前置节点的结果"""
        results = {}
        for node_exec in execution.nodes.values():
            if (node_exec.node_id != node_id and
                node_exec.status == NodeStatus.COMPLETED and
                node_exec.result is not None):
                results[node_exec.node_id] = node_exec.result
        return results
    
    def _get_execution_result(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """获取执行结果"""
        return {
            "status": execution.status.value,
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "nodes": {
                node_id: {
                    "status": node_exec.status.value,
                    "result": node_exec.result,
                    "error": node_exec.error,
                    "retry_count": node_exec.retry_count
                }
                for node_id, node_exec in execution.nodes.items()
            }
        }
    
    def _handle_agent_state_update(self, entry) -> None:
        """处理智能体状态更新"""
        try:
            data = entry.data
            agent_id = data.get("agent_id")
            state = data.get("state", {})
            
            logger.debug(f"智能体状态更新: {agent_id} -> {state}")
            
        except Exception as e:
            logger.error(f"处理智能体状态更新错误: {e}")