#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Learning Engine - Learning Coordinator (基于AgenticX框架重构)
学习协调器：协调五个阶段的学习组件，管理学习流程和状态

重构说明：
- 基于AgenticX的Component基类重构
- 使用AgenticX的事件系统进行学习协调
- 集成AgenticX的工作流引擎进行学习流程管理
- 遵循AgenticX的异步执行和状态管理架构

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Callable, Awaitable
)
import json
import statistics
from uuid import uuid4

from .prior_knowledge import PriorKnowledgeRetriever
from .guided_explorer import GuidedExplorer
from .task_synthesizer import TaskSynthesizer
from .usage_optimizer import UsageOptimizer
from .edge_handler import EdgeHandler
from agenticx.core.component import Component

from core.info_pool import InfoPool, InfoType, InfoPriority
from utils import get_iso_timestamp


class LearningPhase(Enum):
    """学习阶段"""
    PRIOR_KNOWLEDGE = "prior_knowledge"  # 先验知识检索
    GUIDED_EXPLORATION = "guided_exploration"  # 引导探索
    TASK_SYNTHESIS = "task_synthesis"  # 任务合成
    USAGE_OPTIMIZATION = "usage_optimization"  # 使用优化
    EDGE_HANDLING = "edge_handling"  # 边缘处理


class LearningMode(Enum):
    """学习模式"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"  # 并行执行
    ADAPTIVE = "adaptive"  # 自适应执行
    PIPELINE = "pipeline"  # 流水线执行


class LearningStatus(Enum):
    """学习状态"""
    IDLE = "idle"  # 空闲
    INITIALIZING = "initializing"  # 初始化中
    RUNNING = "running"  # 运行中
    PAUSED = "paused"  # 暂停
    COMPLETED = "completed"  # 完成
    ERROR = "error"  # 错误
    SHUTDOWN = "shutdown"  # 关闭


class LearningTaskStatus(Enum):
    """学习任务状态"""
    PENDING = "pending"  # 待处理
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class LearningTaskPriority(Enum):
    """学习任务优先级"""
    LOW = 1  # 低优先级
    MEDIUM = 2  # 中优先级
    HIGH = 3  # 高优先级
    URGENT = 4  # 紧急


@dataclass
class LearningTask:
    """学习任务"""
    task_id: str
    task_type: str
    description: str
    phases: List[LearningPhase]
    priority: int
    context: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningSession:
    """学习会话"""
    session_id: str
    session_type: str
    tasks: List[str]  # task_ids
    mode: LearningMode
    started_at: str
    ended_at: Optional[str] = None
    status: LearningStatus = LearningStatus.INITIALIZING
    progress: Dict[str, float] = field(default_factory=dict)  # phase -> progress
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """学习指标"""
    session_id: str
    phase: LearningPhase
    execution_time: float
    success_rate: float
    knowledge_gained: int
    patterns_discovered: int
    optimizations_applied: int
    errors_handled: int
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)


class LearningCoordinator(Component):
    """学习协调器
    
    协调五个阶段的学习组件，管理学习流程和状态。
    基于AgenticX的Component基类重构。
    """
    
    def __init__(
        self,
        info_pool=None,
        agent_id: Optional[str] = None, # 新增 agent_id 参数
        max_concurrent_tasks: int = 5,
        max_session_history: int = 1000,
        default_mode: LearningMode = LearningMode.ADAPTIVE
    ):
        super().__init__()
        self.info_pool = info_pool
        self.agent_id = agent_id # 存储 agent_id
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_session_history = max_session_history
        self.default_mode = default_mode
        
        # 学习组件
        self.prior_knowledge_retriever = PriorKnowledgeRetriever(info_pool)
        self.guided_explorer = GuidedExplorer(info_pool, self.prior_knowledge_retriever)
        self.task_synthesizer = TaskSynthesizer(info_pool)
        self.usage_optimizer = UsageOptimizer(config={}, info_pool=info_pool)
        self.edge_handler = EdgeHandler(info_pool)
        
        # 核心数据结构
        self.learning_tasks: Dict[str, LearningTask] = {}
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.session_history: List[str] = []  # session_ids
        self.learning_metrics: List[LearningMetrics] = []
        
        # 运行时状态
        self.current_session: Optional[LearningSession] = None
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: deque = deque()
        self.phase_dependencies: Dict[LearningPhase, List[LearningPhase]] = {
            LearningPhase.GUIDED_EXPLORATION: [LearningPhase.PRIOR_KNOWLEDGE],
            LearningPhase.TASK_SYNTHESIS: [LearningPhase.GUIDED_EXPLORATION],
            LearningPhase.USAGE_OPTIMIZATION: [LearningPhase.TASK_SYNTHESIS],
            LearningPhase.EDGE_HANDLING: []  # 可以独立运行
        }
        
        # 配置参数
        self.phase_timeouts: Dict[LearningPhase, float] = {
            LearningPhase.PRIOR_KNOWLEDGE: 30.0,
            LearningPhase.GUIDED_EXPLORATION: 120.0,
            LearningPhase.TASK_SYNTHESIS: 60.0,
            LearningPhase.USAGE_OPTIMIZATION: 45.0,
            LearningPhase.EDGE_HANDLING: 30.0
        }
        
        # 统计信息
        self.coordinator_stats = {
            "total_sessions": 0,
            "completed_sessions": 0,
            "failed_sessions": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_learning_time": 0.0,
            "avg_session_duration": 0.0,
            "knowledge_items_processed": 0,
            "patterns_discovered": 0,
            "optimizations_applied": 0,
            "edge_cases_handled": 0
        }
        
        # 日志
        self.logger = logger
        
        # 事件处理
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """设置事件处理器"""
        if self.info_pool:
            # 监听智能体任务完成事件
            self.info_pool.subscribe(
                InfoType.TASK_COMPLETION,
                self._handle_task_completion
            )
            
            # 监听智能体动作结果事件
            self.info_pool.subscribe(
                InfoType.ACTION_RESULT,
                self._handle_action_result
            )
            
            # 监听反思结果事件
            self.info_pool.subscribe(
                InfoType.REFLECTION_RESULT,
                self._handle_reflection_result
            )
    
    async def _handle_task_completion(self, info: Dict[str, Any]) -> None:
        """处理任务完成事件"""
        try:
            # 触发学习流程
            await self.trigger_learning(
                task_type="task_completion",
                description=f"任务完成学习: {info.get('task_id', 'unknown')}",
                context=info,
                phases=[LearningPhase.TASK_SYNTHESIS, LearningPhase.USAGE_OPTIMIZATION]
            )
        except Exception as e:
            logger.warning(f"处理任务完成事件失败: {e}")
    
    async def _handle_action_result(self, info: Dict[str, Any]) -> None:
        """处理动作结果事件"""
        try:
            # 如果动作失败，触发边缘处理
            if not info.get("success", True):
                await self.trigger_learning(
                    task_type="action_failure",
                    description=f"动作失败学习: {info.get('action_type', 'unknown')}",
                    context=info,
                    phases=[LearningPhase.EDGE_HANDLING]
                )
            else:
                # 成功的动作用于探索学习
                await self.trigger_learning(
                    task_type="action_success",
                    description=f"动作成功学习: {info.get('action_type', 'unknown')}",
                    context=info,
                    phases=[LearningPhase.GUIDED_EXPLORATION]
                )
        except Exception as e:
            logger.warning(f"处理动作结果事件失败: {e}")
    
    async def _handle_reflection_result(self, info: Dict[str, Any]) -> None:
        """处理反思结果事件"""
        try:
            # 反思结果用于知识检索和优化
            await self.trigger_learning(
                task_type="reflection",
                description=f"反思学习: {info.get('reflection_type', 'unknown')}",
                context=info,
                phases=[LearningPhase.PRIOR_KNOWLEDGE, LearningPhase.USAGE_OPTIMIZATION]
            )
        except Exception as e:
            logger.warning(f"处理反思结果事件失败: {e}")
    
    # 新增：统一将 Enum 或普通值转换为字符串或原值，避免 .value 报错
    def _to_value(self, x: Any) -> Any:
        try:
            return x.value if hasattr(x, "value") else x
        except Exception:
            return x

    async def start_learning_session(
        self,
        session_type: str = "general",
        mode: Optional[LearningMode] = None,
        tasks: Optional[List[str]] = None
    ) -> str:
        """启动学习会话"""
        try:
            session_id = str(uuid4())
            
            # 创建学习会话
            session = LearningSession(
                session_id=session_id,
                session_type=session_type,
                tasks=tasks or [],
                mode=mode or self.default_mode,
                started_at=get_iso_timestamp(),
                status=LearningStatus.INITIALIZING
            )
            
            self.learning_sessions[session_id] = session
            self.current_session = session
            self.coordinator_stats["total_sessions"] += 1
            
            # 初始化学习组件
            await self._initialize_learning_components()
            
            # 更新状态
            session.status = LearningStatus.RUNNING
            
            # 发布会话启动事件
            if self.info_pool:
                from agenticx.core.event import Event
                event = Event(
                    type=self._to_value(InfoType.LEARNING_UPDATE),
                    data={
                        "agent_id": "learning_coordinator",
                        "update_type": "session_started",
                        "session_id": session_id,
                        "session_type": session_type,
                        "mode": self._to_value(mode) if mode is not None else self._to_value(self.default_mode),
                        "timestamp": get_iso_timestamp()
                    },
                    agent_id="learning_coordinator"
                )
                self.info_pool.publish(event)
            
            logger.info(f"学习会话已启动: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"启动学习会话失败: {e}")
            raise
    
    async def _initialize_learning_components(self) -> None:
        """初始化学习组件"""
        try:
            # 这里可以添加组件初始化逻辑
            # 例如：加载预训练模型、初始化知识库等
            pass
        except Exception as e:
            logger.warning(f"初始化学习组件失败: {e}")
    
    async def trigger_learning(
        self,
        task_type: str,
        description: str,
        context: Dict[str, Any],
        phases: Optional[List[LearningPhase]] = None,
        priority: int = 5
    ) -> str:
        """触发学习任务"""
        try:
            task_id = str(uuid4())
            
            # 创建学习任务
            learning_task = LearningTask(
                task_id=task_id,
                task_type=task_type,
                description=description,
                phases=phases or list(LearningPhase),
                priority=priority,
                context=context,
                created_at=get_iso_timestamp()
            )
            
            self.learning_tasks[task_id] = learning_task
            self.coordinator_stats["total_tasks"] += 1
            
            # 添加到任务队列
            self.task_queue.append(task_id)
            
            # 如果有活跃会话，添加到会话任务列表
            if self.current_session:
                self.current_session.tasks.append(task_id)
            
            # 尝试执行任务
            await self._process_task_queue()
            
            return task_id
            
        except Exception as e:
            logger.error(f"触发学习任务失败: {e}")
            raise
    
    async def _process_task_queue(self) -> None:
        """处理任务队列"""
        try:
            while (self.task_queue and 
                   len(self.active_tasks) < self.max_concurrent_tasks):
                
                task_id = self.task_queue.popleft()
                learning_task = self.learning_tasks.get(task_id)
                
                if not learning_task or learning_task.status != "pending":
                    continue
                
                # 创建任务协程
                task_coroutine = self._execute_learning_task(learning_task)
                async_task = asyncio.create_task(task_coroutine)
                
                self.active_tasks[task_id] = async_task
                learning_task.status = "running"
                learning_task.started_at = get_iso_timestamp()
                
                # 设置任务完成回调
                async_task.add_done_callback(
                    lambda t, tid=task_id: asyncio.create_task(self._on_task_completed(tid, t))
                )
                
        except Exception as e:
            logger.error(f"处理任务队列失败: {e}")
    
    async def _execute_learning_task(self, learning_task: LearningTask) -> None:
        """执行学习任务"""
        try:
            start_time = datetime.now()
            
            # 根据学习模式执行阶段
            if self.current_session and self.current_session.mode == LearningMode.SEQUENTIAL:
                await self._execute_sequential(learning_task)
            elif self.current_session and self.current_session.mode == LearningMode.PARALLEL:
                await self._execute_parallel(learning_task)
            elif self.current_session and self.current_session.mode == LearningMode.PIPELINE:
                await self._execute_pipeline(learning_task)
            else:  # ADAPTIVE
                await self._execute_adaptive(learning_task)
            
            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 更新任务状态
            learning_task.status = "completed"
            learning_task.completed_at = get_iso_timestamp()
            learning_task.metadata["execution_time"] = execution_time
            
            # 更新统计信息
            self.coordinator_stats["completed_tasks"] += 1
            self.coordinator_stats["total_learning_time"] += execution_time
            
        except Exception as e:
            learning_task.status = "failed"
            learning_task.errors.append(str(e))
            self.coordinator_stats["failed_tasks"] += 1
            logger.error(f"执行学习任务失败: {e}")
            raise
    
    async def _execute_sequential(self, learning_task: LearningTask) -> None:
        """顺序执行学习阶段"""
        for phase in learning_task.phases:
            await self._execute_phase(phase, learning_task)
    
    async def _execute_parallel(self, learning_task: LearningTask) -> None:
        """并行执行学习阶段"""
        # 创建所有阶段的协程
        phase_tasks = [
            self._execute_phase(phase, learning_task)
            for phase in learning_task.phases
        ]
        
        # 并行执行
        await asyncio.gather(*phase_tasks, return_exceptions=True)
    
    async def _execute_pipeline(self, learning_task: LearningTask) -> None:
        """流水线执行学习阶段"""
        # 根据依赖关系排序阶段
        sorted_phases = self._sort_phases_by_dependencies(learning_task.phases)
        
        # 流水线执行
        running_phases = {}
        for phase in sorted_phases:
            # 等待依赖阶段完成
            dependencies = self.phase_dependencies.get(phase, [])
            for dep_phase in dependencies:
                if dep_phase in running_phases:
                    await running_phases[dep_phase]
            
            # 启动当前阶段
            phase_task = asyncio.create_task(self._execute_phase(phase, learning_task))
            running_phases[phase] = phase_task
        
        # 等待所有阶段完成
        await asyncio.gather(*running_phases.values(), return_exceptions=True)
    
    async def _execute_adaptive(self, learning_task: LearningTask) -> None:
        """自适应执行学习阶段"""
        # 根据任务类型和上下文选择最佳执行策略
        task_type = learning_task.task_type
        context = learning_task.context
        
        if task_type == "action_failure":
            # 失败案例优先处理边缘情况
            priority_phases = [LearningPhase.EDGE_HANDLING, LearningPhase.PRIOR_KNOWLEDGE]
            other_phases = [p for p in learning_task.phases if p not in priority_phases]
            
            # 先执行优先阶段
            for phase in priority_phases:
                if phase in learning_task.phases:
                    await self._execute_phase(phase, learning_task)
            
            # 并行执行其他阶段
            if other_phases:
                phase_tasks = [self._execute_phase(phase, learning_task) for phase in other_phases]
                await asyncio.gather(*phase_tasks, return_exceptions=True)
        
        elif task_type == "task_completion":
            # 任务完成优先合成和优化
            await self._execute_sequential(learning_task)
        
        else:
            # 默认使用流水线执行
            await self._execute_pipeline(learning_task)
    
    def _sort_phases_by_dependencies(self, phases: List[LearningPhase]) -> List[LearningPhase]:
        """根据依赖关系排序阶段"""
        sorted_phases = []
        remaining_phases = set(phases)
        
        while remaining_phases:
            # 找到没有未满足依赖的阶段
            ready_phases = []
            for phase in remaining_phases:
                dependencies = self.phase_dependencies.get(phase, [])
                if all(dep not in remaining_phases for dep in dependencies):
                    ready_phases.append(phase)
            
            if not ready_phases:
                # 如果没有就绪的阶段，说明有循环依赖，按原顺序处理
                ready_phases = list(remaining_phases)
            
            # 添加就绪阶段并从剩余阶段中移除
            sorted_phases.extend(ready_phases)
            remaining_phases -= set(ready_phases)
        
        return sorted_phases
    
    async def _execute_phase(self, phase: LearningPhase, learning_task: LearningTask) -> None:
        """执行学习阶段"""
        try:
            start_time = datetime.now()
            phase_timeout = self.phase_timeouts.get(phase, 60.0)
            
            # 根据阶段类型调用相应的学习组件
            if phase == LearningPhase.PRIOR_KNOWLEDGE:
                result = await asyncio.wait_for(
                    self._execute_prior_knowledge_phase(learning_task),
                    timeout=phase_timeout
                )
            elif phase == LearningPhase.GUIDED_EXPLORATION:
                result = await asyncio.wait_for(
                    self._execute_guided_exploration_phase(learning_task),
                    timeout=phase_timeout
                )
            elif phase == LearningPhase.TASK_SYNTHESIS:
                result = await asyncio.wait_for(
                    self._execute_task_synthesis_phase(learning_task),
                    timeout=phase_timeout
                )
            elif phase == LearningPhase.USAGE_OPTIMIZATION:
                result = await asyncio.wait_for(
                    self._execute_usage_optimization_phase(learning_task),
                    timeout=phase_timeout
                )
            elif phase == LearningPhase.EDGE_HANDLING:
                result = await asyncio.wait_for(
                    self._execute_edge_handling_phase(learning_task),
                    timeout=phase_timeout
                )
            else:
                result = {"status": "skipped", "reason": "unknown_phase"}
            
            # 记录阶段结果
            execution_time = (datetime.now() - start_time).total_seconds()
            learning_task.results[self._to_value(phase)] = result
            
            # 创建学习指标
            metrics = LearningMetrics(
                session_id=self.current_session.session_id if self.current_session else "unknown",
                phase=phase,
                execution_time=execution_time,
                success_rate=result.get("success_rate", 0.0),
                knowledge_gained=result.get("knowledge_gained", 0),
                patterns_discovered=result.get("patterns_discovered", 0),
                optimizations_applied=result.get("optimizations_applied", 0),
                errors_handled=result.get("errors_handled", 0),
                timestamp=get_iso_timestamp(),
                details=result
            )
            
            self.learning_metrics.append(metrics)
            
            # 更新会话进度
            if self.current_session:
                self.current_session.progress[self._to_value(phase)] = 1.0
            
        except asyncio.TimeoutError:
            logger.warning(f"学习阶段超时: {self._to_value(phase)}")
            learning_task.errors.append(f"阶段{self._to_value(phase)}执行超时")
        except Exception as e:
            logger.error(f"执行学习阶段失败: {self._to_value(phase)} - {e}")
            learning_task.errors.append(f"阶段{self._to_value(phase)}执行失败: {str(e)}")
    
    async def _execute_prior_knowledge_phase(self, learning_task: LearningTask) -> Dict[str, Any]:
        """执行先验知识检索阶段"""
        try:
            # 从任务上下文中提取查询信息
            query = learning_task.context.get("query", learning_task.description)
            task_type = learning_task.context.get("task_type", "general")
            
            # 检索先验知识
            knowledge_items = await self.prior_knowledge_retriever.retrieve_knowledge(
                query=query,
                task_type=task_type,
                limit=10
            )
            
            # 分析知识质量
            analysis_result = await self.prior_knowledge_retriever.analyze_knowledge_quality(
                knowledge_items
            )
            
            return {
                "status": "completed",
                "knowledge_items": len(knowledge_items),
                "knowledge_gained": len(knowledge_items),
                "success_rate": analysis_result.get("avg_quality", 0.0),
                "details": {
                    "knowledge_items": knowledge_items,
                    "analysis": analysis_result
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "knowledge_gained": 0,
                "success_rate": 0.0
            }
    
    async def _execute_guided_exploration_phase(self, learning_task: LearningTask) -> Dict[str, Any]:
        """执行引导探索阶段"""
        try:
            # 从先验知识结果中获取指导信息
            prior_knowledge = learning_task.results.get("prior_knowledge", {})
            knowledge_items = prior_knowledge.get("details", {}).get("knowledge_items", [])
            
            # 生成探索候选
            candidates = await self.guided_explorer.generate_exploration_candidates(
                task_context=learning_task.context,
                prior_knowledge=knowledge_items,
                num_candidates=5
            )
            
            # 执行探索
            exploration_results = []
            for candidate in candidates[:3]:  # 限制探索数量
                result = await self.guided_explorer.execute_exploration(
                    candidate,
                    learning_task.context
                )
                exploration_results.append(result)
            
            # 评估探索结果
            evaluation = await self.guided_explorer.evaluate_exploration_results(
                exploration_results
            )
            
            return {
                "status": "completed",
                "explorations_executed": len(exploration_results),
                "success_rate": evaluation.get("success_rate", 0.0),
                "patterns_discovered": evaluation.get("patterns_found", 0),
                "details": {
                    "candidates": candidates,
                    "results": exploration_results,
                    "evaluation": evaluation
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "explorations_executed": 0,
                "success_rate": 0.0,
                "patterns_discovered": 0
            }
    
    async def _execute_task_synthesis_phase(self, learning_task: LearningTask) -> Dict[str, Any]:
        """执行任务合成阶段"""
        try:
            # 从探索结果中获取成功模式
            exploration_results = learning_task.results.get("guided_exploration", {})
            exploration_data = exploration_results.get("details", {}).get("results", [])
            
            # 合成任务
            synthesis_result = await self.task_synthesizer.synthesize_from_exploration(
                exploration_results=exploration_data,
                context=learning_task.context
            )
            
            # 获取合成的任务和策略
            synthesized_tasks = await self.task_synthesizer.get_synthesized_tasks(limit=5)
            synthesized_strategies = await self.task_synthesizer.get_synthesized_strategies(limit=5)
            
            return {
                "status": "completed",
                "tasks_synthesized": len(synthesized_tasks),
                "strategies_synthesized": len(synthesized_strategies),
                "success_rate": synthesis_result.get("quality_score", 0.0),
                "patterns_discovered": len(synthesis_result.get("patterns", [])),
                "details": {
                    "synthesis_result": synthesis_result,
                    "tasks": synthesized_tasks,
                    "strategies": synthesized_strategies
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "tasks_synthesized": 0,
                "strategies_synthesized": 0,
                "success_rate": 0.0,
                "patterns_discovered": 0
            }
    
    async def _execute_usage_optimization_phase(self, learning_task: LearningTask) -> Dict[str, Any]:
        """执行使用优化阶段"""
        try:
            # 监控当前性能
            performance_metrics = await self.usage_optimizer.monitor_performance(
                agent_id=learning_task.context.get("agent_id", "unknown"),
                task_type=learning_task.task_type
            )
            
            # 检查优化机会
            optimization_opportunities = await self.usage_optimizer.check_optimization_opportunities(
                performance_metrics
            )
            
            # 生成优化建议
            recommendations = await self.usage_optimizer.generate_optimization_recommendations(
                optimization_opportunities
            )
            
            # 应用优化
            applied_optimizations = 0
            for recommendation in recommendations[:3]:  # 限制优化数量
                success = await self.usage_optimizer.apply_optimization(
                    recommendation,
                    learning_task.context
                )
                if success:
                    applied_optimizations += 1
            
            return {
                "status": "completed",
                "optimizations_applied": applied_optimizations,
                "recommendations_generated": len(recommendations),
                "success_rate": applied_optimizations / max(len(recommendations), 1),
                "details": {
                    "performance_metrics": performance_metrics,
                    "opportunities": optimization_opportunities,
                    "recommendations": recommendations
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "optimizations_applied": 0,
                "recommendations_generated": 0,
                "success_rate": 0.0
            }
    
    async def _execute_edge_handling_phase(self, learning_task: LearningTask) -> Dict[str, Any]:
        """执行边缘处理阶段"""
        try:
            # 检查是否有错误信息
            error_info = learning_task.context.get("error", {})
            
            if error_info:
                # 处理边缘案例
                from edge_handler import EdgeCaseType
                
                # 确定边缘案例类型
                case_type = EdgeCaseType.UNKNOWN_ERROR
                if "timeout" in str(error_info).lower():
                    case_type = EdgeCaseType.TIMEOUT
                elif "network" in str(error_info).lower():
                    case_type = EdgeCaseType.NETWORK_FAILURE
                elif "permission" in str(error_info).lower():
                    case_type = EdgeCaseType.PERMISSION_DENIED
                
                # 处理边缘案例
                edge_case = await self.edge_handler.handle_edge_case(
                    case_type=case_type,
                    description=learning_task.description,
                    context=learning_task.context,
                    agent_id=learning_task.context.get("agent_id"),
                    task_id=learning_task.task_id,
                    error_details=error_info
                )
                
                return {
                    "status": "completed",
                    "errors_handled": 1,
                    "edge_case_resolved": edge_case.resolution_status == "resolved",
                    "success_rate": 1.0 if edge_case.resolution_status == "resolved" else 0.0,
                    "details": {
                        "edge_case": edge_case
                    }
                }
            else:
                return {
                    "status": "skipped",
                    "reason": "no_errors_to_handle",
                    "errors_handled": 0,
                    "success_rate": 1.0
                }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "errors_handled": 0,
                "success_rate": 0.0
            }
    
    async def _on_task_completed(self, task_id: str, async_task: asyncio.Task) -> None:
        """任务完成回调"""
        try:
            # 从活跃任务中移除
            self.active_tasks.pop(task_id, None)
            
            # 检查任务异常
            if async_task.exception():
                logger.error(f"学习任务异常: {task_id} - {async_task.exception()}")
            
            # 继续处理队列
            await self._process_task_queue()
            
            # 检查会话是否完成
            if self.current_session:
                await self._check_session_completion()
            
        except Exception as e:
            logger.error(f"任务完成回调失败: {e}")
    
    async def _check_session_completion(self) -> None:
        """检查会话是否完成"""
        try:
            if not self.current_session:
                return
            
            # 检查所有任务是否完成
            all_completed = True
            for task_id in self.current_session.tasks:
                task = self.learning_tasks.get(task_id)
                if not task or task.status not in ["completed", "failed"]:
                    all_completed = False
                    break
            
            # 检查是否还有活跃任务
            if all_completed and not self.active_tasks and not self.task_queue:
                await self.end_learning_session()
            
        except Exception as e:
            logger.error(f"检查会话完成状态失败: {e}")
    
    async def end_learning_session(self) -> None:
        """结束学习会话"""
        try:
            if not self.current_session:
                return
            
            session = self.current_session
            session.ended_at = get_iso_timestamp()
            session.status = LearningStatus.COMPLETED
            
            # 计算会话指标
            session_duration = 0.0
            if session.started_at and session.ended_at:
                start_time = datetime.fromisoformat(session.started_at.replace("Z", "+00:00"))
                end_time = datetime.fromisoformat(session.ended_at.replace("Z", "+00:00"))
                session_duration = (end_time - start_time).total_seconds()
            
            # 统计会话结果
            completed_tasks = sum(1 for task_id in session.tasks 
                                if self.learning_tasks.get(task_id, {}).get("status") == "completed")
            failed_tasks = sum(1 for task_id in session.tasks 
                             if self.learning_tasks.get(task_id, {}).get("status") == "failed")
            
            session.metrics = {
                "duration": session_duration,
                "total_tasks": len(session.tasks),
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": completed_tasks / max(len(session.tasks), 1)
            }
            
            # 更新统计信息
            self.coordinator_stats["completed_sessions"] += 1
            if failed_tasks > 0:
                self.coordinator_stats["failed_sessions"] += 1
            
            # 更新平均会话时长
            total_sessions = self.coordinator_stats["completed_sessions"]
            current_avg = self.coordinator_stats["avg_session_duration"]
            self.coordinator_stats["avg_session_duration"] = (
                (current_avg * (total_sessions - 1) + session_duration) / total_sessions
            )
            
            # 添加到历史记录
            self.session_history.append(session.session_id)
            if len(self.session_history) > self.max_session_history:
                # 清理旧会话
                old_session_id = self.session_history.pop(0)
                self.learning_sessions.pop(old_session_id, None)
            
            # 发布会话结束事件
            if self.info_pool:
                from agenticx.core.event import Event
                event = Event(
                    type=self._to_value(InfoType.LEARNING_UPDATE),
                    data={
                        "agent_id": "learning_coordinator",
                        "update_type": "session_completed",
                        "session_id": session.session_id,
                        "metrics": session.metrics,
                        "timestamp": get_iso_timestamp()
                    },
                    agent_id="learning_coordinator"
                )
                self.info_pool.publish(event)
            
            self.current_session = None
            logger.info(f"学习会话已结束: {session.session_id}")
            
        except Exception as e:
            logger.error(f"结束学习会话失败: {e}")
    
    async def pause_learning_session(self) -> bool:
        """暂停学习会话"""
        try:
            if not self.current_session or self.current_session.status != LearningStatus.RUNNING:
                return False
            
            self.current_session.status = LearningStatus.PAUSED
            
            # 暂停所有活跃任务（这里只是标记，实际实现可能需要更复杂的逻辑）
            for task_id, async_task in self.active_tasks.items():
                # 在实际实现中，这里可能需要发送暂停信号给任务
                pass
            
            logger.info(f"学习会话已暂停: {self.current_session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"暂停学习会话失败: {e}")
            return False
    
    async def resume_learning_session(self) -> bool:
        """恢复学习会话"""
        try:
            if not self.current_session or self.current_session.status != LearningStatus.PAUSED:
                return False
            
            self.current_session.status = LearningStatus.RUNNING
            
            # 恢复任务处理
            await self._process_task_queue()
            
            logger.info(f"学习会话已恢复: {self.current_session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"恢复学习会话失败: {e}")
            return False
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        try:
            status = {
                "current_session": None,
                "active_tasks": len(self.active_tasks),
                "queued_tasks": len(self.task_queue),
                "total_sessions": len(self.learning_sessions),
                "coordinator_stats": self.coordinator_stats.copy()
            }
            
            if self.current_session:
                status["current_session"] = {
                    "session_id": self.current_session.session_id,
                    "session_type": self.current_session.session_type,
                    "status": self._to_value(self.current_session.status),
                    "mode": self._to_value(self.current_session.mode),
                    "started_at": self.current_session.started_at,
                    "tasks": len(self.current_session.tasks),
                    "progress": self.current_session.progress
                }
            
            return status
            
        except Exception as e:
            logger.error(f"获取学习状态失败: {e}")
            return {"error": str(e)}
    
    def get_learning_metrics(
        self,
        session_id: Optional[str] = None,
        phase: Optional[LearningPhase] = None,
        limit: int = 100
    ) -> List[LearningMetrics]:
        """获取学习指标"""
        try:
            metrics = self.learning_metrics.copy()
            
            if session_id:
                metrics = [m for m in metrics if m.session_id == session_id]
            
            if phase:
                metrics = [m for m in metrics if m.phase == phase]
            
            # 按时间倒序排序
            metrics.sort(key=lambda x: x.timestamp, reverse=True)
            
            return metrics[:limit]
            
        except Exception as e:
            logger.error(f"获取学习指标失败: {e}")
            return []
    
    async def shutdown(self) -> None:
        """关闭学习协调器"""
        try:
            # 结束当前会话
            if self.current_session:
                self.current_session.status = LearningStatus.SHUTDOWN
                await self.end_learning_session()
            
            # 取消所有活跃任务
            for task_id, async_task in self.active_tasks.items():
                async_task.cancel()
            
            # 等待任务取消完成
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
            
            # 关闭学习组件
            await self.prior_knowledge_retriever.shutdown()
            await self.guided_explorer.shutdown()
            await self.task_synthesizer.shutdown()
            await self.usage_optimizer.shutdown()
            await self.edge_handler.shutdown()
            
            # 发布关闭事件
            if self.info_pool:
                from agenticx.core.event import Event
                event = Event(
                    type=self._to_value(InfoType.LEARNING_UPDATE),
                    data={
                        "agent_id": "learning_coordinator",
                        "update_type": "coordinator_shutdown",
                        "final_stats": self.coordinator_stats,
                        "timestamp": get_iso_timestamp()
                    },
                    agent_id="learning_coordinator"
                )
                self.info_pool.publish(event)
            
            logger.info("学习协调器已关闭")
            
        except Exception as e:
            logger.error(f"关闭学习协调器失败: {e}")