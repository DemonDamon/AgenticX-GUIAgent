#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Learning Engine - Main Learning Engine (基于AgenticX框架重构)
学习引擎：整合五个阶段的学习组件，提供统一的学习接口

重构说明：
- 基于AgenticX的Component基类重构
- 使用AgenticX的事件系统进行学习引擎协调
- 集成AgenticX的工作流引擎进行学习流程编排
- 遵循AgenticX的生命周期管理和配置架构

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Callable, Awaitable
)
import json
from uuid import uuid4

from agenticx.core.event import Event

from .learning_coordinator import LearningCoordinator, LearningMode, LearningPhase
from .prior_knowledge import PriorKnowledgeRetriever
from .guided_explorer import GuidedExplorer
from .task_synthesizer import TaskSynthesizer
from .usage_optimizer import UsageOptimizer
from .edge_handler import EdgeHandler
from agenticx.core.component import Component

from core.info_pool import InfoPool, InfoType, InfoPriority
from utils import get_iso_timestamp
# from config import LearningConfig


class LearningEngineStatus(Enum):
    """学习引擎状态"""
    IDLE = "idle"  # 空闲
    INITIALIZING = "initializing"  # 初始化中
    RUNNING = "running"  # 运行中
    LEARNING = "learning"  # 学习中
    OPTIMIZING = "optimizing"  # 优化中
    ERROR = "error"  # 错误
    SHUTDOWN = "shutdown"  # 关闭


class LearningTrigger(Enum):
    """学习触发器"""
    MANUAL = "manual"  # 手动触发
    AUTOMATIC = "automatic"  # 自动触发
    SCHEDULED = "scheduled"  # 定时触发
    EVENT_DRIVEN = "event_driven"  # 事件驱动
    PERFORMANCE_BASED = "performance_based"  # 性能驱动


@dataclass
class LearningConfiguration:
    """学习配置"""
    # 基础配置
    auto_learning_enabled: bool = True
    learning_mode: LearningMode = LearningMode.ADAPTIVE
    max_concurrent_sessions: int = 3
    max_learning_tasks: int = 10
    
    # 触发配置
    learning_triggers: List[LearningTrigger] = field(default_factory=lambda: [
        LearningTrigger.AUTOMATIC,
        LearningTrigger.EVENT_DRIVEN,
        LearningTrigger.PERFORMANCE_BASED
    ])
    
    # 阶段配置
    enabled_phases: List[LearningPhase] = field(default_factory=lambda: list(LearningPhase))
    phase_weights: Dict[str, float] = field(default_factory=lambda: {
        "prior_knowledge": 1.0,
        "guided_exploration": 1.2,
        "task_synthesis": 1.1,
        "usage_optimization": 1.3,
        "edge_handling": 0.8
    })
    
    # 性能配置
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "success_rate_threshold": 0.7,
        "efficiency_threshold": 0.8,
        "error_rate_threshold": 0.1,
        "learning_interval_hours": 1.0
    })
    
    # 资源配置
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 512,
        "max_cpu_percent": 50,
        "max_learning_time_minutes": 30
    })


@dataclass
class LearningResult:
    """学习结果"""
    session_id: str
    trigger: LearningTrigger
    phases_executed: List[str]
    success: bool
    duration: float
    knowledge_gained: int
    patterns_discovered: int
    optimizations_applied: int
    errors_handled: int
    performance_improvement: float
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class LearningEngine(Component):
    """学习引擎
    
    整合五个阶段的学习组件，提供统一的学习接口。
    支持自动学习、手动学习、定时学习等多种学习模式。
    基于AgenticX的Component基类重构。
    """
    
    def __init__(
        self,
        info_pool=None,
        config: Optional[LearningConfiguration] = None
    ):
        super().__init__()
        self.info_pool = info_pool
        self.config = config or LearningConfiguration()
        
        # 核心组件
        self.learning_coordinator = LearningCoordinator(
            info_pool=info_pool,
            max_concurrent_tasks=self.config.max_learning_tasks,
            default_mode=self.config.learning_mode
        )
        
        # 学习组件（通过协调器访问）
        self.prior_knowledge_retriever = self.learning_coordinator.prior_knowledge_retriever
        self.guided_explorer = self.learning_coordinator.guided_explorer
        self.task_synthesizer = self.learning_coordinator.task_synthesizer
        self.usage_optimizer = self.learning_coordinator.usage_optimizer
        self.edge_handler = self.learning_coordinator.edge_handler
        
        # 运行时状态
        self.status = LearningEngineStatus.IDLE
        self.active_sessions: Dict[str, str] = {}  # session_id -> trigger
        self.learning_history: List[LearningResult] = []
        self.performance_baseline: Dict[str, float] = {}
        self.last_learning_time: Optional[datetime] = None
        
        # 定时任务
        self.scheduled_tasks: Dict[str, asyncio.Task] = {}
        self.auto_learning_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.engine_stats = {
            "total_learning_sessions": 0,
            "successful_sessions": 0,
            "failed_sessions": 0,
            "total_learning_time": 0.0,
            "knowledge_items_learned": 0,
            "patterns_discovered": 0,
            "optimizations_applied": 0,
            "edge_cases_handled": 0,
            "performance_improvements": 0,
            "avg_session_duration": 0.0,
            "last_learning_timestamp": None
        }

    # 新增：将 Enum 或普通值统一转换为其字符串值，避免访问 .value 导致异常
    def _to_value(self, x: Any) -> Any:
        try:
            return x.value if hasattr(x, "value") else x
        except Exception:
            return x

    def _publish_event(self, event_type: InfoType, data: Dict[str, Any]) -> None:
        """发布事件的辅助方法"""
        if self.info_pool:
            event = Event(
                type=self._to_value(event_type),
                data=data,
                agent_id="learning_engine"
            )
            self.info_pool.publish(event)

    def _setup_event_handlers(self) -> None:
        """设置事件处理器"""
        if self.info_pool and hasattr(self.info_pool, 'event_bus') and self.info_pool.event_bus:
            # 监听性能指标更新
            self.info_pool.event_bus.subscribe(
                InfoType.PERFORMANCE_METRICS.value,
                self._handle_performance_metrics
            )
            
            # 监听错误事件
            self.info_pool.event_bus.subscribe(
                InfoType.ERROR_REPORT.value,
                self._handle_error_event
            )
            
            # 监听任务完成事件
            self.info_pool.event_bus.subscribe(
                InfoType.TASK_COMPLETION.value,
                self._handle_task_completion_event
            )
    
    async def initialize(self) -> bool:
        """初始化学习引擎"""
        try:
            self.status = LearningEngineStatus.INITIALIZING
            
            # 初始化性能基线
            await self._initialize_performance_baseline()
            
            # 启动自动学习（如果启用）
            if self.config.auto_learning_enabled:
                await self._start_auto_learning()
            
            # 启动定时学习（如果配置）
            if LearningTrigger.SCHEDULED in self.config.learning_triggers:
                await self._start_scheduled_learning()
            
            self.status = LearningEngineStatus.RUNNING
            
            # 发布初始化完成事件
            if self.info_pool:
                # 使用 EventBus 的 publish 方法而不是 publish_info
                event = Event(
                    type=InfoType.LEARNING_UPDATE.value,
                    data={
                        "agent_id": "learning_engine",
                        "update_type": "engine_initialized",
                        "config": {
                            "auto_learning_enabled": self.config.auto_learning_enabled,
                            "learning_mode": self._to_value(self.config.learning_mode),
                            "enabled_phases": [self._to_value(p) for p in self.config.enabled_phases],
                            "triggers": [self._to_value(t) for t in self.config.learning_triggers]
                        },
                        "timestamp": get_iso_timestamp()
                    },
                    agent_id="learning_engine"
                )
                self.info_pool.publish(event)
            
            logger.info("学习引擎初始化完成")
            return True
            
        except Exception as e:
            self.status = LearningEngineStatus.ERROR
            logger.error(f"学习引擎初始化失败: {e}")
            return False
    
    async def _initialize_performance_baseline(self) -> None:
        """初始化性能基线"""
        try:
            # 这里可以从历史数据或配置中加载性能基线
            self.performance_baseline = {
                "success_rate": 0.5,
                "efficiency": 0.5,
                "error_rate": 0.2,
                "response_time": 1.0
            }
        except Exception as e:
            logger.warning(f"初始化性能基线失败: {e}")
    
    async def _start_auto_learning(self) -> None:
        """启动自动学习"""
        try:
            if self.auto_learning_task and not self.auto_learning_task.done():
                return
            
            self.auto_learning_task = asyncio.create_task(self._auto_learning_loop())
            logger.info("自动学习已启动")
            
        except Exception as e:
            logger.error(f"启动自动学习失败: {e}")
    
    async def _auto_learning_loop(self) -> None:
        """自动学习循环"""
        try:
            # 使用默认值，因为 LearningConfig 没有 performance_thresholds 属性
            learning_interval = getattr(self.config, 'performance_thresholds', {}).get("learning_interval_hours", 1.0)
            interval_seconds = learning_interval * 3600
            
            while self.status in [LearningEngineStatus.RUNNING, LearningEngineStatus.LEARNING]:
                try:
                    # 检查是否需要学习
                    if await self._should_trigger_learning():
                        await self.trigger_learning(
                            trigger=LearningTrigger.AUTOMATIC,
                            description="自动学习触发",
                            context={"trigger_reason": "scheduled_interval"}
                        )
                    
                    # 等待下一个检查周期
                    await asyncio.sleep(interval_seconds)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"自动学习循环异常: {e}")
                    await asyncio.sleep(60)  # 出错时等待1分钟再重试
            
        except Exception as e:
            logger.error(f"自动学习循环失败: {e}")
    
    async def _start_scheduled_learning(self) -> None:
        """启动定时学习"""
        try:
            # 这里可以实现更复杂的定时学习逻辑
            # 例如：每天特定时间、每周特定时间等
            pass
        except Exception as e:
            logger.error(f"启动定时学习失败: {e}")
    
    async def _should_trigger_learning(self) -> bool:
        """检查是否应该触发学习"""
        try:
            # 检查时间间隔
            if self.last_learning_time:
                learning_interval = getattr(self.config, 'performance_thresholds', {}).get("learning_interval_hours", 1.0)
                time_since_last = datetime.now() - self.last_learning_time
                if time_since_last.total_seconds() < learning_interval * 3600:
                    return False
            
            # 检查活跃会话数量
            if len(self.active_sessions) >= self.config.max_concurrent_sessions:
                return False
            
            # 检查性能指标
            current_performance = await self._get_current_performance()
            if current_performance:
                success_rate = current_performance.get("success_rate", 1.0)
                error_rate = current_performance.get("error_rate", 0.0)
                
                success_threshold = getattr(self.config, 'performance_thresholds', {}).get("success_rate_threshold", 0.7)
                error_threshold = getattr(self.config, 'performance_thresholds', {}).get("error_rate_threshold", 0.1)
                
                if success_rate < success_threshold or error_rate > error_threshold:
                    return True
            
            return True  # 默认允许学习
            
        except Exception as e:
            logger.warning(f"检查学习触发条件失败: {e}")
            return False
    
    async def _get_current_performance(self) -> Optional[Dict[str, float]]:
        """获取当前性能指标"""
        try:
            # 这里可以从性能监控系统获取实时性能数据
            # 暂时返回模拟数据
            return {
                "success_rate": 0.8,
                "efficiency": 0.75,
                "error_rate": 0.05,
                "response_time": 0.8
            }
        except Exception as e:
            logger.warning(f"获取当前性能失败: {e}")
            return None
    
    async def trigger_learning(
        self,
        trigger: LearningTrigger = LearningTrigger.MANUAL,
        description: str = "手动触发学习",
        context: Optional[Dict[str, Any]] = None,
        phases: Optional[List[LearningPhase]] = None,
        mode: Optional[LearningMode] = None
    ) -> str:
        """触发学习"""
        try:
            if self.status not in [LearningEngineStatus.RUNNING, LearningEngineStatus.LEARNING]:
                raise ValueError(f"学习引擎状态不允许触发学习: {self.status}")
            
            # 检查并发限制
            if len(self.active_sessions) >= self.config.max_concurrent_sessions:
                raise ValueError("已达到最大并发学习会话数量")
            
            # 启动学习会话
            session_id = await self.learning_coordinator.start_learning_session(
                session_type=self._to_value(trigger),
                mode=mode or self.config.learning_mode
            )
            
            # 记录活跃会话
            self.active_sessions[session_id] = self._to_value(trigger)
            
            # 触发学习任务
            task_id = await self.learning_coordinator.trigger_learning(
                task_type=f"learning_{self._to_value(trigger)}",
                description=description,
                context=context or {},
                phases=phases or self.config.enabled_phases,
                priority=self._get_trigger_priority(trigger)
            )
            
            # 更新状态
            if self.status == LearningEngineStatus.RUNNING:
                self.status = LearningEngineStatus.LEARNING
            
            self.last_learning_time = datetime.now()
            self.engine_stats["total_learning_sessions"] += 1
            
            # 设置会话完成回调
            asyncio.create_task(self._monitor_learning_session(session_id, trigger))
            
            # 发布学习触发事件
            self._publish_event(
                InfoType.LEARNING_UPDATE,
                {
                    "agent_id": "learning_engine",
                    "update_type": "learning_triggered",
                    "session_id": session_id,
                    "task_id": task_id,
                    "trigger": self._to_value(trigger),
                    "description": description,
                    "phases": [self._to_value(p) for p in (phases or self.config.enabled_phases)],
                    "timestamp": get_iso_timestamp()
                }
            )
            
            logger.info(f"学习已触发: {session_id} (触发器: {self._to_value(trigger)})")
            return session_id
            
        except Exception as e:
            logger.error(f"触发学习失败: {e}")
            raise
    
    def _get_trigger_priority(self, trigger: LearningTrigger) -> int:
        """获取触发器优先级"""
        priority_map = {
            LearningTrigger.MANUAL: 1,
            LearningTrigger.PERFORMANCE_BASED: 2,
            LearningTrigger.EVENT_DRIVEN: 3,
            LearningTrigger.AUTOMATIC: 4,
            LearningTrigger.SCHEDULED: 5
        }
        return priority_map.get(trigger, 5)
    
    async def _monitor_learning_session(self, session_id: str, trigger: LearningTrigger) -> None:
        """监控学习会话"""
        try:
            start_time = datetime.now()
            
            # 等待会话完成
            while session_id in self.active_sessions:
                await asyncio.sleep(1)
                
                # 检查超时
                max_learning_time = self.config.resource_limits.get("max_learning_time_minutes", 30)
                if (datetime.now() - start_time).total_seconds() > max_learning_time * 60:
                    logger.warning(f"学习会话超时: {session_id}")
                    break
            
            # 处理会话完成
            await self._handle_learning_session_completed(session_id, trigger, start_time)
            
        except Exception as e:
            logger.error(f"监控学习会话失败: {e}")
    
    async def _handle_learning_session_completed(
        self,
        session_id: str,
        trigger: LearningTrigger,
        start_time: datetime
    ) -> None:
        """处理学习会话完成"""
        try:
            # 从活跃会话中移除
            self.active_sessions.pop(session_id, None)
            
            # 获取学习结果
            session_metrics = await self._get_session_metrics(session_id)
            
            # 计算性能改进
            performance_improvement = await self._calculate_performance_improvement(session_metrics)
            
            # 创建学习结果
            duration = (datetime.now() - start_time).total_seconds()
            learning_result = LearningResult(
                session_id=session_id,
                trigger=trigger,
                phases_executed=session_metrics.get("phases_executed", []),
                success=session_metrics.get("success", False),
                duration=duration,
                knowledge_gained=session_metrics.get("knowledge_gained", 0),
                patterns_discovered=session_metrics.get("patterns_discovered", 0),
                optimizations_applied=session_metrics.get("optimizations_applied", 0),
                errors_handled=session_metrics.get("errors_handled", 0),
                performance_improvement=performance_improvement,
                timestamp=get_iso_timestamp(),
                details=session_metrics
            )
            
            # 添加到历史记录
            self.learning_history.append(learning_result)
            
            # 更新统计信息
            if learning_result.success:
                self.engine_stats["successful_sessions"] += 1
            else:
                self.engine_stats["failed_sessions"] += 1
            
            self.engine_stats["total_learning_time"] += duration
            self.engine_stats["knowledge_items_learned"] += learning_result.knowledge_gained
            self.engine_stats["patterns_discovered"] += learning_result.patterns_discovered
            self.engine_stats["optimizations_applied"] += learning_result.optimizations_applied
            self.engine_stats["edge_cases_handled"] += learning_result.errors_handled
            
            if learning_result.performance_improvement > 0:
                self.engine_stats["performance_improvements"] += 1
            
            # 更新平均会话时长
            total_sessions = self.engine_stats["total_learning_sessions"]
            if total_sessions > 0:
                self.engine_stats["avg_session_duration"] = (
                    self.engine_stats["total_learning_time"] / total_sessions
                )
            
            self.engine_stats["last_learning_timestamp"] = get_iso_timestamp()
            
            # 更新引擎状态
            if not self.active_sessions:
                self.status = LearningEngineStatus.RUNNING
            
            # 发布学习完成事件
            self._publish_event(
                InfoType.LEARNING_UPDATE,
                {
                    "agent_id": "learning_engine",
                    "update_type": "learning_completed",
                    "session_id": session_id,
                    "trigger": self._to_value(trigger),
                    "result": {
                        "success": learning_result.success,
                        "duration": learning_result.duration,
                        "knowledge_gained": learning_result.knowledge_gained,
                        "patterns_discovered": learning_result.patterns_discovered,
                        "optimizations_applied": learning_result.optimizations_applied,
                        "performance_improvement": learning_result.performance_improvement
                    },
                    "timestamp": get_iso_timestamp()
                }
            )
            
            logger.info(
                f"学习会话完成: {session_id} "
                f"(成功: {learning_result.success}, "
                f"时长: {learning_result.duration:.2f}s, "
                f"性能改进: {learning_result.performance_improvement:.2%})"
            )
            
        except Exception as e:
            logger.error(f"处理学习会话完成失败: {e}")
    
    async def _get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """获取会话指标"""
        try:
            # 从学习协调器获取会话指标
            learning_metrics = self.learning_coordinator.get_learning_metrics(
                session_id=session_id
            )
            
            if not learning_metrics:
                return {"success": False, "phases_executed": []}
            
            # 聚合指标
            total_knowledge = sum(m.knowledge_gained for m in learning_metrics)
            total_patterns = sum(m.patterns_discovered for m in learning_metrics)
            total_optimizations = sum(m.optimizations_applied for m in learning_metrics)
            total_errors_handled = sum(m.errors_handled for m in learning_metrics)
            avg_success_rate = sum(m.success_rate for m in learning_metrics) / len(learning_metrics)
            phases_executed = [self._to_value(m.phase) for m in learning_metrics]
            
            return {
                "success": avg_success_rate > 0.5,
                "phases_executed": phases_executed,
                "knowledge_gained": total_knowledge,
                "patterns_discovered": total_patterns,
                "optimizations_applied": total_optimizations,
                "errors_handled": total_errors_handled,
                "avg_success_rate": avg_success_rate,
                "metrics_count": len(learning_metrics)
            }
            
        except Exception as e:
            logger.warning(f"获取会话指标失败: {e}")
            return {"success": False, "phases_executed": []}
    
    async def _calculate_performance_improvement(self, session_metrics: Dict[str, Any]) -> float:
        """计算性能改进"""
        try:
            # 这里可以实现更复杂的性能改进计算逻辑
            # 暂时基于成功率和优化数量计算
            success_rate = session_metrics.get("avg_success_rate", 0.0)
            optimizations = session_metrics.get("optimizations_applied", 0)
            
            # 简单的性能改进计算
            improvement = (success_rate - 0.5) * 0.1 + optimizations * 0.05
            return max(0.0, min(1.0, improvement))  # 限制在0-1之间
            
        except Exception as e:
            logger.warning(f"计算性能改进失败: {e}")
            return 0.0
    
    async def _handle_performance_metrics(self, info: Dict[str, Any]) -> None:
        """处理性能指标事件"""
        try:
            if LearningTrigger.PERFORMANCE_BASED not in self.config.learning_triggers:
                return
            
            # 检查性能是否低于阈值
            success_rate = info.get("success_rate", 1.0)
            error_rate = info.get("error_rate", 0.0)
            
            success_threshold = getattr(self.config, 'performance_thresholds', {}).get("success_rate_threshold", 0.7)
            error_threshold = getattr(self.config, 'performance_thresholds', {}).get("error_rate_threshold", 0.1)
            
            if success_rate < success_threshold or error_rate > error_threshold:
                await self.trigger_learning(
                    trigger=LearningTrigger.PERFORMANCE_BASED,
                    description=f"性能指标触发学习 (成功率: {success_rate:.2%}, 错误率: {error_rate:.2%})",
                    context=info,
                    phases=[LearningPhase.USAGE_OPTIMIZATION, LearningPhase.EDGE_HANDLING]
                )
            
        except Exception as e:
            logger.warning(f"处理性能指标事件失败: {e}")
    
    async def _handle_error_event(self, info: Dict[str, Any]) -> None:
        """处理错误事件"""
        try:
            if LearningTrigger.EVENT_DRIVEN not in self.config.learning_triggers:
                return
            
            # 触发边缘处理学习
            await self.trigger_learning(
                trigger=LearningTrigger.EVENT_DRIVEN,
                description=f"错误事件触发学习: {info.get('error_type', 'unknown')}",
                context=info,
                phases=[LearningPhase.EDGE_HANDLING, LearningPhase.PRIOR_KNOWLEDGE]
            )
            
        except Exception as e:
            logger.warning(f"处理错误事件失败: {e}")
    
    async def _handle_task_completion_event(self, info: Dict[str, Any]) -> None:
        """处理任务完成事件"""
        try:
            if LearningTrigger.EVENT_DRIVEN not in self.config.learning_triggers:
                return
            
            # 如果任务成功完成，触发任务合成学习
            if info.get("success", False):
                await self.trigger_learning(
                    trigger=LearningTrigger.EVENT_DRIVEN,
                    description=f"任务完成触发学习: {info.get('task_type', 'unknown')}",
                    context=info,
                    phases=[LearningPhase.TASK_SYNTHESIS, LearningPhase.USAGE_OPTIMIZATION]
                )
            
        except Exception as e:
            logger.warning(f"处理任务完成事件失败: {e}")
    
    async def pause_learning(self) -> bool:
        """暂停学习"""
        try:
            if self.status != LearningEngineStatus.LEARNING:
                return False
            
            # 暂停自动学习
            if self.auto_learning_task and not self.auto_learning_task.done():
                self.auto_learning_task.cancel()
            
            # 暂停学习协调器
            await self.learning_coordinator.pause_learning_session()
            
            self.status = LearningEngineStatus.RUNNING
            logger.info("学习已暂停")
            return True
            
        except Exception as e:
            logger.error(f"暂停学习失败: {e}")
            return False
    
    async def resume_learning(self) -> bool:
        """恢复学习"""
        try:
            if self.status != LearningEngineStatus.RUNNING:
                return False
            
            # 恢复学习协调器
            await self.learning_coordinator.resume_learning_session()
            
            # 重启自动学习
            if self.config.auto_learning_enabled:
                await self._start_auto_learning()
            
            self.status = LearningEngineStatus.LEARNING
            logger.info("学习已恢复")
            return True
            
        except Exception as e:
            logger.error(f"恢复学习失败: {e}")
            return False
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        try:
            coordinator_status = self.learning_coordinator.get_learning_status()
            
            return {
                "engine_status": self._to_value(self.status),
                "active_sessions": len(self.active_sessions),
                "session_details": self.active_sessions.copy(),
                "last_learning_time": self.last_learning_time.isoformat() if self.last_learning_time else None,
                "auto_learning_enabled": self.config.auto_learning_enabled,
                "learning_mode": self._to_value(self.config.learning_mode),
                "enabled_phases": [self._to_value(p) for p in self.config.enabled_phases],
                "learning_triggers": [self._to_value(t) for t in self.config.learning_triggers],
                "coordinator_status": coordinator_status,
                "engine_stats": self.engine_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"获取学习状态失败: {e}")
            return {"error": str(e)}
    
    def get_learning_history(
        self,
        trigger: Optional[LearningTrigger] = None,
        limit: int = 50
    ) -> List[LearningResult]:
        """获取学习历史"""
        try:
            history = self.learning_history.copy()
            
            if trigger:
                history = [r for r in history if r.trigger == trigger]
            
            # 按时间倒序排序
            history.sort(key=lambda x: x.timestamp, reverse=True)
            
            return history[:limit]
            
        except Exception as e:
            logger.error(f"获取学习历史失败: {e}")
            return []
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        try:
            stats = self.engine_stats.copy()
            
            # 添加成功率
            total_sessions = stats["total_learning_sessions"]
            if total_sessions > 0:
                stats["success_rate"] = stats["successful_sessions"] / total_sessions
                stats["failure_rate"] = stats["failed_sessions"] / total_sessions
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
            
            # 添加最近学习结果
            recent_results = self.get_learning_history(limit=10)
            if recent_results:
                stats["recent_avg_duration"] = sum(r.duration for r in recent_results) / len(recent_results)
                stats["recent_success_rate"] = sum(1 for r in recent_results if r.success) / len(recent_results)
                stats["recent_avg_improvement"] = sum(r.performance_improvement for r in recent_results) / len(recent_results)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取学习统计信息失败: {e}")
            return {}
    
    async def update_configuration(self, new_config: LearningConfiguration) -> bool:
        """更新学习配置"""
        try:
            old_auto_learning = self.config.auto_learning_enabled
            self.config = new_config
            
            # 如果自动学习设置发生变化
            if old_auto_learning != new_config.auto_learning_enabled:
                if new_config.auto_learning_enabled:
                    await self._start_auto_learning()
                else:
                    if self.auto_learning_task and not self.auto_learning_task.done():
                        self.auto_learning_task.cancel()
            
            # 发布配置更新事件
            self._publish_event(
                InfoType.LEARNING_UPDATE,
                {
                    "agent_id": "learning_engine",
                    "update_type": "config_updated",
                    "new_config": {
                        "auto_learning_enabled": new_config.auto_learning_enabled,
                        "learning_mode": self._to_value(new_config.learning_mode),
                        "enabled_phases": [self._to_value(p) for p in new_config.enabled_phases],
                        "triggers": [self._to_value(t) for t in new_config.learning_triggers]
                    },
                    "timestamp": get_iso_timestamp()
                }
            )
            
            logger.info("学习配置已更新")
            return True
            
        except Exception as e:
            logger.error(f"更新学习配置失败: {e}")
            return False
    
    async def shutdown(self) -> None:
        """关闭学习引擎"""
        try:
            self.status = LearningEngineStatus.SHUTDOWN
            
            # 取消自动学习任务
            if self.auto_learning_task and not self.auto_learning_task.done():
                self.auto_learning_task.cancel()
            
            # 取消所有定时任务
            for task_id, task in self.scheduled_tasks.items():
                task.cancel()
            
            # 等待任务取消完成
            all_tasks = [self.auto_learning_task] + list(self.scheduled_tasks.values())
            active_tasks = [t for t in all_tasks if t and not t.done()]
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            
            # 关闭学习协调器
            await self.learning_coordinator.shutdown()
            
            # 发布关闭事件
            self._publish_event(
                InfoType.LEARNING_UPDATE,
                {
                    "agent_id": "learning_engine",
                    "update_type": "engine_shutdown",
                    "final_stats": self.engine_stats,
                    "timestamp": get_iso_timestamp()
                }
            )
            
            logger.info("学习引擎已关闭")
            
        except Exception as e:
            logger.error(f"关闭学习引擎失败: {e}")