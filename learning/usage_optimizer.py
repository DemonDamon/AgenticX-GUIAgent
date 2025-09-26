#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent学习引擎 - 使用优化器（第四阶段）(基于AgenticX框架重构)

该模块实现了学习引擎的第四阶段：使用优化器（Usage Optimizer）。
负责优化智能体的使用模式，提高执行效率和成功率。

重构说明：
- 基于AgenticX的Component基类重构
- 使用AgenticX的事件系统进行性能监控和优化通知
- 集成AgenticX的内存组件进行性能数据管理
- 遵循AgenticX的自适应优化和资源管理架构

主要功能：
1. 性能监控和分析
2. 使用模式优化
3. 资源分配优化
4. 执行策略调整
5. 自适应参数调优

Author: AgenticX Team
Date: 2025-01-20
"""

import asyncio
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import random
import json
from datetime import datetime, timedelta, UTC

from agenticx.core.component import Component
from agenticx.core.event import Event

from core.info_pool import InfoPool, InfoType, InfoPriority
from utils import get_iso_timestamp


class OptimizationType(Enum):
    """优化类型"""
    PERFORMANCE = "performance"  # 性能优化
    RESOURCE = "resource"  # 资源优化
    STRATEGY = "strategy"  # 策略优化
    PARAMETER = "parameter"  # 参数优化
    WORKFLOW = "workflow"  # 工作流优化
    ADAPTIVE = "adaptive"  # 自适应优化


class OptimizationStrategy(Enum):
    """优化策略"""
    GREEDY = "greedy"  # 贪心优化
    GRADIENT_BASED = "gradient_based"  # 基于梯度的优化
    EVOLUTIONARY = "evolutionary"  # 进化算法
    SIMULATED_ANNEALING = "simulated_annealing"  # 模拟退火
    BAYESIAN = "bayesian"  # 贝叶斯优化
    REINFORCEMENT = "reinforcement"  # 强化学习优化
    HEURISTIC = "heuristic"  # 启发式优化
    MULTI_OBJECTIVE = "multi_objective"  # 多目标优化


class OptimizationScope(Enum):
    """优化范围"""
    AGENT = "agent"  # 单个智能体
    TASK = "task"  # 特定任务
    SYSTEM = "system"  # 整个系统
    WORKFLOW = "workflow"  # 工作流
    GLOBAL = "global"  # 全局优化


@dataclass
class PerformanceMetrics:
    """性能指标"""
    agent_id: str
    task_type: str
    success_rate: float
    average_duration: float
    error_rate: float
    resource_usage: Dict[str, float]
    efficiency_score: float
    quality_score: float
    timestamp: str
    sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    recommendation_id: str
    optimization_type: OptimizationType
    scope: OptimizationScope
    target_entity: str  # 目标实体（智能体ID、任务类型等）
    description: str
    current_metrics: PerformanceMetrics
    expected_improvement: Dict[str, float]
    implementation_steps: List[Dict[str, Any]]
    priority: float  # 0-1，优先级
    confidence: float  # 0-1，置信度
    estimated_effort: float  # 预估实施工作量
    risk_level: str  # low, medium, high
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """优化结果"""
    optimization_id: str
    recommendation: OptimizationRecommendation
    implementation_status: str  # pending, in_progress, completed, failed
    before_metrics: PerformanceMetrics
    after_metrics: Optional[PerformanceMetrics]
    actual_improvement: Dict[str, float]
    implementation_time: float
    success: bool
    lessons_learned: List[str]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsagePattern:
    """使用模式"""
    pattern_id: str
    pattern_type: str
    agent_id: str
    task_types: List[str]
    frequency: int
    success_rate: float
    average_performance: Dict[str, float]
    resource_consumption: Dict[str, float]
    time_distribution: Dict[str, int]  # 时间分布
    context_conditions: Dict[str, Any]
    optimization_potential: float  # 0-1，优化潜力
    metadata: Dict[str, Any] = field(default_factory=dict)


class UsageOptimizer(Component):
    """使用优化器
    
    负责监控和优化智能体的使用模式，提高整体系统性能。
    基于AgenticX的Component基类重构。
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        info_pool: Optional[InfoPool] = None
    ):
        super().__init__()
        self.config = config
        self.info_pool = info_pool
        self.logger = logger
        
        # 优化参数
        self.optimization_params = config.get("optimization_params", {
            "monitoring_window": 3600,  # 监控窗口（秒）
            "min_samples_for_optimization": 10,  # 最小样本数
            "performance_threshold": 0.7,  # 性能阈值
            "improvement_threshold": 0.1,  # 改进阈值
            "max_concurrent_optimizations": 3,  # 最大并发优化数
            "optimization_cooldown": 1800,  # 优化冷却时间（秒）
        })
        
        # 性能监控
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics: Dict[str, PerformanceMetrics] = {}
        self.usage_patterns: Dict[str, UsagePattern] = {}
        
        # 优化管理
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        self.optimization_results: List[OptimizationResult] = []
        self.active_optimizations: Dict[str, OptimizationResult] = {}
        self.optimization_history: deque = deque(maxlen=500)
        
        # 自适应参数
        self.adaptive_parameters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.parameter_adjustment_history: deque = deque(maxlen=200)
        
        # 统计信息
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_improvement": 0.0,
            "optimization_by_type": defaultdict(int),
            "optimization_by_scope": defaultdict(int),
            "performance_trends": defaultdict(list)
        }
        
        # 初始化
        self._initialize_optimizer()
    
    def _to_value(self, enum_member: Enum) -> Any:
        """安全地获取枚举成员的值"""
        if hasattr(enum_member, 'value'):
            return enum_member.value
        return enum_member

    def _initialize_optimizer(self) -> None:
        """初始化优化器"""
        try:
            # 设置默认自适应参数
            self._setup_default_adaptive_parameters()
            
            # 启动性能监控
            if self.info_pool:
                asyncio.create_task(self._start_performance_monitoring())
            
            logger.info("使用优化器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化使用优化器失败: {e}")
            raise
    
    def _setup_default_adaptive_parameters(self) -> None:
        """设置默认自适应参数"""
        default_params = {
            "exploration_rate": {
                "value": 0.1,
                "min": 0.01,
                "max": 0.5,
                "adjustment_factor": 0.1
            },
            "learning_rate": {
                "value": 0.01,
                "min": 0.001,
                "max": 0.1,
                "adjustment_factor": 0.1
            },
            "retry_attempts": {
                "value": 3,
                "min": 1,
                "max": 10,
                "adjustment_factor": 1
            },
            "timeout_duration": {
                "value": 30.0,
                "min": 5.0,
                "max": 300.0,
                "adjustment_factor": 5.0
            }
        }
        
        for param_name, param_config in default_params.items():
            self.adaptive_parameters["global"][param_name] = param_config.copy()
    
    async def _start_performance_monitoring(self) -> None:
        """启动性能监控"""
        try:
            if not self.info_pool or not hasattr(self.info_pool, 'event_bus'):
                logger.warning("InfoPool或EventBus未初始化，无法订阅事件")
                return

            self.info_pool.event_bus.subscribe(
                event_type=self._to_value(InfoType.TASK_COMPLETION),
                handler=self._handle_task_completion
            )
            self.info_pool.event_bus.subscribe(
                event_type=self._to_value(InfoType.ACTION_RESULT),
                handler=self._handle_action_result
            )
            self.info_pool.event_bus.subscribe(
                event_type=self._to_value(InfoType.AGENT_STATUS),
                handler=self._handle_agent_status
            )
            
            logger.info("性能监控已启动")
            
        except Exception as e:
            logger.error(f"启动性能监控失败: {e}")
    
    async def _handle_task_completion(self, event: Event) -> None:
        """处理任务完成事件""" 
        try:
            if not isinstance(event.data, dict) or 'data' not in event.data:
                logger.warning(f"Invalid event data format in _handle_task_completion: {event.data}")
                return

            task_data = event.data['data']
            agent_id = event.agent_id or task_data.get("agent_id")

            if not isinstance(task_data, dict):
                logger.warning(f"task_data is not a dict: {task_data}")
                return

            # HACK: task_type is not in TaskEndEvent, need to get it from somewhere else.
            # Maybe from the result or we need to add it to the event.
            result = task_data.get("result", {})
            task_type = result.get("task_type", "unknown_task")
            success = task_data.get("success", False)
            duration = result.get("duration", 0.0)
            
            if agent_id and task_type:
                await self._update_performance_metrics(
                    agent_id,
                    task_type,
                    success,
                    duration,
                    task_data
                )
                
        except Exception as e:
            logger.warning(f"处理任务完成事件失败: {e}", exc_info=True)
    
    async def _handle_action_result(self, event: Event) -> None:
        """处理动作结果事件"""
        try:
            if not isinstance(event.data, dict) or 'data' not in event.data:
                logger.warning(f"Invalid event data format in _handle_action_result: {event.data}")
                return

            action_record = event.data['data']
            agent_id = event.agent_id
            
            if not isinstance(action_record, dict):
                logger.warning(f"action_record is not a dict: {action_record}")
                return

            action_type = action_record.get("task_type")
            success = action_record.get("success", False)
            
            # HACK: execution_time is not available in action_record, using 0.0 for now
            # The result from executor_agent's actions should probably include duration.
            result = action_record.get("result", {})
            execution_time = result.get("duration", 0.0)

            if agent_id and action_type:
                await self._update_action_metrics(
                    agent_id,
                    action_type,
                    success,
                    execution_time,
                    action_record
                )
        except Exception as e:
            logger.warning(f"处理动作结果事件失败: {e}", exc_info=True)
    
    async def _handle_agent_status(self, event: Event) -> None:
        """处理智能体状态事件"""
        try:
            if not isinstance(event.data, dict) or 'data' not in event.data:
                logger.warning(f"Invalid event data format in _handle_agent_status: {event.data}")
                return

            status_data = event.data['data']
            agent_id = event.agent_id or status_data.get("agent_id")

            if not isinstance(status_data, dict):
                logger.warning(f"status_data is not a dict: {status_data}")
                return

            status = status_data.get("status")
            resource_usage = status_data.get("resource_usage", {})
            
            if agent_id and status:
                await self._update_agent_metrics(
                    agent_id,
                    status,
                    resource_usage,
                    status_data
                )
                
        except Exception as e:
            logger.warning(f"处理智能体状态事件失败: {e}", exc_info=True)

    async def _update_performance_metrics(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
        duration: float,
        context: Dict[str, Any]
    ) -> None:
        """更新性能指标"""
        try:
            key = f"{agent_id}_{task_type}"
            
            # 获取或创建性能指标
            if key not in self.current_metrics:
                self.current_metrics[key] = PerformanceMetrics(
                    agent_id=agent_id,
                    task_type=task_type,
                    success_rate=0.0,
                    average_duration=0.0,
                    error_rate=0.0,
                    resource_usage={},
                    efficiency_score=0.0,
                    quality_score=0.0,
                    timestamp=get_iso_timestamp(),
                    sample_count=0
                )
            
            metrics = self.current_metrics[key]
            
            # 更新指标
            metrics.sample_count += 1
            
            # 更新成功率
            old_success_rate = metrics.success_rate
            metrics.success_rate = (
                (old_success_rate * (metrics.sample_count - 1) + (1.0 if success else 0.0)) /
                metrics.sample_count
            )
            
            # 更新平均持续时间
            old_duration = metrics.average_duration
            metrics.average_duration = (
                (old_duration * (metrics.sample_count - 1) + duration) /
                metrics.sample_count
            )
            
            # 更新错误率
            metrics.error_rate = 1.0 - metrics.success_rate
            
            # 更新资源使用
            resource_usage = context.get("resource_usage", {})
            for resource, usage in resource_usage.items():
                if resource not in metrics.resource_usage:
                    metrics.resource_usage[resource] = usage
                else:
                    old_usage = metrics.resource_usage[resource]
                    metrics.resource_usage[resource] = (
                        (old_usage * (metrics.sample_count - 1) + usage) /
                        metrics.sample_count
                    )
            
            # 计算效率分数
            metrics.efficiency_score = self._calculate_efficiency_score(metrics)
            
            # 计算质量分数
            metrics.quality_score = self._calculate_quality_score(metrics, context)
            
            # 更新时间戳
            metrics.timestamp = get_iso_timestamp()
            
            # 添加到历史记录
            self.performance_history[key].append({
                "timestamp": get_iso_timestamp(),
                "success": success,
                "duration": duration,
                "metrics": metrics
            })
            
            # 检查是否需要优化
            if metrics.sample_count >= self.optimization_params["min_samples_for_optimization"]:
                await self._check_optimization_opportunity(key, metrics)
            
        except Exception as e:
            logger.warning(f"更新性能指标失败: {e}")
    
    def _calculate_efficiency_score(
        self,
        metrics: PerformanceMetrics
    ) -> float:
        """计算效率分数"""
        try:
            # 基于成功率和执行时间计算效率
            success_weight = 0.6
            speed_weight = 0.4
            
            success_score = metrics.success_rate
            
            # 速度分数（假设30秒为基准）
            baseline_duration = 30.0
            if metrics.average_duration > 0:
                speed_score = min(1.0, baseline_duration / metrics.average_duration)
            else:
                speed_score = 1.0
            
            efficiency_score = (
                success_score * success_weight +
                speed_score * speed_weight
            )
            
            return min(1.0, max(0.0, efficiency_score))
            
        except Exception as e:
            logger.warning(f"计算效率分数失败: {e}")
            return 0.5
    
    def _calculate_quality_score(
        self,
        metrics: PerformanceMetrics,
        context: Dict[str, Any]
    ) -> float:
        """计算质量分数"""
        try:
            # 基于多个维度计算质量分数
            accuracy_score = metrics.success_rate
            reliability_score = 1.0 - metrics.error_rate
            consistency_score = self._calculate_consistency_score(metrics)
            
            quality_score = (
                accuracy_score * 0.4 +
                reliability_score * 0.3 +
                consistency_score * 0.3
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"计算质量分数失败: {e}")
            return 0.5
    
    def _calculate_consistency_score(
        self,
        metrics: PerformanceMetrics
    ) -> float:
        """计算一致性分数"""
        try:
            key = f"{metrics.agent_id}_{metrics.task_type}"
            history = self.performance_history.get(key, [])
            
            if len(history) < 5:
                return 0.5  # 样本不足
            
            # 计算最近执行的成功率方差
            recent_successes = [1.0 if record["success"] else 0.0 for record in list(history)[-10:]]
            
            if len(recent_successes) > 1:
                variance = statistics.variance(recent_successes)
                consistency_score = 1.0 - min(1.0, variance * 2)  # 方差越小，一致性越高
            else:
                consistency_score = 0.5
            
            return max(0.0, consistency_score)
            
        except Exception as e:
            logger.warning(f"计算一致性分数失败: {e}")
            return 0.5
    
    async def _update_action_metrics(
        self,
        agent_id: str,
        action_type: str,
        success: bool,
        execution_time: float,
        context: Dict[str, Any]
    ) -> None:
        """更新动作指标"""
        try:
            # 更新动作级别的性能指标
            await self._update_performance_metrics(
                agent_id,
                f"action_{action_type}",
                success,
                execution_time,
                context
            )
            
        except Exception as e:
            logger.warning(f"更新动作指标失败: {e}")
    
    async def _update_agent_metrics(
        self,
        agent_id: str,
        status: str,
        resource_usage: Dict[str, float],
        context: Dict[str, Any]
    ) -> None:
        """更新智能体指标"""
        try:
            # 更新智能体级别的资源使用指标
            key = f"{agent_id}_agent_status"
            
            if key not in self.current_metrics:
                self.current_metrics[key] = PerformanceMetrics(
                    agent_id=agent_id,
                    task_type="agent_status",
                    success_rate=1.0,
                    average_duration=0.0,
                    error_rate=0.0,
                    resource_usage=resource_usage.copy(),
                    efficiency_score=0.5,
                    quality_score=0.5,
                    timestamp=get_iso_timestamp(),
                    sample_count=1
                )
            else:
                metrics = self.current_metrics[key]
                metrics.sample_count += 1
                
                # 更新资源使用
                for resource, usage in resource_usage.items():
                    if resource not in metrics.resource_usage:
                        metrics.resource_usage[resource] = usage
                    else:
                        old_usage = metrics.resource_usage[resource]
                        metrics.resource_usage[resource] = (
                            (old_usage * (metrics.sample_count - 1) + usage) /
                            metrics.sample_count
                        )
                
                metrics.timestamp = get_iso_timestamp()
            
        except Exception as e:
            logger.warning(f"更新智能体指标失败: {e}")
    
    async def _check_optimization_opportunity(
        self,
        metrics_key: str,
        metrics: PerformanceMetrics
    ) -> None:
        """检查优化机会"""
        try:
            # 检查性能是否低于阈值
            performance_threshold = self.optimization_params["performance_threshold"]
            
            needs_optimization = (
                metrics.success_rate < performance_threshold or
                metrics.efficiency_score < performance_threshold or
                metrics.quality_score < performance_threshold
            )
            
            if needs_optimization:
                # 生成优化建议
                recommendation = await self._generate_optimization_recommendation(
                    metrics_key,
                    metrics
                )
                
                if recommendation:
                    self.optimization_recommendations.append(recommendation)
                    
                    # 发布优化建议
                    if self.info_pool:
                        await self.info_pool.publish_info(
                            InfoType.LEARNING_UPDATE,
                            {
                                "agent_id": "usage_optimizer",
                                "update_type": "optimization_recommendation",
                                "recommendation": recommendation,
                                "timestamp": get_iso_timestamp()
                            },
                            InfoPriority.MEDIUM
                        )
            
        except Exception as e:
            logger.warning(f"检查优化机会失败: {e}")
    
    async def _generate_optimization_recommendation(
        self,
        metrics_key: str,
        metrics: PerformanceMetrics
    ) -> Optional[OptimizationRecommendation]:
        """生成优化建议"""
        try:
            # 分析性能问题
            issues = self._analyze_performance_issues(metrics)
            
            if not issues:
                return None
            
            # 确定优化类型和范围
            optimization_type, scope = self._determine_optimization_approach(issues, metrics)
            
            # 生成实施步骤
            implementation_steps = self._generate_implementation_steps(
                optimization_type,
                issues,
                metrics
            )
            
            # 估算改进效果
            expected_improvement = self._estimate_improvement(
                optimization_type,
                issues,
                metrics
            )
            
            # 计算优先级和置信度
            priority = self._calculate_optimization_priority(issues, metrics)
            confidence = self._calculate_optimization_confidence(optimization_type, metrics)
            
            recommendation = OptimizationRecommendation(
                recommendation_id=f"opt_rec_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                optimization_type=optimization_type,
                scope=scope,
                target_entity=metrics.agent_id,
                description=f"优化 {metrics.agent_id} 的 {metrics.task_type} 性能",
                current_metrics=metrics,
                expected_improvement=expected_improvement,
                implementation_steps=implementation_steps,
                priority=priority,
                confidence=confidence,
                estimated_effort=self._estimate_implementation_effort(implementation_steps),
                risk_level=self._assess_risk_level(optimization_type, metrics),
                metadata={
                    "issues_identified": issues,
                    "metrics_key": metrics_key,
                    "generation_timestamp": get_iso_timestamp()
                }
            )
            
            return recommendation
            
        except Exception as e:
            logger.warning(f"生成优化建议失败: {e}")
            return None
    
    def _analyze_performance_issues(
        self,
        metrics: PerformanceMetrics
    ) -> List[Dict[str, Any]]:
        """分析性能问题"""
        issues = []
        
        # 检查成功率问题
        if metrics.success_rate < 0.8:
            issues.append({
                "type": "low_success_rate",
                "severity": "high" if metrics.success_rate < 0.5 else "medium",
                "value": metrics.success_rate,
                "description": f"成功率过低: {metrics.success_rate:.2f}"
            })
        
        # 检查执行时间问题
        if metrics.average_duration > 60.0:  # 超过1分钟
            issues.append({
                "type": "slow_execution",
                "severity": "high" if metrics.average_duration > 120.0 else "medium",
                "value": metrics.average_duration,
                "description": f"执行时间过长: {metrics.average_duration:.2f}秒"
            })
        
        # 检查错误率问题
        if metrics.error_rate > 0.2:
            issues.append({
                "type": "high_error_rate",
                "severity": "high" if metrics.error_rate > 0.5 else "medium",
                "value": metrics.error_rate,
                "description": f"错误率过高: {metrics.error_rate:.2f}"
            })
        
        # 检查效率问题
        if metrics.efficiency_score < 0.6:
            issues.append({
                "type": "low_efficiency",
                "severity": "medium",
                "value": metrics.efficiency_score,
                "description": f"效率分数过低: {metrics.efficiency_score:.2f}"
            })
        
        # 检查资源使用问题
        for resource, usage in metrics.resource_usage.items():
            if usage > 0.8:  # 资源使用率超过80%
                issues.append({
                    "type": "high_resource_usage",
                    "severity": "high" if usage > 0.9 else "medium",
                    "value": usage,
                    "resource": resource,
                    "description": f"{resource}使用率过高: {usage:.2f}"
                })
        
        return issues
    
    def _determine_optimization_approach(
        self,
        issues: List[Dict[str, Any]],
        metrics: PerformanceMetrics
    ) -> Tuple[OptimizationType, OptimizationScope]:
        """确定优化方法"""
        # 根据问题类型确定优化类型
        issue_types = [issue["type"] for issue in issues]
        
        if "high_resource_usage" in issue_types:
            optimization_type = OptimizationType.RESOURCE
        elif "slow_execution" in issue_types:
            optimization_type = OptimizationType.PERFORMANCE
        elif "low_success_rate" in issue_types or "high_error_rate" in issue_types:
            optimization_type = OptimizationType.STRATEGY
        else:
            optimization_type = OptimizationType.PARAMETER
        
        # 确定优化范围
        if metrics.task_type.startswith("action_"):
            scope = OptimizationScope.TASK
        else:
            scope = OptimizationScope.AGENT
        
        return optimization_type, scope
    
    def _generate_implementation_steps(
        self,
        optimization_type: OptimizationType,
        issues: List[Dict[str, Any]],
        metrics: PerformanceMetrics
    ) -> List[Dict[str, Any]]:
        """生成实施步骤"""
        steps = []
        
        if optimization_type == OptimizationType.PERFORMANCE:
            steps.extend([
                {
                    "step": "analyze_bottlenecks",
                    "description": "分析性能瓶颈",
                    "estimated_time": 300  # 5分钟
                },
                {
                    "step": "optimize_algorithms",
                    "description": "优化算法实现",
                    "estimated_time": 1800  # 30分钟
                },
                {
                    "step": "test_performance",
                    "description": "测试性能改进",
                    "estimated_time": 600  # 10分钟
                }
            ])
        
        elif optimization_type == OptimizationType.RESOURCE:
            steps.extend([
                {
                    "step": "monitor_resource_usage",
                    "description": "监控资源使用情况",
                    "estimated_time": 180  # 3分钟
                },
                {
                    "step": "optimize_resource_allocation",
                    "description": "优化资源分配",
                    "estimated_time": 900  # 15分钟
                },
                {
                    "step": "implement_resource_limits",
                    "description": "实施资源限制",
                    "estimated_time": 600  # 10分钟
                }
            ])
        
        elif optimization_type == OptimizationType.STRATEGY:
            steps.extend([
                {
                    "step": "analyze_failure_patterns",
                    "description": "分析失败模式",
                    "estimated_time": 600  # 10分钟
                },
                {
                    "step": "adjust_strategy_parameters",
                    "description": "调整策略参数",
                    "estimated_time": 1200  # 20分钟
                },
                {
                    "step": "implement_fallback_strategies",
                    "description": "实施备用策略",
                    "estimated_time": 1800  # 30分钟
                }
            ])
        
        elif optimization_type == OptimizationType.PARAMETER:
            steps.extend([
                {
                    "step": "identify_key_parameters",
                    "description": "识别关键参数",
                    "estimated_time": 300  # 5分钟
                },
                {
                    "step": "tune_parameters",
                    "description": "调优参数",
                    "estimated_time": 900  # 15分钟
                },
                {
                    "step": "validate_parameter_changes",
                    "description": "验证参数变更",
                    "estimated_time": 600  # 10分钟
                }
            ])
        
        return steps
    
    def _estimate_improvement(
        self,
        optimization_type: OptimizationType,
        issues: List[Dict[str, Any]],
        metrics: PerformanceMetrics
    ) -> Dict[str, float]:
        """估算改进效果"""
        improvement = {
            "success_rate": 0.0,
            "efficiency_score": 0.0,
            "average_duration": 0.0,
            "error_rate": 0.0
        }
        
        # 根据优化类型估算改进
        if optimization_type == OptimizationType.PERFORMANCE:
            improvement["average_duration"] = -0.2  # 减少20%执行时间
            improvement["efficiency_score"] = 0.15  # 提高15%效率
        
        elif optimization_type == OptimizationType.STRATEGY:
            improvement["success_rate"] = 0.1  # 提高10%成功率
            improvement["error_rate"] = -0.1  # 减少10%错误率
        
        elif optimization_type == OptimizationType.PARAMETER:
            improvement["success_rate"] = 0.05  # 提高5%成功率
            improvement["efficiency_score"] = 0.1  # 提高10%效率
        
        # 根据问题严重程度调整估算
        high_severity_count = sum(1 for issue in issues if issue.get("severity") == "high")
        if high_severity_count > 0:
            # 严重问题更多，改进潜力更大
            for key in improvement:
                improvement[key] *= (1 + high_severity_count * 0.2)
        
        return improvement
    
    def _calculate_optimization_priority(
        self,
        issues: List[Dict[str, Any]],
        metrics: PerformanceMetrics
    ) -> float:
        """计算优化优先级"""
        # 基于问题严重程度和影响范围计算优先级
        priority_score = 0.0
        
        for issue in issues:
            severity = issue.get("severity", "low")
            if severity == "high":
                priority_score += 0.3
            elif severity == "medium":
                priority_score += 0.2
            else:
                priority_score += 0.1
        
        # 考虑样本数量（更多样本意味着更可靠的统计）
        sample_factor = min(1.0, metrics.sample_count / 50.0)
        priority_score *= sample_factor
        
        return min(1.0, priority_score)
    
    def _calculate_optimization_confidence(
        self,
        optimization_type: OptimizationType,
        metrics: PerformanceMetrics
    ) -> float:
        """计算优化置信度"""
        # 基于历史数据和优化类型计算置信度
        base_confidence = 0.6
        
        # 样本数量影响置信度
        sample_factor = min(1.0, metrics.sample_count / 100.0)
        
        # 优化类型影响置信度
        type_factor = {
            OptimizationType.PARAMETER: 0.8,
            OptimizationType.PERFORMANCE: 0.7,
            OptimizationType.STRATEGY: 0.6,
            OptimizationType.RESOURCE: 0.7
        }.get(optimization_type, 0.5)
        
        confidence = base_confidence * sample_factor * type_factor
        
        return min(1.0, max(0.1, confidence))
    
    def _estimate_implementation_effort(
        self,
        implementation_steps: List[Dict[str, Any]]
    ) -> float:
        """估算实施工作量（小时）"""
        total_time = sum(step.get("estimated_time", 600) for step in implementation_steps)
        return total_time / 3600.0  # 转换为小时
    
    def _assess_risk_level(
        self,
        optimization_type: OptimizationType,
        metrics: PerformanceMetrics
    ) -> str:
        """评估风险级别"""
        # 根据优化类型和当前性能评估风险
        if optimization_type == OptimizationType.STRATEGY:
            return "medium"  # 策略变更有中等风险
        elif optimization_type == OptimizationType.RESOURCE:
            return "high" if any(usage > 0.9 for usage in metrics.resource_usage.values()) else "medium"
        else:
            return "low"  # 参数和性能优化风险较低
    
    async def optimize_usage(
        self,
        target_entity: str,
        optimization_types: List[OptimizationType],
        scope: OptimizationScope = OptimizationScope.AGENT
    ) -> Dict[str, Any]:
        """执行使用优化"""
        try:
            # 检查并发优化限制
            if len(self.active_optimizations) >= self.optimization_params["max_concurrent_optimizations"]:
                return {
                    "success": False,
                    "error": "已达到最大并发优化数量限制",
                    "active_optimizations": len(self.active_optimizations)
                }
            
            # 查找相关的优化建议
            relevant_recommendations = [
                rec for rec in self.optimization_recommendations
                if rec.target_entity == target_entity and
                rec.optimization_type in optimization_types and
                rec.scope == scope
            ]
            
            if not relevant_recommendations:
                return {
                    "success": False,
                    "error": "未找到相关的优化建议",
                    "target_entity": target_entity
                }
            
            # 选择最高优先级的建议
            best_recommendation = max(
                relevant_recommendations,
                key=lambda x: x.priority * x.confidence
            )
            
            # 开始优化
            optimization_result = await self._execute_optimization(best_recommendation)
            
            return {
                "success": True,
                "optimization_id": optimization_result.optimization_id,
                "recommendation": best_recommendation,
                "status": optimization_result.implementation_status
            }
            
        except Exception as e:
            logger.error(f"执行使用优化失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_optimization(
        self,
        recommendation: OptimizationRecommendation
    ) -> OptimizationResult:
        """执行优化"""
        optimization_id = f"opt_{get_iso_timestamp()}_{random.randint(1000, 9999)}"
        
        # 创建优化结果记录
        optimization_result = OptimizationResult(
            optimization_id=optimization_id,
            recommendation=recommendation,
            implementation_status="in_progress",
            before_metrics=recommendation.current_metrics,
            after_metrics=None,
            actual_improvement={},
            implementation_time=0.0,
            success=False,
            lessons_learned=[],
            timestamp=get_iso_timestamp()
        )
        
        # 添加到活跃优化
        self.active_optimizations[optimization_id] = optimization_result
        
        try:
            start_time = datetime.now()
            
            # 执行优化步骤
            for step in recommendation.implementation_steps:
                await self._execute_optimization_step(step, recommendation)
            
            # 等待一段时间收集新的性能数据
            await asyncio.sleep(30)  # 等待30秒
            
            # 评估优化效果
            after_metrics = await self._collect_post_optimization_metrics(
                recommendation.target_entity,
                recommendation.current_metrics.task_type
            )
            
            if after_metrics:
                optimization_result.after_metrics = after_metrics
                optimization_result.actual_improvement = self._calculate_actual_improvement(
                    recommendation.current_metrics,
                    after_metrics
                )
                optimization_result.success = self._evaluate_optimization_success(
                    recommendation.expected_improvement,
                    optimization_result.actual_improvement
                )
            
            optimization_result.implementation_time = (
                datetime.now() - start_time
            ).total_seconds()
            optimization_result.implementation_status = "completed"
            
            # 更新统计信息
            self._update_optimization_stats(optimization_result)
            
            # 发布优化结果
            if self.info_pool:
                await self.info_pool.publish_info(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "usage_optimizer",
                        "update_type": "optimization_completed",
                        "optimization_result": optimization_result,
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.MEDIUM
                )
            
        except Exception as e:
            optimization_result.implementation_status = "failed"
            optimization_result.lessons_learned.append(f"优化执行失败: {str(e)}")
            logger.error(f"执行优化失败: {e}")
        
        finally:
            # 从活跃优化中移除
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
            
            # 添加到历史记录
            self.optimization_results.append(optimization_result)
            self.optimization_history.append({
                "optimization_id": optimization_id,
                "timestamp": get_iso_timestamp(),
                "success": optimization_result.success,
                "improvement": optimization_result.actual_improvement
            })
        
        return optimization_result
    
    async def _execute_optimization_step(
        self,
        step: Dict[str, Any],
        recommendation: OptimizationRecommendation
    ) -> None:
        """执行优化步骤"""
        try:
            step_type = step.get("step")
            
            if step_type == "tune_parameters":
                await self._tune_parameters(recommendation)
            elif step_type == "adjust_strategy_parameters":
                await self._adjust_strategy_parameters(recommendation)
            elif step_type == "optimize_resource_allocation":
                await self._optimize_resource_allocation(recommendation)
            elif step_type == "optimize_algorithms":
                await self._optimize_algorithms(recommendation)
            else:
                # 通用步骤处理
                await self._execute_generic_step(step, recommendation)
            
            logger.info(f"完成优化步骤: {step_type}")
            
        except Exception as e:
            logger.warning(f"执行优化步骤失败 {step.get('step')}: {e}")
    
    async def _tune_parameters(
        self,
        recommendation: OptimizationRecommendation
    ) -> None:
        """调优参数"""
        try:
            target_entity = recommendation.target_entity
            
            # 获取当前参数
            current_params = self.adaptive_parameters.get(target_entity, {})
            
            # 基于性能问题调整参数
            issues = recommendation.metadata.get("issues_identified", [])
            
            for issue in issues:
                if issue["type"] == "low_success_rate":
                    # 增加重试次数
                    if "retry_attempts" in current_params:
                        current_params["retry_attempts"]["value"] = min(
                            current_params["retry_attempts"]["max"],
                            current_params["retry_attempts"]["value"] + 1
                        )
                    
                    # 增加超时时间
                    if "timeout_duration" in current_params:
                        current_params["timeout_duration"]["value"] = min(
                            current_params["timeout_duration"]["max"],
                            current_params["timeout_duration"]["value"] * 1.2
                        )
                
                elif issue["type"] == "slow_execution":
                    # 减少探索率以提高执行速度
                    if "exploration_rate" in current_params:
                        current_params["exploration_rate"]["value"] = max(
                            current_params["exploration_rate"]["min"],
                            current_params["exploration_rate"]["value"] * 0.8
                        )
            
            # 记录参数调整
            self.parameter_adjustment_history.append({
                "timestamp": get_iso_timestamp(),
                "target_entity": target_entity,
                "adjustments": current_params,
                "reason": "optimization_tuning"
            })
            
        except Exception as e:
            logger.warning(f"参数调优失败: {e}")
    
    async def _adjust_strategy_parameters(
        self,
        recommendation: OptimizationRecommendation
    ) -> None:
        """调整策略参数"""
        try:
            # 发布策略调整信息到InfoPool
            if self.info_pool:
                await self.info_pool.publish_info(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "usage_optimizer",
                        "update_type": "strategy_adjustment",
                        "target_entity": recommendation.target_entity,
                        "adjustments": {
                            "increase_retry_attempts": True,
                            "enable_fallback_strategies": True,
                            "adjust_timeout_thresholds": True
                        },
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.MEDIUM
                )
            
        except Exception as e:
            logger.warning(f"调整策略参数失败: {e}")
    
    async def _optimize_resource_allocation(
        self,
        recommendation: OptimizationRecommendation
    ) -> None:
        """优化资源分配"""
        try:
            # 发布资源优化信息到InfoPool
            if self.info_pool:
                await self.info_pool.publish_info(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "usage_optimizer",
                        "update_type": "resource_optimization",
                        "target_entity": recommendation.target_entity,
                        "optimizations": {
                            "reduce_memory_usage": True,
                            "optimize_cpu_allocation": True,
                            "implement_resource_pooling": True
                        },
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.MEDIUM
                )
            
        except Exception as e:
            logger.warning(f"优化资源分配失败: {e}")
    
    async def _optimize_algorithms(
        self,
        recommendation: OptimizationRecommendation
    ) -> None:
        """优化算法"""
        try:
            # 发布算法优化信息到InfoPool
            if self.info_pool:
                await self.info_pool.publish_info(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "usage_optimizer",
                        "update_type": "algorithm_optimization",
                        "target_entity": recommendation.target_entity,
                        "optimizations": {
                            "improve_search_algorithms": True,
                            "optimize_decision_making": True,
                            "enhance_caching_strategies": True
                        },
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.MEDIUM
                )
            
        except Exception as e:
            logger.warning(f"优化算法失败: {e}")
    
    async def _execute_generic_step(
        self,
        step: Dict[str, Any],
        recommendation: OptimizationRecommendation
    ) -> None:
        """执行通用步骤"""
        try:
            # 模拟步骤执行时间
            estimated_time = step.get("estimated_time", 60)
            await asyncio.sleep(min(5, estimated_time / 60))  # 最多等待5秒
            
            logger.info(f"执行通用优化步骤: {step.get('description', '未知步骤')}")
            
        except Exception as e:
            logger.warning(f"执行通用步骤失败: {e}")
    
    async def _collect_post_optimization_metrics(
        self,
        target_entity: str,
        task_type: str
    ) -> Optional[PerformanceMetrics]:
        """收集优化后的性能指标"""
        try:
            key = f"{target_entity}_{task_type}"
            
            # 等待新的性能数据
            await asyncio.sleep(10)
            
            # 返回当前指标（在实际实现中，这里应该收集新的性能数据）
            return self.current_metrics.get(key)
            
        except Exception as e:
            logger.warning(f"收集优化后指标失败: {e}")
            return None
    
    def _calculate_actual_improvement(
        self,
        before_metrics: PerformanceMetrics,
        after_metrics: PerformanceMetrics
    ) -> Dict[str, float]:
        """计算实际改进"""
        improvement = {}
        
        # 计算各项指标的改进
        improvement["success_rate"] = after_metrics.success_rate - before_metrics.success_rate
        improvement["efficiency_score"] = after_metrics.efficiency_score - before_metrics.efficiency_score
        improvement["average_duration"] = (
            before_metrics.average_duration - after_metrics.average_duration
        ) / before_metrics.average_duration if before_metrics.average_duration > 0 else 0.0
        improvement["error_rate"] = before_metrics.error_rate - after_metrics.error_rate
        improvement["quality_score"] = after_metrics.quality_score - before_metrics.quality_score
        
        return improvement
    
    def _evaluate_optimization_success(
        self,
        expected_improvement: Dict[str, float],
        actual_improvement: Dict[str, float]
    ) -> bool:
        """评估优化是否成功"""
        try:
            # 检查是否达到预期改进的阈值
            threshold = self.optimization_params["improvement_threshold"]
            
            success_count = 0
            total_count = 0
            
            for metric, expected in expected_improvement.items():
                if metric in actual_improvement:
                    actual = actual_improvement[metric]
                    total_count += 1
                    
                    # 对于负向指标（如执行时间、错误率），期望值为负
                    if expected < 0:
                        if actual <= expected * (1 - threshold):
                            success_count += 1
                    else:
                        if actual >= expected * (1 - threshold):
                            success_count += 1
            
            # 至少50%的指标达到预期才算成功
            return success_count / total_count >= 0.5 if total_count > 0 else False
            
        except Exception as e:
            logger.warning(f"评估优化成功性失败: {e}")
            return False
    
    def _update_optimization_stats(
        self,
        optimization_result: OptimizationResult
    ) -> None:
        """更新优化统计信息"""
        try:
            self.optimization_stats["total_optimizations"] += 1
            
            if optimization_result.success:
                self.optimization_stats["successful_optimizations"] += 1
                
                # 更新平均改进
                if optimization_result.actual_improvement:
                    improvements = list(optimization_result.actual_improvement.values())
                    avg_improvement = sum(improvements) / len(improvements)
                    
                    current_avg = self.optimization_stats["average_improvement"]
                    total_opts = self.optimization_stats["successful_optimizations"]
                    
                    self.optimization_stats["average_improvement"] = (
                        (current_avg * (total_opts - 1) + avg_improvement) / total_opts
                    )
            else:
                self.optimization_stats["failed_optimizations"] += 1
            
            # 按类型统计
            opt_type = optimization_result.recommendation.optimization_type.value
            self.optimization_stats["optimization_by_type"][opt_type] += 1
            
            # 按范围统计
            opt_scope = optimization_result.recommendation.scope.value
            self.optimization_stats["optimization_by_scope"][opt_scope] += 1
            
        except Exception as e:
            logger.warning(f"更新优化统计信息失败: {e}")
    
    async def get_usage_patterns(
        self,
        agent_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> List[UsagePattern]:
        """获取使用模式"""
        try:
            patterns = []
            
            # 分析性能历史数据生成使用模式
            for key, history in self.performance_history.items():
                if len(history) < 5:  # 样本不足
                    continue
                
                parts = key.split("_", 1)
                if len(parts) != 2:
                    continue
                
                pattern_agent_id, pattern_task_type = parts
                
                # 过滤条件
                if agent_id and pattern_agent_id != agent_id:
                    continue
                if task_type and pattern_task_type != task_type:
                    continue
                
                # 生成使用模式
                pattern = await self._generate_usage_pattern(key, history)
                if pattern:
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"获取使用模式失败: {e}")
            return []
    
    async def _generate_usage_pattern(
        self,
        key: str,
        history: deque
    ) -> Optional[UsagePattern]:
        """生成使用模式"""
        try:
            parts = key.split("_", 1)
            agent_id, task_type = parts
            
            # 计算统计信息
            total_executions = len(history)
            successful_executions = sum(1 for record in history if record["success"])
            success_rate = successful_executions / total_executions
            
            durations = [record["duration"] for record in history]
            avg_duration = statistics.mean(durations)
            
            # 分析时间分布
            time_distribution = self._analyze_time_distribution(history)
            
            # 计算资源消耗
            resource_consumption = self._calculate_resource_consumption(history)
            
            # 计算优化潜力
            optimization_potential = self._calculate_optimization_potential(
                success_rate,
                avg_duration,
                durations
            )
            
            pattern = UsagePattern(
                pattern_id=f"pattern_{key}_{get_iso_timestamp()}",
                pattern_type="execution_pattern",
                agent_id=agent_id,
                task_types=[task_type],
                frequency=total_executions,
                success_rate=success_rate,
                average_performance={
                    "duration": avg_duration,
                    "success_rate": success_rate
                },
                resource_consumption=resource_consumption,
                time_distribution=time_distribution,
                context_conditions={},
                optimization_potential=optimization_potential,
                metadata={
                    "analysis_timestamp": get_iso_timestamp(),
                    "sample_count": total_executions
                }
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"生成使用模式失败: {e}")
            return None
    
    def _analyze_time_distribution(
        self,
        history: deque
    ) -> Dict[str, int]:
        """分析时间分布"""
        time_distribution = {
            "morning": 0,    # 6-12
            "afternoon": 0,  # 12-18
            "evening": 0,    # 18-24
            "night": 0       # 0-6
        }
        
        for record in history:
            try:
                timestamp = record["timestamp"]
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                hour = dt.hour
                
                if 6 <= hour < 12:
                    time_distribution["morning"] += 1
                elif 12 <= hour < 18:
                    time_distribution["afternoon"] += 1
                elif 18 <= hour < 24:
                    time_distribution["evening"] += 1
                else:
                    time_distribution["night"] += 1
                    
            except Exception:
                continue
        
        return time_distribution
    
    def _calculate_resource_consumption(
        self,
        history: deque
    ) -> Dict[str, float]:
        """计算资源消耗"""
        resource_consumption = {
            "cpu": 0.0,
            "memory": 0.0,
            "network": 0.0
        }
        
        total_records = 0
        for record in history:
            try:
                metrics = record.get("metrics")
                if metrics and hasattr(metrics, 'resource_usage'):
                    for resource, usage in metrics.resource_usage.items():
                        if resource in resource_consumption:
                            resource_consumption[resource] += usage
                            total_records += 1
            except Exception:
                continue
        
        # 计算平均值
        if total_records > 0:
            for resource in resource_consumption:
                resource_consumption[resource] /= total_records
        
        return resource_consumption
    
    def _calculate_optimization_potential(
        self,
        success_rate: float,
        avg_duration: float,
        durations: List[float]
    ) -> float:
        """计算优化潜力"""
        try:
            # 基于成功率、执行时间变异性等计算优化潜力
            potential = 0.0
            
            # 成功率低表示有优化空间
            if success_rate < 0.8:
                potential += (0.8 - success_rate) * 0.5
            
            # 执行时间变异性高表示有优化空间
            if len(durations) > 1:
                duration_variance = statistics.variance(durations)
                normalized_variance = min(1.0, duration_variance / (avg_duration ** 2))
                potential += normalized_variance * 0.3
            
            # 执行时间过长表示有优化空间
            if avg_duration > 30.0:  # 超过30秒
                potential += min(0.2, (avg_duration - 30.0) / 120.0)
            
            return min(1.0, potential)
            
        except Exception as e:
            logger.warning(f"计算优化潜力失败: {e}")
            return 0.5
    
    async def get_optimization_recommendations(
        self,
        target_entity: Optional[str] = None,
        optimization_type: Optional[OptimizationType] = None,
        min_priority: float = 0.0
    ) -> List[OptimizationRecommendation]:
        """获取优化建议"""
        try:
            recommendations = self.optimization_recommendations.copy()
            
            # 过滤条件
            if target_entity:
                recommendations = [
                    rec for rec in recommendations
                    if rec.target_entity == target_entity
                ]
            
            if optimization_type:
                recommendations = [
                    rec for rec in recommendations
                    if rec.optimization_type == optimization_type
                ]
            
            if min_priority > 0.0:
                recommendations = [
                    rec for rec in recommendations
                    if rec.priority >= min_priority
                ]
            
            # 按优先级排序
            recommendations.sort(
                key=lambda x: x.priority * x.confidence,
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"获取优化建议失败: {e}")
            return []
    
    async def get_optimization_results(
        self,
        optimization_id: Optional[str] = None,
        success_only: bool = False
    ) -> List[OptimizationResult]:
        """获取优化结果"""
        try:
            results = self.optimization_results.copy()
            
            if optimization_id:
                results = [
                    result for result in results
                    if result.optimization_id == optimization_id
                ]
            
            if success_only:
                results = [
                    result for result in results
                    if result.success
                ]
            
            return results
            
        except Exception as e:
            logger.error(f"获取优化结果失败: {e}")
            return []
    
    def get_adaptive_parameters(
        self,
        entity: str = "global"
    ) -> Dict[str, Any]:
        """获取自适应参数"""
        return self.adaptive_parameters.get(entity, {}).copy()
    
    def update_adaptive_parameter(
        self,
        entity: str,
        parameter_name: str,
        new_value: Any,
        reason: str = "manual_update"
    ) -> bool:
        """更新自适应参数"""
        try:
            if entity not in self.adaptive_parameters:
                self.adaptive_parameters[entity] = {}
            
            if parameter_name not in self.adaptive_parameters[entity]:
                # 创建新参数
                self.adaptive_parameters[entity][parameter_name] = {
                    "value": new_value,
                    "min": new_value * 0.1 if isinstance(new_value, (int, float)) else new_value,
                    "max": new_value * 10 if isinstance(new_value, (int, float)) else new_value,
                    "adjustment_factor": 0.1 if isinstance(new_value, (int, float)) else 1
                }
            else:
                # 更新现有参数
                param_config = self.adaptive_parameters[entity][parameter_name]
                
                # 检查边界
                if isinstance(new_value, (int, float)):
                    min_val = param_config.get("min", float('-inf'))
                    max_val = param_config.get("max", float('inf'))
                    new_value = max(min_val, min(max_val, new_value))
                
                param_config["value"] = new_value
            
            # 记录调整历史
            self.parameter_adjustment_history.append({
                "timestamp": get_iso_timestamp(),
                "entity": entity,
                "parameter_name": parameter_name,
                "new_value": new_value,
                "reason": reason
            })
            
            return True
            
        except Exception as e:
            logger.error(f"更新自适应参数失败: {e}")
            return False
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = self.optimization_stats.copy()
        
        # 添加实时统计
        stats["active_optimizations"] = len(self.active_optimizations)
        stats["pending_recommendations"] = len(self.optimization_recommendations)
        stats["total_performance_metrics"] = len(self.current_metrics)
        stats["total_usage_patterns"] = len(self.usage_patterns)
        
        # 计算成功率
        if stats["total_optimizations"] > 0:
            stats["success_rate"] = (
                stats["successful_optimizations"] / stats["total_optimizations"]
            )
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    async def clear_optimization_cache(
        self,
        older_than_hours: int = 24
    ) -> Dict[str, int]:
        """清理优化缓存"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            
            # 清理优化建议
            old_recommendations = len(self.optimization_recommendations)
            self.optimization_recommendations = [
                rec for rec in self.optimization_recommendations
                if datetime.fromisoformat(rec.metadata.get("generation_timestamp", get_iso_timestamp()).replace("Z", "+00:00")) > cutoff_time
            ]
            cleared_recommendations = old_recommendations - len(self.optimization_recommendations)
            
            # 清理优化结果（保留最近的）
            old_results = len(self.optimization_results)
            self.optimization_results = [
                result for result in self.optimization_results
                if datetime.fromisoformat(result.timestamp.replace("Z", "+00:00")) > cutoff_time
            ]
            cleared_results = old_results - len(self.optimization_results)
            
            # 清理性能历史（保留最近的）
            cleared_history = 0
            for key in list(self.performance_history.keys()):
                history = self.performance_history[key]
                old_count = len(history)
                
                # 过滤旧记录
                filtered_history = deque([
                    record for record in history
                    if datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")) > cutoff_time
                ], maxlen=1000)
                
                self.performance_history[key] = filtered_history
                cleared_history += old_count - len(filtered_history)
            
            return {
                "cleared_recommendations": cleared_recommendations,
                "cleared_results": cleared_results,
                "cleared_history_records": cleared_history
            }
            
        except Exception as e:
            logger.error(f"清理优化缓存失败: {e}")
            return {
                "cleared_recommendations": 0,
                "cleared_results": 0,
                "cleared_history_records": 0
            }
    
    async def shutdown(self) -> None:
        """关闭使用优化器"""
        try:
            # 停止所有活跃的优化
            for optimization_id in list(self.active_optimizations.keys()):
                optimization_result = self.active_optimizations[optimization_id]
                optimization_result.implementation_status = "cancelled"
                optimization_result.lessons_learned.append("优化因系统关闭而取消")
            
            # 发布最终统计信息
            if self.info_pool:
                await self.info_pool.publish_info(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "usage_optimizer",
                        "update_type": "optimizer_shutdown",
                        "final_stats": self.get_optimization_stats(),
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.LOW
                )
            
            logger.info("使用优化器已关闭")
            
        except Exception as e:
            logger.error(f"关闭使用优化器失败: {e}")