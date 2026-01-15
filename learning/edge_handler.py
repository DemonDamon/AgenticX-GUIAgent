#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Learning Engine - Edge Handler (基于AgenticX框架重构)
边缘处理器：处理异常情况、边缘案例和错误恢复

重构说明：
- 基于AgenticX的Component基类重构
- 使用AgenticX的事件系统进行边缘案例通知
- 集成AgenticX的错误处理和恢复机制
- 遵循AgenticX的异常管理和监控架构

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
import traceback
from uuid import uuid4

from agenticx.core.component import Component

from core.info_pool import InfoPool, InfoType, InfoPriority
from utils import get_iso_timestamp


class EdgeCaseType(Enum):
    """边缘案例类型"""
    UNKNOWN_ERROR = "unknown_error"  # 未知错误
    TIMEOUT = "timeout"  # 超时
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # 资源耗尽
    NETWORK_FAILURE = "network_failure"  # 网络故障
    PERMISSION_DENIED = "permission_denied"  # 权限拒绝
    INVALID_INPUT = "invalid_input"  # 无效输入
    STATE_INCONSISTENCY = "state_inconsistency"  # 状态不一致
    CONCURRENT_CONFLICT = "concurrent_conflict"  # 并发冲突
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"  # 外部服务故障
    DATA_CORRUPTION = "data_corruption"  # 数据损坏
    CONFIGURATION_ERROR = "configuration_error"  # 配置错误
    HARDWARE_FAILURE = "hardware_failure"  # 硬件故障


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"  # 重试
    FALLBACK = "fallback"  # 回退
    SKIP = "skip"  # 跳过
    ABORT = "abort"  # 中止
    ESCALATE = "escalate"  # 升级
    COMPENSATE = "compensate"  # 补偿
    RESTART = "restart"  # 重启
    ROLLBACK = "rollback"  # 回滚
    ALTERNATIVE = "alternative"  # 替代方案
    MANUAL_INTERVENTION = "manual_intervention"  # 人工干预


class EdgeCaseSeverity(Enum):
    """边缘案例严重程度"""
    LOW = "low"  # 低
    MEDIUM = "medium"  # 中
    HIGH = "high"  # 高
    CRITICAL = "critical"  # 严重


class EdgeHandlingStrategy(Enum):
    """边缘处理策略"""
    PROACTIVE = "proactive"  # 主动处理
    REACTIVE = "reactive"  # 被动处理
    PREDICTIVE = "predictive"  # 预测性处理
    ADAPTIVE = "adaptive"  # 自适应处理
    PREVENTIVE = "preventive"  # 预防性处理
    CORRECTIVE = "corrective"  # 纠正性处理
    ESCALATION = "escalation"  # 升级处理
    ISOLATION = "isolation"  # 隔离处理


@dataclass
class EdgeCase:
    """边缘案例"""
    case_id: str
    case_type: EdgeCaseType
    severity: EdgeCaseSeverity
    description: str
    context: Dict[str, Any]
    timestamp: str
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolution_status: str = "unresolved"  # unresolved, resolved, escalated
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """恢复动作"""
    action_id: str
    strategy: RecoveryStrategy
    description: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    expected_outcome: str
    timeout_seconds: float
    retry_count: int = 0
    max_retries: int = 3
    success_criteria: List[str] = field(default_factory=list)
    failure_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """恢复结果"""
    recovery_id: str
    edge_case_id: str
    action_id: str
    success: bool
    execution_time: float
    outcome_description: str
    side_effects: List[str]
    lessons_learned: List[str]
    timestamp: str
    error_details: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgePattern:
    """边缘模式"""
    pattern_id: str
    pattern_type: EdgeCaseType
    frequency: int
    success_rate: float
    common_contexts: List[Dict[str, Any]]
    effective_strategies: List[RecoveryStrategy]
    prevention_measures: List[str]
    detection_rules: List[str]
    confidence: float
    last_updated: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeHandlingResult:
    """边缘处理结果"""
    handling_id: str
    edge_case: EdgeCase
    strategy: EdgeHandlingStrategy
    recovery_actions: List[RecoveryAction]
    recovery_results: List[RecoveryResult]
    overall_success: bool
    total_handling_time: float
    lessons_learned: List[str]
    patterns_detected: List[EdgePattern]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EdgeHandler(Component):
    """边缘处理器
    
    负责处理异常情况、边缘案例和错误恢复的学习引擎组件。
    基于AgenticX的Component基类重构。
    """
    
    def __init__(
        self,
        info_pool=None,
        max_edge_cases: int = 10000,
        max_recovery_history: int = 5000,
        pattern_detection_threshold: int = 5,
        auto_recovery_enabled: bool = True
    ):
        super().__init__()
        self.info_pool = info_pool
        self.max_edge_cases = max_edge_cases
        self.max_recovery_history = max_recovery_history
        self.pattern_detection_threshold = pattern_detection_threshold
        self.auto_recovery_enabled = auto_recovery_enabled
        
        # 核心数据结构
        self.edge_cases: List[EdgeCase] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.recovery_results: List[RecoveryResult] = []
        self.edge_patterns: Dict[str, EdgePattern] = {}
        
        # 实时监控
        self.active_recoveries: Dict[str, RecoveryResult] = {}
        self.case_frequency: Dict[EdgeCaseType, int] = defaultdict(int)
        self.strategy_effectiveness: Dict[RecoveryStrategy, Dict[str, float]] = defaultdict(lambda: {"success_rate": 0.0, "avg_time": 0.0})
        
        # 预定义恢复策略
        self.default_strategies: Dict[EdgeCaseType, List[RecoveryStrategy]] = {
            EdgeCaseType.TIMEOUT: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            EdgeCaseType.NETWORK_FAILURE: [RecoveryStrategy.RETRY, RecoveryStrategy.ALTERNATIVE],
            EdgeCaseType.PERMISSION_DENIED: [RecoveryStrategy.ESCALATE, RecoveryStrategy.ALTERNATIVE],
            EdgeCaseType.RESOURCE_EXHAUSTION: [RecoveryStrategy.RESTART, RecoveryStrategy.COMPENSATE],
            EdgeCaseType.INVALID_INPUT: [RecoveryStrategy.SKIP, RecoveryStrategy.COMPENSATE],
            EdgeCaseType.STATE_INCONSISTENCY: [RecoveryStrategy.ROLLBACK, RecoveryStrategy.RESTART],
            EdgeCaseType.CONCURRENT_CONFLICT: [RecoveryStrategy.RETRY, RecoveryStrategy.ROLLBACK],
            EdgeCaseType.EXTERNAL_SERVICE_FAILURE: [RecoveryStrategy.FALLBACK, RecoveryStrategy.ALTERNATIVE],
            EdgeCaseType.DATA_CORRUPTION: [RecoveryStrategy.ROLLBACK, RecoveryStrategy.MANUAL_INTERVENTION],
            EdgeCaseType.CONFIGURATION_ERROR: [RecoveryStrategy.RESTART, RecoveryStrategy.MANUAL_INTERVENTION],
            EdgeCaseType.HARDWARE_FAILURE: [RecoveryStrategy.ALTERNATIVE, RecoveryStrategy.MANUAL_INTERVENTION],
            EdgeCaseType.UNKNOWN_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.ESCALATE]
        }
        
        # 统计信息
        self.edge_stats = {
            "total_edge_cases": 0,
            "resolved_cases": 0,
            "escalated_cases": 0,
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "patterns_detected": 0,
            "auto_recoveries": 0,
            "manual_interventions": 0
        }
        
        # 配置
        self.recovery_timeout = 300.0  # 5分钟
        self.pattern_confidence_threshold = 0.7
        self.escalation_threshold = 3  # 连续失败次数
        
        # 日志
        self.logger = logger
        
        # 初始化预定义恢复动作
        self._initialize_recovery_actions()
    
    def _initialize_recovery_actions(self) -> None:
        """初始化预定义恢复动作"""
        # 重试动作
        self.recovery_actions["retry_action"] = RecoveryAction(
            action_id="retry_action",
            strategy=RecoveryStrategy.RETRY,
            description="重试失败的操作",
            parameters={"max_retries": 3, "backoff_factor": 2.0},
            preconditions=["error_is_transient"],
            expected_outcome="操作成功完成",
            timeout_seconds=60.0,
            success_criteria=["no_error_occurred"],
            failure_criteria=["max_retries_exceeded"]
        )
        
        # 回退动作
        self.recovery_actions["fallback_action"] = RecoveryAction(
            action_id="fallback_action",
            strategy=RecoveryStrategy.FALLBACK,
            description="使用备用方案",
            parameters={"fallback_method": "alternative_implementation"},
            preconditions=["fallback_available"],
            expected_outcome="使用备用方案完成任务",
            timeout_seconds=120.0,
            success_criteria=["task_completed_with_fallback"],
            failure_criteria=["fallback_also_failed"]
        )
        
        # 跳过动作
        self.recovery_actions["skip_action"] = RecoveryAction(
            action_id="skip_action",
            strategy=RecoveryStrategy.SKIP,
            description="跳过当前操作",
            parameters={"skip_reason": "non_critical_operation"},
            preconditions=["operation_is_optional"],
            expected_outcome="跳过操作并继续执行",
            timeout_seconds=5.0,
            success_criteria=["operation_skipped"],
            failure_criteria=["skip_not_allowed"]
        )
        
        # 重启动作
        self.recovery_actions["restart_action"] = RecoveryAction(
            action_id="restart_action",
            strategy=RecoveryStrategy.RESTART,
            description="重启相关组件",
            parameters={"restart_scope": "component"},
            preconditions=["restart_permission"],
            expected_outcome="组件重启并恢复正常",
            timeout_seconds=180.0,
            success_criteria=["component_restarted", "health_check_passed"],
            failure_criteria=["restart_failed"]
        )
        
        # 回滚动作
        self.recovery_actions["rollback_action"] = RecoveryAction(
            action_id="rollback_action",
            strategy=RecoveryStrategy.ROLLBACK,
            description="回滚到之前的状态",
            parameters={"rollback_point": "last_known_good_state"},
            preconditions=["rollback_point_available"],
            expected_outcome="成功回滚到稳定状态",
            timeout_seconds=240.0,
            success_criteria=["state_rolled_back", "consistency_verified"],
            failure_criteria=["rollback_failed"]
        )
    
    async def handle_edge_case(
        self,
        case_type: EdgeCaseType,
        description: str,
        context: Dict[str, Any],
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        auto_recover: bool = True
    ) -> EdgeCase:
        """处理边缘案例"""
        try:
            # 创建边缘案例
            edge_case = EdgeCase(
                case_id=str(uuid4()),
                case_type=case_type,
                severity=self._determine_severity(case_type, context, error_details),
                description=description,
                context=context,
                timestamp=get_iso_timestamp(),
                agent_id=agent_id,
                task_id=task_id,
                error_details=error_details,
                stack_trace=traceback.format_exc() if error_details else None
            )
            
            # 记录边缘案例
            self.edge_cases.append(edge_case)
            self.case_frequency[case_type] += 1
            self.edge_stats["total_edge_cases"] += 1
            
            # 限制边缘案例数量
            if len(self.edge_cases) > self.max_edge_cases:
                self.edge_cases = self.edge_cases[-self.max_edge_cases:]
            
            # 检测模式
            await self._detect_edge_patterns(edge_case)
            
            # 自动恢复
            if auto_recover and self.auto_recovery_enabled:
                recovery_result = await self._attempt_auto_recovery(edge_case)
                if recovery_result and recovery_result.success:
                    edge_case.resolution_status = "resolved"
                    self.edge_stats["resolved_cases"] += 1
                    self.edge_stats["auto_recoveries"] += 1
                else:
                    # 考虑升级
                    if self._should_escalate(edge_case):
                        await self._escalate_edge_case(edge_case)
            
            # 发布边缘案例信息
            if self.info_pool:
                await self.info_pool.publish(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "edge_handler",
                        "update_type": "edge_case_detected",
                        "edge_case": {
                            "case_id": edge_case.case_id,
                            "case_type": edge_case.case_type.value,
                            "severity": edge_case.severity.value,
                            "description": edge_case.description,
                            "agent_id": edge_case.agent_id,
                            "task_id": edge_case.task_id
                        },
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.HIGH if edge_case.severity in [EdgeCaseSeverity.HIGH, EdgeCaseSeverity.CRITICAL] else InfoPriority.MEDIUM
                )
            
            return edge_case
            
        except Exception as e:
            logger.error(f"处理边缘案例失败: {e}")
            # 创建一个基本的边缘案例记录
            return EdgeCase(
                case_id=str(uuid4()),
                case_type=EdgeCaseType.UNKNOWN_ERROR,
                severity=EdgeCaseSeverity.HIGH,
                description=f"处理边缘案例时发生错误: {str(e)}",
                context=context,
                timestamp=get_iso_timestamp(),
                agent_id=agent_id,
                task_id=task_id
            )
    
    def _determine_severity(
        self,
        case_type: EdgeCaseType,
        context: Dict[str, Any],
        error_details: Optional[Dict[str, Any]]
    ) -> EdgeCaseSeverity:
        """确定边缘案例严重程度"""
        # 基于案例类型的基础严重程度
        base_severity = {
            EdgeCaseType.HARDWARE_FAILURE: EdgeCaseSeverity.CRITICAL,
            EdgeCaseType.DATA_CORRUPTION: EdgeCaseSeverity.CRITICAL,
            EdgeCaseType.CONFIGURATION_ERROR: EdgeCaseSeverity.HIGH,
            EdgeCaseType.EXTERNAL_SERVICE_FAILURE: EdgeCaseSeverity.HIGH,
            EdgeCaseType.RESOURCE_EXHAUSTION: EdgeCaseSeverity.HIGH,
            EdgeCaseType.STATE_INCONSISTENCY: EdgeCaseSeverity.HIGH,
            EdgeCaseType.PERMISSION_DENIED: EdgeCaseSeverity.MEDIUM,
            EdgeCaseType.NETWORK_FAILURE: EdgeCaseSeverity.MEDIUM,
            EdgeCaseType.CONCURRENT_CONFLICT: EdgeCaseSeverity.MEDIUM,
            EdgeCaseType.TIMEOUT: EdgeCaseSeverity.MEDIUM,
            EdgeCaseType.INVALID_INPUT: EdgeCaseSeverity.LOW,
            EdgeCaseType.UNKNOWN_ERROR: EdgeCaseSeverity.MEDIUM
        }.get(case_type, EdgeCaseSeverity.MEDIUM)
        
        # 基于上下文调整严重程度
        if context.get("critical_task", False):
            if base_severity == EdgeCaseSeverity.LOW:
                base_severity = EdgeCaseSeverity.MEDIUM
            elif base_severity == EdgeCaseSeverity.MEDIUM:
                base_severity = EdgeCaseSeverity.HIGH
        
        if context.get("user_facing", False):
            if base_severity == EdgeCaseSeverity.LOW:
                base_severity = EdgeCaseSeverity.MEDIUM
        
        # 基于错误频率调整
        if self.case_frequency[case_type] > 10:  # 频繁出现
            if base_severity == EdgeCaseSeverity.LOW:
                base_severity = EdgeCaseSeverity.MEDIUM
            elif base_severity == EdgeCaseSeverity.MEDIUM:
                base_severity = EdgeCaseSeverity.HIGH
        
        return base_severity
    
    async def _detect_edge_patterns(self, edge_case: EdgeCase) -> None:
        """检测边缘模式"""
        try:
            # 查找相似的边缘案例
            similar_cases = [
                case for case in self.edge_cases[-100:]  # 检查最近100个案例
                if case.case_type == edge_case.case_type
                and case.case_id != edge_case.case_id
            ]
            
            if len(similar_cases) >= self.pattern_detection_threshold:
                pattern_id = f"pattern_{edge_case.case_type.value}_{len(self.edge_patterns)}"
                
                # 分析共同上下文
                common_contexts = self._find_common_contexts(similar_cases + [edge_case])
                
                # 分析有效策略
                effective_strategies = self._analyze_effective_strategies(similar_cases)
                
                # 计算成功率
                resolved_cases = [case for case in similar_cases if case.resolution_status == "resolved"]
                success_rate = len(resolved_cases) / len(similar_cases) if similar_cases else 0.0
                
                # 创建或更新模式
                if pattern_id not in self.edge_patterns:
                    pattern = EdgePattern(
                        pattern_id=pattern_id,
                        pattern_type=edge_case.case_type,
                        frequency=len(similar_cases) + 1,
                        success_rate=success_rate,
                        common_contexts=common_contexts,
                        effective_strategies=effective_strategies,
                        prevention_measures=self._generate_prevention_measures(edge_case.case_type, common_contexts),
                        detection_rules=self._generate_detection_rules(edge_case.case_type, common_contexts),
                        confidence=min(1.0, len(similar_cases) / 20.0),
                        last_updated=get_iso_timestamp()
                    )
                    
                    self.edge_patterns[pattern_id] = pattern
                    self.edge_stats["patterns_detected"] += 1
                    
                    # 发布模式检测信息
                    if self.info_pool:
                        await self.info_pool.publish(
                            InfoType.LEARNING_UPDATE,
                            {
                                "agent_id": "edge_handler",
                                "update_type": "pattern_detected",
                                "pattern": {
                                    "pattern_id": pattern.pattern_id,
                                    "pattern_type": pattern.pattern_type.value,
                                    "frequency": pattern.frequency,
                                    "success_rate": pattern.success_rate,
                                    "confidence": pattern.confidence
                                },
                                "timestamp": get_iso_timestamp()
                            },
                            InfoPriority.MEDIUM
                        )
                
        except Exception as e:
            logger.warning(f"检测边缘模式失败: {e}")
    
    def _find_common_contexts(self, cases: List[EdgeCase]) -> List[Dict[str, Any]]:
        """查找共同上下文"""
        if not cases:
            return []
        
        # 收集所有上下文键
        all_keys = set()
        for case in cases:
            all_keys.update(case.context.keys())
        
        common_contexts = []
        for key in all_keys:
            values = [case.context.get(key) for case in cases if key in case.context]
            if len(values) >= len(cases) * 0.7:  # 至少70%的案例包含此键
                # 查找最常见的值
                value_counts = {}
                for value in values:
                    if value is not None:
                        value_str = str(value)
                        value_counts[value_str] = value_counts.get(value_str, 0) + 1
                
                if value_counts:
                    most_common_value = max(value_counts, key=value_counts.get)
                    if value_counts[most_common_value] >= len(cases) * 0.5:  # 至少50%的案例有相同值
                        common_contexts.append({
                            "key": key,
                            "value": most_common_value,
                            "frequency": value_counts[most_common_value] / len(cases)
                        })
        
        return common_contexts
    
    def _analyze_effective_strategies(self, cases: List[EdgeCase]) -> List[RecoveryStrategy]:
        """分析有效策略"""
        strategy_success = defaultdict(lambda: {"total": 0, "success": 0})
        
        for case in cases:
            for attempt in case.recovery_attempts:
                strategy = attempt.get("strategy")
                success = attempt.get("success", False)
                
                if strategy:
                    try:
                        strategy_enum = RecoveryStrategy(strategy)
                        strategy_success[strategy_enum]["total"] += 1
                        if success:
                            strategy_success[strategy_enum]["success"] += 1
                    except ValueError:
                        continue
        
        # 按成功率排序
        effective_strategies = []
        for strategy, stats in strategy_success.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                if success_rate > 0.5:  # 成功率超过50%
                    effective_strategies.append((strategy, success_rate))
        
        effective_strategies.sort(key=lambda x: x[1], reverse=True)
        return [strategy for strategy, _ in effective_strategies]
    
    def _generate_prevention_measures(self, case_type: EdgeCaseType, common_contexts: List[Dict[str, Any]]) -> List[str]:
        """生成预防措施"""
        measures = []
        
        # 基于案例类型的通用预防措施
        type_measures = {
            EdgeCaseType.TIMEOUT: [
                "增加超时时间",
                "实现异步处理",
                "添加进度监控"
            ],
            EdgeCaseType.NETWORK_FAILURE: [
                "实现网络重试机制",
                "添加网络状态检查",
                "使用备用网络路径"
            ],
            EdgeCaseType.RESOURCE_EXHAUSTION: [
                "实现资源监控",
                "添加资源限制",
                "优化资源使用"
            ],
            EdgeCaseType.PERMISSION_DENIED: [
                "验证权限配置",
                "实现权限检查",
                "添加权限申请流程"
            ],
            EdgeCaseType.INVALID_INPUT: [
                "加强输入验证",
                "添加输入清理",
                "实现输入格式检查"
            ]
        }
        
        measures.extend(type_measures.get(case_type, []))
        
        # 基于共同上下文的特定预防措施
        for context in common_contexts:
            key = context["key"]
            value = context["value"]
            
            if key == "operation_type" and "write" in str(value).lower():
                measures.append("实现写操作前的状态检查")
            elif key == "data_size" and isinstance(value, str) and "large" in value:
                measures.append("实现大数据处理优化")
            elif key == "concurrent_users" and isinstance(value, str) and int(value) > 10:
                measures.append("实现并发控制机制")
        
        return list(set(measures))  # 去重
    
    def _generate_detection_rules(self, case_type: EdgeCaseType, common_contexts: List[Dict[str, Any]]) -> List[str]:
        """生成检测规则"""
        rules = []
        
        # 基于案例类型的检测规则
        type_rules = {
            EdgeCaseType.TIMEOUT: [
                "监控操作执行时间",
                "检测长时间运行的任务"
            ],
            EdgeCaseType.NETWORK_FAILURE: [
                "监控网络连接状态",
                "检测网络延迟异常"
            ],
            EdgeCaseType.RESOURCE_EXHAUSTION: [
                "监控CPU和内存使用率",
                "检测资源使用趋势"
            ],
            EdgeCaseType.PERMISSION_DENIED: [
                "监控权限验证失败",
                "检测权限配置变更"
            ]
        }
        
        rules.extend(type_rules.get(case_type, []))
        
        # 基于共同上下文的检测规则
        for context in common_contexts:
            key = context["key"]
            value = context["value"]
            frequency = context["frequency"]
            
            if frequency > 0.8:  # 高频率上下文
                rules.append(f"监控{key}={value}的情况")
        
        return list(set(rules))  # 去重
    
    async def _attempt_auto_recovery(self, edge_case: EdgeCase) -> Optional[RecoveryResult]:
        """尝试自动恢复"""
        try:
            # 获取推荐的恢复策略
            strategies = self._get_recommended_strategies(edge_case)
            
            for strategy in strategies:
                # 获取对应的恢复动作
                action = self._get_recovery_action(strategy, edge_case)
                if not action:
                    continue
                
                # 检查前置条件
                if not self._check_preconditions(action, edge_case):
                    continue
                
                # 执行恢复动作
                recovery_result = await self._execute_recovery_action(action, edge_case)
                
                # 记录恢复尝试
                edge_case.recovery_attempts.append({
                    "strategy": strategy.value,
                    "action_id": action.action_id,
                    "success": recovery_result.success,
                    "timestamp": recovery_result.timestamp,
                    "execution_time": recovery_result.execution_time
                })
                
                if recovery_result.success:
                    return recovery_result
            
            return None
            
        except Exception as e:
            logger.error(f"自动恢复失败: {e}")
            return None
    
    def _get_recommended_strategies(self, edge_case: EdgeCase) -> List[RecoveryStrategy]:
        """获取推荐的恢复策略"""
        # 首先检查是否有匹配的模式
        for pattern in self.edge_patterns.values():
            if (pattern.pattern_type == edge_case.case_type and 
                pattern.confidence >= self.pattern_confidence_threshold):
                if pattern.effective_strategies:
                    return pattern.effective_strategies[:3]  # 最多3个策略
        
        # 使用默认策略
        return self.default_strategies.get(edge_case.case_type, [RecoveryStrategy.RETRY])
    
    def _get_recovery_action(self, strategy: RecoveryStrategy, edge_case: EdgeCase) -> Optional[RecoveryAction]:
        """获取恢复动作"""
        # 查找匹配的预定义动作
        for action in self.recovery_actions.values():
            if action.strategy == strategy:
                return action
        
        # 动态创建恢复动作
        return self._create_dynamic_recovery_action(strategy, edge_case)
    
    def _create_dynamic_recovery_action(self, strategy: RecoveryStrategy, edge_case: EdgeCase) -> Optional[RecoveryAction]:
        """动态创建恢复动作"""
        try:
            action_id = f"dynamic_{strategy.value}_{edge_case.case_id[:8]}"
            
            if strategy == RecoveryStrategy.RETRY:
                return RecoveryAction(
                    action_id=action_id,
                    strategy=strategy,
                    description=f"重试失败的操作: {edge_case.description}",
                    parameters={"max_retries": 3, "backoff_factor": 1.5},
                    preconditions=["error_is_transient"],
                    expected_outcome="操作成功完成",
                    timeout_seconds=60.0
                )
            elif strategy == RecoveryStrategy.SKIP:
                return RecoveryAction(
                    action_id=action_id,
                    strategy=strategy,
                    description=f"跳过失败的操作: {edge_case.description}",
                    parameters={"skip_reason": "non_critical_failure"},
                    preconditions=["operation_is_optional"],
                    expected_outcome="跳过操作并继续",
                    timeout_seconds=5.0
                )
            elif strategy == RecoveryStrategy.ESCALATE:
                return RecoveryAction(
                    action_id=action_id,
                    strategy=strategy,
                    description=f"升级处理: {edge_case.description}",
                    parameters={"escalation_level": "supervisor"},
                    preconditions=["escalation_allowed"],
                    expected_outcome="问题被升级处理",
                    timeout_seconds=10.0
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"创建动态恢复动作失败: {e}")
            return None
    
    def _check_preconditions(self, action: RecoveryAction, edge_case: EdgeCase) -> bool:
        """检查前置条件"""
        try:
            for condition in action.preconditions:
                if condition == "error_is_transient":
                    # 检查错误是否是暂时性的
                    transient_types = {
                        EdgeCaseType.TIMEOUT,
                        EdgeCaseType.NETWORK_FAILURE,
                        EdgeCaseType.CONCURRENT_CONFLICT
                    }
                    if edge_case.case_type not in transient_types:
                        return False
                
                elif condition == "operation_is_optional":
                    # 检查操作是否是可选的
                    if edge_case.context.get("critical_task", False):
                        return False
                
                elif condition == "escalation_allowed":
                    # 检查是否允许升级
                    if edge_case.severity == EdgeCaseSeverity.LOW:
                        return False
                
                elif condition == "fallback_available":
                    # 检查是否有备用方案
                    if not edge_case.context.get("has_fallback", False):
                        return False
                
                elif condition == "restart_permission":
                    # 检查是否有重启权限
                    if not edge_case.context.get("can_restart", False):
                        return False
                
                elif condition == "rollback_point_available":
                    # 检查是否有回滚点
                    if not edge_case.context.get("has_checkpoint", False):
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"检查前置条件失败: {e}")
            return False
    
    async def _execute_recovery_action(self, action: RecoveryAction, edge_case: EdgeCase) -> RecoveryResult:
        """执行恢复动作"""
        start_time = datetime.now()
        recovery_id = str(uuid4())
        
        try:
            # 创建恢复结果
            recovery_result = RecoveryResult(
                recovery_id=recovery_id,
                edge_case_id=edge_case.case_id,
                action_id=action.action_id,
                success=False,
                execution_time=0.0,
                outcome_description="",
                side_effects=[],
                lessons_learned=[],
                timestamp=get_iso_timestamp()
            )
            
            # 添加到活跃恢复
            self.active_recoveries[recovery_id] = recovery_result
            
            # 模拟恢复动作执行
            success = await self._simulate_recovery_execution(action, edge_case)
            
            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 更新结果
            recovery_result.success = success
            recovery_result.execution_time = execution_time
            
            if success:
                recovery_result.outcome_description = f"成功执行{action.strategy.value}策略"
                recovery_result.lessons_learned.append(f"{action.strategy.value}策略对{edge_case.case_type.value}类型有效")
            else:
                recovery_result.outcome_description = f"执行{action.strategy.value}策略失败"
                recovery_result.lessons_learned.append(f"{action.strategy.value}策略对{edge_case.case_type.value}类型无效")
            
            # 更新统计信息
            self.edge_stats["total_recoveries"] += 1
            if success:
                self.edge_stats["successful_recoveries"] += 1
            
            # 更新策略效果统计
            strategy_stats = self.strategy_effectiveness[action.strategy]
            current_total = strategy_stats.get("total_attempts", 0)
            current_success = strategy_stats.get("successful_attempts", 0)
            current_time_sum = strategy_stats.get("total_time", 0.0)
            
            strategy_stats["total_attempts"] = current_total + 1
            if success:
                strategy_stats["successful_attempts"] = current_success + 1
            strategy_stats["total_time"] = current_time_sum + execution_time
            strategy_stats["success_rate"] = strategy_stats["successful_attempts"] / strategy_stats["total_attempts"]
            strategy_stats["avg_time"] = strategy_stats["total_time"] / strategy_stats["total_attempts"]
            
            # 记录恢复结果
            self.recovery_results.append(recovery_result)
            
            # 限制恢复历史数量
            if len(self.recovery_results) > self.max_recovery_history:
                self.recovery_results = self.recovery_results[-self.max_recovery_history:]
            
            # 从活跃恢复中移除
            self.active_recoveries.pop(recovery_id, None)
            
            return recovery_result
            
        except Exception as e:
            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 创建失败结果
            recovery_result = RecoveryResult(
                recovery_id=recovery_id,
                edge_case_id=edge_case.case_id,
                action_id=action.action_id,
                success=False,
                execution_time=execution_time,
                outcome_description=f"执行恢复动作时发生错误: {str(e)}",
                side_effects=["recovery_execution_error"],
                lessons_learned=["需要改进恢复动作的错误处理"],
                timestamp=get_iso_timestamp(),
                error_details={"error": str(e), "type": type(e).__name__}
            )
            
            # 从活跃恢复中移除
            self.active_recoveries.pop(recovery_id, None)
            
            logger.error(f"执行恢复动作失败: {e}")
            return recovery_result
    
    async def _simulate_recovery_execution(self, action: RecoveryAction, edge_case: EdgeCase) -> bool:
        """模拟恢复动作执行"""
        try:
            # 基于策略类型和案例类型模拟成功概率
            success_probability = 0.5  # 默认成功率
            
            # 基于策略调整成功率
            strategy_success_rates = {
                RecoveryStrategy.RETRY: 0.7,
                RecoveryStrategy.SKIP: 0.9,
                RecoveryStrategy.FALLBACK: 0.8,
                RecoveryStrategy.RESTART: 0.6,
                RecoveryStrategy.ROLLBACK: 0.7,
                RecoveryStrategy.ESCALATE: 0.95,
                RecoveryStrategy.ABORT: 1.0,
                RecoveryStrategy.COMPENSATE: 0.6,
                RecoveryStrategy.ALTERNATIVE: 0.7,
                RecoveryStrategy.MANUAL_INTERVENTION: 0.9
            }
            
            success_probability = strategy_success_rates.get(action.strategy, 0.5)
            
            # 基于案例严重程度调整
            if edge_case.severity == EdgeCaseSeverity.LOW:
                success_probability += 0.2
            elif edge_case.severity == EdgeCaseSeverity.CRITICAL:
                success_probability -= 0.2
            
            # 基于历史成功率调整
            strategy_stats = self.strategy_effectiveness.get(action.strategy, {})
            if "success_rate" in strategy_stats and strategy_stats["total_attempts"] > 5:
                historical_rate = strategy_stats["success_rate"]
                success_probability = (success_probability + historical_rate) / 2
            
            # 确保概率在合理范围内
            success_probability = max(0.1, min(0.95, success_probability))
            
            # 模拟执行时间
            import random
            await asyncio.sleep(random.uniform(0.1, 1.0))  # 模拟执行时间
            
            # 基于概率决定成功或失败
            return random.random() < success_probability
            
        except Exception as e:
            logger.warning(f"模拟恢复执行失败: {e}")
            return False
    
    def _should_escalate(self, edge_case: EdgeCase) -> bool:
        """判断是否应该升级"""
        # 检查严重程度
        if edge_case.severity == EdgeCaseSeverity.CRITICAL:
            return True
        
        # 检查恢复尝试次数
        failed_attempts = sum(1 for attempt in edge_case.recovery_attempts if not attempt.get("success", False))
        if failed_attempts >= self.escalation_threshold:
            return True
        
        # 检查案例类型
        escalation_types = {
            EdgeCaseType.HARDWARE_FAILURE,
            EdgeCaseType.DATA_CORRUPTION,
            EdgeCaseType.CONFIGURATION_ERROR
        }
        if edge_case.case_type in escalation_types:
            return True
        
        return False
    
    async def _escalate_edge_case(self, edge_case: EdgeCase) -> None:
        """升级边缘案例"""
        try:
            edge_case.resolution_status = "escalated"
            self.edge_stats["escalated_cases"] += 1
            
            # 发布升级信息
            if self.info_pool:
                await self.info_pool.publish(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "edge_handler",
                        "update_type": "edge_case_escalated",
                        "edge_case": {
                            "case_id": edge_case.case_id,
                            "case_type": edge_case.case_type.value,
                            "severity": edge_case.severity.value,
                            "description": edge_case.description,
                            "failed_attempts": len(edge_case.recovery_attempts)
                        },
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.HIGH
                )
            
            logger.warning(f"边缘案例已升级: {edge_case.case_id} - {edge_case.description}")
            
        except Exception as e:
            logger.error(f"升级边缘案例失败: {e}")
    
    async def get_edge_cases(
        self,
        case_type: Optional[EdgeCaseType] = None,
        severity: Optional[EdgeCaseSeverity] = None,
        resolution_status: Optional[str] = None,
        limit: int = 100
    ) -> List[EdgeCase]:
        """获取边缘案例"""
        try:
            cases = self.edge_cases.copy()
            
            # 过滤条件
            if case_type:
                cases = [case for case in cases if case.case_type == case_type]
            
            if severity:
                cases = [case for case in cases if case.severity == severity]
            
            if resolution_status:
                cases = [case for case in cases if case.resolution_status == resolution_status]
            
            # 按时间倒序排序
            cases.sort(key=lambda x: x.timestamp, reverse=True)
            
            return cases[:limit]
            
        except Exception as e:
            logger.error(f"获取边缘案例失败: {e}")
            return []
    
    async def get_recovery_results(
        self,
        edge_case_id: Optional[str] = None,
        success_only: bool = False,
        limit: int = 100
    ) -> List[RecoveryResult]:
        """获取恢复结果"""
        try:
            results = self.recovery_results.copy()
            
            if edge_case_id:
                results = [result for result in results if result.edge_case_id == edge_case_id]
            
            if success_only:
                results = [result for result in results if result.success]
            
            # 按时间倒序排序
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"获取恢复结果失败: {e}")
            return []
    
    def get_edge_patterns(
        self,
        pattern_type: Optional[EdgeCaseType] = None,
        min_confidence: float = 0.0
    ) -> List[EdgePattern]:
        """获取边缘模式"""
        try:
            patterns = list(self.edge_patterns.values())
            
            if pattern_type:
                patterns = [pattern for pattern in patterns if pattern.pattern_type == pattern_type]
            
            if min_confidence > 0.0:
                patterns = [pattern for pattern in patterns if pattern.confidence >= min_confidence]
            
            # 按置信度排序
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            return patterns
            
        except Exception as e:
            logger.error(f"获取边缘模式失败: {e}")
            return []
    
    def get_strategy_effectiveness(self) -> Dict[RecoveryStrategy, Dict[str, float]]:
        """获取策略效果统计"""
        return dict(self.strategy_effectiveness)
    
    def get_edge_stats(self) -> Dict[str, Any]:
        """获取边缘处理统计信息"""
        stats = self.edge_stats.copy()
        
        # 添加实时统计
        stats["active_recoveries"] = len(self.active_recoveries)
        stats["total_patterns"] = len(self.edge_patterns)
        stats["case_type_distribution"] = dict(self.case_frequency)
        
        # 计算成功率
        if stats["total_recoveries"] > 0:
            stats["recovery_success_rate"] = stats["successful_recoveries"] / stats["total_recoveries"]
        else:
            stats["recovery_success_rate"] = 0.0
        
        if stats["total_edge_cases"] > 0:
            stats["resolution_rate"] = (stats["resolved_cases"] + stats["escalated_cases"]) / stats["total_edge_cases"]
        else:
            stats["resolution_rate"] = 0.0
        
        return stats
    
    async def clear_edge_cache(
        self,
        older_than_hours: int = 72
    ) -> Dict[str, int]:
        """清理边缘案例缓存"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            
            # 清理边缘案例（保留未解决的）
            old_cases = len(self.edge_cases)
            self.edge_cases = [
                case for case in self.edge_cases
                if (case.resolution_status == "unresolved" or 
                    datetime.fromisoformat(case.timestamp.replace("Z", "+00:00")) > cutoff_time)
            ]
            cleared_cases = old_cases - len(self.edge_cases)
            
            # 清理恢复结果
            old_results = len(self.recovery_results)
            self.recovery_results = [
                result for result in self.recovery_results
                if datetime.fromisoformat(result.timestamp.replace("Z", "+00:00")) > cutoff_time
            ]
            cleared_results = old_results - len(self.recovery_results)
            
            # 清理过时的模式
            old_patterns = len(self.edge_patterns)
            patterns_to_remove = []
            for pattern_id, pattern in self.edge_patterns.items():
                if datetime.fromisoformat(pattern.last_updated.replace("Z", "+00:00")) < cutoff_time:
                    patterns_to_remove.append(pattern_id)
            
            for pattern_id in patterns_to_remove:
                del self.edge_patterns[pattern_id]
            
            cleared_patterns = old_patterns - len(self.edge_patterns)
            
            return {
                "cleared_cases": cleared_cases,
                "cleared_results": cleared_results,
                "cleared_patterns": cleared_patterns
            }
            
        except Exception as e:
            logger.error(f"清理边缘案例缓存失败: {e}")
            return {
                "cleared_cases": 0,
                "cleared_results": 0,
                "cleared_patterns": 0
            }
    
    async def shutdown(self) -> None:
        """关闭边缘处理器"""
        try:
            # 停止所有活跃的恢复
            for recovery_id, recovery_result in self.active_recoveries.items():
                recovery_result.outcome_description = "恢复因系统关闭而中断"
                recovery_result.side_effects.append("recovery_interrupted")
            
            # 发布最终统计信息
            if self.info_pool:
                await self.info_pool.publish(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "edge_handler",
                        "update_type": "edge_handler_shutdown",
                        "final_stats": self.get_edge_stats(),
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.LOW
                )
            
            logger.info("边缘处理器已关闭")
            
        except Exception as e:
            logger.error(f"关闭边缘处理器失败: {e}")