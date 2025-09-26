#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Tool Monitor
工具监控器：负责工具执行的实时监控、性能分析和异常检测

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Deque

from .gui_tools import (
    GUITool, ToolParameters, ToolResult, ToolError,
    ToolType, ToolStatus
)
from utils import get_iso_timestamp, setup_logger


class MonitorLevel(Enum):
    """监控级别"""
    BASIC = "basic"          # 基础监控
    DETAILED = "detailed"    # 详细监控
    COMPREHENSIVE = "comprehensive"  # 全面监控
    DEBUG = "debug"          # 调试监控


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 计数器
    GAUGE = "gauge"          # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    TIMER = "timer"          # 计时器


@dataclass
class MonitorEvent:
    """监控事件"""
    event_id: str
    event_type: str  # execution_start, execution_end, error, warning, metric_update
    tool_id: str
    tool_name: str
    timestamp: str = field(default_factory=get_iso_timestamp)
    data: Optional[Dict[str, Any]] = None
    severity: AlertSeverity = AlertSeverity.INFO
    message: Optional[str] = None


@dataclass
class PerformanceMetric:
    """性能指标"""
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: str = field(default_factory=get_iso_timestamp)
    tags: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class ExecutionTrace:
    """执行跟踪"""
    trace_id: str
    tool_id: str
    tool_name: str
    start_time: str
    end_time: Optional[str] = None
    status: ToolStatus = ToolStatus.IDLE
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: List[PerformanceMetric] = field(default_factory=list)
    events: List[MonitorEvent] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """执行持续时间（秒）"""
        if self.start_time and self.end_time:
            try:
                # 简化的时间计算，实际应该使用proper datetime parsing
                return float(self.end_time.split('T')[1].split(':')[2].split('Z')[0]) - \
                       float(self.start_time.split('T')[1].split(':')[2].split('Z')[0])
            except Exception:
                return None
        return None


@dataclass
class Alert:
    """告警"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    tool_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    timestamp: str = field(default_factory=get_iso_timestamp)
    acknowledged: bool = False
    resolved: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """指标收集器"""
    
    def __init__(self, max_history_size: int = 10000):
        self.metrics: Dict[str, Deque[PerformanceMetric]] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.lock = threading.Lock()
        self.logger = logger
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> None:
        """记录指标"""
        metric = PerformanceMetric(
            metric_name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            tags=tags or {},
            description=description
        )
        
        with self.lock:
            self.metrics[name].append(metric)
            self._update_aggregated_metrics(name, value, metric_type)
    
    def get_metric_history(
        self,
        name: str,
        limit: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """获取指标历史"""
        with self.lock:
            history = list(self.metrics.get(name, []))
            if limit:
                return history[-limit:]
            return history
    
    def get_aggregated_metrics(self, name: str) -> Dict[str, float]:
        """获取聚合指标"""
        with self.lock:
            return self.aggregated_metrics.get(name, {}).copy()
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取所有聚合指标"""
        with self.lock:
            return {name: metrics.copy() for name, metrics in self.aggregated_metrics.items()}
    
    def _update_aggregated_metrics(self, name: str, value: float, metric_type: MetricType) -> None:
        """更新聚合指标"""
        if name not in self.aggregated_metrics:
            self.aggregated_metrics[name] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0.0,
                'latest': 0.0
            }
        
        agg = self.aggregated_metrics[name]
        
        if metric_type == MetricType.COUNTER:
            agg['sum'] += value
            agg['latest'] = agg['sum']
        else:
            agg['count'] += 1
            agg['sum'] += value
            agg['min'] = min(agg['min'], value)
            agg['max'] = max(agg['max'], value)
            agg['avg'] = agg['sum'] / agg['count']
            agg['latest'] = value


class AlertManager:
    """告警管理器"""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts: Deque[Alert] = deque(maxlen=max_alerts)
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.Lock()
        self.logger = logger
    
    def add_alert_rule(
        self,
        metric_name: str,
        condition: str,  # ">", "<", ">=", "<=", "==", "!="
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        message_template: str = "Metric {metric_name} {condition} {threshold}"
    ) -> None:
        """添加告警规则"""
        rule = {
            'metric_name': metric_name,
            'condition': condition,
            'threshold': threshold,
            'severity': severity,
            'message_template': message_template
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {metric_name} {condition} {threshold}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def check_metrics(self, metrics: Dict[str, Dict[str, float]]) -> List[Alert]:
        """检查指标并生成告警"""
        alerts = []
        
        for rule in self.alert_rules:
            metric_name = rule['metric_name']
            condition = rule['condition']
            threshold = rule['threshold']
            
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                current_value = metric_data.get('latest', 0)
                
                if self._evaluate_condition(current_value, condition, threshold):
                    alert = Alert(
                        alert_id=f"alert_{int(time.time() * 1000)}",
                        severity=rule['severity'],
                        title=f"Metric Alert: {metric_name}",
                        message=rule['message_template'].format(
                            metric_name=metric_name,
                            condition=condition,
                            threshold=threshold,
                            current_value=current_value
                        ),
                        metric_name=metric_name,
                        threshold_value=threshold,
                        actual_value=current_value
                    )
                    alerts.append(alert)
        
        # 记录和处理告警
        for alert in alerts:
            self._record_alert(alert)
            self._handle_alert(alert)
        
        return alerts
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """获取告警列表"""
        with self.lock:
            alerts = list(self.alerts)
        
        # 过滤条件
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        # 按时间倒序排列
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            return alerts[:limit]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        return False
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """评估条件"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 1e-9
        elif condition == "!=":
            return abs(value - threshold) >= 1e-9
        return False
    
    def _record_alert(self, alert: Alert) -> None:
        """记录告警"""
        with self.lock:
            self.alerts.append(alert)
        logger.warning(f"Alert generated: {alert.title} - {alert.message}")
    
    def _handle_alert(self, alert: Alert) -> None:
        """处理告警"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")


class ExecutionTracker:
    """执行跟踪器"""
    
    def __init__(self, max_traces: int = 1000):
        self.traces: Dict[str, ExecutionTrace] = {}
        self.completed_traces: Deque[ExecutionTrace] = deque(maxlen=max_traces)
        self.lock = threading.Lock()
        self.logger = logger
    
    def start_trace(self, tool: GUITool, parameters: Optional[ToolParameters] = None) -> str:
        """开始跟踪"""
        trace_id = f"trace_{tool.tool_id}_{int(time.time() * 1000)}"
        
        trace = ExecutionTrace(
            trace_id=trace_id,
            tool_id=tool.tool_id,
            tool_name=tool.name,
            start_time=get_iso_timestamp(),
            status=ToolStatus.RUNNING,
            parameters=parameters.data if parameters else None
        )
        
        with self.lock:
            self.traces[trace_id] = trace
        
        logger.debug(f"Started trace: {trace_id} for tool {tool.name}")
        return trace_id
    
    def end_trace(
        self,
        trace_id: str,
        result: Optional[ToolResult] = None,
        error: Optional[str] = None
    ) -> None:
        """结束跟踪"""
        with self.lock:
            if trace_id in self.traces:
                trace = self.traces[trace_id]
                trace.end_time = get_iso_timestamp()
                
                if result:
                    trace.status = result.status
                    trace.result = result.data
                    if result.error_message:
                        trace.error = result.error_message
                elif error:
                    trace.status = ToolStatus.FAILED
                    trace.error = error
                else:
                    trace.status = ToolStatus.COMPLETED
                
                # 移动到已完成跟踪
                self.completed_traces.append(trace)
                del self.traces[trace_id]
                
                logger.debug(f"Ended trace: {trace_id}, status: {trace.status.value}")
    
    def add_trace_event(self, trace_id: str, event: MonitorEvent) -> None:
        """添加跟踪事件"""
        with self.lock:
            if trace_id in self.traces:
                self.traces[trace_id].events.append(event)
    
    def add_trace_metric(self, trace_id: str, metric: PerformanceMetric) -> None:
        """添加跟踪指标"""
        with self.lock:
            if trace_id in self.traces:
                self.traces[trace_id].metrics.append(metric)
    
    def get_active_traces(self) -> List[ExecutionTrace]:
        """获取活跃跟踪"""
        with self.lock:
            return list(self.traces.values())
    
    def get_completed_traces(
        self,
        tool_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExecutionTrace]:
        """获取已完成跟踪"""
        with self.lock:
            traces = list(self.completed_traces)
        
        if tool_id:
            traces = [t for t in traces if t.tool_id == tool_id]
        
        # 按开始时间倒序排列
        traces.sort(key=lambda x: x.start_time, reverse=True)
        
        if limit:
            return traces[:limit]
        
        return traces
    
    def get_trace_by_id(self, trace_id: str) -> Optional[ExecutionTrace]:
        """根据ID获取跟踪"""
        with self.lock:
            # 检查活跃跟踪
            if trace_id in self.traces:
                return self.traces[trace_id]
            
            # 检查已完成跟踪
            for trace in self.completed_traces:
                if trace.trace_id == trace_id:
                    return trace
        
        return None


class ToolMonitor:
    """工具监控器"""
    
    def __init__(self, level: MonitorLevel = MonitorLevel.DETAILED):
        self.level = level
        self.logger = logger
        
        # 组件
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.execution_tracker = ExecutionTracker()
        
        # 事件处理
        self.event_handlers: List[Callable[[MonitorEvent], None]] = []
        self.events: Deque[MonitorEvent] = deque(maxlen=10000)
        
        # 监控状态
        self.monitoring_enabled = True
        self.monitoring_start_time = get_iso_timestamp()
        
        # 统计信息
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # 设置默认告警规则
        self._setup_default_alert_rules()
        
        # 启动监控循环
        self._start_monitoring_loop()
    
    def start_tool_execution(self, tool: GUITool, parameters: Optional[ToolParameters] = None) -> str:
        """开始工具执行监控"""
        if not self.monitoring_enabled:
            return ""
        
        # 开始跟踪
        trace_id = self.execution_tracker.start_trace(tool, parameters)
        
        # 记录开始事件
        event = MonitorEvent(
            event_id=f"event_{int(time.time() * 1000)}",
            event_type="execution_start",
            tool_id=tool.tool_id,
            tool_name=tool.name,
            data={
                'trace_id': trace_id,
                'parameters': parameters.data if parameters else None
            },
            message=f"Started execution of tool {tool.name}"
        )
        
        self._record_event(event)
        
        # 记录指标
        self.metric_collector.record_metric(
            f"tool.{tool.name}.executions",
            1,
            MetricType.COUNTER,
            "count",
            {'tool_id': tool.tool_id}
        )
        
        return trace_id
    
    def end_tool_execution(
        self,
        trace_id: str,
        tool: GUITool,
        result: Optional[ToolResult] = None,
        error: Optional[str] = None
    ) -> None:
        """结束工具执行监控"""
        if not self.monitoring_enabled or not trace_id:
            return
        
        # 结束跟踪
        self.execution_tracker.end_trace(trace_id, result, error)
        
        # 记录结束事件
        event_type = "execution_end"
        severity = AlertSeverity.INFO
        message = f"Completed execution of tool {tool.name}"
        
        if error:
            event_type = "execution_error"
            severity = AlertSeverity.ERROR
            message = f"Failed execution of tool {tool.name}: {error}"
        elif result and not result.success:
            event_type = "execution_failure"
            severity = AlertSeverity.WARNING
            message = f"Unsuccessful execution of tool {tool.name}"
        
        event = MonitorEvent(
            event_id=f"event_{int(time.time() * 1000)}",
            event_type=event_type,
            tool_id=tool.tool_id,
            tool_name=tool.name,
            severity=severity,
            data={
                'trace_id': trace_id,
                'success': result.success if result else False,
                'error': error or (result.error_message if result else None)
            },
            message=message
        )
        
        self._record_event(event)
        
        # 记录性能指标
        if result and hasattr(result, 'execution_time'):
            execution_time = getattr(result, 'execution_time', 0)
            
            self.metric_collector.record_metric(
                f"tool.{tool.name}.execution_time",
                execution_time,
                MetricType.TIMER,
                "seconds",
                {'tool_id': tool.tool_id}
            )
            
            # 更新统计信息
            self._update_stats(result.success, execution_time)
        
        # 记录成功/失败指标
        if result:
            metric_name = f"tool.{tool.name}.success" if result.success else f"tool.{tool.name}.failure"
            self.metric_collector.record_metric(
                metric_name,
                1,
                MetricType.COUNTER,
                "count",
                {'tool_id': tool.tool_id}
            )
    
    def record_custom_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """记录自定义指标"""
        if self.monitoring_enabled:
            self.metric_collector.record_metric(name, value, metric_type, unit, tags)
    
    def add_event_handler(self, handler: Callable[[MonitorEvent], None]) -> None:
        """添加事件处理器"""
        self.event_handlers.append(handler)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """添加告警处理器"""
        self.alert_manager.add_alert_handler(handler)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        # 获取基础统计
        stats = self.stats.copy()
        
        # 获取活跃跟踪
        active_traces = self.execution_tracker.get_active_traces()
        
        # 获取最近的告警
        recent_alerts = self.alert_manager.get_alerts(limit=10)
        
        # 获取关键指标
        key_metrics = {}
        all_metrics = self.metric_collector.get_all_metrics()
        
        for metric_name, metric_data in all_metrics.items():
            if any(keyword in metric_name for keyword in ['execution_time', 'success', 'failure', 'executions']):
                key_metrics[metric_name] = metric_data
        
        # 计算健康状态
        health_status = self._calculate_health_status()
        
        return {
            'monitoring_status': {
                'enabled': self.monitoring_enabled,
                'start_time': self.monitoring_start_time,
                'level': self.level.value,
                'health': health_status
            },
            'statistics': stats,
            'active_executions': len(active_traces),
            'recent_alerts': [{
                'id': alert.alert_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'timestamp': alert.timestamp,
                'resolved': alert.resolved
            } for alert in recent_alerts],
            'key_metrics': key_metrics,
            'active_traces': [{
                'trace_id': trace.trace_id,
                'tool_name': trace.tool_name,
                'start_time': trace.start_time,
                'status': trace.status.value
            } for trace in active_traces]
        }
    
    def get_tool_performance_report(self, tool_id: str) -> Dict[str, Any]:
        """获取工具性能报告"""
        # 获取工具的执行跟踪
        traces = self.execution_tracker.get_completed_traces(tool_id=tool_id, limit=100)
        
        if not traces:
            return {'error': 'No execution data found for tool'}
        
        # 计算性能指标
        total_executions = len(traces)
        successful_executions = sum(1 for t in traces if t.status == ToolStatus.COMPLETED)
        failed_executions = total_executions - successful_executions
        
        # 执行时间分析
        execution_times = [t.duration for t in traces if t.duration is not None]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        
        # 错误分析
        error_types = {}
        for trace in traces:
            if trace.error:
                error_type = trace.error.split(':')[0] if ':' in trace.error else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # 获取相关指标
        tool_name = traces[0].tool_name if traces else 'Unknown'
        tool_metrics = {}
        all_metrics = self.metric_collector.get_all_metrics()
        
        for metric_name, metric_data in all_metrics.items():
            if tool_name in metric_name:
                tool_metrics[metric_name] = metric_data
        
        return {
            'tool_id': tool_id,
            'tool_name': tool_name,
            'performance_summary': {
                'total_executions': total_executions,
                'success_rate': successful_executions / total_executions,
                'failure_rate': failed_executions / total_executions,
                'average_execution_time': avg_execution_time,
                'min_execution_time': min_execution_time,
                'max_execution_time': max_execution_time
            },
            'error_analysis': error_types,
            'metrics': tool_metrics,
            'recent_traces': traces[:10]  # 最近10次执行
        }
    
    def enable_monitoring(self) -> None:
        """启用监控"""
        self.monitoring_enabled = True
        logger.info("Tool monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """禁用监控"""
        self.monitoring_enabled = False
        logger.info("Tool monitoring disabled")
    
    def _setup_default_alert_rules(self) -> None:
        """设置默认告警规则"""
        # 执行时间告警
        self.alert_manager.add_alert_rule(
            "execution_time",
            ">",
            30.0,  # 30秒
            AlertSeverity.WARNING,
            "Tool execution time exceeded 30 seconds: {current_value}s"
        )
        
        self.alert_manager.add_alert_rule(
            "execution_time",
            ">",
            60.0,  # 1分钟
            AlertSeverity.ERROR,
            "Tool execution time exceeded 1 minute: {current_value}s"
        )
        
        # 失败率告警
        self.alert_manager.add_alert_rule(
            "failure_rate",
            ">",
            0.1,  # 10%
            AlertSeverity.WARNING,
            "Tool failure rate exceeded 10%: {current_value}"
        )
        
        self.alert_manager.add_alert_rule(
            "failure_rate",
            ">",
            0.2,  # 20%
            AlertSeverity.ERROR,
            "Tool failure rate exceeded 20%: {current_value}"
        )
    
    def _start_monitoring_loop(self) -> None:
        """启动监控循环"""
        def monitoring_loop():
            while True:
                try:
                    if self.monitoring_enabled:
                        # 检查指标并生成告警
                        metrics = self.metric_collector.get_all_metrics()
                        self.alert_manager.check_metrics(metrics)
                        
                        # 清理过期的跟踪
                        self._cleanup_expired_traces()
                    
                    time.sleep(10)  # 每10秒检查一次
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(30)  # 出错时等待30秒
        
        # 在后台线程中运行监控循环
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def _record_event(self, event: MonitorEvent) -> None:
        """记录事件"""
        self.events.append(event)
        
        # 调用事件处理器
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
        
        # 记录到日志
        if event.severity == AlertSeverity.ERROR:
            logger.error(f"Monitor event: {event.message}")
        elif event.severity == AlertSeverity.WARNING:
            logger.warning(f"Monitor event: {event.message}")
        else:
            logger.debug(f"Monitor event: {event.message}")
    
    def _update_stats(self, success: bool, execution_time: float) -> None:
        """更新统计信息"""
        self.stats['total_executions'] += 1
        
        if success:
            self.stats['successful_executions'] += 1
        else:
            self.stats['failed_executions'] += 1
        
        self.stats['total_execution_time'] += execution_time
        self.stats['average_execution_time'] = (
            self.stats['total_execution_time'] / self.stats['total_executions']
        )
    
    def _calculate_health_status(self) -> str:
        """计算健康状态"""
        if not self.monitoring_enabled:
            return "disabled"
        
        # 检查最近的告警
        recent_alerts = self.alert_manager.get_alerts(limit=10)
        critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved]
        error_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.ERROR and not a.resolved]
        
        if critical_alerts:
            return "critical"
        elif error_alerts:
            return "unhealthy"
        elif len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING and not a.resolved]) > 5:
            return "warning"
        else:
            return "healthy"
    
    def _cleanup_expired_traces(self) -> None:
        """清理过期的跟踪"""
        current_time = time.time()
        expired_trace_ids = []
        
        for trace_id, trace in self.execution_tracker.traces.items():
            # 如果跟踪超过1小时仍未完成，认为是过期的
            try:
                start_timestamp = time.mktime(time.strptime(trace.start_time, "%Y-%m-%dT%H:%M:%SZ"))
                if current_time - start_timestamp > 3600:  # 1小时
                    expired_trace_ids.append(trace_id)
            except Exception:
                # 如果时间解析失败，也认为是过期的
                expired_trace_ids.append(trace_id)
        
        # 清理过期跟踪
        for trace_id in expired_trace_ids:
            self.execution_tracker.end_trace(trace_id, error="Trace expired")
            logger.warning(f"Cleaned up expired trace: {trace_id}")