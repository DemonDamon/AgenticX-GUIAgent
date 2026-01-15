#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Performance Evaluator
基于AgenticX框架的性能评估器：提供全面的性能评估和分析功能

重构说明：
- 使用AgenticX的Component作为基类
- 集成AgenticX的事件系统进行性能监控
- 使用AgenticX的工具框架进行指标计算
- 避免重复实现性能监控基础设施

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from loguru import logger

# 使用AgenticX核心组件
from agenticx.core.component import Component
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from agenticx.core.tool import BaseTool

from utils import get_iso_timestamp, setup_logger
from .test_environment import TestResult, TestStatus


class MetricType(Enum):
    """指标类型"""
    ACCURACY = "accuracy"              # 准确率
    PRECISION = "precision"            # 精确率
    RECALL = "recall"                  # 召回率
    F1_SCORE = "f1_score"              # F1分数
    SUCCESS_RATE = "success_rate"      # 成功率
    ERROR_RATE = "error_rate"          # 错误率
    RESPONSE_TIME = "response_time"    # 响应时间
    THROUGHPUT = "throughput"          # 吞吐量
    LATENCY = "latency"                # 延迟
    MEMORY_USAGE = "memory_usage"      # 内存使用
    CPU_USAGE = "cpu_usage"            # CPU使用
    RELIABILITY = "reliability"        # 可靠性
    AVAILABILITY = "availability"      # 可用性
    SCALABILITY = "scalability"        # 可扩展性
    EFFICIENCY = "efficiency"          # 效率
    ROBUSTNESS = "robustness"          # 鲁棒性


class EvaluationLevel(Enum):
    """评估级别"""
    BASIC = "basic"          # 基础评估
    STANDARD = "standard"    # 标准评估
    COMPREHENSIVE = "comprehensive"  # 全面评估
    EXPERT = "expert"        # 专家评估


class ComparisonType(Enum):
    """比较类型"""
    BASELINE = "baseline"    # 基线比较
    HISTORICAL = "historical"  # 历史比较
    PEER = "peer"            # 同级比较
    TARGET = "target"        # 目标比较


@dataclass
class MetricValue:
    """指标值"""
    metric_type: MetricType
    value: float
    unit: str = ""
    timestamp: str = field(default_factory=get_iso_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None  # 置信度
    
    def __str__(self) -> str:
        unit_str = f" {self.unit}" if self.unit else ""
        confidence_str = f" (confidence: {self.confidence:.2%})" if self.confidence else ""
        return f"{self.metric_type.value}: {self.value:.4f}{unit_str}{confidence_str}"


@dataclass
class PerformanceReport:
    """性能报告"""
    report_id: str
    name: str
    description: str
    evaluation_level: EvaluationLevel
    metrics: List[MetricValue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=get_iso_timestamp)
    duration: Optional[float] = None
    
    def add_metric(self, metric: MetricValue) -> None:
        """添加指标"""
        self.metrics.append(metric)
    
    def get_metric(self, metric_type: MetricType) -> Optional[MetricValue]:
        """获取指标"""
        for metric in self.metrics:
            if metric.metric_type == metric_type:
                return metric
        return None
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricValue]:
        """获取指定类型的所有指标"""
        return [metric for metric in self.metrics if metric.metric_type == metric_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'report_id': self.report_id,
            'name': self.name,
            'description': self.description,
            'evaluation_level': self.evaluation_level.value,
            'metrics': [{
                'type': metric.metric_type.value,
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp,
                'confidence': metric.confidence,
                'metadata': metric.metadata
            } for metric in self.metrics],
            'summary': self.summary,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp,
            'duration': self.duration
        }


class BaseMetric(ABC):
    """基础指标计算器"""
    
    def __init__(self, metric_type: MetricType, name: str, description: str = ""):
        self.metric_type = metric_type
        self.name = name
        self.description = description
        self.logger = logger
    
    @abstractmethod
    async def calculate(self, data: Any, **kwargs) -> MetricValue:
        """计算指标值"""
        pass
    
    def validate_data(self, data: Any) -> bool:
        """验证输入数据"""
        return data is not None


class AccuracyMetric(BaseMetric):
    """准确率指标"""
    
    def __init__(self):
        super().__init__(MetricType.ACCURACY, "Accuracy", "测试准确率")
    
    async def calculate(self, data: List[TestResult], **kwargs) -> MetricValue:
        """计算准确率"""
        if not self.validate_data(data) or not data:
            return MetricValue(self.metric_type, 0.0, "%")
        
        total_tests = len(data)
        passed_tests = sum(1 for result in data if result.status == TestStatus.PASSED)
        
        accuracy = (passed_tests / total_tests) * 100 if total_tests > 0 else 0.0
        
        return MetricValue(
            metric_type=self.metric_type,
            value=accuracy,
            unit="%",
            metadata={
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests
            }
        )


class ResponseTimeMetric(BaseMetric):
    """响应时间指标"""
    
    def __init__(self):
        super().__init__(MetricType.RESPONSE_TIME, "ResponseTime", "平均响应时间")
    
    async def calculate(self, data: List[TestResult], **kwargs) -> MetricValue:
        """计算平均响应时间"""
        if not self.validate_data(data) or not data:
            return MetricValue(self.metric_type, 0.0, "ms")
        
        durations = [result.duration for result in data if result.duration is not None]
        
        if not durations:
            return MetricValue(self.metric_type, 0.0, "ms")
        
        avg_duration = statistics.mean(durations) * 1000  # 转换为毫秒
        median_duration = statistics.median(durations) * 1000
        std_duration = statistics.stdev(durations) * 1000 if len(durations) > 1 else 0
        
        return MetricValue(
            metric_type=self.metric_type,
            value=avg_duration,
            unit="ms",
            metadata={
                'median': median_duration,
                'std_dev': std_duration,
                'min': min(durations) * 1000,
                'max': max(durations) * 1000,
                'count': len(durations)
            }
        )


class ThroughputMetric(BaseMetric):
    """吞吐量指标"""
    
    def __init__(self):
        super().__init__(MetricType.THROUGHPUT, "Throughput", "系统吞吐量")
    
    async def calculate(self, data: List[TestResult], **kwargs) -> MetricValue:
        """计算吞吐量"""
        if not self.validate_data(data) or not data:
            return MetricValue(self.metric_type, 0.0, "ops/s")
        
        total_duration = kwargs.get('total_duration', None)
        if total_duration is None:
            # 计算总持续时间
            durations = [result.duration for result in data if result.duration is not None]
            total_duration = sum(durations) if durations else 1.0
        
        throughput = len(data) / total_duration if total_duration > 0 else 0.0
        
        return MetricValue(
            metric_type=self.metric_type,
            value=throughput,
            unit="ops/s",
            metadata={
                'total_operations': len(data),
                'total_duration': total_duration,
                'avg_operation_time': total_duration / len(data) if data else 0
            }
        )


class ReliabilityMetric(BaseMetric):
    """可靠性指标"""
    
    def __init__(self):
        super().__init__(MetricType.RELIABILITY, "Reliability", "系统可靠性")
    
    async def calculate(self, data: List[TestResult], **kwargs) -> MetricValue:
        """计算可靠性"""
        if not self.validate_data(data) or not data:
            return MetricValue(self.metric_type, 0.0, "%")
        
        total_tests = len(data)
        successful_tests = sum(1 for result in data if result.status == TestStatus.PASSED)
        error_tests = sum(1 for result in data if result.status == TestStatus.ERROR)
        timeout_tests = sum(1 for result in data if result.status == TestStatus.TIMEOUT)
        
        # 可靠性 = (成功测试数 / 总测试数) * 100
        reliability = (successful_tests / total_tests) * 100 if total_tests > 0 else 0.0
        
        return MetricValue(
            metric_type=self.metric_type,
            value=reliability,
            unit="%",
            metadata={
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'error_tests': error_tests,
                'timeout_tests': timeout_tests,
                'error_rate': (error_tests / total_tests) * 100 if total_tests > 0 else 0,
                'timeout_rate': (timeout_tests / total_tests) * 100 if total_tests > 0 else 0
            }
        )


class EfficiencyMetric(BaseMetric):
    """效率指标"""
    
    def __init__(self):
        super().__init__(MetricType.EFFICIENCY, "Efficiency", "系统效率")
    
    async def calculate(self, data: List[TestResult], **kwargs) -> MetricValue:
        """计算效率"""
        if not self.validate_data(data) or not data:
            return MetricValue(self.metric_type, 0.0, "%")
        
        # 效率 = 成功率 / 平均执行时间
        total_tests = len(data)
        successful_tests = sum(1 for result in data if result.status == TestStatus.PASSED)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        durations = [result.duration for result in data if result.duration is not None]
        avg_duration = statistics.mean(durations) if durations else 1.0
        
        # 归一化效率分数 (0-100)
        efficiency = (success_rate / max(avg_duration, 0.001)) * 100
        
        return MetricValue(
            metric_type=self.metric_type,
            value=min(efficiency, 100.0),  # 限制最大值为100
            unit="%",
            metadata={
                'success_rate': success_rate * 100,
                'avg_duration': avg_duration,
                'total_tests': total_tests,
                'successful_tests': successful_tests
            }
        )


class RobustnessMetric(BaseMetric):
    """鲁棒性指标"""
    
    def __init__(self):
        super().__init__(MetricType.ROBUSTNESS, "Robustness", "系统鲁棒性")
    
    async def calculate(self, data: List[TestResult], **kwargs) -> MetricValue:
        """计算鲁棒性"""
        if not self.validate_data(data) or not data:
            return MetricValue(self.metric_type, 0.0, "%")
        
        total_tests = len(data)
        
        # 统计各种状态的测试
        passed_tests = sum(1 for result in data if result.status == TestStatus.PASSED)
        failed_tests = sum(1 for result in data if result.status == TestStatus.FAILED)
        error_tests = sum(1 for result in data if result.status == TestStatus.ERROR)
        timeout_tests = sum(1 for result in data if result.status == TestStatus.TIMEOUT)
        
        # 鲁棒性考虑系统在异常情况下的表现
        # 鲁棒性 = (通过测试 + 优雅失败测试) / 总测试数
        # 这里假设FAILED状态是优雅失败，ERROR和TIMEOUT是非优雅失败
        graceful_handling = passed_tests + failed_tests
        robustness = (graceful_handling / total_tests) * 100 if total_tests > 0 else 0.0
        
        return MetricValue(
            metric_type=self.metric_type,
            value=robustness,
            unit="%",
            metadata={
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'timeout_tests': timeout_tests,
                'graceful_handling_rate': robustness,
                'catastrophic_failure_rate': ((error_tests + timeout_tests) / total_tests) * 100 if total_tests > 0 else 0
            }
        )


class PerformanceEvaluator(Component):
    """性能评估器 - 基于AgenticX Component"""
    
    def __init__(self, output_dir: Optional[str] = None, event_bus: Optional[EventBus] = None):
        super().__init__(name="performance_evaluator")
        
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_bus = event_bus or EventBus()
        
        self.logger = logger
        
        # 注册默认指标
        self.metrics: Dict[MetricType, BaseMetric] = {
            MetricType.ACCURACY: AccuracyMetric(),
            MetricType.RESPONSE_TIME: ResponseTimeMetric(),
            MetricType.THROUGHPUT: ThroughputMetric(),
            MetricType.RELIABILITY: ReliabilityMetric(),
            MetricType.EFFICIENCY: EfficiencyMetric(),
            MetricType.ROBUSTNESS: RobustnessMetric()
        }
        
        # 评估历史
        self.evaluation_history: List[PerformanceReport] = []
        
        # 基线数据
        self.baselines: Dict[str, PerformanceReport] = {}
    
    def register_metric(self, metric: BaseMetric) -> None:
        """注册自定义指标"""
        self.metrics[metric.metric_type] = metric
        logger.info(f"Registered metric: {metric.name}")
    
    def unregister_metric(self, metric_type: MetricType) -> None:
        """注销指标"""
        if metric_type in self.metrics:
            del self.metrics[metric_type]
            logger.info(f"Unregistered metric: {metric_type.value}")
    
    async def evaluate(self, 
                      data: List[TestResult], 
                      name: str,
                      description: str = "",
                      level: EvaluationLevel = EvaluationLevel.STANDARD,
                      selected_metrics: Optional[List[MetricType]] = None,
                      **kwargs) -> PerformanceReport:
        """执行性能评估"""
        start_time = time.time()
        
        logger.info(f"Starting performance evaluation: {name}")
        
        # 确定要计算的指标
        if selected_metrics is None:
            selected_metrics = self._get_metrics_for_level(level)
        
        # 创建报告
        report = PerformanceReport(
            report_id=f"eval_{int(time.time())}",
            name=name,
            description=description,
            evaluation_level=level
        )
        
        # 计算指标
        for metric_type in selected_metrics:
            if metric_type in self.metrics:
                try:
                    metric_value = await self.metrics[metric_type].calculate(data, **kwargs)
                    report.add_metric(metric_value)
                    logger.debug(f"Calculated metric: {metric_value}")
                except Exception as e:
                    logger.error(f"Error calculating metric {metric_type.value}: {e}")
        
        # 生成摘要
        report.summary = self._generate_summary(report)
        
        # 生成建议
        report.recommendations = self._generate_recommendations(report)
        
        # 设置持续时间
        report.duration = time.time() - start_time
        
        # 保存到历史
        self.evaluation_history.append(report)
        
        # 保存报告
        await self._save_report(report)
        
        logger.info(f"Performance evaluation completed: {name}")
        return report
    
    async def compare_with_baseline(self, 
                                   current_report: PerformanceReport,
                                   baseline_name: str) -> Dict[str, Any]:
        """与基线比较"""
        if baseline_name not in self.baselines:
            raise ValueError(f"Baseline not found: {baseline_name}")
        
        baseline_report = self.baselines[baseline_name]
        comparison = {
            'baseline_name': baseline_name,
            'current_report': current_report.name,
            'comparison_type': ComparisonType.BASELINE.value,
            'metrics_comparison': {},
            'overall_improvement': 0.0,
            'recommendations': []
        }
        
        improvements = []
        
        for current_metric in current_report.metrics:
            baseline_metric = baseline_report.get_metric(current_metric.metric_type)
            if baseline_metric:
                improvement = self._calculate_improvement(
                    current_metric, baseline_metric
                )
                comparison['metrics_comparison'][current_metric.metric_type.value] = {
                    'current_value': current_metric.value,
                    'baseline_value': baseline_metric.value,
                    'improvement': improvement,
                    'improvement_percentage': (improvement / baseline_metric.value) * 100 if baseline_metric.value != 0 else 0
                }
                improvements.append(improvement)
        
        # 计算总体改进
        if improvements:
            comparison['overall_improvement'] = statistics.mean(improvements)
        
        # 生成比较建议
        comparison['recommendations'] = self._generate_comparison_recommendations(comparison)
        
        return comparison
    
    def set_baseline(self, report: PerformanceReport, baseline_name: str) -> None:
        """设置基线"""
        self.baselines[baseline_name] = report
        logger.info(f"Set baseline: {baseline_name}")
    
    def get_baseline(self, baseline_name: str) -> Optional[PerformanceReport]:
        """获取基线"""
        return self.baselines.get(baseline_name)
    
    def get_evaluation_history(self, limit: Optional[int] = None) -> List[PerformanceReport]:
        """获取评估历史"""
        if limit:
            return self.evaluation_history[-limit:]
        return self.evaluation_history.copy()
    
    async def generate_trend_analysis(self, 
                                     metric_type: MetricType,
                                     window_size: int = 10) -> Dict[str, Any]:
        """生成趋势分析"""
        if len(self.evaluation_history) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # 获取最近的评估数据
        recent_reports = self.evaluation_history[-window_size:]
        
        values = []
        timestamps = []
        
        for report in recent_reports:
            metric = report.get_metric(metric_type)
            if metric:
                values.append(metric.value)
                timestamps.append(metric.timestamp)
        
        if len(values) < 2:
            return {'error': f'Insufficient data for metric {metric_type.value}'}
        
        # 计算趋势
        trend_analysis = {
            'metric_type': metric_type.value,
            'data_points': len(values),
            'values': values,
            'timestamps': timestamps,
            'trend': self._calculate_trend(values),
            'statistics': {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'latest': values[-1],
                'change_from_first': values[-1] - values[0],
                'change_percentage': ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
            }
        }
        
        return trend_analysis
    
    def _get_metrics_for_level(self, level: EvaluationLevel) -> List[MetricType]:
        """根据评估级别获取指标列表"""
        if level == EvaluationLevel.BASIC:
            return [MetricType.ACCURACY, MetricType.RESPONSE_TIME]
        elif level == EvaluationLevel.STANDARD:
            return [MetricType.ACCURACY, MetricType.RESPONSE_TIME, MetricType.THROUGHPUT, MetricType.RELIABILITY]
        elif level == EvaluationLevel.COMPREHENSIVE:
            return [MetricType.ACCURACY, MetricType.RESPONSE_TIME, MetricType.THROUGHPUT, 
                   MetricType.RELIABILITY, MetricType.EFFICIENCY, MetricType.ROBUSTNESS]
        else:  # EXPERT
            return list(self.metrics.keys())
    
    def _generate_summary(self, report: PerformanceReport) -> Dict[str, Any]:
        """生成报告摘要"""
        summary = {
            'total_metrics': len(report.metrics),
            'evaluation_level': report.evaluation_level.value,
            'key_findings': []
        }
        
        # 分析关键发现
        for metric in report.metrics:
            if metric.metric_type == MetricType.ACCURACY:
                if metric.value >= 95:
                    summary['key_findings'].append(f"Excellent accuracy: {metric.value:.2f}%")
                elif metric.value >= 80:
                    summary['key_findings'].append(f"Good accuracy: {metric.value:.2f}%")
                else:
                    summary['key_findings'].append(f"Low accuracy: {metric.value:.2f}%")
            
            elif metric.metric_type == MetricType.RESPONSE_TIME:
                if metric.value <= 100:  # ms
                    summary['key_findings'].append(f"Fast response time: {metric.value:.2f}ms")
                elif metric.value <= 1000:
                    summary['key_findings'].append(f"Acceptable response time: {metric.value:.2f}ms")
                else:
                    summary['key_findings'].append(f"Slow response time: {metric.value:.2f}ms")
            
            elif metric.metric_type == MetricType.RELIABILITY:
                if metric.value >= 99:
                    summary['key_findings'].append(f"Excellent reliability: {metric.value:.2f}%")
                elif metric.value >= 95:
                    summary['key_findings'].append(f"Good reliability: {metric.value:.2f}%")
                else:
                    summary['key_findings'].append(f"Poor reliability: {metric.value:.2f}%")
        
        return summary
    
    def _generate_recommendations(self, report: PerformanceReport) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        for metric in report.metrics:
            if metric.metric_type == MetricType.ACCURACY and metric.value < 90:
                recommendations.append("Consider improving test case design or system accuracy")
            
            elif metric.metric_type == MetricType.RESPONSE_TIME and metric.value > 1000:
                recommendations.append("Optimize system performance to reduce response time")
            
            elif metric.metric_type == MetricType.RELIABILITY and metric.value < 95:
                recommendations.append("Improve system stability and error handling")
            
            elif metric.metric_type == MetricType.EFFICIENCY and metric.value < 70:
                recommendations.append("Optimize resource utilization and execution efficiency")
            
            elif metric.metric_type == MetricType.ROBUSTNESS and metric.value < 80:
                recommendations.append("Enhance error recovery and fault tolerance mechanisms")
        
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges")
        
        return recommendations
    
    def _calculate_improvement(self, current: MetricValue, baseline: MetricValue) -> float:
        """计算改进程度"""
        # 对于某些指标，值越高越好（如准确率）
        # 对于某些指标，值越低越好（如响应时间）
        
        if current.metric_type in [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
                                  MetricType.F1_SCORE, MetricType.SUCCESS_RATE, MetricType.THROUGHPUT,
                                  MetricType.RELIABILITY, MetricType.AVAILABILITY, MetricType.EFFICIENCY,
                                  MetricType.ROBUSTNESS]:
            # 值越高越好
            return current.value - baseline.value
        else:
            # 值越低越好（如响应时间、错误率等）
            return baseline.value - current.value
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """生成比较建议"""
        recommendations = []
        overall_improvement = comparison['overall_improvement']
        
        if overall_improvement > 0:
            recommendations.append(f"Overall performance improved by {overall_improvement:.2f}")
        elif overall_improvement < 0:
            recommendations.append(f"Overall performance decreased by {abs(overall_improvement):.2f}")
        else:
            recommendations.append("Performance remained stable compared to baseline")
        
        # 分析具体指标的改进
        for metric_name, metric_comparison in comparison['metrics_comparison'].items():
            improvement_pct = metric_comparison['improvement_percentage']
            if abs(improvement_pct) > 10:  # 显著变化
                if improvement_pct > 0:
                    recommendations.append(f"{metric_name} improved significantly by {improvement_pct:.1f}%")
                else:
                    recommendations.append(f"{metric_name} degraded by {abs(improvement_pct):.1f}%")
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "insufficient_data"
        
        # 简单的线性趋势计算
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.05:  # 5%阈值
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    async def _save_report(self, report: PerformanceReport) -> None:
        """保存报告"""
        try:
            report_file = self.output_dir / f"{report.report_id}_performance_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved performance report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取评估器状态"""
        return {
            'registered_metrics': list(self.metrics.keys()),
            'evaluation_history_count': len(self.evaluation_history),
            'baselines_count': len(self.baselines),
            'output_directory': str(self.output_dir)
        }