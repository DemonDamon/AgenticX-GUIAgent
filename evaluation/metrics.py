#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Evaluation Metrics
评估指标：定义各种评估指标和分析方法

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import math
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from loguru import logger

from utils import get_iso_timestamp
from utils import setup_logger


class MetricCategory(Enum):
    """指标类别"""
    ACCURACY = "accuracy"              # 准确性
    PERFORMANCE = "performance"        # 性能
    RELIABILITY = "reliability"        # 可靠性
    EFFICIENCY = "efficiency"          # 效率
    USABILITY = "usability"            # 可用性
    ROBUSTNESS = "robustness"          # 鲁棒性
    SCALABILITY = "scalability"        # 可扩展性
    SECURITY = "security"              # 安全性
    COMPATIBILITY = "compatibility"    # 兼容性
    MAINTAINABILITY = "maintainability" # 可维护性


class MetricUnit(Enum):
    """指标单位"""
    PERCENTAGE = "percentage"          # 百分比
    SECONDS = "seconds"                # 秒
    MILLISECONDS = "milliseconds"      # 毫秒
    COUNT = "count"                    # 计数
    RATE = "rate"                      # 比率
    SCORE = "score"                    # 分数
    BYTES = "bytes"                    # 字节
    OPERATIONS_PER_SECOND = "ops"      # 每秒操作数
    REQUESTS_PER_SECOND = "rps"        # 每秒请求数
    DIMENSIONLESS = "dimensionless"    # 无量纲


class AggregationType(Enum):
    """聚合类型"""
    MEAN = "mean"                      # 平均值
    MEDIAN = "median"                  # 中位数
    MIN = "min"                        # 最小值
    MAX = "max"                        # 最大值
    SUM = "sum"                        # 总和
    COUNT = "count"                    # 计数
    PERCENTILE_95 = "p95"              # 95百分位
    PERCENTILE_99 = "p99"              # 99百分位
    STANDARD_DEVIATION = "std"         # 标准差
    VARIANCE = "variance"              # 方差


@dataclass
class MetricThreshold:
    """指标阈值"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    def check_threshold(self, value: float) -> Tuple[bool, str]:
        """检查阈值"""
        if self.critical_threshold is not None:
            if value >= self.critical_threshold:
                return False, "critical"
        
        if self.warning_threshold is not None:
            if value >= self.warning_threshold:
                return False, "warning"
        
        if self.min_value is not None and value < self.min_value:
            return False, "below_minimum"
        
        if self.max_value is not None and value > self.max_value:
            return False, "above_maximum"
        
        return True, "normal"


@dataclass
class MetricValue:
    """指标值"""
    value: Union[float, int, str, bool]
    timestamp: str = field(default_factory=get_iso_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class MetricResult:
    """指标结果"""
    metric_name: str
    category: MetricCategory
    unit: MetricUnit
    value: MetricValue
    aggregated_values: Dict[AggregationType, float] = field(default_factory=dict)
    threshold_status: Tuple[bool, str] = field(default=(True, "normal"))
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'category': self.category.value,
            'unit': self.unit.value,
            'value': self.value.to_dict(),
            'aggregated_values': {k.value: v for k, v in self.aggregated_values.items()},
            'threshold_status': {
                'is_within_threshold': self.threshold_status[0],
                'status': self.threshold_status[1]
            },
            'trend': self.trend,
            'confidence': self.confidence
        }


class BaseMetric(ABC):
    """基础指标类"""
    
    def __init__(self, 
                 name: str,
                 category: MetricCategory,
                 unit: MetricUnit,
                 description: str = "",
                 threshold: Optional[MetricThreshold] = None):
        self.name = name
        self.category = category
        self.unit = unit
        self.description = description
        self.threshold = threshold or MetricThreshold()
        self.logger = logger
        
        # 历史数据
        self.history: List[MetricValue] = []
        self.max_history_size = 1000
        
        # 配置
        self.enabled = True
        self.collection_interval = 1.0  # 秒
        self.last_collection_time = 0.0
    
    @abstractmethod
    async def calculate(self, data: Any) -> MetricValue:
        """计算指标值"""
        pass
    
    def add_value(self, value: MetricValue) -> None:
        """添加指标值"""
        self.history.append(value)
        
        # 限制历史记录大小
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
    
    def get_aggregated_values(self, 
                            aggregation_types: List[AggregationType] = None) -> Dict[AggregationType, float]:
        """获取聚合值"""
        if not self.history:
            return {}
        
        if aggregation_types is None:
            aggregation_types = [AggregationType.MEAN, AggregationType.MIN, AggregationType.MAX]
        
        numeric_values = []
        for metric_value in self.history:
            if isinstance(metric_value.value, (int, float)):
                numeric_values.append(float(metric_value.value))
        
        if not numeric_values:
            return {}
        
        aggregated = {}
        
        for agg_type in aggregation_types:
            try:
                if agg_type == AggregationType.MEAN:
                    aggregated[agg_type] = statistics.mean(numeric_values)
                elif agg_type == AggregationType.MEDIAN:
                    aggregated[agg_type] = statistics.median(numeric_values)
                elif agg_type == AggregationType.MIN:
                    aggregated[agg_type] = min(numeric_values)
                elif agg_type == AggregationType.MAX:
                    aggregated[agg_type] = max(numeric_values)
                elif agg_type == AggregationType.SUM:
                    aggregated[agg_type] = sum(numeric_values)
                elif agg_type == AggregationType.COUNT:
                    aggregated[agg_type] = len(numeric_values)
                elif agg_type == AggregationType.PERCENTILE_95:
                    aggregated[agg_type] = self._percentile(numeric_values, 95)
                elif agg_type == AggregationType.PERCENTILE_99:
                    aggregated[agg_type] = self._percentile(numeric_values, 99)
                elif agg_type == AggregationType.STANDARD_DEVIATION:
                    aggregated[agg_type] = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
                elif agg_type == AggregationType.VARIANCE:
                    aggregated[agg_type] = statistics.variance(numeric_values) if len(numeric_values) > 1 else 0
            except Exception as e:
                logger.warning(f"Failed to calculate {agg_type.value}: {e}")
        
        return aggregated
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index >= len(sorted_values):
                return sorted_values[lower_index]
            
            weight = index - lower_index
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def get_trend(self, window_size: int = 10) -> Optional[str]:
        """获取趋势"""
        if len(self.history) < window_size:
            return None
        
        recent_values = []
        for metric_value in self.history[-window_size:]:
            if isinstance(metric_value.value, (int, float)):
                recent_values.append(float(metric_value.value))
        
        if len(recent_values) < 2:
            return None
        
        # 简单线性趋势分析
        x = list(range(len(recent_values)))
        y = recent_values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # 计算斜率
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if abs(slope) < 0.01:  # 阈值可调
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    async def collect_and_calculate(self, data: Any) -> Optional[MetricResult]:
        """收集并计算指标"""
        if not self.enabled:
            return None
        
        current_time = time.time()
        if current_time - self.last_collection_time < self.collection_interval:
            return None
        
        try:
            metric_value = await self.calculate(data)
            self.add_value(metric_value)
            
            # 检查阈值
            threshold_status = (True, "normal")
            if isinstance(metric_value.value, (int, float)):
                threshold_status = self.threshold.check_threshold(float(metric_value.value))
            
            # 获取聚合值
            aggregated_values = self.get_aggregated_values()
            
            # 获取趋势
            trend = self.get_trend()
            
            result = MetricResult(
                metric_name=self.name,
                category=self.category,
                unit=self.unit,
                value=metric_value,
                aggregated_values=aggregated_values,
                threshold_status=threshold_status,
                trend=trend
            )
            
            self.last_collection_time = current_time
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate metric {self.name}: {e}")
            return None
    
    def reset(self) -> None:
        """重置指标"""
        self.history.clear()
        self.last_collection_time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """获取指标信息"""
        return {
            'name': self.name,
            'category': self.category.value,
            'unit': self.unit.value,
            'description': self.description,
            'enabled': self.enabled,
            'collection_interval': self.collection_interval,
            'history_size': len(self.history),
            'max_history_size': self.max_history_size
        }


class AccuracyMetric(BaseMetric):
    """准确性指标"""
    
    def __init__(self, name: str = "accuracy"):
        threshold = MetricThreshold(
            min_value=0.0,
            max_value=1.0,
            target_value=0.95,
            warning_threshold=0.8,
            critical_threshold=0.7
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.ACCURACY,
            unit=MetricUnit.PERCENTAGE,
            description="Accuracy of task execution",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算准确性"""
        if isinstance(data, dict):
            correct = data.get('correct', 0)
            total = data.get('total', 1)
            
            if total == 0:
                accuracy = 0.0
            else:
                accuracy = correct / total
            
            return MetricValue(
                value=accuracy,
                metadata={
                    'correct': correct,
                    'total': total,
                    'incorrect': total - correct
                }
            )
        
        # 默认返回1.0（完全准确）
        return MetricValue(value=1.0)


class ResponseTimeMetric(BaseMetric):
    """响应时间指标"""
    
    def __init__(self, name: str = "response_time"):
        threshold = MetricThreshold(
            min_value=0.0,
            target_value=1.0,
            warning_threshold=3.0,
            critical_threshold=5.0
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.PERFORMANCE,
            unit=MetricUnit.SECONDS,
            description="Response time of operations",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算响应时间"""
        if isinstance(data, dict):
            start_time = data.get('start_time', 0)
            end_time = data.get('end_time', 0)
            
            if start_time and end_time:
                response_time = end_time - start_time
            else:
                response_time = data.get('response_time', 0)
            
            return MetricValue(
                value=response_time,
                metadata={
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        
        elif isinstance(data, (int, float)):
            return MetricValue(value=float(data))
        
        return MetricValue(value=0.0)


class ThroughputMetric(BaseMetric):
    """吞吐量指标"""
    
    def __init__(self, name: str = "throughput"):
        threshold = MetricThreshold(
            min_value=0.0,
            target_value=100.0,
            warning_threshold=50.0,
            critical_threshold=10.0
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.PERFORMANCE,
            unit=MetricUnit.OPERATIONS_PER_SECOND,
            description="Throughput of operations per second",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算吞吐量"""
        if isinstance(data, dict):
            operations = data.get('operations', 0)
            duration = data.get('duration', 1)
            
            if duration == 0:
                throughput = 0.0
            else:
                throughput = operations / duration
            
            return MetricValue(
                value=throughput,
                metadata={
                    'operations': operations,
                    'duration': duration
                }
            )
        
        return MetricValue(value=0.0)


class ErrorRateMetric(BaseMetric):
    """错误率指标"""
    
    def __init__(self, name: str = "error_rate"):
        threshold = MetricThreshold(
            min_value=0.0,
            max_value=1.0,
            target_value=0.01,
            warning_threshold=0.05,
            critical_threshold=0.1
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.RELIABILITY,
            unit=MetricUnit.PERCENTAGE,
            description="Error rate of operations",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算错误率"""
        if isinstance(data, dict):
            errors = data.get('errors', 0)
            total = data.get('total', 1)
            
            if total == 0:
                error_rate = 0.0
            else:
                error_rate = errors / total
            
            return MetricValue(
                value=error_rate,
                metadata={
                    'errors': errors,
                    'total': total,
                    'success': total - errors
                }
            )
        
        return MetricValue(value=0.0)


class MemoryUsageMetric(BaseMetric):
    """内存使用指标"""
    
    def __init__(self, name: str = "memory_usage"):
        threshold = MetricThreshold(
            min_value=0.0,
            warning_threshold=1024 * 1024 * 512,  # 512MB
            critical_threshold=1024 * 1024 * 1024  # 1GB
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.EFFICIENCY,
            unit=MetricUnit.BYTES,
            description="Memory usage in bytes",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算内存使用"""
        if isinstance(data, dict):
            memory_usage = data.get('memory_usage', 0)
            
            return MetricValue(
                value=memory_usage,
                metadata={
                    'memory_mb': memory_usage / (1024 * 1024),
                    'memory_gb': memory_usage / (1024 * 1024 * 1024)
                }
            )
        
        elif isinstance(data, (int, float)):
            return MetricValue(value=float(data))
        
        return MetricValue(value=0.0)


class CPUUsageMetric(BaseMetric):
    """CPU使用指标"""
    
    def __init__(self, name: str = "cpu_usage"):
        threshold = MetricThreshold(
            min_value=0.0,
            max_value=1.0,
            target_value=0.5,
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.EFFICIENCY,
            unit=MetricUnit.PERCENTAGE,
            description="CPU usage percentage",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算CPU使用率"""
        if isinstance(data, dict):
            cpu_usage = data.get('cpu_usage', 0.0)
            
            return MetricValue(
                value=cpu_usage,
                metadata={
                    'cpu_percent': cpu_usage * 100
                }
            )
        
        elif isinstance(data, (int, float)):
            return MetricValue(value=float(data))
        
        return MetricValue(value=0.0)


class SuccessRateMetric(BaseMetric):
    """成功率指标"""
    
    def __init__(self, name: str = "success_rate"):
        threshold = MetricThreshold(
            min_value=0.0,
            max_value=1.0,
            target_value=0.99,
            warning_threshold=0.95,
            critical_threshold=0.9
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.RELIABILITY,
            unit=MetricUnit.PERCENTAGE,
            description="Success rate of operations",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算成功率"""
        if isinstance(data, dict):
            success = data.get('success', 0)
            total = data.get('total', 1)
            
            if total == 0:
                success_rate = 0.0
            else:
                success_rate = success / total
            
            return MetricValue(
                value=success_rate,
                metadata={
                    'success': success,
                    'total': total,
                    'failed': total - success
                }
            )
        
        return MetricValue(value=1.0)


class AvailabilityMetric(BaseMetric):
    """可用性指标"""
    
    def __init__(self, name: str = "availability"):
        threshold = MetricThreshold(
            min_value=0.0,
            max_value=1.0,
            target_value=0.999,
            warning_threshold=0.99,
            critical_threshold=0.95
        )
        
        super().__init__(
            name=name,
            category=MetricCategory.RELIABILITY,
            unit=MetricUnit.PERCENTAGE,
            description="System availability percentage",
            threshold=threshold
        )
    
    async def calculate(self, data: Any) -> MetricValue:
        """计算可用性"""
        if isinstance(data, dict):
            uptime = data.get('uptime', 0)
            total_time = data.get('total_time', 1)
            
            if total_time == 0:
                availability = 0.0
            else:
                availability = uptime / total_time
            
            return MetricValue(
                value=availability,
                metadata={
                    'uptime': uptime,
                    'total_time': total_time,
                    'downtime': total_time - uptime
                }
            )
        
        return MetricValue(value=1.0)


class MetricCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, BaseMetric] = {}
        self.logger = logger
        
        # 收集配置
        self.collection_enabled = True
        self.collection_interval = 1.0
        self.auto_collection = False
        
        # 收集任务
        self._collection_task: Optional[asyncio.Task] = None
        self._stop_collection = False
    
    def register_metric(self, metric: BaseMetric) -> None:
        """注册指标"""
        self.metrics[metric.name] = metric
        logger.info(f"Registered metric: {metric.name}")
    
    def unregister_metric(self, metric_name: str) -> None:
        """注销指标"""
        if metric_name in self.metrics:
            del self.metrics[metric_name]
            logger.info(f"Unregistered metric: {metric_name}")
    
    def get_metric(self, metric_name: str) -> Optional[BaseMetric]:
        """获取指标"""
        return self.metrics.get(metric_name)
    
    def list_metrics(self) -> List[str]:
        """列出所有指标"""
        return list(self.metrics.keys())
    
    async def collect_metric(self, metric_name: str, data: Any) -> Optional[MetricResult]:
        """收集单个指标"""
        if not self.collection_enabled:
            return None
        
        metric = self.get_metric(metric_name)
        if not metric:
            logger.warning(f"Metric not found: {metric_name}")
            return None
        
        return await metric.collect_and_calculate(data)
    
    async def collect_all_metrics(self, data: Dict[str, Any]) -> Dict[str, MetricResult]:
        """收集所有指标"""
        results = {}
        
        for metric_name, metric in self.metrics.items():
            metric_data = data.get(metric_name, data)
            result = await metric.collect_and_calculate(metric_data)
            
            if result:
                results[metric_name] = result
        
        return results
    
    def start_auto_collection(self, data_source: Callable[[], Dict[str, Any]]) -> None:
        """启动自动收集"""
        if self._collection_task and not self._collection_task.done():
            logger.warning("Auto collection already running")
            return
        
        self.auto_collection = True
        self._stop_collection = False
        self._collection_task = asyncio.create_task(
            self._auto_collection_loop(data_source)
        )
        
        logger.info("Started auto collection")
    
    def stop_auto_collection(self) -> None:
        """停止自动收集"""
        self._stop_collection = True
        self.auto_collection = False
        
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
        
        logger.info("Stopped auto collection")
    
    async def _auto_collection_loop(self, data_source: Callable[[], Dict[str, Any]]) -> None:
        """自动收集循环"""
        while not self._stop_collection:
            try:
                data = data_source()
                await self.collect_all_metrics(data)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto collection: {e}")
                await asyncio.sleep(1.0)
    
    def reset_all_metrics(self) -> None:
        """重置所有指标"""
        for metric in self.metrics.values():
            metric.reset()
        
        logger.info("Reset all metrics")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {
            'total_metrics': len(self.metrics),
            'enabled_metrics': sum(1 for m in self.metrics.values() if m.enabled),
            'categories': {},
            'units': {},
            'metrics': {}
        }
        
        for metric_name, metric in self.metrics.items():
            # 统计类别
            category = metric.category.value
            summary['categories'][category] = summary['categories'].get(category, 0) + 1
            
            # 统计单位
            unit = metric.unit.value
            summary['units'][unit] = summary['units'].get(unit, 0) + 1
            
            # 指标信息
            summary['metrics'][metric_name] = metric.get_info()
        
        return summary


class MetricAnalyzer:
    """指标分析器"""
    
    def __init__(self):
        self.logger = logger
    
    def analyze_metric_results(self, results: Dict[str, MetricResult]) -> Dict[str, Any]:
        """分析指标结果"""
        analysis = {
            'total_metrics': len(results),
            'healthy_metrics': 0,
            'warning_metrics': 0,
            'critical_metrics': 0,
            'categories': {},
            'trends': {},
            'recommendations': []
        }
        
        for metric_name, result in results.items():
            # 健康状态统计
            is_healthy, status = result.threshold_status
            if is_healthy:
                analysis['healthy_metrics'] += 1
            elif status == 'warning':
                analysis['warning_metrics'] += 1
            else:
                analysis['critical_metrics'] += 1
            
            # 类别统计
            category = result.category.value
            if category not in analysis['categories']:
                analysis['categories'][category] = {
                    'total': 0,
                    'healthy': 0,
                    'warning': 0,
                    'critical': 0
                }
            
            analysis['categories'][category]['total'] += 1
            if is_healthy:
                analysis['categories'][category]['healthy'] += 1
            elif status == 'warning':
                analysis['categories'][category]['warning'] += 1
            else:
                analysis['categories'][category]['critical'] += 1
            
            # 趋势统计
            if result.trend:
                analysis['trends'][result.trend] = analysis['trends'].get(result.trend, 0) + 1
        
        # 生成建议
        analysis['recommendations'] = self._generate_recommendations(results)
        
        return analysis
    
    def _generate_recommendations(self, results: Dict[str, MetricResult]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        for metric_name, result in results.items():
            is_healthy, status = result.threshold_status
            
            if not is_healthy:
                if result.category == MetricCategory.PERFORMANCE:
                    if 'response_time' in metric_name.lower():
                        recommendations.append(f"Consider optimizing {metric_name} - current response time is {status}")
                    elif 'throughput' in metric_name.lower():
                        recommendations.append(f"Consider scaling resources to improve {metric_name}")
                
                elif result.category == MetricCategory.RELIABILITY:
                    recommendations.append(f"Investigate reliability issues with {metric_name}")
                
                elif result.category == MetricCategory.EFFICIENCY:
                    recommendations.append(f"Optimize resource usage for {metric_name}")
            
            # 趋势建议
            if result.trend == 'increasing' and result.category in [MetricCategory.PERFORMANCE, MetricCategory.EFFICIENCY]:
                if 'error' in metric_name.lower() or 'usage' in metric_name.lower():
                    recommendations.append(f"Monitor increasing trend in {metric_name}")
        
        return recommendations
    
    def compare_metric_results(self, 
                             current_results: Dict[str, MetricResult],
                             baseline_results: Dict[str, MetricResult]) -> Dict[str, Any]:
        """比较指标结果"""
        comparison = {
            'improved_metrics': [],
            'degraded_metrics': [],
            'stable_metrics': [],
            'new_metrics': [],
            'removed_metrics': []
        }
        
        current_names = set(current_results.keys())
        baseline_names = set(baseline_results.keys())
        
        # 新增和移除的指标
        comparison['new_metrics'] = list(current_names - baseline_names)
        comparison['removed_metrics'] = list(baseline_names - current_names)
        
        # 比较共同指标
        common_metrics = current_names & baseline_names
        
        for metric_name in common_metrics:
            current = current_results[metric_name]
            baseline = baseline_results[metric_name]
            
            if isinstance(current.value.value, (int, float)) and isinstance(baseline.value.value, (int, float)):
                current_val = float(current.value.value)
                baseline_val = float(baseline.value.value)
                
                # 计算变化百分比
                if baseline_val != 0:
                    change_percent = (current_val - baseline_val) / baseline_val * 100
                else:
                    change_percent = 0
                
                # 判断改进或退化（根据指标类型）
                is_improvement = self._is_improvement(current.category, current_val, baseline_val)
                
                if abs(change_percent) < 5:  # 5%以内认为稳定
                    comparison['stable_metrics'].append({
                        'name': metric_name,
                        'change_percent': change_percent
                    })
                elif is_improvement:
                    comparison['improved_metrics'].append({
                        'name': metric_name,
                        'change_percent': change_percent
                    })
                else:
                    comparison['degraded_metrics'].append({
                        'name': metric_name,
                        'change_percent': change_percent
                    })
        
        return comparison
    
    def _is_improvement(self, category: MetricCategory, current_val: float, baseline_val: float) -> bool:
        """判断是否为改进"""
        # 对于这些类别，值越大越好
        positive_categories = [
            MetricCategory.ACCURACY,
            MetricCategory.RELIABILITY,
            MetricCategory.USABILITY,
            MetricCategory.SCALABILITY
        ]
        
        # 对于这些类别，值越小越好
        negative_categories = [
            MetricCategory.PERFORMANCE,  # 响应时间等
            MetricCategory.EFFICIENCY    # 资源使用等
        ]
        
        if category in positive_categories:
            return current_val > baseline_val
        elif category in negative_categories:
            return current_val < baseline_val
        else:
            # 默认认为值越大越好
            return current_val > baseline_val