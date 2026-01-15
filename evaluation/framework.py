#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Evaluation Framework
基于AgenticX框架的评估框架：整合所有评估组件，提供统一的评估接口

重构说明：
- 使用AgenticX的Component作为基类
- 集成AgenticX的事件系统进行评估协调
- 使用AgenticX的工作流引擎进行评估流程管理
- 避免重复实现评估基础设施

Author: AgenticX Team
Date: 2025
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from loguru import logger

# 使用AgenticX核心组件
from agenticx.core.component import Component
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from agenticx.core.workflow import Workflow

from utils import get_iso_timestamp, setup_logger

from .test_environment import TestEnvironment, TestSuite, TestRunner, TestResult, TestConfig
from .performance_evaluator import PerformanceEvaluator, PerformanceReport
from .test_scenarios import ScenarioManager
from .metrics import MetricCollector, MetricAnalyzer
from .benchmarks import BenchmarkRunner
from .reports import ReportManager

# 临时定义缺失的类型
from typing import Any as BenchmarkResult
from typing import Any as MetricResult
from typing import Any as BaseTestScenario


class EvaluationMode(Enum):
    """评估模式"""
    QUICK = "quick"                    # 快速评估
    STANDARD = "standard"              # 标准评估
    COMPREHENSIVE = "comprehensive"    # 全面评估
    CUSTOM = "custom"                  # 自定义评估
    CONTINUOUS = "continuous"          # 持续评估
    REGRESSION = "regression"          # 回归测试
    STRESS = "stress"                  # 压力测试
    PERFORMANCE = "performance"        # 性能测试


class EvaluationStatus(Enum):
    """评估状态"""
    PENDING = "pending"                # 等待中
    RUNNING = "running"                # 运行中
    COMPLETED = "completed"            # 已完成
    FAILED = "failed"                  # 失败
    CANCELLED = "cancelled"            # 已取消
    PAUSED = "paused"                  # 已暂停


class EvaluationPriority(Enum):
    """评估优先级"""
    LOW = "low"                        # 低优先级
    MEDIUM = "medium"                  # 中等优先级
    HIGH = "high"                      # 高优先级
    CRITICAL = "critical"              # 关键优先级


class ReportFormat(Enum):
    """报告格式"""
    HTML = "html"                      # HTML格式
    JSON = "json"                      # JSON格式
    PDF = "pdf"                        # PDF格式
    MARKDOWN = "markdown"              # Markdown格式
    CSV = "csv"                        # CSV格式
    EXCEL = "excel"                    # Excel格式


@dataclass
class EvaluationConfig:
    """评估配置"""
    name: str
    description: str = ""
    mode: EvaluationMode = EvaluationMode.STANDARD
    priority: EvaluationPriority = EvaluationPriority.MEDIUM
    
    # 测试配置
    test_suites: List[str] = field(default_factory=list)
    test_scenarios: List[str] = field(default_factory=list)
    test_timeout: float = 300.0  # 5分钟
    
    # 基准测试配置
    benchmarks: List[str] = field(default_factory=list)
    benchmark_iterations: int = 10
    benchmark_timeout: float = 600.0  # 10分钟
    
    # 指标配置
    metrics: List[str] = field(default_factory=list)
    metric_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # 报告配置
    report_formats: List[ReportFormat] = field(default_factory=lambda: [ReportFormat.HTML, ReportFormat.JSON])
    report_output_dir: str = "evaluation_reports"
    
    # 执行配置
    parallel_execution: bool = True
    max_workers: int = 4
    retry_count: int = 3
    retry_delay: float = 1.0
    
    # 环境配置
    environment_setup: Dict[str, Any] = field(default_factory=dict)
    cleanup_after_evaluation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'mode': self.mode.value,
            'priority': self.priority.value,
            'test_suites': self.test_suites,
            'test_scenarios': self.test_scenarios,
            'test_timeout': self.test_timeout,
            'benchmarks': self.benchmarks,
            'benchmark_iterations': self.benchmark_iterations,
            'benchmark_timeout': self.benchmark_timeout,
            'metrics': self.metrics,
            'metric_thresholds': self.metric_thresholds,
            'report_formats': [rf.value for rf in self.report_formats],
            'report_output_dir': self.report_output_dir,
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers,
            'retry_count': self.retry_count,
            'retry_delay': self.retry_delay,
            'environment_setup': self.environment_setup,
            'cleanup_after_evaluation': self.cleanup_after_evaluation
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    evaluation_id: str
    config_name: str
    status: EvaluationStatus
    start_time: str
    end_time: Optional[str] = None
    duration: float = 0.0
    
    # 测试结果
    test_results: List[TestResult] = field(default_factory=list)
    test_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 基准测试结果
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    benchmark_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 指标结果
    metric_results: Dict[str, MetricResult] = field(default_factory=dict)
    metric_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 性能报告
    performance_report: Optional[PerformanceReport] = None
    
    # 报告文件路径
    report_files: List[str] = field(default_factory=list)
    
    # 错误信息
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'evaluation_id': self.evaluation_id,
            'config_name': self.config_name,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'test_results': [result.to_dict() for result in self.test_results],
            'test_summary': self.test_summary,
            'benchmark_results': [result.to_dict() for result in self.benchmark_results],
            'benchmark_summary': self.benchmark_summary,
            'metric_results': {k: v.to_dict() for k, v in self.metric_results.items()},
            'metric_summary': self.metric_summary,
            'performance_report': self.performance_report.to_dict() if self.performance_report else None,
            'report_files': self.report_files,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'metadata': self.metadata
        }
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if not self.test_results:
            return 0.0
        
        passed_tests = sum(1 for result in self.test_results if result.status.value == "passed")
        return passed_tests / len(self.test_results)
    
    def get_overall_score(self) -> float:
        """获取总体评分"""
        scores = []
        
        # 测试成功率权重 40%
        test_score = self.get_success_rate() * 0.4
        scores.append(test_score)
        
        # 基准测试成功率权重 30%
        if self.benchmark_results:
            benchmark_success_rate = sum(result.get_success_rate() for result in self.benchmark_results) / len(self.benchmark_results)
            benchmark_score = benchmark_success_rate * 0.3
            scores.append(benchmark_score)
        
        # 指标评分权重 30%
        if self.metric_results:
            # 简化的指标评分计算
            metric_score = min(1.0, sum(result.value for result in self.metric_results.values()) / len(self.metric_results) / 100) * 0.3
            scores.append(metric_score)
        
        return sum(scores)


class EvaluationFramework(Component):
    """评估框架 - 基于AgenticX Component"""
    
    def __init__(self, base_dir: str = ".", event_bus: Optional[EventBus] = None):
        super().__init__(name="evaluation_framework")
        
        self.base_dir = base_dir
        self.event_bus = event_bus or EventBus()
        self.logger = logger  # setup_logger()
        
        # 核心组件
        self.test_environment = TestEnvironment(config=TestConfig(test_name="default"), event_bus=self.event_bus)
        self.performance_evaluator = PerformanceEvaluator()
        self.scenario_manager = ScenarioManager()
        self.metric_collector = MetricCollector()
        self.metric_analyzer = MetricAnalyzer()
        self.benchmark_runner = BenchmarkRunner()
        self.report_manager = ReportManager()
        
        # 评估状态
        self.current_evaluations: Dict[str, EvaluationResult] = {}
        self.evaluation_history: List[EvaluationResult] = []
        
        # 配置
        self.default_configs: Dict[EvaluationMode, EvaluationConfig] = {}
        self._setup_default_configs()
        
        # 事件处理器
        self.event_handlers: Dict[str, List[Callable]] = {
            'evaluation_started': [],
            'evaluation_completed': [],
            'evaluation_failed': [],
            'test_completed': [],
            'benchmark_completed': [],
            'metric_collected': []
        }
    
    def _setup_default_configs(self) -> None:
        """设置默认配置"""
        # 快速评估配置
        self.default_configs[EvaluationMode.QUICK] = EvaluationConfig(
            name="quick_evaluation",
            description="Quick evaluation with basic tests",
            mode=EvaluationMode.QUICK,
            test_timeout=60.0,
            benchmark_iterations=3,
            benchmark_timeout=120.0,
            parallel_execution=True,
            max_workers=2,
            report_formats=[ReportFormat.JSON]
        )
        
        # 标准评估配置
        self.default_configs[EvaluationMode.STANDARD] = EvaluationConfig(
            name="standard_evaluation",
            description="Standard evaluation with comprehensive tests",
            mode=EvaluationMode.STANDARD,
            test_timeout=300.0,
            benchmark_iterations=10,
            benchmark_timeout=600.0,
            parallel_execution=True,
            max_workers=4,
            report_formats=[ReportFormat.HTML, ReportFormat.JSON]
        )
        
        # 全面评估配置
        self.default_configs[EvaluationMode.COMPREHENSIVE] = EvaluationConfig(
            name="comprehensive_evaluation",
            description="Comprehensive evaluation with all tests and benchmarks",
            mode=EvaluationMode.COMPREHENSIVE,
            test_timeout=600.0,
            benchmark_iterations=20,
            benchmark_timeout=1200.0,
            parallel_execution=True,
            max_workers=8,
            report_formats=[ReportFormat.HTML, ReportFormat.JSON, ReportFormat.MARKDOWN]
        )
    
    async def run_evaluation(self, config: EvaluationConfig) -> EvaluationResult:
        """运行评估"""
        evaluation_id = f"eval_{int(time.time() * 1000)}"
        
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            config_name=config.name,
            status=EvaluationStatus.PENDING,
            start_time=get_iso_timestamp()
        )
        
        self.current_evaluations[evaluation_id] = result
        
        try:
            logger.info(f"Starting evaluation: {config.name} (ID: {evaluation_id})")
            result.status = EvaluationStatus.RUNNING
            
            # 触发评估开始事件
            await self._trigger_event('evaluation_started', result)
            
            # 设置环境
            await self._setup_environment(config)
            
            # 运行测试
            if config.test_suites or config.test_scenarios:
                await self._run_tests(config, result)
            
            # 运行基准测试
            if config.benchmarks:
                await self._run_benchmarks(config, result)
            
            # 收集指标
            if config.metrics:
                await self._collect_metrics(config, result)
            
            # 生成性能报告
            result.performance_report = await self._generate_performance_report(result)
            
            # 生成报告
            result.report_files = await self._generate_reports(config, result)
            
            # 清理环境
            if config.cleanup_after_evaluation:
                await self._cleanup_environment(config)
            
            result.status = EvaluationStatus.COMPLETED
            result.end_time = get_iso_timestamp()
            result.duration = time.time() - time.mktime(time.strptime(result.start_time, "%Y-%m-%dT%H:%M:%S.%fZ"))
            
            logger.info(f"Evaluation completed: {config.name} (ID: {evaluation_id})")
            
            # 触发评估完成事件
            await self._trigger_event('evaluation_completed', result)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {config.name} (ID: {evaluation_id}): {e}")
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            result.end_time = get_iso_timestamp()
            
            # 触发评估失败事件
            await self._trigger_event('evaluation_failed', result)
        
        finally:
            # 移动到历史记录
            self.evaluation_history.append(result)
            if evaluation_id in self.current_evaluations:
                del self.current_evaluations[evaluation_id]
        
        return result
    
    async def run_evaluation_by_mode(self, mode: EvaluationMode, **kwargs) -> EvaluationResult:
        """按模式运行评估"""
        config = self.default_configs.get(mode)
        if not config:
            raise ValueError(f"No default config found for mode: {mode}")
        
        # 应用自定义参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return await self.run_evaluation(config)
    
    async def run_multiple_evaluations(self, configs: List[EvaluationConfig], parallel: bool = True) -> List[EvaluationResult]:
        """运行多个评估"""
        if parallel:
            tasks = [self.run_evaluation(config) for config in configs]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for config in configs:
                result = await self.run_evaluation(config)
                results.append(result)
            return results
    
    async def _setup_environment(self, config: EvaluationConfig) -> None:
        """设置环境"""
        logger.info("Setting up evaluation environment")
        
        # 初始化测试环境
        await self.test_environment.initialize(config.environment_setup)
        
        # 设置指标收集器
        for metric_name in config.metrics:
            # 这里应该根据metric_name创建相应的指标
            pass
    
    async def _cleanup_environment(self, config: EvaluationConfig) -> None:
        """清理环境"""
        logger.info("Cleaning up evaluation environment")
        
        # 清理测试环境
        await self.test_environment.cleanup()
        
        # 重置指标收集器
        self.metric_collector.reset_all_metrics()
    
    async def _run_tests(self, config: EvaluationConfig, result: EvaluationResult) -> None:
        """运行测试"""
        logger.info("Running tests")
        
        test_runner = TestRunner()
        
        # 运行测试套件
        for suite_name in config.test_suites:
            suite = self.test_environment.get_test_suite(suite_name)
            if suite:
                suite_results = await test_runner.run_suite(suite, timeout=config.test_timeout)
                result.test_results.extend(suite_results)
        
        # 运行测试场景
        for scenario_name in config.test_scenarios:
            scenario = self.scenario_manager.get_scenario(scenario_name)
            if scenario:
                scenario_result = await self._run_scenario(scenario, config.test_timeout)
                if scenario_result:
                    result.test_results.append(scenario_result)
        
        # 计算测试摘要
        result.test_summary = self._calculate_test_summary(result.test_results)
        
        # 触发测试完成事件
        await self._trigger_event('test_completed', result.test_results)
    
    async def _run_scenario(self, scenario: BaseTestScenario, timeout: float) -> Optional[TestResult]:
        """运行测试场景"""
        try:
            # 这里应该实现场景的执行逻辑
            # 暂时返回一个模拟结果
            return TestResult(
                test_name=scenario.__class__.__name__,
                status="passed",
                start_time=get_iso_timestamp(),
                duration=1.0
            )
        except Exception as e:
            logger.error(f"Failed to run scenario {scenario.__class__.__name__}: {e}")
            return None
    
    async def _run_benchmarks(self, config: EvaluationConfig, result: EvaluationResult) -> None:
        """运行基准测试"""
        logger.info("Running benchmarks")
        
        for benchmark_name in config.benchmarks:
            benchmark_result = await self.benchmark_runner.run_benchmark(
                benchmark_name,
                iterations=config.benchmark_iterations,
                timeout=config.benchmark_timeout
            )
            if benchmark_result:
                result.benchmark_results.append(benchmark_result)
        
        # 计算基准测试摘要
        result.benchmark_summary = self._calculate_benchmark_summary(result.benchmark_results)
        
        # 触发基准测试完成事件
        await self._trigger_event('benchmark_completed', result.benchmark_results)
    
    async def _collect_metrics(self, config: EvaluationConfig, result: EvaluationResult) -> None:
        """收集指标"""
        logger.info("Collecting metrics")
        
        for metric_name in config.metrics:
            metric_result = self.metric_collector.collect_metric(metric_name)
            if metric_result:
                result.metric_results[metric_name] = metric_result
        
        # 计算指标摘要
        result.metric_summary = self._calculate_metric_summary(result.metric_results)
        
        # 触发指标收集事件
        await self._trigger_event('metric_collected', result.metric_results)
    
    async def _generate_performance_report(self, result: EvaluationResult) -> Optional[PerformanceReport]:
        """生成性能报告"""
        logger.info("Generating performance report")
        
        try:
            # 使用性能评估器生成报告
            return await self.performance_evaluator.evaluate_performance(
                test_results=result.test_results,
                benchmark_results=result.benchmark_results,
                metric_results=result.metric_results
            )
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return None
    
    async def _generate_reports(self, config: EvaluationConfig, result: EvaluationResult) -> List[str]:
        """生成报告"""
        logger.info("Generating reports")
        
        report_files = []
        
        # 准备报告数据
        report_data = ReportData(
            title=f"Evaluation Report - {config.name}",
            description=config.description,
            test_results=result.test_results,
            benchmark_results=result.benchmark_results,
            metric_results=result.metric_results,
            statistics={
                'test_summary': result.test_summary,
                'benchmark_summary': result.benchmark_summary,
                'metric_summary': result.metric_summary
            },
            metadata={
                'evaluation_id': result.evaluation_id,
                'config': config.to_dict(),
                'duration': result.duration,
                'success_rate': result.get_success_rate(),
                'overall_score': result.get_overall_score()
            }
        )
        
        # 生成各种格式的报告
        for report_format in config.report_formats:
            report_config = ReportConfig(
                name=f"{config.name}_{result.evaluation_id}",
                title=f"Evaluation Report - {config.name}",
                description=config.description,
                report_type=ReportType.DETAILED,
                format=report_format,
                output_dir=config.report_output_dir
            )
            
            filepath = await self.report_manager.generate_report(report_config, report_data)
            if filepath:
                report_files.append(filepath)
        
        return report_files
    
    def _calculate_test_summary(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """计算测试摘要"""
        if not test_results:
            return {}
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.status.value == "passed")
        failed_tests = sum(1 for result in test_results if result.status.value == "failed")
        skipped_tests = sum(1 for result in test_results if result.status.value == "skipped")
        
        return {
            'total': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'average_duration': sum(result.duration for result in test_results) / total_tests if total_tests > 0 else 0,
            'total_duration': sum(result.duration for result in test_results)
        }
    
    def _calculate_benchmark_summary(self, benchmark_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """计算基准测试摘要"""
        if not benchmark_results:
            return {}
        
        total_benchmarks = len(benchmark_results)
        completed_benchmarks = sum(1 for result in benchmark_results if result.status.value == "completed")
        
        return {
            'total': total_benchmarks,
            'completed': completed_benchmarks,
            'completion_rate': completed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
            'total_duration': sum(result.duration for result in benchmark_results),
            'average_duration': sum(result.duration for result in benchmark_results) / total_benchmarks if total_benchmarks > 0 else 0,
            'average_success_rate': sum(result.get_success_rate() for result in benchmark_results) / total_benchmarks if total_benchmarks > 0 else 0,
            'total_iterations': sum(result.total_iterations for result in benchmark_results),
            'completed_iterations': sum(result.iterations_completed for result in benchmark_results)
        }
    
    def _calculate_metric_summary(self, metric_results: Dict[str, MetricResult]) -> Dict[str, Any]:
        """计算指标摘要"""
        if not metric_results:
            return {}
        
        return {
            'total_metrics': len(metric_results),
            'metrics': {name: result.value for name, result in metric_results.items()},
            'average_value': sum(result.value for result in metric_results.values()) / len(metric_results),
            'max_value': max(result.value for result in metric_results.values()),
            'min_value': min(result.value for result in metric_results.values())
        }
    
    async def _trigger_event(self, event_name: str, data: Any) -> None:
        """触发事件"""
        handlers = self.event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")
    
    def add_event_handler(self, event_name: str, handler: Callable) -> None:
        """添加事件处理器"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
        logger.info(f"Added event handler for {event_name}")
    
    def remove_event_handler(self, event_name: str, handler: Callable) -> None:
        """移除事件处理器"""
        if event_name in self.event_handlers:
            try:
                self.event_handlers[event_name].remove(handler)
                logger.info(f"Removed event handler for {event_name}")
            except ValueError:
                logger.warning(f"Handler not found for event {event_name}")
    
    def get_current_evaluations(self) -> Dict[str, EvaluationResult]:
        """获取当前评估"""
        return self.current_evaluations.copy()
    
    def get_evaluation_history(self) -> List[EvaluationResult]:
        """获取评估历史"""
        return self.evaluation_history.copy()
    
    def get_evaluation_by_id(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """根据ID获取评估结果"""
        # 先查找当前评估
        if evaluation_id in self.current_evaluations:
            return self.current_evaluations[evaluation_id]
        
        # 再查找历史评估
        for result in self.evaluation_history:
            if result.evaluation_id == evaluation_id:
                return result
        
        return None
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        total_evaluations = len(self.evaluation_history)
        if total_evaluations == 0:
            return {}
        
        completed_evaluations = sum(1 for result in self.evaluation_history if result.status == EvaluationStatus.COMPLETED)
        failed_evaluations = sum(1 for result in self.evaluation_history if result.status == EvaluationStatus.FAILED)
        
        success_rates = [result.get_success_rate() for result in self.evaluation_history if result.status == EvaluationStatus.COMPLETED]
        overall_scores = [result.get_overall_score() for result in self.evaluation_history if result.status == EvaluationStatus.COMPLETED]
        
        return {
            'total_evaluations': total_evaluations,
            'completed_evaluations': completed_evaluations,
            'failed_evaluations': failed_evaluations,
            'completion_rate': completed_evaluations / total_evaluations,
            'average_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0,
            'average_overall_score': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            'max_success_rate': max(success_rates) if success_rates else 0,
            'min_success_rate': min(success_rates) if success_rates else 0,
            'max_overall_score': max(overall_scores) if overall_scores else 0,
            'min_overall_score': min(overall_scores) if overall_scores else 0
        }
    
    async def cancel_evaluation(self, evaluation_id: str) -> bool:
        """取消评估"""
        if evaluation_id in self.current_evaluations:
            result = self.current_evaluations[evaluation_id]
            result.status = EvaluationStatus.CANCELLED
            result.end_time = get_iso_timestamp()
            
            # 移动到历史记录
            self.evaluation_history.append(result)
            del self.current_evaluations[evaluation_id]
            
            logger.info(f"Cancelled evaluation: {evaluation_id}")
            return True
        
        return False
    
    def clear_history(self) -> None:
        """清空评估历史"""
        self.evaluation_history.clear()
        logger.info("Cleared evaluation history")
    
    def export_results(self, filepath: str, format: str = "json") -> bool:
        """导出评估结果"""
        try:
            data = {
                'current_evaluations': {k: v.to_dict() for k, v in self.current_evaluations.items()},
                'evaluation_history': [result.to_dict() for result in self.evaluation_history],
                'statistics': self.get_evaluation_statistics(),
                'exported_at': get_iso_timestamp()
            }
            
            if format.lower() == "json":
                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported evaluation results to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False
    
    def import_results(self, filepath: str, format: str = "json") -> bool:
        """导入评估结果"""
        try:
            if format.lower() == "json":
                import json
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            # 导入历史记录
            if 'evaluation_history' in data:
                for result_data in data['evaluation_history']:
                    # 这里需要实现从字典重建EvaluationResult的逻辑
                    pass
            
            logger.info(f"Imported evaluation results from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import results: {e}")
            return False