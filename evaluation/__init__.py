#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Evaluation Module
基于AgenticX框架的测试环境与评估系统：提供完整的测试环境和性能评估功能

注意：此模块已重构以使用AgenticX框架的核心组件，避免重复实现。
主要功能：
- 智能体性能评估
- GUI操作测试
- 基准测试和回归测试
- 评估报告生成

Author: AgenticX Team
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "AgenticX Team"
__description__ = "AgenticX-GUIAgent测试环境与评估系统 - 基于AgenticX框架重构"

# 核心评估组件
from .test_environment import (
    TestEnvironment,
    TestCase,
    TestSuite,
    TestResult,
    TestRunner,
    TestConfig
)

from .performance_evaluator import (
    PerformanceEvaluator,
    PerformanceReport,
    MetricValue,
    BaseMetric,
    AccuracyMetric,
    ResponseTimeMetric,
    ThroughputMetric,
    ReliabilityMetric,
    EfficiencyMetric,
    RobustnessMetric
)

from .framework import (
    EvaluationFramework,
    EvaluationResult,
    EvaluationConfig,
    EvaluationMode,
    EvaluationStatus,
    ReportFormat
)

# 测试数据和场景
from .test_scenarios import (
    BaseTestScenario,
    ScenarioFactory,
    ScenarioManager,
    LoginScenario,
    SearchScenario,
    PurchaseScenario,
    ErrorHandlingScenario,
    PerformanceTestScenario
)

# 评估指标和分析
from .metrics import (
    AccuracyMetric,
    ResponseTimeMetric,
    ThroughputMetric,
    ErrorRateMetric,
    MemoryUsageMetric,
    CPUUsageMetric,
    SuccessRateMetric,
    AvailabilityMetric,
    MetricCollector,
    MetricAnalyzer,
    BaseMetric,
    MetricResult
)

# 报告生成
# 报告生成模块暂时跳过，避免导入错误
# from .reports import (...)

# 基准测试
from .benchmarks import (
    BaseBenchmark,
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkSuite,
    PerformanceBenchmark,
    LoadBenchmark
)

# 导出的主要类和函数
__all__ = [
    # 核心组件
    "TestEnvironment",
    "TestCase",
    "TestSuite",
    "TestResult",
    "TestRunner",
    "TestConfig",
    
    # 性能评估
    "PerformanceEvaluator",
    "PerformanceReport",
    "MetricValue",
    "BaseMetric",
    "AccuracyMetric",
    "ResponseTimeMetric",
    "ThroughputMetric",
    "ReliabilityMetric",
    "EfficiencyMetric",
    "RobustnessMetric",
    
    # 评估框架
    "EvaluationFramework",
    "EvaluationResult",
    "EvaluationConfig",
    "EvaluationMode",
    "EvaluationStatus",
    "ReportFormat",
    
    # 测试场景
    "BaseTestScenario",
    "ScenarioFactory",
    "ScenarioManager",
    "LoginScenario",
    "SearchScenario",
    "PurchaseScenario",
    "ErrorHandlingScenario",
    "PerformanceTestScenario",
    
    # 评估指标
    "AccuracyMetric",
    "ResponseTimeMetric",
    "ThroughputMetric",
    "ErrorRateMetric",
    "MemoryUsageMetric",
    "CPUUsageMetric",
    "SuccessRateMetric",
    "AvailabilityMetric",
    "MetricCollector",
    "MetricAnalyzer",
    "BaseMetric",
    "MetricResult",
    
    # 报告生成
    # 报告生成类暂时跳过
    
    # 基准测试
    "BaseBenchmark",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkSuite",
    "PerformanceBenchmark",
    "LoadBenchmark",
]

# 便捷函数
def create_test_environment(config: dict) -> TestEnvironment:
    """创建测试环境"""
    return TestEnvironment(TestConfig(**config))

def create_performance_evaluator(metrics: list = None) -> PerformanceEvaluator:
    """创建性能评估器"""
    return PerformanceEvaluator(metrics or [])

def create_evaluation_framework(tasks: list = None) -> EvaluationFramework:
    """创建评估框架"""
    return EvaluationFramework(tasks or [])

def run_benchmark(benchmark_name: str, config: dict = None) -> BenchmarkResult:
    """运行基准测试"""
    benchmark = StandardBenchmark.get_benchmark(benchmark_name)
    runner = BenchmarkRunner(config or {})
    return runner.run(benchmark)

def generate_report(results: list, format: str = "html") -> str:
    """生成评估报告"""
    if format.lower() == "html":
        generator = HTMLReportGenerator()
    elif format.lower() == "json":
        generator = JSONReportGenerator()
    elif format.lower() == "pdf":
        generator = PDFReportGenerator()
    else:
        raise ValueError(f"Unsupported report format: {format}")
    
    return generator.generate(results)

# 版本信息
def get_version() -> str:
    """获取版本信息"""
    return __version__

def get_supported_platforms() -> list:
    """获取支持的平台"""
    return ["android", "ios", "web", "desktop"]

def get_available_metrics() -> list:
    """获取可用的评估指标"""
    return [
        "accuracy",
        "efficiency", 
        "robustness",
        "usability",
        "performance",
        "reliability",
        "scalability"
    ]

def get_benchmark_suites() -> list:
    """获取可用的基准测试套件"""
    return [
        "mobile_gui_automation",
        "web_interaction",
        "game_automation",
        "productivity_tasks",
        "stress_testing",
        "compatibility_testing"
    ]