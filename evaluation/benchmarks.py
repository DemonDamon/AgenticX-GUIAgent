#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Evaluation Benchmarks
基准测试：定义基准测试套件和性能基准

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from loguru import logger

from utils import get_iso_timestamp
from utils import setup_logger
from .metrics import MetricResult, MetricCollector
from .test_environment import TestEnvironment, TestResult, TestStatus


class BenchmarkType(Enum):
    """基准测试类型"""
    PERFORMANCE = "performance"        # 性能基准
    ACCURACY = "accuracy"              # 准确性基准
    RELIABILITY = "reliability"        # 可靠性基准
    SCALABILITY = "scalability"        # 可扩展性基准
    STRESS = "stress"                  # 压力测试
    LOAD = "load"                      # 负载测试
    ENDURANCE = "endurance"            # 耐久性测试
    COMPATIBILITY = "compatibility"    # 兼容性测试
    SECURITY = "security"              # 安全性测试
    USABILITY = "usability"            # 可用性测试


class BenchmarkStatus(Enum):
    """基准测试状态"""
    PENDING = "pending"                # 待执行
    RUNNING = "running"                # 运行中
    COMPLETED = "completed"            # 已完成
    FAILED = "failed"                  # 失败
    CANCELLED = "cancelled"            # 已取消
    TIMEOUT = "timeout"                # 超时


class BenchmarkLevel(Enum):
    """基准测试级别"""
    BASIC = "basic"                    # 基础测试
    STANDARD = "standard"              # 标准测试
    ADVANCED = "advanced"              # 高级测试
    COMPREHENSIVE = "comprehensive"    # 综合测试


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    description: str = ""
    benchmark_type: BenchmarkType = BenchmarkType.PERFORMANCE
    level: BenchmarkLevel = BenchmarkLevel.STANDARD
    timeout: float = 300.0  # 5分钟
    iterations: int = 1
    warmup_iterations: int = 0
    parallel_execution: bool = False
    max_parallel_tasks: int = 5
    collect_metrics: bool = True
    save_results: bool = True
    
    # 环境配置
    environment_config: Dict[str, Any] = field(default_factory=dict)
    
    # 数据配置
    test_data: Dict[str, Any] = field(default_factory=dict)
    
    # 阈值配置
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'benchmark_type': self.benchmark_type.value,
            'level': self.level.value,
            'timeout': self.timeout,
            'iterations': self.iterations,
            'warmup_iterations': self.warmup_iterations,
            'parallel_execution': self.parallel_execution,
            'max_parallel_tasks': self.max_parallel_tasks,
            'collect_metrics': self.collect_metrics,
            'save_results': self.save_results,
            'environment_config': self.environment_config,
            'test_data': self.test_data,
            'performance_thresholds': self.performance_thresholds
        }


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    benchmark_name: str
    status: BenchmarkStatus
    start_time: str
    end_time: Optional[str] = None
    duration: float = 0.0
    iterations_completed: int = 0
    total_iterations: int = 0
    
    # 测试结果
    test_results: List[TestResult] = field(default_factory=list)
    
    # 指标结果
    metric_results: Dict[str, MetricResult] = field(default_factory=dict)
    
    # 性能统计
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark_name': self.benchmark_name,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'iterations_completed': self.iterations_completed,
            'total_iterations': self.total_iterations,
            'test_results': [result.to_dict() for result in self.test_results],
            'metric_results': {k: v.to_dict() for k, v in self.metric_results.items()},
            'performance_stats': self.performance_stats,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'metadata': self.metadata
        }
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if not self.test_results:
            return 0.0
        
        successful = sum(1 for result in self.test_results if result.status == TestStatus.PASSED)
        return successful / len(self.test_results)
    
    def get_average_duration(self) -> float:
        """获取平均执行时间"""
        if not self.test_results:
            return 0.0
        
        total_duration = sum(result.duration for result in self.test_results)
        return total_duration / len(self.test_results)
    
    def get_error_summary(self) -> Dict[str, int]:
        """获取错误摘要"""
        error_summary = {}
        
        for result in self.test_results:
            if result.status == TestStatus.FAILED and result.error_message:
                error_type = result.error_message.split(':')[0] if ':' in result.error_message else 'Unknown'
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        return error_summary


class BaseBenchmark(ABC):
    """基础基准测试类"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logger
        
        # 状态
        self.status = BenchmarkStatus.PENDING
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        
        # 结果
        self.result: Optional[BenchmarkResult] = None
        
        # 组件
        self.test_environment: Optional[TestEnvironment] = None
        self.metric_collector: Optional[MetricCollector] = None
        
        # 控制
        self._stop_requested = False
        self._execution_task: Optional[asyncio.Task] = None
    
    @abstractmethod
    async def setup(self) -> None:
        """设置基准测试"""
        pass
    
    @abstractmethod
    async def execute_iteration(self, iteration: int) -> TestResult:
        """执行单次迭代"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理基准测试"""
        pass
    
    async def run(self) -> BenchmarkResult:
        """运行基准测试"""
        logger.info(f"Starting benchmark: {self.config.name}")
        
        self.status = BenchmarkStatus.RUNNING
        self.start_time = get_iso_timestamp()
        
        # 初始化结果
        self.result = BenchmarkResult(
            benchmark_name=self.config.name,
            status=self.status,
            start_time=self.start_time,
            total_iterations=self.config.iterations
        )
        
        try:
            # 设置
            await self.setup()
            
            # 预热
            if self.config.warmup_iterations > 0:
                logger.info(f"Running {self.config.warmup_iterations} warmup iterations")
                for i in range(self.config.warmup_iterations):
                    if self._stop_requested:
                        break
                    await self.execute_iteration(i)
            
            # 执行测试
            if self.config.parallel_execution:
                await self._run_parallel()
            else:
                await self._run_sequential()
            
            # 收集指标
            if self.config.collect_metrics and self.metric_collector:
                metric_data = self._prepare_metric_data()
                self.result.metric_results = await self.metric_collector.collect_all_metrics(metric_data)
            
            # 计算性能统计
            self.result.performance_stats = self._calculate_performance_stats()
            
            # 设置状态
            if self._stop_requested:
                self.result.status = BenchmarkStatus.CANCELLED
            else:
                self.result.status = BenchmarkStatus.COMPLETED
            
        except asyncio.TimeoutError:
            self.result.status = BenchmarkStatus.TIMEOUT
            self.result.error_message = "Benchmark execution timed out"
            logger.error(f"Benchmark {self.config.name} timed out")
            
        except Exception as e:
            self.result.status = BenchmarkStatus.FAILED
            self.result.error_message = str(e)
            self.result.error_details = {'exception_type': type(e).__name__}
            logger.error(f"Benchmark {self.config.name} failed: {e}")
            
        finally:
            # 清理
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
            
            # 完成
            self.end_time = get_iso_timestamp()
            self.result.end_time = self.end_time
            
            if self.start_time and self.end_time:
                start_ts = time.fromisoformat(self.start_time.replace('Z', '+00:00')).timestamp()
                end_ts = time.fromisoformat(self.end_time.replace('Z', '+00:00')).timestamp()
                self.result.duration = end_ts - start_ts
            
            self.status = self.result.status
            
            logger.info(f"Benchmark {self.config.name} completed with status: {self.status.value}")
        
        return self.result
    
    async def _run_sequential(self) -> None:
        """顺序执行"""
        for i in range(self.config.iterations):
            if self._stop_requested:
                break
            
            try:
                # 执行迭代
                test_result = await asyncio.wait_for(
                    self.execute_iteration(i),
                    timeout=self.config.timeout / self.config.iterations
                )
                
                self.result.test_results.append(test_result)
                self.result.iterations_completed += 1
                
                logger.debug(f"Completed iteration {i + 1}/{self.config.iterations}")
                
            except asyncio.TimeoutError:
                logger.warning(f"Iteration {i + 1} timed out")
                break
            except Exception as e:
                logger.error(f"Iteration {i + 1} failed: {e}")
                # 继续执行其他迭代
    
    async def _run_parallel(self) -> None:
        """并行执行"""
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)
        
        async def run_iteration(iteration: int) -> Optional[TestResult]:
            async with semaphore:
                if self._stop_requested:
                    return None
                
                try:
                    return await asyncio.wait_for(
                        self.execute_iteration(iteration),
                        timeout=self.config.timeout / self.config.iterations
                    )
                except Exception as e:
                    logger.error(f"Iteration {iteration + 1} failed: {e}")
                    return None
        
        # 创建任务
        tasks = [run_iteration(i) for i in range(self.config.iterations)]
        
        # 执行任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(results):
            if isinstance(result, TestResult):
                self.result.test_results.append(result)
                self.result.iterations_completed += 1
            elif isinstance(result, Exception):
                logger.error(f"Iteration {i + 1} failed with exception: {result}")
    
    def _prepare_metric_data(self) -> Dict[str, Any]:
        """准备指标数据"""
        if not self.result.test_results:
            return {}
        
        # 计算基础统计
        total_tests = len(self.result.test_results)
        successful_tests = sum(1 for r in self.result.test_results if r.status == TestStatus.PASSED)
        failed_tests = total_tests - successful_tests
        
        durations = [r.duration for r in self.result.test_results]
        
        return {
            'accuracy': {
                'correct': successful_tests,
                'total': total_tests
            },
            'response_time': {
                'response_time': sum(durations) / len(durations) if durations else 0
            },
            'throughput': {
                'operations': total_tests,
                'duration': self.result.duration if self.result.duration > 0 else 1
            },
            'error_rate': {
                'errors': failed_tests,
                'total': total_tests
            },
            'success_rate': {
                'success': successful_tests,
                'total': total_tests
            }
        }
    
    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """计算性能统计"""
        if not self.result.test_results:
            return {}
        
        durations = [r.duration for r in self.result.test_results]
        
        stats = {
            'total_tests': len(self.result.test_results),
            'successful_tests': sum(1 for r in self.result.test_results if r.status == TestStatus.PASSED),
            'failed_tests': sum(1 for r in self.result.test_results if r.status == TestStatus.FAILED),
            'success_rate': self.result.get_success_rate(),
            'average_duration': self.result.get_average_duration(),
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'total_duration': self.result.duration,
            'throughput': len(self.result.test_results) / self.result.duration if self.result.duration > 0 else 0
        }
        
        # 计算百分位数
        if durations:
            sorted_durations = sorted(durations)
            stats['p50_duration'] = self._percentile(sorted_durations, 50)
            stats['p95_duration'] = self._percentile(sorted_durations, 95)
            stats['p99_duration'] = self._percentile(sorted_durations, 99)
        
        return stats
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        
        index = (percentile / 100.0) * (len(values) - 1)
        
        if index.is_integer():
            return values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index >= len(values):
                return values[lower_index]
            
            weight = index - lower_index
            return values[lower_index] * (1 - weight) + values[upper_index] * weight
    
    def stop(self) -> None:
        """停止基准测试"""
        self._stop_requested = True
        
        if self._execution_task and not self._execution_task.done():
            self._execution_task.cancel()
        
        logger.info(f"Stop requested for benchmark: {self.config.name}")
    
    def get_status(self) -> BenchmarkStatus:
        """获取状态"""
        return self.status
    
    def get_result(self) -> Optional[BenchmarkResult]:
        """获取结果"""
        return self.result


class PerformanceBenchmark(BaseBenchmark):
    """性能基准测试"""
    
    def __init__(self, config: BenchmarkConfig, test_function: Callable[[], Any]):
        super().__init__(config)
        self.test_function = test_function
    
    async def setup(self) -> None:
        """设置性能测试"""
        logger.info("Setting up performance benchmark")
        
        # 初始化测试环境
        if self.config.environment_config:
            self.test_environment = TestEnvironment(self.config.environment_config)
            await self.test_environment.initialize()
        
        # 初始化指标收集器
        if self.config.collect_metrics:
            from .metrics import (
                MetricCollector, ResponseTimeMetric, ThroughputMetric,
                AccuracyMetric, ErrorRateMetric, SuccessRateMetric
            )
            
            self.metric_collector = MetricCollector()
            self.metric_collector.register_metric(ResponseTimeMetric())
            self.metric_collector.register_metric(ThroughputMetric())
            self.metric_collector.register_metric(AccuracyMetric())
            self.metric_collector.register_metric(ErrorRateMetric())
            self.metric_collector.register_metric(SuccessRateMetric())
    
    async def execute_iteration(self, iteration: int) -> TestResult:
        """执行性能测试迭代"""
        start_time = time.time()
        
        try:
            # 执行测试函数
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.test_function
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            return TestResult(
                test_name=f"{self.config.name}_iteration_{iteration}",
                status=TestStatus.PASSED,
                start_time=get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                duration=duration,
                result_data={'result': result}
            )
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            return TestResult(
                test_name=f"{self.config.name}_iteration_{iteration}",
                status=TestStatus.FAILED,
                start_time=get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                duration=duration,
                error_message=str(e)
            )
    
    async def cleanup(self) -> None:
        """清理性能测试"""
        logger.info("Cleaning up performance benchmark")
        
        if self.test_environment:
            await self.test_environment.cleanup()
        
        if self.metric_collector:
            self.metric_collector.stop_auto_collection()


class LoadBenchmark(BaseBenchmark):
    """负载基准测试"""
    
    def __init__(self, 
                 config: BenchmarkConfig,
                 test_function: Callable[[], Any],
                 concurrent_users: int = 10,
                 ramp_up_time: float = 10.0):
        super().__init__(config)
        self.test_function = test_function
        self.concurrent_users = concurrent_users
        self.ramp_up_time = ramp_up_time
        
        # 负载测试特定配置
        self.user_tasks: List[asyncio.Task] = []
        self.results_queue: asyncio.Queue = asyncio.Queue()
    
    async def setup(self) -> None:
        """设置负载测试"""
        logger.info(f"Setting up load benchmark with {self.concurrent_users} concurrent users")
        
        # 初始化测试环境
        if self.config.environment_config:
            self.test_environment = TestEnvironment(self.config.environment_config)
            await self.test_environment.initialize()
        
        # 初始化指标收集器
        if self.config.collect_metrics:
            from .metrics import (
                MetricCollector, ResponseTimeMetric, ThroughputMetric,
                ErrorRateMetric, SuccessRateMetric
            )
            
            self.metric_collector = MetricCollector()
            self.metric_collector.register_metric(ResponseTimeMetric())
            self.metric_collector.register_metric(ThroughputMetric())
            self.metric_collector.register_metric(ErrorRateMetric())
            self.metric_collector.register_metric(SuccessRateMetric())
    
    async def execute_iteration(self, iteration: int) -> TestResult:
        """执行负载测试迭代"""
        start_time = time.time()
        
        try:
            # 启动并发用户
            await self._start_concurrent_users()
            
            # 等待所有用户完成
            await self._wait_for_completion()
            
            # 收集结果
            results = await self._collect_results()
            
            end_time = time.time()
            duration = end_time - start_time
            
            return TestResult(
                test_name=f"{self.config.name}_load_iteration_{iteration}",
                status=TestStatus.PASSED,
                start_time=get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                duration=duration,
                result_data={
                    'concurrent_users': self.concurrent_users,
                    'total_requests': len(results),
                    'successful_requests': sum(1 for r in results if r.get('success', False)),
                    'failed_requests': sum(1 for r in results if not r.get('success', True)),
                    'average_response_time': sum(r.get('duration', 0) for r in results) / len(results) if results else 0
                }
            )
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            return TestResult(
                test_name=f"{self.config.name}_load_iteration_{iteration}",
                status=TestStatus.FAILED,
                start_time=get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                duration=duration,
                error_message=str(e)
            )
    
    async def _start_concurrent_users(self) -> None:
        """启动并发用户"""
        ramp_up_delay = self.ramp_up_time / self.concurrent_users if self.concurrent_users > 0 else 0
        
        for i in range(self.concurrent_users):
            if self._stop_requested:
                break
            
            # 创建用户任务
            task = asyncio.create_task(self._simulate_user(i))
            self.user_tasks.append(task)
            
            # 渐进启动
            if ramp_up_delay > 0:
                await asyncio.sleep(ramp_up_delay)
    
    async def _simulate_user(self, user_id: int) -> None:
        """模拟用户行为"""
        try:
            for request_id in range(self.config.iterations):
                if self._stop_requested:
                    break
                
                start_time = time.time()
                
                try:
                    # 执行测试函数
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.test_function
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    await self.results_queue.put({
                        'user_id': user_id,
                        'request_id': request_id,
                        'success': True,
                        'duration': duration,
                        'result': result
                    })
                    
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    await self.results_queue.put({
                        'user_id': user_id,
                        'request_id': request_id,
                        'success': False,
                        'duration': duration,
                        'error': str(e)
                    })
                
                # 用户间隔
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"User {user_id} simulation failed: {e}")
    
    async def _wait_for_completion(self) -> None:
        """等待所有用户完成"""
        if self.user_tasks:
            await asyncio.gather(*self.user_tasks, return_exceptions=True)
    
    async def _collect_results(self) -> List[Dict[str, Any]]:
        """收集结果"""
        results = []
        
        while not self.results_queue.empty():
            try:
                result = await asyncio.wait_for(self.results_queue.get(), timeout=1.0)
                results.append(result)
            except asyncio.TimeoutError:
                break
        
        return results
    
    async def cleanup(self) -> None:
        """清理负载测试"""
        logger.info("Cleaning up load benchmark")
        
        # 停止用户任务
        for task in self.user_tasks:
            if not task.done():
                task.cancel()
        
        if self.user_tasks:
            await asyncio.gather(*self.user_tasks, return_exceptions=True)
        
        self.user_tasks.clear()
        
        # 清空结果队列
        while not self.results_queue.empty():
            try:
                await asyncio.wait_for(self.results_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                break
        
        if self.test_environment:
            await self.test_environment.cleanup()
        
        if self.metric_collector:
            self.metric_collector.stop_auto_collection()


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.benchmarks: List[BaseBenchmark] = []
        self.logger = logger
        
        # 执行配置
        self.parallel_execution = False
        self.stop_on_failure = False
        self.timeout = 3600.0  # 1小时
        
        # 状态
        self.status = BenchmarkStatus.PENDING
        self.results: List[BenchmarkResult] = []
    
    def add_benchmark(self, benchmark: BaseBenchmark) -> None:
        """添加基准测试"""
        self.benchmarks.append(benchmark)
        logger.info(f"Added benchmark: {benchmark.config.name}")
    
    def remove_benchmark(self, benchmark_name: str) -> None:
        """移除基准测试"""
        self.benchmarks = [b for b in self.benchmarks if b.config.name != benchmark_name]
        logger.info(f"Removed benchmark: {benchmark_name}")
    
    async def run(self) -> List[BenchmarkResult]:
        """运行基准测试套件"""
        logger.info(f"Starting benchmark suite: {self.name}")
        
        self.status = BenchmarkStatus.RUNNING
        self.results.clear()
        
        try:
            if self.parallel_execution:
                await self._run_parallel()
            else:
                await self._run_sequential()
            
            self.status = BenchmarkStatus.COMPLETED
            
        except Exception as e:
            self.status = BenchmarkStatus.FAILED
            logger.error(f"Benchmark suite {self.name} failed: {e}")
        
        logger.info(f"Benchmark suite {self.name} completed with status: {self.status.value}")
        return self.results
    
    async def _run_sequential(self) -> None:
        """顺序执行"""
        for benchmark in self.benchmarks:
            try:
                result = await asyncio.wait_for(
                    benchmark.run(),
                    timeout=self.timeout
                )
                
                self.results.append(result)
                
                if self.stop_on_failure and result.status == BenchmarkStatus.FAILED:
                    logger.warning(f"Stopping suite due to benchmark failure: {benchmark.config.name}")
                    break
                    
            except asyncio.TimeoutError:
                logger.error(f"Benchmark {benchmark.config.name} timed out")
                if self.stop_on_failure:
                    break
            except Exception as e:
                logger.error(f"Benchmark {benchmark.config.name} failed: {e}")
                if self.stop_on_failure:
                    break
    
    async def _run_parallel(self) -> None:
        """并行执行"""
        tasks = []
        
        for benchmark in self.benchmarks:
            task = asyncio.create_task(benchmark.run())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, BenchmarkResult):
                self.results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Benchmark {self.benchmarks[i].config.name} failed: {result}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        if not self.results:
            return {
                'suite_name': self.name,
                'total_benchmarks': len(self.benchmarks),
                'completed_benchmarks': 0,
                'status': self.status.value
            }
        
        completed = sum(1 for r in self.results if r.status == BenchmarkStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == BenchmarkStatus.FAILED)
        
        total_duration = sum(r.duration for r in self.results)
        average_success_rate = sum(r.get_success_rate() for r in self.results) / len(self.results)
        
        return {
            'suite_name': self.name,
            'description': self.description,
            'status': self.status.value,
            'total_benchmarks': len(self.benchmarks),
            'completed_benchmarks': completed,
            'failed_benchmarks': failed,
            'total_duration': total_duration,
            'average_success_rate': average_success_rate,
            'results': [r.to_dict() for r in self.results]
        }


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.logger = logger
        
        # 注册的基准测试
        self.benchmarks: Dict[str, BaseBenchmark] = {}
        self.suites: Dict[str, BenchmarkSuite] = {}
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        
        # 配置
        self.max_concurrent_benchmarks = 5
        self.default_timeout = 3600.0
    
    def register_benchmark(self, benchmark: BaseBenchmark) -> None:
        """注册基准测试"""
        self.benchmarks[benchmark.config.name] = benchmark
        logger.info(f"Registered benchmark: {benchmark.config.name}")
    
    def register_suite(self, suite: BenchmarkSuite) -> None:
        """注册基准测试套件"""
        self.suites[suite.name] = suite
        logger.info(f"Registered benchmark suite: {suite.name}")
    
    async def run_benchmark(self, benchmark_name: str) -> Optional[BenchmarkResult]:
        """运行单个基准测试"""
        benchmark = self.benchmarks.get(benchmark_name)
        if not benchmark:
            logger.error(f"Benchmark not found: {benchmark_name}")
            return None
        
        try:
            result = await benchmark.run()
            
            # 记录执行历史
            self.execution_history.append({
                'type': 'benchmark',
                'name': benchmark_name,
                'timestamp': get_iso_timestamp(),
                'status': result.status.value,
                'duration': result.duration
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
            return None
    
    async def run_suite(self, suite_name: str) -> Optional[List[BenchmarkResult]]:
        """运行基准测试套件"""
        suite = self.suites.get(suite_name)
        if not suite:
            logger.error(f"Benchmark suite not found: {suite_name}")
            return None
        
        try:
            results = await suite.run()
            
            # 记录执行历史
            self.execution_history.append({
                'type': 'suite',
                'name': suite_name,
                'timestamp': get_iso_timestamp(),
                'status': suite.status.value,
                'total_benchmarks': len(suite.benchmarks),
                'completed_benchmarks': len(results)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run benchmark suite {suite_name}: {e}")
            return None
    
    def list_benchmarks(self) -> List[str]:
        """列出所有基准测试"""
        return list(self.benchmarks.keys())
    
    def list_suites(self) -> List[str]:
        """列出所有基准测试套件"""
        return list(self.suites.keys())
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history.copy()
    
    def clear_history(self) -> None:
        """清空执行历史"""
        self.execution_history.clear()
        logger.info("Cleared execution history")