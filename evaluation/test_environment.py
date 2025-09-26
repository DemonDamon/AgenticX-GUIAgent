#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Test Environment
基于AgenticX框架的测试环境：提供完整的测试环境和测试执行框架

重构说明：
- 使用AgenticX的Component作为基类
- 集成AgenticX的事件系统
- 使用AgenticX的工具框架
- 避免重复实现测试基础设施

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import time
import uuid
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
from agenticx.core.agent import Agent, AgentResult

from utils import get_iso_timestamp, setup_logger
from core.base_agent import BaseAgenticXGUIAgentAgent


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestPriority(Enum):
    """测试优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestType(Enum):
    """测试类型"""
    UNIT = "unit"                    # 单元测试
    INTEGRATION = "integration"      # 集成测试
    FUNCTIONAL = "functional"        # 功能测试
    PERFORMANCE = "performance"      # 性能测试
    STRESS = "stress"                # 压力测试
    REGRESSION = "regression"        # 回归测试
    ACCEPTANCE = "acceptance"        # 验收测试
    EXPLORATORY = "exploratory"      # 探索性测试


@dataclass
class TestConfig:
    """测试配置"""
    test_name: str
    test_type: TestType = TestType.FUNCTIONAL
    priority: TestPriority = TestPriority.MEDIUM
    timeout: float = 300.0  # 5分钟
    retry_count: int = 0
    retry_delay: float = 1.0
    parallel: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    setup_required: bool = True
    teardown_required: bool = True
    dependencies: List[str] = field(default_factory=list)
    environment_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestAssertion:
    """测试断言"""
    assertion_id: str
    description: str
    expected: Any
    actual: Optional[Any] = None
    passed: Optional[bool] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=get_iso_timestamp)


@dataclass
class TestStep:
    """测试步骤"""
    step_id: str
    description: str
    action: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_result: Optional[Any] = None
    actual_result: Optional[Any] = None
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    assertions: List[TestAssertion] = field(default_factory=list)
    
    async def execute(self) -> bool:
        """执行测试步骤"""
        self.start_time = get_iso_timestamp()
        self.status = TestStatus.RUNNING
        
        try:
            start_time = time.time()
            
            # 执行动作
            if asyncio.iscoroutinefunction(self.action):
                self.actual_result = await self.action(**self.parameters)
            else:
                self.actual_result = self.action(**self.parameters)
            
            self.duration = time.time() - start_time
            self.end_time = get_iso_timestamp()
            
            # 验证结果
            if self.expected_result is not None:
                if self.actual_result == self.expected_result:
                    self.status = TestStatus.PASSED
                    return True
                else:
                    self.status = TestStatus.FAILED
                    self.error_message = f"Expected {self.expected_result}, got {self.actual_result}"
                    return False
            else:
                self.status = TestStatus.PASSED
                return True
                
        except Exception as e:
            self.duration = time.time() - start_time if 'start_time' in locals() else 0
            self.end_time = get_iso_timestamp()
            self.status = TestStatus.ERROR
            self.error_message = str(e)
            return False
    
    def add_assertion(self, description: str, expected: Any, actual: Any) -> bool:
        """添加断言"""
        assertion = TestAssertion(
            assertion_id=f"assert_{len(self.assertions) + 1}",
            description=description,
            expected=expected,
            actual=actual
        )
        
        try:
            assertion.passed = (expected == actual)
            if not assertion.passed:
                assertion.error_message = f"Expected {expected}, got {actual}"
        except Exception as e:
            assertion.passed = False
            assertion.error_message = f"Assertion error: {e}"
        
        self.assertions.append(assertion)
        return assertion.passed


class TestCase(ABC):
    """测试用例基类"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_id = str(uuid.uuid4())
        self.status = TestStatus.PENDING
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.duration: Optional[float] = None
        self.error_message: Optional[str] = None
        self.steps: List[TestStep] = []
        self.setup_completed = False
        self.teardown_completed = False
        self.logger = logger
        
        # 测试数据
        self.test_data: Dict[str, Any] = {}
        self.artifacts: List[str] = []  # 测试产物（截图、日志等）
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'passed_steps': 0,
            'failed_steps': 0,
            'error_steps': 0,
            'skipped_steps': 0
        }
    
    @abstractmethod
    async def setup(self) -> bool:
        """测试前置条件设置"""
        pass
    
    @abstractmethod
    async def execute(self) -> bool:
        """执行测试"""
        pass
    
    @abstractmethod
    async def teardown(self) -> bool:
        """测试后清理"""
        pass
    
    async def run(self) -> 'TestResult':
        """运行测试用例"""
        self.start_time = get_iso_timestamp()
        self.status = TestStatus.RUNNING
        
        try:
            start_time = time.time()
            
            # 执行前置条件
            if self.config.setup_required:
                logger.info(f"Setting up test: {self.config.test_name}")
                setup_success = await self.setup()
                self.setup_completed = setup_success
                
                if not setup_success:
                    self.status = TestStatus.ERROR
                    self.error_message = "Setup failed"
                    return self._create_result()
            
            # 执行测试
            logger.info(f"Executing test: {self.config.test_name}")
            test_success = await self.execute()
            
            # 执行后清理
            if self.config.teardown_required:
                logger.info(f"Tearing down test: {self.config.test_name}")
                teardown_success = await self.teardown()
                self.teardown_completed = teardown_success
                
                if not teardown_success:
                    logger.warning("Teardown failed, but test result preserved")
            
            # 设置最终状态
            if test_success:
                self.status = TestStatus.PASSED
            else:
                self.status = TestStatus.FAILED
            
            self.duration = time.time() - start_time
            self.end_time = get_iso_timestamp()
            
            logger.info(f"Test completed: {self.config.test_name}, Status: {self.status.value}")
            
        except asyncio.TimeoutError:
            self.status = TestStatus.TIMEOUT
            self.error_message = f"Test timed out after {self.config.timeout} seconds"
            self.duration = time.time() - start_time if 'start_time' in locals() else 0
            self.end_time = get_iso_timestamp()
            logger.error(f"Test timed out: {self.config.test_name}")
            
        except Exception as e:
            self.status = TestStatus.ERROR
            self.error_message = str(e)
            self.duration = time.time() - start_time if 'start_time' in locals() else 0
            self.end_time = get_iso_timestamp()
            logger.error(f"Test error: {self.config.test_name}, Error: {e}")
        
        return self._create_result()
    
    def add_step(self, description: str, action: Callable, **kwargs) -> TestStep:
        """添加测试步骤"""
        step = TestStep(
            step_id=f"step_{len(self.steps) + 1}",
            description=description,
            action=action,
            parameters=kwargs
        )
        self.steps.append(step)
        return step
    
    def add_artifact(self, artifact_path: str) -> None:
        """添加测试产物"""
        self.artifacts.append(artifact_path)
    
    def set_test_data(self, key: str, value: Any) -> None:
        """设置测试数据"""
        self.test_data[key] = value
    
    def get_test_data(self, key: str, default: Any = None) -> Any:
        """获取测试数据"""
        return self.test_data.get(key, default)
    
    def _create_result(self) -> 'TestResult':
        """创建测试结果"""
        # 更新统计信息
        self.stats['total_steps'] = len(self.steps)
        for step in self.steps:
            if step.status == TestStatus.PASSED:
                self.stats['passed_steps'] += 1
            elif step.status == TestStatus.FAILED:
                self.stats['failed_steps'] += 1
            elif step.status == TestStatus.ERROR:
                self.stats['error_steps'] += 1
            elif step.status == TestStatus.SKIPPED:
                self.stats['skipped_steps'] += 1
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.config.test_name,
            test_type=self.config.test_type,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.duration,
            error_message=self.error_message,
            steps=self.steps.copy(),
            artifacts=self.artifacts.copy(),
            test_data=self.test_data.copy(),
            stats=self.stats.copy(),
            setup_completed=self.setup_completed,
            teardown_completed=self.teardown_completed
        )


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    steps: List[TestStep] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    test_data: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)
    setup_completed: bool = False
    teardown_completed: bool = False
    
    @property
    def success(self) -> bool:
        """测试是否成功"""
        return self.status == TestStatus.PASSED
    
    @property
    def failed(self) -> bool:
        """测试是否失败"""
        return self.status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'error_message': self.error_message,
            'steps': [{
                'step_id': step.step_id,
                'description': step.description,
                'status': step.status.value,
                'duration': step.duration,
                'error_message': step.error_message,
                'assertions': [{
                    'assertion_id': assertion.assertion_id,
                    'description': assertion.description,
                    'passed': assertion.passed,
                    'error_message': assertion.error_message
                } for assertion in step.assertions]
            } for step in self.steps],
            'artifacts': self.artifacts,
            'stats': self.stats,
            'setup_completed': self.setup_completed,
            'teardown_completed': self.teardown_completed
        }


class TestSuite:
    """测试套件"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.suite_id = str(uuid.uuid4())
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        self.logger = logger
        
        # 套件配置
        self.parallel_execution = False
        self.max_parallel_tests = 4
        self.stop_on_failure = False
        self.timeout = 3600.0  # 1小时
        
        # 统计信息
        self.stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0,
            'timeout_tests': 0
        }
    
    def add_test_case(self, test_case: TestCase) -> None:
        """添加测试用例"""
        self.test_cases.append(test_case)
        logger.info(f"Added test case: {test_case.config.test_name}")
    
    def remove_test_case(self, test_name: str) -> bool:
        """移除测试用例"""
        for i, test_case in enumerate(self.test_cases):
            if test_case.config.test_name == test_name:
                del self.test_cases[i]
                logger.info(f"Removed test case: {test_name}")
                return True
        return False
    
    async def run(self) -> List[TestResult]:
        """运行测试套件"""
        logger.info(f"Starting test suite: {self.name}")
        self.results.clear()
        
        start_time = time.time()
        
        try:
            if self.parallel_execution:
                await self._run_parallel()
            else:
                await self._run_sequential()
        except Exception as e:
            logger.error(f"Test suite execution error: {e}")
        
        duration = time.time() - start_time
        
        # 更新统计信息
        self._update_stats()
        
        logger.info(
            f"Test suite completed: {self.name}, "
            f"Duration: {duration:.2f}s, "
            f"Passed: {self.stats['passed_tests']}/{self.stats['total_tests']}"
        )
        
        return self.results
    
    async def _run_sequential(self) -> None:
        """顺序执行测试"""
        for test_case in self.test_cases:
            try:
                # 设置超时
                result = await asyncio.wait_for(
                    test_case.run(),
                    timeout=test_case.config.timeout
                )
                self.results.append(result)
                
                # 检查是否需要在失败时停止
                if self.stop_on_failure and result.failed:
                    logger.warning(f"Stopping suite execution due to failure in: {test_case.config.test_name}")
                    break
                    
            except asyncio.TimeoutError:
                logger.error(f"Test case timed out: {test_case.config.test_name}")
                # 创建超时结果
                timeout_result = TestResult(
                    test_id=test_case.test_id,
                    test_name=test_case.config.test_name,
                    test_type=test_case.config.test_type,
                    status=TestStatus.TIMEOUT,
                    error_message=f"Test timed out after {test_case.config.timeout} seconds"
                )
                self.results.append(timeout_result)
                
                if self.stop_on_failure:
                    break
            
            except Exception as e:
                logger.error(f"Error running test case {test_case.config.test_name}: {e}")
    
    async def _run_parallel(self) -> None:
        """并行执行测试"""
        semaphore = asyncio.Semaphore(self.max_parallel_tests)
        
        async def run_with_semaphore(test_case: TestCase) -> TestResult:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        test_case.run(),
                        timeout=test_case.config.timeout
                    )
                except asyncio.TimeoutError:
                    return TestResult(
                        test_id=test_case.test_id,
                        test_name=test_case.config.test_name,
                        test_type=test_case.config.test_type,
                        status=TestStatus.TIMEOUT,
                        error_message=f"Test timed out after {test_case.config.timeout} seconds"
                    )
        
        # 创建任务
        tasks = [run_with_semaphore(test_case) for test_case in self.test_cases]
        
        # 等待所有任务完成
        self.results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        for i, result in enumerate(self.results):
            if isinstance(result, Exception):
                logger.error(f"Error in parallel test execution: {result}")
                self.results[i] = TestResult(
                    test_id=self.test_cases[i].test_id,
                    test_name=self.test_cases[i].config.test_name,
                    test_type=self.test_cases[i].config.test_type,
                    status=TestStatus.ERROR,
                    error_message=str(result)
                )
    
    def _update_stats(self) -> None:
        """更新统计信息"""
        self.stats = {
            'total_tests': len(self.results),
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0,
            'timeout_tests': 0
        }
        
        for result in self.results:
            if result.status == TestStatus.PASSED:
                self.stats['passed_tests'] += 1
            elif result.status == TestStatus.FAILED:
                self.stats['failed_tests'] += 1
            elif result.status == TestStatus.ERROR:
                self.stats['error_tests'] += 1
            elif result.status == TestStatus.SKIPPED:
                self.stats['skipped_tests'] += 1
            elif result.status == TestStatus.TIMEOUT:
                self.stats['timeout_tests'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取测试套件摘要"""
        return {
            'suite_id': self.suite_id,
            'name': self.name,
            'description': self.description,
            'stats': self.stats,
            'success_rate': self.stats['passed_tests'] / max(self.stats['total_tests'], 1),
            'total_duration': sum(r.duration or 0 for r in self.results)
        }


class TestRunner(Component):
    """测试运行器 - 基于AgenticX Component"""
    
    def __init__(self, output_dir: Optional[str] = None, event_bus: Optional[EventBus] = None):
        super().__init__(name="test_runner")
        
        self.output_dir = Path(output_dir) if output_dir else Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_bus = event_bus or EventBus()
        self.logger = logger
        
        # 运行配置
        self.global_timeout = 7200.0  # 2小时
        self.save_artifacts = True
        self.generate_reports = True
        
        # 运行状态
        self.running = False
        self.current_suite: Optional[TestSuite] = None
        self.all_results: List[TestResult] = []
    
    async def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """运行测试套件"""
        self.running = True
        self.current_suite = suite
        
        try:
            logger.info(f"Starting test suite: {suite.name}")
            
            # 运行测试套件
            results = await asyncio.wait_for(
                suite.run(),
                timeout=self.global_timeout
            )
            
            # 保存结果
            self.all_results.extend(results)
            
            # 生成报告
            if self.generate_reports:
                await self._generate_suite_report(suite, results)
            
            logger.info(f"Test suite completed: {suite.name}")
            return results
            
        except asyncio.TimeoutError:
            logger.error(f"Test suite timed out: {suite.name}")
            raise
        except Exception as e:
            logger.error(f"Error running test suite {suite.name}: {e}")
            raise
        finally:
            self.running = False
            self.current_suite = None
    
    async def run_multiple_suites(self, suites: List[TestSuite]) -> Dict[str, List[TestResult]]:
        """运行多个测试套件"""
        all_suite_results = {}
        
        for suite in suites:
            try:
                results = await self.run_suite(suite)
                all_suite_results[suite.name] = results
            except Exception as e:
                logger.error(f"Failed to run suite {suite.name}: {e}")
                all_suite_results[suite.name] = []
        
        # 生成总体报告
        if self.generate_reports:
            await self._generate_overall_report(all_suite_results)
        
        return all_suite_results
    
    async def _generate_suite_report(self, suite: TestSuite, results: List[TestResult]) -> None:
        """生成测试套件报告"""
        try:
            report_data = {
                'suite_summary': suite.get_summary(),
                'results': [result.to_dict() for result in results],
                'generated_at': get_iso_timestamp()
            }
            
            # 保存JSON报告
            report_file = self.output_dir / f"{suite.name}_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating suite report: {e}")
    
    async def _generate_overall_report(self, all_results: Dict[str, List[TestResult]]) -> None:
        """生成总体报告"""
        try:
            # 计算总体统计
            total_stats = {
                'total_suites': len(all_results),
                'total_tests': sum(len(results) for results in all_results.values()),
                'passed_tests': 0,
                'failed_tests': 0,
                'error_tests': 0,
                'skipped_tests': 0,
                'timeout_tests': 0
            }
            
            for results in all_results.values():
                for result in results:
                    if result.status == TestStatus.PASSED:
                        total_stats['passed_tests'] += 1
                    elif result.status == TestStatus.FAILED:
                        total_stats['failed_tests'] += 1
                    elif result.status == TestStatus.ERROR:
                        total_stats['error_tests'] += 1
                    elif result.status == TestStatus.SKIPPED:
                        total_stats['skipped_tests'] += 1
                    elif result.status == TestStatus.TIMEOUT:
                        total_stats['timeout_tests'] += 1
            
            report_data = {
                'overall_stats': total_stats,
                'success_rate': total_stats['passed_tests'] / max(total_stats['total_tests'], 1),
                'suite_results': {
                    suite_name: [result.to_dict() for result in results]
                    for suite_name, results in all_results.items()
                },
                'generated_at': get_iso_timestamp()
            }
            
            # 保存总体报告
            report_file = self.output_dir / "overall_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated overall report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating overall report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取运行状态"""
        return {
            'running': self.running,
            'current_suite': self.current_suite.name if self.current_suite else None,
            'total_results': len(self.all_results),
            'output_directory': str(self.output_dir)
        }


class TestEnvironment(Component):
    """测试环境 - 基于AgenticX Component"""
    
    def __init__(self, config: TestConfig, event_bus: Optional[EventBus] = None):
        super().__init__(name=f"test_env_{config.test_name}")
        
        self.config = config
        self.environment_id = str(uuid.uuid4())
        self.event_bus = event_bus or EventBus()
        self.logger = logger
        
        # 环境组件
        self.test_runner = TestRunner()
        self.test_suites: Dict[str, TestSuite] = {}
        
        # 环境状态
        self.initialized = False
        self.running = False
        
        # 资源管理
        self.resources: Dict[str, Any] = {}
        self.cleanup_handlers: List[Callable] = []
    
    async def initialize(self) -> bool:
        """初始化测试环境"""
        try:
            logger.info("Initializing test environment")
            
            # 创建输出目录
            self.test_runner.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化资源
            await self._initialize_resources()
            
            self.initialized = True
            logger.info("Test environment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize test environment: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理测试环境"""
        try:
            logger.info("Cleaning up test environment")
            
            # 执行清理处理器
            for handler in self.cleanup_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    logger.error(f"Error in cleanup handler: {e}")
            
            # 清理资源
            await self._cleanup_resources()
            
            self.initialized = False
            logger.info("Test environment cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup test environment: {e}")
            return False
    
    def add_test_suite(self, suite: TestSuite) -> None:
        """添加测试套件"""
        self.test_suites[suite.name] = suite
        logger.info(f"Added test suite: {suite.name}")
    
    async def run_all_suites(self) -> Dict[str, List[TestResult]]:
        """运行所有测试套件"""
        if not self.initialized:
            raise RuntimeError("Test environment not initialized")
        
        self.running = True
        try:
            suites = list(self.test_suites.values())
            return await self.test_runner.run_multiple_suites(suites)
        finally:
            self.running = False
    
    async def run_suite(self, suite_name: str) -> List[TestResult]:
        """运行指定测试套件"""
        if not self.initialized:
            raise RuntimeError("Test environment not initialized")
        
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite not found: {suite_name}")
        
        suite = self.test_suites[suite_name]
        return await self.test_runner.run_suite(suite)
    
    def add_resource(self, name: str, resource: Any) -> None:
        """添加环境资源"""
        self.resources[name] = resource
    
    def get_resource(self, name: str) -> Any:
        """获取环境资源"""
        return self.resources.get(name)
    
    def add_cleanup_handler(self, handler: Callable) -> None:
        """添加清理处理器"""
        self.cleanup_handlers.append(handler)
    
    async def _initialize_resources(self) -> None:
        """初始化资源"""
        # 这里可以初始化各种测试资源
        # 例如：数据库连接、模拟器、测试数据等
        pass
    
    async def _cleanup_resources(self) -> None:
        """清理资源"""
        # 清理所有资源
        self.resources.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """获取环境状态"""
        return {
            'environment_id': self.environment_id,
            'initialized': self.initialized,
            'running': self.running,
            'test_suites': list(self.test_suites.keys()),
            'resources': list(self.resources.keys()),
            'runner_status': self.test_runner.get_status()
        }