#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Test Scenarios
测试场景：定义各种测试场景和用例

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from loguru import logger

from utils import get_iso_timestamp
from utils import setup_logger
from core.base_agent import BaseAgenticXGUIAgentAgent
from tools.gui_tools import GUITool, ToolResult, Coordinate
from .test_environment import TestCase, TestConfig, TestType, TestPriority, TestStep


class ScenarioType(Enum):
    """场景类型"""
    BASIC_INTERACTION = "basic_interaction"      # 基础交互
    COMPLEX_WORKFLOW = "complex_workflow"        # 复杂工作流
    ERROR_HANDLING = "error_handling"            # 错误处理
    PERFORMANCE_TEST = "performance_test"        # 性能测试
    STRESS_TEST = "stress_test"                  # 压力测试
    REGRESSION_TEST = "regression_test"          # 回归测试
    USER_JOURNEY = "user_journey"                # 用户旅程
    EDGE_CASE = "edge_case"                      # 边缘情况
    INTEGRATION_TEST = "integration_test"        # 集成测试
    ACCESSIBILITY_TEST = "accessibility_test"    # 可访问性测试


class AppType(Enum):
    """应用类型"""
    SOCIAL_MEDIA = "social_media"        # 社交媒体
    E_COMMERCE = "e_commerce"            # 电商
    PRODUCTIVITY = "productivity"        # 生产力工具
    ENTERTAINMENT = "entertainment"      # 娱乐
    FINANCE = "finance"                  # 金融
    EDUCATION = "education"              # 教育
    HEALTH = "health"                    # 健康
    TRAVEL = "travel"                    # 旅行
    NEWS = "news"                        # 新闻
    UTILITY = "utility"                  # 工具
    GAME = "game"                        # 游戏
    COMMUNICATION = "communication"      # 通讯


@dataclass
class ScenarioContext:
    """场景上下文"""
    app_type: AppType
    app_name: str
    app_version: str = "1.0.0"
    device_type: str = "mobile"
    os_version: str = "latest"
    screen_size: Tuple[int, int] = (375, 812)  # iPhone X size
    network_condition: str = "wifi"  # wifi, 4g, 3g, offline
    user_profile: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'app_type': self.app_type.value,
            'app_name': self.app_name,
            'app_version': self.app_version,
            'device_type': self.device_type,
            'os_version': self.os_version,
            'screen_size': self.screen_size,
            'network_condition': self.network_condition,
            'user_profile': self.user_profile,
            'environment_variables': self.environment_variables
        }


@dataclass
class ExpectedOutcome:
    """预期结果"""
    success_criteria: List[str] = field(default_factory=list)
    failure_criteria: List[str] = field(default_factory=list)
    performance_criteria: Dict[str, float] = field(default_factory=dict)
    ui_state_criteria: Dict[str, Any] = field(default_factory=dict)
    data_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, actual_result: Any) -> Tuple[bool, List[str]]:
        """验证结果"""
        validation_errors = []
        
        # 这里可以实现具体的验证逻辑
        # 目前返回简单的模拟结果
        if not actual_result:
            validation_errors.append("No actual result provided")
        
        return len(validation_errors) == 0, validation_errors


class BaseTestScenario(ABC):
    """基础测试场景"""
    
    def __init__(self, 
                 scenario_type: ScenarioType,
                 name: str,
                 description: str,
                 context: ScenarioContext,
                 expected_outcome: ExpectedOutcome):
        self.scenario_type = scenario_type
        self.name = name
        self.description = description
        self.context = context
        self.expected_outcome = expected_outcome
        self.logger = logger
        
        # 场景配置
        self.timeout = 300.0  # 5分钟
        self.retry_count = 1
        self.setup_required = True
        self.teardown_required = True
        
        # 场景数据
        self.test_data: Dict[str, Any] = {}
        self.artifacts: List[str] = []
    
    @abstractmethod
    async def setup_scenario(self) -> bool:
        """设置场景"""
        pass
    
    @abstractmethod
    async def execute_scenario(self) -> Any:
        """执行场景"""
        pass
    
    @abstractmethod
    async def teardown_scenario(self) -> bool:
        """清理场景"""
        pass
    
    def create_test_case(self) -> TestCase:
        """创建测试用例"""
        config = TestConfig(
            test_name=self.name,
            test_type=TestType.FUNCTIONAL,
            priority=TestPriority.MEDIUM,
            timeout=self.timeout,
            retry_count=self.retry_count,
            setup_required=self.setup_required,
            teardown_required=self.teardown_required,
            tags=[self.scenario_type.value, self.context.app_type.value],
            metadata={
                'scenario_type': self.scenario_type.value,
                'context': self.context.to_dict(),
                'expected_outcome': {
                    'success_criteria': self.expected_outcome.success_criteria,
                    'failure_criteria': self.expected_outcome.failure_criteria,
                    'performance_criteria': self.expected_outcome.performance_criteria
                }
            }
        )
        
        return ScenarioTestCase(config, self)
    
    def set_test_data(self, key: str, value: Any) -> None:
        """设置测试数据"""
        self.test_data[key] = value
    
    def get_test_data(self, key: str, default: Any = None) -> Any:
        """获取测试数据"""
        return self.test_data.get(key, default)


class ScenarioTestCase(TestCase):
    """场景测试用例"""
    
    def __init__(self, config: TestConfig, scenario: BaseTestScenario):
        super().__init__(config)
        self.scenario = scenario
    
    async def setup(self) -> bool:
        """设置测试"""
        try:
            return await self.scenario.setup_scenario()
        except Exception as e:
            logger.error(f"Scenario setup failed: {e}")
            return False
    
    async def execute(self) -> bool:
        """执行测试"""
        try:
            result = await self.scenario.execute_scenario()
            
            # 验证结果
            is_valid, errors = self.scenario.expected_outcome.validate(result)
            
            if not is_valid:
                self.error_message = f"Validation failed: {', '.join(errors)}"
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Scenario execution failed: {e}")
            self.error_message = str(e)
            return False
    
    async def teardown(self) -> bool:
        """清理测试"""
        try:
            return await self.scenario.teardown_scenario()
        except Exception as e:
            logger.error(f"Scenario teardown failed: {e}")
            return False


class LoginScenario(BaseTestScenario):
    """登录场景"""
    
    def __init__(self, context: ScenarioContext, username: str, password: str):
        expected_outcome = ExpectedOutcome(
            success_criteria=[
                "User successfully logged in",
                "Main screen is displayed",
                "User profile is loaded"
            ],
            failure_criteria=[
                "Login failed with invalid credentials",
                "Network error during login",
                "App crashed during login"
            ],
            performance_criteria={
                "login_time": 5.0,  # 最大5秒
                "ui_response_time": 1.0  # UI响应时间1秒内
            }
        )
        
        super().__init__(
            ScenarioType.BASIC_INTERACTION,
            "User Login",
            "Test user login functionality",
            context,
            expected_outcome
        )
        
        self.username = username
        self.password = password
    
    async def setup_scenario(self) -> bool:
        """设置登录场景"""
        logger.info("Setting up login scenario")
        
        # 模拟应用启动
        await asyncio.sleep(0.5)
        
        # 检查应用是否在登录页面
        self.set_test_data("app_state", "login_screen")
        self.set_test_data("username", self.username)
        self.set_test_data("password", self.password)
        
        return True
    
    async def execute_scenario(self) -> Any:
        """执行登录场景"""
        logger.info(f"Executing login for user: {self.username}")
        
        start_time = time.time()
        
        # 模拟登录步骤
        steps = [
            ("Find username field", 0.2),
            ("Enter username", 0.3),
            ("Find password field", 0.2),
            ("Enter password", 0.3),
            ("Tap login button", 0.5),
            ("Wait for response", 2.0),
            ("Verify login success", 0.5)
        ]
        
        for step_name, duration in steps:
            logger.debug(f"Executing step: {step_name}")
            await asyncio.sleep(duration)
        
        login_time = time.time() - start_time
        
        # 模拟登录结果
        if self.username == "invalid_user":
            raise Exception("Invalid credentials")
        
        result = {
            "success": True,
            "login_time": login_time,
            "user_id": f"user_{hash(self.username) % 10000}",
            "session_token": f"token_{int(time.time())}",
            "user_profile": {
                "username": self.username,
                "display_name": f"User {self.username}",
                "avatar_url": "https://example.com/avatar.jpg"
            }
        }
        
        self.set_test_data("login_result", result)
        return result
    
    async def teardown_scenario(self) -> bool:
        """清理登录场景"""
        logger.info("Tearing down login scenario")
        
        # 清理会话数据
        self.test_data.clear()
        
        return True


class SearchScenario(BaseTestScenario):
    """搜索场景"""
    
    def __init__(self, context: ScenarioContext, search_query: str):
        expected_outcome = ExpectedOutcome(
            success_criteria=[
                "Search results are displayed",
                "Results are relevant to query",
                "Search completed within time limit"
            ],
            performance_criteria={
                "search_time": 3.0,
                "result_count": 10
            }
        )
        
        super().__init__(
            ScenarioType.BASIC_INTERACTION,
            "Search Functionality",
            "Test search functionality",
            context,
            expected_outcome
        )
        
        self.search_query = search_query
    
    async def setup_scenario(self) -> bool:
        """设置搜索场景"""
        logger.info("Setting up search scenario")
        
        # 确保在主页面
        self.set_test_data("app_state", "main_screen")
        self.set_test_data("search_query", self.search_query)
        
        return True
    
    async def execute_scenario(self) -> Any:
        """执行搜索场景"""
        logger.info(f"Executing search for: {self.search_query}")
        
        start_time = time.time()
        
        # 模拟搜索步骤
        await asyncio.sleep(0.3)  # 找到搜索框
        await asyncio.sleep(0.5)  # 输入搜索词
        await asyncio.sleep(0.2)  # 点击搜索按钮
        await asyncio.sleep(1.5)  # 等待搜索结果
        
        search_time = time.time() - start_time
        
        # 模拟搜索结果
        result_count = random.randint(5, 20)
        results = []
        
        for i in range(result_count):
            results.append({
                "id": f"result_{i}",
                "title": f"Result {i} for {self.search_query}",
                "description": f"Description for result {i}",
                "relevance_score": random.uniform(0.6, 1.0)
            })
        
        search_result = {
            "success": True,
            "search_time": search_time,
            "query": self.search_query,
            "result_count": result_count,
            "results": results
        }
        
        self.set_test_data("search_result", search_result)
        return search_result
    
    async def teardown_scenario(self) -> bool:
        """清理搜索场景"""
        logger.info("Tearing down search scenario")
        return True


class PurchaseScenario(BaseTestScenario):
    """购买场景"""
    
    def __init__(self, context: ScenarioContext, product_id: str, quantity: int = 1):
        expected_outcome = ExpectedOutcome(
            success_criteria=[
                "Product added to cart",
                "Checkout process completed",
                "Payment processed successfully",
                "Order confirmation received"
            ],
            failure_criteria=[
                "Product out of stock",
                "Payment failed",
                "Network error during checkout"
            ],
            performance_criteria={
                "checkout_time": 30.0,
                "payment_processing_time": 10.0
            }
        )
        
        super().__init__(
            ScenarioType.COMPLEX_WORKFLOW,
            "Product Purchase",
            "Test complete purchase workflow",
            context,
            expected_outcome
        )
        
        self.product_id = product_id
        self.quantity = quantity
    
    async def setup_scenario(self) -> bool:
        """设置购买场景"""
        logger.info("Setting up purchase scenario")
        
        # 确保用户已登录
        self.set_test_data("user_logged_in", True)
        self.set_test_data("product_id", self.product_id)
        self.set_test_data("quantity", self.quantity)
        
        return True
    
    async def execute_scenario(self) -> Any:
        """执行购买场景"""
        logger.info(f"Executing purchase for product: {self.product_id}")
        
        start_time = time.time()
        
        # 模拟购买流程
        steps = [
            ("Navigate to product page", 1.0),
            ("Select quantity", 0.5),
            ("Add to cart", 0.8),
            ("Go to cart", 0.5),
            ("Proceed to checkout", 1.0),
            ("Enter shipping info", 2.0),
            ("Select payment method", 1.0),
            ("Process payment", 3.0),
            ("Confirm order", 1.0)
        ]
        
        for step_name, duration in steps:
            logger.debug(f"Executing step: {step_name}")
            await asyncio.sleep(duration)
            
            # 模拟可能的失败
            if step_name == "Process payment" and random.random() < 0.1:  # 10%失败率
                raise Exception("Payment processing failed")
        
        total_time = time.time() - start_time
        
        # 生成订单结果
        order_id = f"order_{int(time.time())}_{random.randint(1000, 9999)}"
        
        result = {
            "success": True,
            "order_id": order_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "total_price": random.uniform(10.0, 100.0),
            "checkout_time": total_time,
            "payment_method": "credit_card",
            "shipping_address": "123 Test Street, Test City",
            "estimated_delivery": "3-5 business days"
        }
        
        self.set_test_data("purchase_result", result)
        return result
    
    async def teardown_scenario(self) -> bool:
        """清理购买场景"""
        logger.info("Tearing down purchase scenario")
        
        # 清理购物车
        self.set_test_data("cart_cleared", True)
        
        return True


class ErrorHandlingScenario(BaseTestScenario):
    """错误处理场景"""
    
    def __init__(self, context: ScenarioContext, error_type: str):
        expected_outcome = ExpectedOutcome(
            success_criteria=[
                "Error is handled gracefully",
                "User receives appropriate feedback",
                "App remains stable",
                "Recovery option is provided"
            ],
            failure_criteria=[
                "App crashes",
                "No error feedback",
                "Data loss occurs"
            ]
        )
        
        super().__init__(
            ScenarioType.ERROR_HANDLING,
            f"Error Handling - {error_type}",
            f"Test error handling for {error_type}",
            context,
            expected_outcome
        )
        
        self.error_type = error_type
    
    async def setup_scenario(self) -> bool:
        """设置错误处理场景"""
        logger.info(f"Setting up error handling scenario for: {self.error_type}")
        
        self.set_test_data("error_type", self.error_type)
        self.set_test_data("error_injected", False)
        
        return True
    
    async def execute_scenario(self) -> Any:
        """执行错误处理场景"""
        logger.info(f"Executing error handling for: {self.error_type}")
        
        # 模拟不同类型的错误
        if self.error_type == "network_error":
            await self._simulate_network_error()
        elif self.error_type == "server_error":
            await self._simulate_server_error()
        elif self.error_type == "invalid_input":
            await self._simulate_invalid_input()
        elif self.error_type == "timeout":
            await self._simulate_timeout()
        else:
            await self._simulate_generic_error()
        
        # 检查错误处理结果
        result = {
            "error_type": self.error_type,
            "error_handled": True,
            "app_stable": True,
            "user_feedback_provided": True,
            "recovery_option_available": True,
            "error_message": f"Simulated {self.error_type} handled successfully"
        }
        
        self.set_test_data("error_handling_result", result)
        return result
    
    async def teardown_scenario(self) -> bool:
        """清理错误处理场景"""
        logger.info("Tearing down error handling scenario")
        
        # 恢复正常状态
        self.set_test_data("error_recovered", True)
        
        return True
    
    async def _simulate_network_error(self) -> None:
        """模拟网络错误"""
        logger.debug("Simulating network error")
        await asyncio.sleep(1.0)
        # 模拟网络超时和重试
        await asyncio.sleep(0.5)
    
    async def _simulate_server_error(self) -> None:
        """模拟服务器错误"""
        logger.debug("Simulating server error")
        await asyncio.sleep(0.8)
        # 模拟服务器500错误处理
    
    async def _simulate_invalid_input(self) -> None:
        """模拟无效输入"""
        logger.debug("Simulating invalid input")
        await asyncio.sleep(0.3)
        # 模拟输入验证错误
    
    async def _simulate_timeout(self) -> None:
        """模拟超时"""
        logger.debug("Simulating timeout")
        await asyncio.sleep(2.0)
        # 模拟请求超时
    
    async def _simulate_generic_error(self) -> None:
        """模拟通用错误"""
        logger.debug("Simulating generic error")
        await asyncio.sleep(0.5)


class PerformanceTestScenario(BaseTestScenario):
    """性能测试场景"""
    
    def __init__(self, context: ScenarioContext, operation_count: int = 100):
        expected_outcome = ExpectedOutcome(
            performance_criteria={
                "avg_response_time": 1.0,  # 平均响应时间1秒
                "max_response_time": 5.0,  # 最大响应时间5秒
                "throughput": 50.0,        # 每秒50个操作
                "error_rate": 0.05         # 错误率5%以下
            }
        )
        
        super().__init__(
            ScenarioType.PERFORMANCE_TEST,
            "Performance Test",
            "Test system performance under load",
            context,
            expected_outcome
        )
        
        self.operation_count = operation_count
    
    async def setup_scenario(self) -> bool:
        """设置性能测试场景"""
        logger.info(f"Setting up performance test with {self.operation_count} operations")
        
        self.set_test_data("operation_count", self.operation_count)
        self.set_test_data("operations_completed", 0)
        self.set_test_data("response_times", [])
        self.set_test_data("errors", [])
        
        return True
    
    async def execute_scenario(self) -> Any:
        """执行性能测试场景"""
        logger.info(f"Executing performance test with {self.operation_count} operations")
        
        start_time = time.time()
        response_times = []
        errors = []
        
        # 执行多个操作
        for i in range(self.operation_count):
            operation_start = time.time()
            
            try:
                # 模拟操作
                await self._simulate_operation(i)
                
                operation_time = time.time() - operation_start
                response_times.append(operation_time)
                
                # 模拟错误
                if random.random() < 0.02:  # 2%错误率
                    errors.append(f"Error in operation {i}")
                
            except Exception as e:
                errors.append(str(e))
            
            # 更新进度
            if (i + 1) % 10 == 0:
                logger.debug(f"Completed {i + 1}/{self.operation_count} operations")
        
        total_time = time.time() - start_time
        
        # 计算性能指标
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        throughput = self.operation_count / total_time if total_time > 0 else 0
        error_rate = len(errors) / self.operation_count if self.operation_count > 0 else 0
        
        result = {
            "operation_count": self.operation_count,
            "total_time": total_time,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "throughput": throughput,
            "error_rate": error_rate,
            "error_count": len(errors),
            "response_times": response_times,
            "errors": errors
        }
        
        self.set_test_data("performance_result", result)
        return result
    
    async def teardown_scenario(self) -> bool:
        """清理性能测试场景"""
        logger.info("Tearing down performance test scenario")
        return True
    
    async def _simulate_operation(self, operation_id: int) -> None:
        """模拟单个操作"""
        # 模拟不同复杂度的操作
        complexity = random.choice(["simple", "medium", "complex"])
        
        if complexity == "simple":
            await asyncio.sleep(random.uniform(0.1, 0.3))
        elif complexity == "medium":
            await asyncio.sleep(random.uniform(0.3, 0.8))
        else:  # complex
            await asyncio.sleep(random.uniform(0.8, 2.0))


class ScenarioFactory:
    """场景工厂"""
    
    @staticmethod
    def create_login_scenario(app_type: AppType, username: str, password: str) -> LoginScenario:
        """创建登录场景"""
        context = ScenarioContext(
            app_type=app_type,
            app_name=f"{app_type.value}_app",
            user_profile={"username": username}
        )
        return LoginScenario(context, username, password)
    
    @staticmethod
    def create_search_scenario(app_type: AppType, search_query: str) -> SearchScenario:
        """创建搜索场景"""
        context = ScenarioContext(
            app_type=app_type,
            app_name=f"{app_type.value}_app"
        )
        return SearchScenario(context, search_query)
    
    @staticmethod
    def create_purchase_scenario(product_id: str, quantity: int = 1) -> PurchaseScenario:
        """创建购买场景"""
        context = ScenarioContext(
            app_type=AppType.E_COMMERCE,
            app_name="ecommerce_app"
        )
        return PurchaseScenario(context, product_id, quantity)
    
    @staticmethod
    def create_error_handling_scenario(app_type: AppType, error_type: str) -> ErrorHandlingScenario:
        """创建错误处理场景"""
        context = ScenarioContext(
            app_type=app_type,
            app_name=f"{app_type.value}_app"
        )
        return ErrorHandlingScenario(context, error_type)
    
    @staticmethod
    def create_performance_test_scenario(app_type: AppType, operation_count: int = 100) -> PerformanceTestScenario:
        """创建性能测试场景"""
        context = ScenarioContext(
            app_type=app_type,
            app_name=f"{app_type.value}_app"
        )
        return PerformanceTestScenario(context, operation_count)
    
    @staticmethod
    def create_scenario_suite(app_type: AppType) -> List[BaseTestScenario]:
        """创建场景套件"""
        scenarios = []
        
        # 基础场景
        scenarios.append(ScenarioFactory.create_login_scenario(app_type, "test_user", "password123"))
        scenarios.append(ScenarioFactory.create_search_scenario(app_type, "test query"))
        
        # 错误处理场景
        error_types = ["network_error", "server_error", "invalid_input", "timeout"]
        for error_type in error_types:
            scenarios.append(ScenarioFactory.create_error_handling_scenario(app_type, error_type))
        
        # 性能测试场景
        scenarios.append(ScenarioFactory.create_performance_test_scenario(app_type, 50))
        
        # 电商特定场景
        if app_type == AppType.E_COMMERCE:
            scenarios.append(ScenarioFactory.create_purchase_scenario("product_123", 2))
        
        return scenarios


class ScenarioManager:
    """场景管理器"""
    
    def __init__(self):
        self.scenarios: Dict[str, BaseTestScenario] = {}
        self.scenario_suites: Dict[str, List[BaseTestScenario]] = {}
        self.logger = logger
    
    def register_scenario(self, scenario: BaseTestScenario) -> None:
        """注册场景"""
        self.scenarios[scenario.name] = scenario
        logger.info(f"Registered scenario: {scenario.name}")
    
    def register_scenario_suite(self, suite_name: str, scenarios: List[BaseTestScenario]) -> None:
        """注册场景套件"""
        self.scenario_suites[suite_name] = scenarios
        for scenario in scenarios:
            self.register_scenario(scenario)
        logger.info(f"Registered scenario suite: {suite_name} with {len(scenarios)} scenarios")
    
    def get_scenario(self, name: str) -> Optional[BaseTestScenario]:
        """获取场景"""
        return self.scenarios.get(name)
    
    def get_scenario_suite(self, suite_name: str) -> List[BaseTestScenario]:
        """获取场景套件"""
        return self.scenario_suites.get(suite_name, [])
    
    def list_scenarios(self) -> List[str]:
        """列出所有场景"""
        return list(self.scenarios.keys())
    
    def list_scenario_suites(self) -> List[str]:
        """列出所有场景套件"""
        return list(self.scenario_suites.keys())
    
    def create_test_cases_from_scenarios(self, scenario_names: List[str]) -> List[TestCase]:
        """从场景创建测试用例"""
        test_cases = []
        
        for name in scenario_names:
            scenario = self.get_scenario(name)
            if scenario:
                test_case = scenario.create_test_case()
                test_cases.append(test_case)
            else:
                logger.warning(f"Scenario not found: {name}")
        
        return test_cases
    
    def create_test_cases_from_suite(self, suite_name: str) -> List[TestCase]:
        """从场景套件创建测试用例"""
        scenarios = self.get_scenario_suite(suite_name)
        scenario_names = [scenario.name for scenario in scenarios]
        return self.create_test_cases_from_scenarios(scenario_names)
    
    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[BaseTestScenario]:
        """根据类型获取场景"""
        return [scenario for scenario in self.scenarios.values() 
                if scenario.scenario_type == scenario_type]
    
    def get_scenarios_by_app_type(self, app_type: AppType) -> List[BaseTestScenario]:
        """根据应用类型获取场景"""
        return [scenario for scenario in self.scenarios.values() 
                if scenario.context.app_type == app_type]
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            'total_scenarios': len(self.scenarios),
            'total_suites': len(self.scenario_suites),
            'scenario_types': list(set(scenario.scenario_type.value for scenario in self.scenarios.values())),
            'app_types': list(set(scenario.context.app_type.value for scenario in self.scenarios.values()))
        }