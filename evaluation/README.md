# AgenticX-GUIAgent Evaluation Module

## 概述

AgenticX-GUIAgent评估模块提供了基于AgenticX框架的完整测试环境和性能评估功能。该模块已重构以避免重复实现，充分利用AgenticX的核心组件。

## 重构说明

### 主要变更

1. **基于AgenticX Component**: 所有核心类都继承自`agenticx.core.component.Component`
2. **事件驱动架构**: 使用`agenticx.core.event.EventBus`进行组件间通信
3. **工具框架集成**: 使用`agenticx.core.tool.BaseTool`进行指标计算
4. **避免重复实现**: 移除与AgenticX重复的基础设施代码

### 核心组件

#### 1. TestEnvironment
- **功能**: 提供完整的测试环境和测试执行框架
- **基类**: `agenticx.core.component.Component`
- **特性**: 事件驱动的测试执行、资源管理、状态同步

#### 2. PerformanceEvaluator
- **功能**: 全面的性能评估和分析
- **基类**: `agenticx.core.component.Component`
- **特性**: 多维度指标计算、趋势分析、基线比较

#### 3. EvaluationFramework
- **功能**: 整合所有评估组件的统一接口
- **基类**: `agenticx.core.component.Component`
- **特性**: 工作流管理、评估协调、报告生成

## 使用示例

### 基本使用

```python
from agenticx.core.event_bus import EventBus
from evaluation import (
    EvaluationFramework,
    EvaluationConfig,
    EvaluationMode,
    TestConfig
)

# 创建事件总线
event_bus = EventBus()

# 创建评估框架
framework = EvaluationFramework(
    base_dir="./evaluation_workspace",
    event_bus=event_bus
)

# 配置评估
config = EvaluationConfig(
    name="agent_performance_test",
    description="AgenticX-GUIAgent智能体性能评估",
    mode=EvaluationMode.STANDARD,
    test_timeout=300.0,
    metrics=["accuracy", "response_time", "efficiency"]
)

# 运行评估
result = await framework.run_evaluation(config)
print(f"评估完成，成功率: {result.get_success_rate():.2%}")
```

### 性能评估

```python
from evaluation import (
    PerformanceEvaluator,
    MetricType,
    EvaluationLevel
)

# 创建性能评估器
evaluator = PerformanceEvaluator(
    output_dir="./performance_reports",
    event_bus=event_bus
)

# 运行性能评估
test_results = []  # 从测试环境获取的结果
report = await evaluator.evaluate(
    data=test_results,
    name="智能体性能评估",
    level=EvaluationLevel.COMPREHENSIVE,
    selected_metrics=[
        MetricType.ACCURACY,
        MetricType.RESPONSE_TIME,
        MetricType.EFFICIENCY
    ]
)

print(f"性能报告: {report.name}")
for metric in report.metrics:
    print(f"  {metric}")
```

### 测试环境

```python
from evaluation import (
    TestEnvironment,
    TestConfig,
    TestType,
    TestPriority
)

# 创建测试配置
test_config = TestConfig(
    test_name="gui_operation_test",
    test_type=TestType.FUNCTIONAL,
    priority=TestPriority.HIGH,
    timeout=120.0
)

# 创建测试环境
test_env = TestEnvironment(
    config=test_config,
    event_bus=event_bus
)

# 初始化并运行测试
await test_env.initialize()
results = await test_env.run_all_suites()
await test_env.cleanup()
```

## 指标类型

### 准确性指标
- `ACCURACY`: 准确率
- `PRECISION`: 精确率
- `RECALL`: 召回率
- `F1_SCORE`: F1分数

### 性能指标
- `RESPONSE_TIME`: 响应时间
- `THROUGHPUT`: 吞吐量
- `LATENCY`: 延迟
- `EFFICIENCY`: 效率

### 可靠性指标
- `SUCCESS_RATE`: 成功率
- `ERROR_RATE`: 错误率
- `RELIABILITY`: 可靠性
- `ROBUSTNESS`: 鲁棒性

## 评估模式

- `QUICK`: 快速评估 - 基本指标，短时间
- `STANDARD`: 标准评估 - 常用指标，中等时间
- `COMPREHENSIVE`: 全面评估 - 所有指标，长时间
- `PERFORMANCE`: 性能测试 - 专注性能指标
- `STRESS`: 压力测试 - 极限条件测试

## 事件系统

评估模块通过AgenticX事件系统进行通信：

### 发布的事件
- `evaluation_started`: 评估开始
- `evaluation_completed`: 评估完成
- `test_started`: 测试开始
- `test_completed`: 测试完成
- `metric_calculated`: 指标计算完成
- `report_generated`: 报告生成完成

### 订阅事件
```python
# 订阅评估事件
def on_evaluation_completed(event):
    result = event.data
    print(f"评估完成: {result['evaluation_id']}")

event_bus.subscribe("evaluation_completed", on_evaluation_completed)
```

## 扩展性

### 自定义指标

```python
from evaluation.performance_evaluator import BaseMetric, MetricValue

class CustomMetric(BaseMetric):
    def __init__(self):
        super().__init__(MetricType.CUSTOM, "自定义指标")
    
    async def calculate(self, data, **kwargs):
        # 实现自定义计算逻辑
        value = self._custom_calculation(data)
        return MetricValue(
            metric_type=self.metric_type,
            value=value,
            unit="custom_unit"
        )

# 注册自定义指标
evaluator.register_metric(CustomMetric())
```

### 自定义测试用例

```python
from evaluation.test_environment import TestCase

class CustomTestCase(TestCase):
    async def setup(self):
        # 测试前准备
        return True
    
    async def execute(self):
        # 执行测试逻辑
        return True
    
    async def teardown(self):
        # 测试后清理
        return True
```

## 最佳实践

1. **使用事件总线**: 通过EventBus进行组件间通信
2. **配置管理**: 使用EvaluationConfig统一管理评估配置
3. **资源清理**: 确保测试后正确清理资源
4. **指标选择**: 根据评估目标选择合适的指标
5. **基线设置**: 建立性能基线进行对比分析

## 注意事项

1. **依赖关系**: 确保AgenticX框架已正确安装
2. **资源管理**: 长时间评估可能消耗大量资源
3. **并发控制**: 注意并发测试的资源竞争
4. **数据持久化**: 重要的评估结果应及时保存

## 未来规划

1. **机器学习集成**: 使用ML模型进行智能评估
2. **云端评估**: 支持分布式云端评估
3. **实时监控**: 实时性能监控和告警
4. **自动优化**: 基于评估结果的自动优化建议