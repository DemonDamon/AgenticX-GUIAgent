# AgenticX-GUIAgent Tools Module

基于AgenticX框架的GUI操作工具集

## 概述

AgenticX-GUIAgent Tools模块已完全基于AgenticX框架重构，提供了统一、标准化的移动设备GUI操作工具集。本模块展示了如何正确基于AgenticX框架构建专业的工具系统。

## 主要变更

### 🔄 架构重构

- **继承AgenticX BaseTool**: 所有工具类都继承自`agenticx.tools.base.BaseTool`
- **集成Component系统**: 管理器类基于`agenticx.core.component.Component`
- **事件驱动架构**: 使用`agenticx.core.event_bus.EventBus`实现事件通信
- **标准化参数验证**: 使用Pydantic模型进行参数验证
- **统一错误处理**: 基于AgenticX的错误处理机制

### 📦 核心组件

#### 1. **GUITool基类** (`gui_tools.py`)
```python
class GUITool(BaseTool):
    """GUI工具基类 - 基于AgenticX BaseTool"""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        platform: Platform = Platform.UNIVERSAL,
        tool_type: ToolType = ToolType.BASIC,
        timeout: Optional[float] = 30.0,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            description=description,
            args_schema=GUIToolParameters,
            timeout=timeout,
            **kwargs
        )
```

#### 2. **GUIToolManager** (`gui_tools.py`)
```python
class GUIToolManager(Component):
    """GUI工具管理器 - 基于AgenticX Component"""
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        enable_monitoring: bool = True,
        enable_caching: bool = True,
        **kwargs
    ):
        super().__init__(name="GUIToolManager", **kwargs)
```

#### 3. **基础工具集** (`basic_tools.py`)
- `ClickTool`: 点击操作工具
- `SwipeTool`: 滑动操作工具
- `TextInputTool`: 文本输入工具
- `KeyPressTool`: 按键操作工具
- `WaitTool`: 等待工具

#### 4. **高级工具集** (`advanced_tools.py`)
- `ScreenshotTool`: 截图工具
- `ElementDetectionTool`: 元素检测工具
- `OCRTool`: 文字识别工具
- `ImageComparisonTool`: 图像比较工具

#### 5. **智能工具集** (`smart_tools.py`)
- `SmartClickTool`: 智能点击工具
- `SmartScrollTool`: 智能滚动工具
- `SmartInputTool`: 智能输入工具

#### 6. **平台适配器** (`tool_adapters.py`)
- `AndroidAdapter`: Android设备适配器
- `iOSAdapter`: iOS设备适配器
- `DesktopAdapter`: 桌面平台适配器
- `AdapterFactory`: 适配器工厂

#### 7. **执行管理** (`tool_executor.py`)
- `ToolExecutor`: 工具执行器
- `ExecutionQueue`: 执行队列
- `BatchExecutor`: 批量执行器

#### 8. **验证系统** (`tool_validator.py`)
- `ToolValidator`: 工具验证器
- `ParameterValidator`: 参数验证器
- `ResultValidator`: 结果验证器

#### 9. **监控系统** (`tool_monitor.py`)
- `ToolMonitor`: 工具监控器
- `MetricCollector`: 指标收集器
- `AlertManager`: 告警管理器

#### 10. **通信模块** (`communication.py`)
- `InfoPool`: 信息池（基于AgenticX Component）

## 使用示例

### 基本使用

```python
from agenticx.core.event_bus import EventBus
from tools import (
    GUIToolManager, ClickTool, ToolParameters, Coordinate
)

# 创建事件总线
event_bus = EventBus()

# 创建工具管理器
tool_manager = GUIToolManager(event_bus=event_bus)
await tool_manager.initialize()

# 创建点击工具
click_tool = ClickTool(event_bus=event_bus)

# 注册工具
tool_manager.register_tool(click_tool)

# 执行点击操作
result = await click_tool.arun(
    target=Coordinate(x=100, y=200),
    wait_after=1.0
)

print(f"点击结果: {result}")
```

### 事件监听

```python
# 订阅工具事件
def on_tool_executed(event):
    print(f"工具执行完成: {event.data}")

event_bus.subscribe('tool_execution_end', on_tool_executed)
```

### 批量执行

```python
from tools import BatchExecutor

batch_executor = BatchExecutor(tool_manager)

# 顺序执行多个操作
tasks = [
    (click_tool, {'target': Coordinate(x=100, y=200)}, None),
    (swipe_tool, {'start': Coordinate(x=100, y=300), 'end': Coordinate(x=100, y=100)}, None)
]

results = await batch_executor.execute_sequential(tasks)
```

### 智能工具使用

```python
from tools import SmartClickTool

smart_click = SmartClickTool(event_bus=event_bus)

# 通过文本智能点击
result = await smart_click.arun(
    text="登录按钮",
    validate=True,
    screenshot=True
)
```

## 事件系统

### 工具事件

- `tool_execution_start`: 工具开始执行
- `tool_execution_end`: 工具执行结束
- `tool_success`: 工具执行成功
- `tool_failure`: 工具执行失败
- `tool_timeout`: 工具执行超时
- `tool_retry`: 工具重试

### 管理器事件

- `manager_started`: 管理器启动
- `manager_stopped`: 管理器停止
- `tool_executed`: 工具执行完成

### 信息池事件

- `info_added`: 信息添加
- `entries_cleaned`: 条目清理

## 平台支持

- **Android**: 通过ADB和UI Automator
- **iOS**: 通过WebDriverAgent
- **Desktop**: 通过系统API
- **Web**: 通过浏览器自动化

## 配置选项

### 工具配置

```python
tool = ClickTool(
    timeout=30.0,           # 超时时间
    retry_count=3,          # 重试次数
    enable_validation=True, # 启用参数验证
    enable_screenshot=True, # 启用截图
    event_bus=event_bus    # 事件总线
)
```

### 管理器配置

```python
manager = GUIToolManager(
    event_bus=event_bus,
    enable_monitoring=True,  # 启用监控
    enable_caching=True,     # 启用缓存
    max_concurrent_tasks=10  # 最大并发任务数
)
```

## 性能优化

### 缓存机制

- 结果缓存：缓存工具执行结果
- 元素缓存：缓存UI元素检测结果
- 截图缓存：缓存屏幕截图

### 并发执行

- 异步执行：支持异步工具执行
- 批量处理：支持批量任务执行
- 队列管理：智能任务队列调度

### 监控指标

- 执行时间：工具执行耗时统计
- 成功率：工具执行成功率
- 错误分析：详细的错误分类和统计

## 最佳实践

### 1. 事件驱动设计

```python
# 使用事件总线进行组件间通信
event_bus = EventBus()

# 所有组件共享同一个事件总线
tool_manager = GUIToolManager(event_bus=event_bus)
tools = [ClickTool(event_bus=event_bus), SwipeTool(event_bus=event_bus)]
```

### 2. 错误处理

```python
try:
    result = await tool.arun(**params)
except GUIToolError as e:
    print(f"工具错误: {e.message}")
except ToolTimeoutError as e:
    print(f"执行超时: {e.message}")
```

### 3. 参数验证

```python
# 使用Pydantic模型进行参数验证
class CustomToolParameters(GUIToolParameters):
    custom_field: str = Field(..., description="自定义字段")
    
class CustomTool(GUITool):
    def __init__(self, **kwargs):
        super().__init__(
            args_schema=CustomToolParameters,
            **kwargs
        )
```

### 4. 资源管理

```python
# 正确的生命周期管理
async with tool_manager:
    # 使用工具管理器
    await tool_manager.execute_tool(tool_id, parameters)
# 自动清理资源
```

## 扩展开发

### 自定义工具

```python
class CustomTool(GUITool):
    def __init__(self, **kwargs):
        super().__init__(
            name="CustomTool",
            description="自定义工具",
            tool_type=ToolType.CUSTOM,
            **kwargs
        )
    
    async def execute_gui_tool(self, parameters, context=None):
        # 实现自定义逻辑
        return ToolResult(
            tool_id=self.tool_id,
            tool_type=self.tool_type.value,
            status=ToolStatus.COMPLETED,
            success=True,
            start_time=get_iso_timestamp(),
            end_time=get_iso_timestamp()
        )
```

### 自定义适配器

```python
class CustomAdapter(ToolAdapter):
    def __init__(self):
        super().__init__(
            platform=Platform.CUSTOM,
            adapter_name="CustomAdapter"
        )
    
    async def initialize(self):
        # 初始化适配器
        return True
    
    async def click(self, x, y, **kwargs):
        # 实现点击逻辑
        return {'success': True}
```

## 注意事项

1. **事件总线**: 确保所有组件共享同一个EventBus实例
2. **资源清理**: 使用完毕后正确清理工具和管理器资源
3. **错误处理**: 妥善处理各种工具执行异常
4. **性能监控**: 启用监控以跟踪工具性能
5. **参数验证**: 使用Pydantic模型确保参数正确性

## 未来规划

- **AI增强**: 集成AI能力提升工具智能化水平
- **云端执行**: 支持云端工具执行和管理
- **可视化界面**: 提供工具执行的可视化监控界面
- **插件系统**: 支持第三方工具插件扩展
- **性能优化**: 持续优化工具执行性能和资源使用

---

本模块展示了如何正确基于AgenticX框架构建专业的工具系统，为AgenticX-GUIAgent项目提供了强大的GUI操作能力。