#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Tools Module
GUI操作工具集：基于AgenticX框架的移动设备操作工具

本模块已完全基于AgenticX框架重构，提供：
- 基于AgenticX BaseTool的统一工具接口
- 集成AgenticX Component和EventBus的事件驱动架构
- 支持AgenticX工具生态系统的标准化工具实现
- 与AgenticX embodiment模块的无缝集成

Author: AgenticX Team
Date: 2025
Version: 1.0.0 (基于AgenticX框架重构)
"""

__version__ = "1.0.0"
__author__ = "AgenticX Team"
__description__ = "基于AgenticX框架的GUI操作工具集，提供移动设备的原子操作能力"

# 导入AgenticX核心组件
from agenticx.tools.base import BaseTool, ToolError, ToolTimeoutError, ToolValidationError
from agenticx.core.component import Component
from agenticx.core.event_bus import EventBus
from agenticx.core.event import Event

# 导入核心工具组件
from .gui_tools import (
    GUIToolManager,
    GUITool,
    ToolResult,
    ToolType,
    ToolStatus,
    Coordinate,
    Rectangle,
    Platform
)

# 导入基础操作工具
from .basic_tools import (
    ClickTool,
    SwipeTool,
    TextInputTool,
    KeyPressTool,
    WaitTool
)

# 导入高级操作工具
from .advanced_tools import (
    ScreenshotTool,
    ElementDetectionTool,
    OCRTool,
    ImageComparisonTool
)

# 导入智能操作工具
from .smart_tools import (
    SmartClickTool,
    SmartScrollTool,
    SmartInputTool
)

# 导入工具适配器
from .tool_adapters import (
    ToolAdapter,
    AndroidAdapter,
    iOSAdapter,
    DesktopAdapter,
    AdapterFactory,
    AdaptedGUITool
)

# 导入工具执行器
from .tool_executor import (
    ToolExecutor,
    ExecutionTask,
    ExecutionBatch,
    ExecutionQueue,
    BatchExecutor
)

# 导入工具验证器
from .tool_validator import (
    ToolValidator,
    ValidationReport,
    ValidationResult,
    ValidationIssue
)

# 导入工具监控器
from .tool_monitor import (
    ToolMonitor,
    PerformanceMetric,
    MonitorEvent,
    Alert
)

__all__ = [
    # AgenticX核心组件
    'BaseTool',
    'ToolError',
    'ToolTimeoutError', 
    'ToolValidationError',
    'Component',
    'EventBus',
    'Event',
    
    # 核心组件
    'GUIToolManager',
    'GUITool',
    'ToolResult',
    'ToolType',
    'ToolStatus',
    'Coordinate',
    'Rectangle',
    'Platform',
    
    # 基础操作
    'ClickTool',
    'SwipeTool',
    'TextInputTool',
    'KeyPressTool',
    'WaitTool',
    
    # 高级操作
    'ScreenshotTool',
    'ElementDetectionTool',
    'OCRTool',
    'ImageComparisonTool',
    
    # 智能操作
    'SmartClickTool',
    'SmartScrollTool',
    'SmartInputTool',
    
    # 适配器
    'ToolAdapter',
    'AndroidAdapter',
    'iOSAdapter',
    'DesktopAdapter',
    'AdapterFactory',
    'AdaptedGUITool',
    
    # 执行器
    'ToolExecutor',
    'ExecutionTask',
    'ExecutionBatch',
    'ExecutionQueue',
    'BatchExecutor',
    
    # 验证器
    'ToolValidator',
    'ValidationReport',
    'ValidationResult',
    'ValidationIssue',
    
    # 监控器
    'ToolMonitor',
    'PerformanceMetric',
    'MonitorEvent',
    'Alert'
]

# 版本信息
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'components': {
        'gui_tools': 'GUI工具核心管理器',
        'basic_operations': '基础操作工具集',
        'advanced_operations': '高级操作工具集',
        'smart_operations': '智能操作工具集',
        'tool_adapters': '平台适配器',
        'tool_executor': '工具执行器',
        'tool_validator': '工具验证器',
        'tool_monitor': '工具监控器'
    }
}

# 支持的平台
SUPPORTED_PLATFORMS = {
    'android': 'Android移动设备',
    'ios': 'iOS移动设备',
    'web': 'Web浏览器',
    'universal': '通用平台'
}

# 工具类型映射
TOOL_TYPE_MAPPING = {
    'click': ClickTool,
    'swipe': SwipeTool,
    'type': TextInputTool,
    'key_press': KeyPressTool,
    'wait': WaitTool,
    'screenshot': ScreenshotTool,
    'element_detection': ElementDetectionTool,
    'ocr': OCRTool,
    'image_comparison': ImageComparisonTool,
    'smart_click': SmartClickTool,
    'smart_scroll': SmartScrollTool,
    'smart_input': SmartInputTool
}


def get_tool_by_type(tool_type: str) -> type:
    """根据类型获取工具类"""
    return TOOL_TYPE_MAPPING.get(tool_type)


def get_supported_tools() -> list:
    """获取支持的工具列表"""
    return list(TOOL_TYPE_MAPPING.keys())


def get_version_info() -> dict:
    """获取版本信息"""
    return VERSION_INFO.copy()


def get_supported_platforms() -> dict:
    """获取支持的平台"""
    return SUPPORTED_PLATFORMS.copy()