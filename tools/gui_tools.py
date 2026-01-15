#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent GUI Tools Core
GUI工具核心：基于AgenticX框架的工具基础架构

本模块已完全基于AgenticX框架重构：
- 继承AgenticX BaseTool提供统一工具接口
- 集成AgenticX Component和EventBus实现事件驱动
- 使用AgenticX标准化的错误处理和参数验证
- 与AgenticX工具生态系统完全兼容

Author: AgenticX Team
Date: 2025
Version: 1.0.0 (基于AgenticX框架重构)
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Callable, Awaitable, Type
)
from uuid import uuid4
from loguru import logger

# AgenticX核心组件
from agenticx.tools.base import BaseTool, ToolError, ToolTimeoutError, ToolValidationError
from agenticx.core.component import Component
from agenticx.core.event_bus import EventBus
from agenticx.core.event import Event
from pydantic import BaseModel, Field

# AgenticX-GUIAgent工具
from utils import get_iso_timestamp, setup_logger


class ToolType(Enum):
    """工具类型"""
    BASIC = "basic"              # 基础操作
    ADVANCED = "advanced"        # 高级操作
    SMART = "smart"              # 智能操作
    COMPOSITE = "composite"      # 复合操作
    CUSTOM = "custom"            # 自定义操作


class ToolStatus(Enum):
    """工具状态"""
    IDLE = "idle"                # 空闲
    RUNNING = "running"          # 运行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消
    TIMEOUT = "timeout"          # 超时


class Platform(Enum):
    """平台类型"""
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"
    UNIVERSAL = "universal"


class ExecutionMode(Enum):
    """执行模式"""
    SYNC = "sync"                # 同步执行
    ASYNC = "async"              # 异步执行
    BATCH = "batch"              # 批量执行
    PIPELINE = "pipeline"        # 流水线执行


@dataclass
class Coordinate:
    """坐标"""
    x: float
    y: float
    
    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
    
    def distance_to(self, other: 'Coordinate') -> float:
        """计算到另一个坐标的距离"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def to_dict(self) -> Dict[str, float]:
        return {'x': self.x, 'y': self.y}


@dataclass
class Rectangle:
    """矩形区域"""
    left: float
    top: float
    right: float
    bottom: float
    
    def __post_init__(self):
        self.left = float(self.left)
        self.top = float(self.top)
        self.right = float(self.right)
        self.bottom = float(self.bottom)
    
    @property
    def width(self) -> float:
        return self.right - self.left
    
    @property
    def height(self) -> float:
        return self.bottom - self.top
    
    @property
    def center(self) -> Coordinate:
        return Coordinate(
            (self.left + self.right) / 2,
            (self.top + self.bottom) / 2
        )
    
    def contains(self, point: Coordinate) -> bool:
        """检查是否包含指定点"""
        return (
            self.left <= point.x <= self.right and
            self.top <= point.y <= self.bottom
        )
    
    def intersects(self, other: 'Rectangle') -> bool:
        """检查是否与另一个矩形相交"""
        return not (
            self.right < other.left or
            other.right < self.left or
            self.bottom < other.top or
            other.bottom < self.top
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'left': self.left,
            'top': self.top,
            'right': self.right,
            'bottom': self.bottom
        }


@dataclass
class ToolParameters:
    """工具参数"""
    target: Optional[Union[Coordinate, Rectangle, str]] = None
    text: Optional[str] = None
    duration: Optional[float] = None
    force: Optional[float] = None
    direction: Optional[str] = None
    distance: Optional[float] = None
    speed: Optional[float] = None
    timeout: Optional[float] = None
    retry_count: Optional[int] = None
    wait_before: Optional[float] = None
    wait_after: Optional[float] = None
    validate: Optional[bool] = None
    screenshot: Optional[bool] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolParameters':
        """从字典创建"""
        params = cls()
        for key, value in data.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_id: str
    tool_type: str
    status: ToolStatus
    success: bool
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    screenshot_path: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            start = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
            self.duration = (end - start).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'tool_id': self.tool_id,
            'tool_type': self.tool_type,
            'status': self.status.value,
            'success': self.success,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'result_data': self.result_data,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'screenshot_path': self.screenshot_path,
            'validation_results': self.validation_results,
            'performance_metrics': self.performance_metrics,
            'metadata': self.metadata
        }


# 使用AgenticX的ToolError，这里定义GUI特定的错误类
class GUIToolError(ToolError):
    """GUI工具特定错误"""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        error_code: str = "GUI_TOOL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, tool_name, details)
        self.error_code = error_code
        self.timestamp = get_iso_timestamp()


class GUIToolParameters(BaseModel):
    """GUI工具参数基类"""
    target: Optional[Union[str, Dict[str, Any]]] = Field(None, description="操作目标")
    text: Optional[str] = Field(None, description="文本内容")
    duration: Optional[float] = Field(None, description="操作持续时间")
    force: Optional[float] = Field(None, description="操作力度")
    direction: Optional[str] = Field(None, description="操作方向")
    distance: Optional[float] = Field(None, description="操作距离")
    speed: Optional[float] = Field(None, description="操作速度")
    wait_before: Optional[float] = Field(None, description="执行前等待时间")
    wait_after: Optional[float] = Field(None, description="执行后等待时间")
    validation_enabled: Optional[bool] = Field(False, description="是否验证结果")
    screenshot: Optional[bool] = Field(False, description="是否截图")
    custom_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="自定义参数")


class GUITool(BaseTool):
    """GUI工具基类 - 基于AgenticX BaseTool"""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        platform: Platform = Platform.UNIVERSAL,
        tool_type: ToolType = ToolType.BASIC,
        timeout: Optional[float] = 30.0,
        retry_count: int = 3,
        enable_validation: bool = True,
        enable_screenshot: bool = False,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ):
        # 调用AgenticX BaseTool的初始化
        super().__init__(
            name=name,
            description=description,
            args_schema=GUIToolParameters,
            timeout=timeout,
            **kwargs
        )
        
        # GUI工具特定属性
        self.tool_id = str(uuid4())
        self.platform = platform
        self.tool_type = tool_type
        self.retry_count = retry_count
        self.enable_validation = enable_validation
        self.enable_screenshot = enable_screenshot
        self.event_bus = event_bus
        
        self.logger = logger
        self.status = ToolStatus.IDLE
        self.current_execution_id: Optional[str] = None
        self.execution_history: List[ToolResult] = []
        self.performance_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_duration': 0.0,
            'last_execution': None
        }
    
    def _run(self, **kwargs) -> Any:
        """同步执行工具 - 实现AgenticX BaseTool抽象方法"""
        # 将同步调用转换为异步调用
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建一个任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(**kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**kwargs))
        except RuntimeError:
            # 如果没有事件循环，创建新的
            return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> Any:
        """异步执行工具 - 重写AgenticX BaseTool方法"""
        # 转换参数格式
        parameters = ToolParameters.from_dict(kwargs)
        context = kwargs.get('context')
        
        # 执行GUI工具逻辑
        return await self.execute_gui_tool(parameters, context)
    
    @abstractmethod
    async def execute_gui_tool(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行GUI工具操作 - 子类必须实现"""
        pass
    
    async def validate_gui_parameters(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证GUI工具参数 - 子类可重写"""
        return True
    
    async def execute_with_retry(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """带重试的执行"""
        execution_id = str(uuid4())
        self.current_execution_id = execution_id
        
        start_time = get_iso_timestamp()
        result = None
        last_error = None
        
        try:
            # 触发执行前事件
            await self._publish_event('tool_execution_start', {
                'execution_id': execution_id,
                'tool_id': self.tool_id,
                'tool_name': self.name,
                'parameters': parameters.to_dict() if hasattr(parameters, 'to_dict') else str(parameters),
                'context': context
            })
            
            # 验证参数
            if self.enable_validation:
                if not await self.validate_gui_parameters(parameters, context):
                    raise GUIToolError(
                        "Parameter validation failed",
                        self.name,
                        "VALIDATION_ERROR"
                    )
            
            # 执行重试逻辑
            for attempt in range(self.retry_count + 1):
                try:
                    self.status = ToolStatus.RUNNING
                    
                    # 执行工具
                    result = await asyncio.wait_for(
                        self.execute_gui_tool(parameters, context),
                        timeout=self.timeout
                    )
                    
                    if result.success:
                        self.status = ToolStatus.COMPLETED
                        break
                    else:
                        last_error = GUIToolError(
                            result.error_message or "Execution failed",
                            self.name,
                            result.error_code or "EXECUTION_ERROR"
                        )
                        
                        if attempt < self.retry_count:
                            await self._publish_event('tool_retry', {
                                'execution_id': execution_id,
                                'tool_id': self.tool_id,
                                'tool_name': self.name,
                                'attempt': attempt + 1,
                                'error': str(last_error)
                            })
                            await asyncio.sleep(min(2 ** attempt, 10))  # 指数退避
                
                except asyncio.TimeoutError:
                    self.status = ToolStatus.TIMEOUT
                    last_error = GUIToolError(
                        f"Tool execution timeout after {self.timeout}s",
                        self.name,
                        "TIMEOUT_ERROR"
                    )
                    
                    await self._publish_event('tool_timeout', {
                        'execution_id': execution_id,
                        'tool_id': self.tool_id,
                        'tool_name': self.name,
                        'timeout': self.timeout
                    })
                    
                    if attempt < self.retry_count:
                        await self._publish_event('tool_retry', {
                            'execution_id': execution_id,
                            'tool_id': self.tool_id,
                            'tool_name': self.name,
                            'attempt': attempt + 1,
                            'error': str(last_error)
                        })
                        await asyncio.sleep(min(2 ** attempt, 10))
                
                except Exception as e:
                    last_error = GUIToolError(
                        str(e),
                        self.name,
                        "EXECUTION_ERROR",
                        {'exception_type': type(e).__name__}
                    )
                    
                    if attempt < self.retry_count:
                        await self._publish_event('tool_retry', {
                            'execution_id': execution_id,
                            'tool_id': self.tool_id,
                            'tool_name': self.name,
                            'attempt': attempt + 1,
                            'error': str(last_error)
                        })
                        await asyncio.sleep(min(2 ** attempt, 10))
            
            # 如果所有重试都失败了
            if not result or not result.success:
                self.status = ToolStatus.FAILED
                
                if not result:
                    result = ToolResult(
                        tool_id=self.tool_id,
                        tool_type=self.tool_type.value,
                        status=self.status,
                        success=False,
                        start_time=start_time,
                        end_time=get_iso_timestamp(),
                        error_message=last_error.message if last_error else "Unknown error",
                        error_code=last_error.error_code if last_error else "UNKNOWN_ERROR"
                    )
                
                await self._publish_event('tool_failure', {
                    'execution_id': execution_id,
                    'tool_id': self.tool_id,
                    'tool_name': self.name,
                    'result': result.to_dict() if result and hasattr(result, 'to_dict') else str(result),
                    'error': str(last_error) if last_error else 'Unknown error'
                })
            else:
                await self._publish_event('tool_success', {
                    'execution_id': execution_id,
                    'tool_id': self.tool_id,
                    'tool_name': self.name,
                    'result': result.to_dict() if hasattr(result, 'to_dict') else str(result)
                })
            
            # 触发执行后事件
            await self._publish_event('tool_execution_end', {
                'execution_id': execution_id,
                'tool_id': self.tool_id,
                'tool_name': self.name,
                'result': result.to_dict() if hasattr(result, 'to_dict') else str(result)
            })
            
            # 更新统计信息
            self._update_performance_stats(result)
            
            # 保存执行历史
            self.execution_history.append(result)
            if len(self.execution_history) > 100:  # 限制历史记录数量
                self.execution_history.pop(0)
            
            return result
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            error_result = ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=self.status,
                success=False,
                start_time=start_time,
                end_time=get_iso_timestamp(),
                error_message=str(e),
                error_code="UNEXPECTED_ERROR"
            )
            
            self._update_performance_stats(error_result)
            self.execution_history.append(error_result)
            
            return error_result
        
        finally:
            self.current_execution_id = None
            if self.status == ToolStatus.RUNNING:
                self.status = ToolStatus.IDLE
    
    async def _publish_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """发布事件到AgenticX EventBus"""
        try:
            if self.event_bus:
                event = Event(
                    type=event_type,
                    data=data,
                    source=self.name,
                    timestamp=get_iso_timestamp()
                )
                await self.event_bus.publish_async(event)
        except Exception as e:
            logger.error(f"Error publishing event {event_type}: {e}")
    
    def subscribe_to_events(
        self,
        event_type: str,
        handler: Callable,
        async_handler: bool = True
    ) -> None:
        """订阅事件"""
        if self.event_bus:
            self.event_bus.subscribe(event_type, handler, async_handler)
    
    def unsubscribe_from_events(
        self,
        event_type: str,
        handler: Callable,
        async_handler: bool = True
    ) -> None:
        """取消订阅事件"""
        if self.event_bus:
            self.event_bus.unsubscribe(event_type, handler, async_handler)
    
    def _update_performance_stats(self, result: ToolResult) -> None:
        """更新性能统计"""
        try:
            self.performance_stats['total_executions'] += 1
            
            if result.success:
                self.performance_stats['successful_executions'] += 1
            else:
                self.performance_stats['failed_executions'] += 1
            
            if result.duration:
                total_duration = (
                    self.performance_stats['average_duration'] * 
                    (self.performance_stats['total_executions'] - 1) +
                    result.duration
                )
                self.performance_stats['average_duration'] = (
                    total_duration / self.performance_stats['total_executions']
                )
            
            self.performance_stats['last_execution'] = result.end_time
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def get_execution_history(
        self,
        limit: Optional[int] = None
    ) -> List[ToolResult]:
        """获取执行历史"""
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history.copy()
    
    def clear_execution_history(self) -> None:
        """清空执行历史"""
        self.execution_history.clear()
    
    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            'tool_id': self.tool_id,
            'name': self.name,
            'description': self.description,
            'platform': self.platform.value,
            'tool_type': self.tool_type.value,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'enable_validation': self.enable_validation,
            'enable_screenshot': self.enable_screenshot,
            'status': self.status.value,
            'performance_stats': self.performance_stats
        }


class GUIToolManager(Component):
    """GUI工具管理器 - 基于AgenticX Component"""
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        enable_monitoring: bool = True,
        enable_caching: bool = True,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name or "GUIToolManager", **kwargs)
        self.logger = logger
        self.event_bus = event_bus
        self.enable_monitoring = enable_monitoring
        self.enable_caching = enable_caching
        
        # 工具注册表
        self.tools: Dict[str, GUITool] = {}
        self.tool_types: Dict[str, Type[GUITool]] = {}
        
        # 执行管理
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_queue: List[Dict[str, Any]] = []
        self.max_concurrent_executions = 10
        
        # 缓存
        self.result_cache: Dict[str, Tuple[ToolResult, datetime]] = {}
        self.cache_ttl = 300  # 5分钟
        
        # 统计信息
        self.manager_stats = {
            'total_tools': 0,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'last_execution': None
        }
        
        # 运行状态
        self._running = False
        self._execution_task: Optional[asyncio.Task] = None
    
    async def _setup(self) -> None:
        """组件初始化设置"""
        self._running = True
        
        # 启动执行任务
        self._execution_task = asyncio.create_task(self._execution_loop())
        
        # 发布启动事件
        await self._publish_event('manager_started', {
            'timestamp': get_iso_timestamp()
        })
        
        logger.info("GUI tool manager initialized")
    
    async def cleanup(self) -> None:
        """组件清理"""
        self._running = False
        
        # 停止执行任务
        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有活跃执行
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_execution(execution_id)
        
        # 发布停止事件
        await self._publish_event('manager_stopped', {
            'timestamp': get_iso_timestamp()
        })
        
        logger.info("GUI tool manager cleaned up")
        await super().cleanup()
    
    def register_tool(self, tool: GUITool) -> bool:
        """注册工具"""
        try:
            if tool.tool_id in self.tools:
                logger.warning(f"Tool {tool.tool_id} already registered")
                return False
            
            self.tools[tool.tool_id] = tool
            self.tool_types[tool.name] = type(tool)
            
            # 设置工具的EventBus
            if self.event_bus and not tool.event_bus:
                tool.event_bus = self.event_bus
            
            # 订阅工具事件
            if tool.event_bus:
                tool.subscribe_to_events('tool_execution_end', self._on_tool_executed)
            
            # 更新统计
            self.manager_stats['total_tools'] += 1
            
            logger.info(f"Tool registered: {tool.name} ({tool.tool_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool: {e}")
            return False
    
    def unregister_tool(self, tool_id: str) -> bool:
        """取消注册工具"""
        try:
            if tool_id not in self.tools:
                return False
            
            tool = self.tools[tool_id]
            del self.tools[tool_id]
            
            # 从类型映射中移除
            if tool.name in self.tool_types:
                del self.tool_types[tool.name]
            
            # 更新统计
            self.manager_stats['total_tools'] = max(
                0, self.manager_stats['total_tools'] - 1
            )
            
            logger.info(f"Tool unregistered: {tool.name} ({tool_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister tool {tool_id}: {e}")
            return False
    
    def get_tool(self, tool_id: str) -> Optional[GUITool]:
        """获取工具"""
        return self.tools.get(tool_id)
    
    def get_tool_by_name(self, tool_name: str) -> Optional[GUITool]:
        """根据名称获取工具"""
        for tool in self.tools.values():
            if tool.name == tool_name:
                return tool
        return None
    
    def list_tools(
        self,
        platform: Optional[Platform] = None,
        tool_type: Optional[ToolType] = None
    ) -> List[GUITool]:
        """列出工具"""
        tools = list(self.tools.values())
        
        if platform:
            tools = [t for t in tools if t.platform == platform or t.platform == Platform.UNIVERSAL]
        
        if tool_type:
            tools = [t for t in tools if t.tool_type == tool_type]
        
        return tools
    
    async def execute_tool(
        self,
        tool_id: str,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None,
        execution_mode: ExecutionMode = ExecutionMode.ASYNC
    ) -> Union[ToolResult, str]:
        """执行工具"""
        try:
            tool = self.get_tool(tool_id)
            if not tool:
                raise ToolError(
                    f"Tool not found: {tool_id}",
                    "TOOL_NOT_FOUND"
                )
            
            # 检查缓存
            if self.enable_caching:
                cache_key = self._generate_cache_key(tool_id, parameters)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    return cached_result
            
            execution_id = str(uuid4())
            
            if execution_mode == ExecutionMode.SYNC:
                # 同步执行
                result = await tool.execute_with_retry(parameters, context)
                
                # 缓存结果
                if self.enable_caching and result.success:
                    cache_key = self._generate_cache_key(tool_id, parameters)
                    self._cache_result(cache_key, result)
                
                return result
            
            elif execution_mode == ExecutionMode.ASYNC:
                # 异步执行
                execution_info = {
                    'execution_id': execution_id,
                    'tool_id': tool_id,
                    'tool': tool,
                    'parameters': parameters,
                    'context': context,
                    'start_time': get_iso_timestamp(),
                    'status': 'queued'
                }
                
                self.execution_queue.append(execution_info)
                
                return execution_id
            
            elif execution_mode == ExecutionMode.BATCH:
                # 批量执行（暂时按异步处理）
                return await self.execute_tool(
                    tool_id, parameters, context, ExecutionMode.ASYNC
                )
            
            elif execution_mode == ExecutionMode.PIPELINE:
                # 流水线执行（暂时按异步处理）
                return await self.execute_tool(
                    tool_id, parameters, context, ExecutionMode.ASYNC
                )
            
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_id}: {e}")
            return ToolResult(
                tool_id=tool_id,
                tool_type="unknown",
                status=ToolStatus.FAILED,
                success=False,
                start_time=get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    async def get_execution_result(self, execution_id: str) -> Optional[ToolResult]:
        """获取执行结果"""
        try:
            # 检查活跃执行
            if execution_id in self.active_executions:
                execution_info = self.active_executions[execution_id]
                if 'result' in execution_info:
                    return execution_info['result']
                else:
                    # 执行还在进行中
                    return None
            
            # 检查所有工具的执行历史
            for tool in self.tools.values():
                for result in tool.execution_history:
                    if result.metadata.get('execution_id') == execution_id:
                        return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get execution result {execution_id}: {e}")
            return None
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        try:
            if execution_id in self.active_executions:
                execution_info = self.active_executions[execution_id]
                execution_info['status'] = 'cancelled'
                
                # 如果有任务，取消它
                if 'task' in execution_info:
                    execution_info['task'].cancel()
                
                del self.active_executions[execution_id]
                
                logger.info(f"Execution cancelled: {execution_id}")
                return True
            
            # 从队列中移除
            self.execution_queue = [
                item for item in self.execution_queue
                if item['execution_id'] != execution_id
            ]
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False
    
    async def _execution_loop(self) -> None:
        """执行循环"""
        while self._running:
            try:
                # 处理队列中的执行请求
                while (
                    self.execution_queue and
                    len(self.active_executions) < self.max_concurrent_executions
                ):
                    execution_info = self.execution_queue.pop(0)
                    await self._start_execution(execution_info)
                
                # 清理完成的执行
                await self._cleanup_completed_executions()
                
                await asyncio.sleep(0.1)  # 短暂休眠
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1)
    
    async def _start_execution(self, execution_info: Dict[str, Any]) -> None:
        """开始执行"""
        try:
            execution_id = execution_info['execution_id']
            tool = execution_info['tool']
            parameters = execution_info['parameters']
            context = execution_info['context']
            
            # 创建执行任务
            task = asyncio.create_task(
                self._execute_tool_async(execution_id, tool, parameters, context)
            )
            
            execution_info['task'] = task
            execution_info['status'] = 'running'
            
            self.active_executions[execution_id] = execution_info
            
        except Exception as e:
            logger.error(f"Failed to start execution: {e}")
    
    async def _execute_tool_async(
        self,
        execution_id: str,
        tool: GUITool,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """异步执行工具"""
        try:
            # 执行工具
            result = await tool.execute_with_retry(parameters, context)
            
            # 添加执行ID到结果元数据
            result.metadata['execution_id'] = execution_id
            
            # 保存结果
            if execution_id in self.active_executions:
                self.active_executions[execution_id]['result'] = result
                self.active_executions[execution_id]['status'] = 'completed'
            
            # 缓存结果
            if self.enable_caching and result.success:
                cache_key = self._generate_cache_key(tool.tool_id, parameters)
                self._cache_result(cache_key, result)
            
        except Exception as e:
            logger.error(f"Error in async tool execution {execution_id}: {e}")
            
            # 创建错误结果
            error_result = ToolResult(
                tool_id=tool.tool_id,
                tool_type=tool.tool_type.value,
                status=ToolStatus.FAILED,
                success=False,
                start_time=get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                error_message=str(e),
                error_code="ASYNC_EXECUTION_ERROR",
                metadata={'execution_id': execution_id}
            )
            
            if execution_id in self.active_executions:
                self.active_executions[execution_id]['result'] = error_result
                self.active_executions[execution_id]['status'] = 'failed'
    
    async def _cleanup_completed_executions(self) -> None:
        """清理完成的执行"""
        try:
            completed_executions = []
            
            for execution_id, execution_info in self.active_executions.items():
                if execution_info['status'] in ['completed', 'failed', 'cancelled']:
                    completed_executions.append(execution_id)
            
            # 移除完成的执行（保留结果一段时间）
            current_time = datetime.now()
            for execution_id in completed_executions:
                execution_info = self.active_executions[execution_id]
                start_time = datetime.fromisoformat(
                    execution_info['start_time'].replace('Z', '+00:00')
                )
                
                # 保留结果10分钟
                if current_time - start_time > timedelta(minutes=10):
                    del self.active_executions[execution_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up executions: {e}")
    
    async def _on_tool_executed(self, event: Event) -> None:
        """工具执行事件处理"""
        try:
            data = event.data
            result_data = data.get('result')
            
            if result_data:
                # 更新管理器统计
                self.manager_stats['total_executions'] += 1
                
                # 从result_data中提取信息
                if isinstance(result_data, dict):
                    success = result_data.get('success', False)
                    duration = result_data.get('duration')
                    end_time = result_data.get('end_time')
                elif hasattr(result_data, 'success'):
                    success = result_data.success
                    duration = getattr(result_data, 'duration', None)
                    end_time = getattr(result_data, 'end_time', None)
                else:
                    success = False
                    duration = None
                    end_time = None
                
                if success:
                    self.manager_stats['successful_executions'] += 1
                else:
                    self.manager_stats['failed_executions'] += 1
                
                if duration:
                    total_duration = (
                        self.manager_stats['average_execution_time'] * 
                        (self.manager_stats['total_executions'] - 1) +
                        duration
                    )
                    self.manager_stats['average_execution_time'] = (
                        total_duration / self.manager_stats['total_executions']
                    )
                
                self.manager_stats['last_execution'] = end_time
                
                # 发布事件
                await self._publish_event('tool_executed', {
                    'tool_id': data.get('tool_id'),
                    'tool_name': data.get('tool_name'),
                    'success': success,
                    'duration': duration,
                    'timestamp': end_time or get_iso_timestamp()
                })
            
        except Exception as e:
            logger.error(f"Error handling tool executed event: {e}")
    
    def _generate_cache_key(
        self,
        tool_id: str,
        parameters: ToolParameters
    ) -> str:
        """生成缓存键"""
        try:
            cache_data = {
                'tool_id': tool_id,
                'parameters': parameters.to_dict()
            }
            return str(hash(json.dumps(cache_data, sort_keys=True)))
        except Exception:
            return f"{tool_id}_{hash(str(parameters))}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ToolResult]:
        """获取缓存结果"""
        if cache_key in self.result_cache:
            result, cached_time = self.result_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                return result
            else:
                del self.result_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: ToolResult) -> None:
        """缓存结果"""
        self.result_cache[cache_key] = (result, datetime.now())
        
        # 限制缓存大小
        if len(self.result_cache) > 1000:
            oldest_key = min(
                self.result_cache.keys(),
                key=lambda k: self.result_cache[k][1]
            )
            del self.result_cache[oldest_key]
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """发布事件到AgenticX EventBus"""
        try:
            event_bus = self.event_bus  # 获取引用避免竞态条件
            if event_bus:
                event = Event(
                    type=event_type,
                    data=data,
                    source=self.name,
                    timestamp=get_iso_timestamp()
                )
                await event_bus.publish_async(event)
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计"""
        return {
            **self.manager_stats,
            'active_executions': len(self.active_executions),
            'queued_executions': len(self.execution_queue),
            'cache_size': len(self.result_cache),
            'timestamp': get_iso_timestamp()
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.result_cache.clear()
        logger.info("Tool manager cache cleared")
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """获取工具统计"""
        tool_stats = {}
        for tool_id, tool in self.tools.items():
            tool_stats[tool_id] = tool.get_performance_stats()
        
        return {
            'manager_stats': self.get_manager_stats(),
            'tool_stats': tool_stats
        }