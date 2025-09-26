#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Basic GUI Tools
基础GUI工具：基于AgenticX框架的移动设备原子操作

本模块已完全基于AgenticX框架重构：
- 继承AgenticX BaseTool和GUITool提供统一接口
- 集成AgenticX事件系统实现操作监控
- 使用AgenticX标准化的参数验证和错误处理
- 与AgenticX工具生态系统完全兼容

Author: AgenticX Team
Date: 2025
Version: 1.0.0 (基于AgenticX框架重构)
"""

import asyncio
import base64
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

from .gui_tools import (
    GUITool, ToolParameters, ToolResult, GUIToolError,
    Coordinate, Rectangle, Platform, ToolType, ToolStatus
)
from utils import get_iso_timestamp, setup_logger


class ClickTool(GUITool):
    """点击工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ClickTool",
            description="Perform click operations on mobile devices",
            tool_type=ToolType.BASIC,
            **kwargs
        )
    
    async def validate_gui_parameters(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证点击参数"""
        try:
            if not parameters.target:
                logger.error("Click target is required")
                return False
            
            if isinstance(parameters.target, Coordinate):
                # 验证坐标范围
                if parameters.target.x < 0 or parameters.target.y < 0:
                    logger.error("Invalid coordinate: negative values")
                    return False
                
                # 如果有屏幕尺寸信息，验证坐标是否在屏幕范围内
                if context and 'screen_size' in context:
                    screen_size = context['screen_size']
                    if (
                        parameters.target.x > screen_size.get('width', float('inf')) or
                        parameters.target.y > screen_size.get('height', float('inf'))
                    ):
                        logger.error("Coordinate outside screen bounds")
                        return False
            
            elif isinstance(parameters.target, str):
                # 验证元素选择器
                if not parameters.target.strip():
                    logger.error("Empty element selector")
                    return False
            
            else:
                logger.error(f"Invalid target type: {type(parameters.target)}")
                return False
            
            # 验证可选参数
            if parameters.duration and parameters.duration < 0:
                logger.error("Invalid duration: negative value")
                return False
            
            if parameters.force and (parameters.force < 0 or parameters.force > 1):
                logger.error("Invalid force: must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute_gui_tool(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行点击操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 解析目标
            if isinstance(parameters.target, Coordinate):
                click_point = parameters.target
            elif isinstance(parameters.target, str):
                # 通过元素选择器查找坐标
                click_point = await self._find_element_coordinate(
                    parameters.target, context
                )
                if not click_point:
                    raise GUIToolError(
                        f"Element not found: {parameters.target}",
                        self.name,
                        "ELEMENT_NOT_FOUND"
                    )
            else:
                raise GUIToolError(
                    f"Invalid target type: {type(parameters.target)}",
                    self.name,
                    "INVALID_TARGET"
                )
            
            # 执行前等待
            if parameters.wait_before:
                await asyncio.sleep(parameters.wait_before)
            
            # 执行点击
            success = await self._perform_click(
                click_point,
                duration=parameters.duration or 0.1,
                force=parameters.force or 0.5,
                context=context
            )
            
            # 执行后等待
            if parameters.wait_after:
                await asyncio.sleep(parameters.wait_after)
            
            # 验证点击结果
            validation_results = None
            if parameters.validate:
                validation_results = await self._validate_click_result(
                    click_point, context
                )
            
            # 截图
            screenshot_path = None
            if parameters.screenshot or self.enable_screenshot:
                screenshot_path = await self._take_screenshot(context)
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'click_point': click_point.to_dict(),
                    'duration': parameters.duration or 0.1,
                    'force': parameters.force or 0.5
                },
                validation_results=validation_results,
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'CLICK_ERROR')
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.FAILED,
                success=False,
                start_time=start_time,
                end_time=end_time,
                error_message=error_msg,
                error_code=error_code
            )
    
    async def _find_element_coordinate(
        self,
        selector: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Coordinate]:
        """通过选择器查找元素坐标"""
        try:
            # 这里应该集成实际的元素查找逻辑
            # 例如通过UI自动化框架或OCR技术
            
            # 模拟实现
            if context and 'elements' in context:
                elements = context['elements']
                for element in elements:
                    if (
                        element.get('id') == selector or
                        element.get('text') == selector or
                        element.get('class') == selector
                    ):
                        bounds = element.get('bounds')
                        if bounds:
                            return Coordinate(
                                (bounds['left'] + bounds['right']) / 2,
                                (bounds['top'] + bounds['bottom']) / 2
                            )
            
            # 如果没有找到，返回None
            return None
            
        except Exception as e:
            logger.error(f"Error finding element coordinate: {e}")
            return None
    
    async def _perform_click(
        self,
        point: Coordinate,
        duration: float,
        force: float,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """执行实际的点击操作"""
        try:
            # 这里应该集成实际的点击实现
            # 例如通过ADB、iOS自动化工具等
            
            logger.info(
                f"Performing click at ({point.x}, {point.y}) "
                f"with duration {duration}s and force {force}"
            )
            
            # 模拟点击延迟
            await asyncio.sleep(duration)
            
            # 模拟成功
            return True
            
        except Exception as e:
            logger.error(f"Error performing click: {e}")
            return False
    
    async def _validate_click_result(
        self,
        point: Coordinate,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证点击结果"""
        try:
            # 这里可以实现点击后的验证逻辑
            # 例如检查UI状态变化、页面跳转等
            
            return {
                'validated': True,
                'validation_time': get_iso_timestamp(),
                'details': 'Click validation passed'
            }
            
        except Exception as e:
            return {
                'validated': False,
                'validation_time': get_iso_timestamp(),
                'error': str(e)
            }
    
    async def _take_screenshot(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """截图"""
        try:
            # 这里应该集成实际的截图实现
            timestamp = int(time.time())
            screenshot_path = f"/tmp/screenshot_{timestamp}.png"
            
            # 模拟截图
            if Image:
                # 创建一个模拟截图
                img = Image.new('RGB', (1080, 1920), color='white')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"Screenshot at {timestamp}", fill='black')
                img.save(screenshot_path)
                
                return screenshot_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None


class SwipeTool(GUITool):
    """滑动工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="SwipeTool",
            description="Perform swipe operations on mobile devices",
            tool_type=ToolType.BASIC,
            **kwargs
        )
    
    async def validate_gui_parameters(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证滑动参数"""
        try:
            if not parameters.target:
                logger.error("Swipe start point is required")
                return False
            
            if not isinstance(parameters.target, (Coordinate, Rectangle)):
                logger.error("Invalid target type for swipe")
                return False
            
            if not parameters.direction:
                logger.error("Swipe direction is required")
                return False
            
            valid_directions = ['up', 'down', 'left', 'right', 'custom']
            if parameters.direction not in valid_directions:
                logger.error(f"Invalid direction: {parameters.direction}")
                return False
            
            if parameters.distance and parameters.distance <= 0:
                logger.error("Invalid distance: must be positive")
                return False
            
            if parameters.duration and parameters.duration <= 0:
                logger.error("Invalid duration: must be positive")
                return False
            
            if parameters.speed and parameters.speed <= 0:
                logger.error("Invalid speed: must be positive")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute_gui_tool(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行滑动操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 计算起始点和结束点
            start_point, end_point = self._calculate_swipe_points(
                parameters, context
            )
            
            # 执行前等待
            if parameters.wait_before:
                await asyncio.sleep(parameters.wait_before)
            
            # 执行滑动
            success = await self._perform_swipe(
                start_point,
                end_point,
                duration=parameters.duration or 0.5,
                context=context
            )
            
            # 执行后等待
            if parameters.wait_after:
                await asyncio.sleep(parameters.wait_after)
            
            # 验证滑动结果
            validation_results = None
            if parameters.validate:
                validation_results = await self._validate_swipe_result(
                    start_point, end_point, context
                )
            
            # 截图
            screenshot_path = None
            if parameters.screenshot or self.enable_screenshot:
                screenshot_path = await self._take_screenshot(context)
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'start_point': start_point.to_dict(),
                    'end_point': end_point.to_dict(),
                    'direction': parameters.direction,
                    'distance': parameters.distance,
                    'duration': parameters.duration or 0.5
                },
                validation_results=validation_results,
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'SWIPE_ERROR')
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.FAILED,
                success=False,
                start_time=start_time,
                end_time=end_time,
                error_message=error_msg,
                error_code=error_code
            )
    
    def _calculate_swipe_points(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Coordinate, Coordinate]:
        """计算滑动起始点和结束点"""
        try:
            # 确定起始点
            if isinstance(parameters.target, Coordinate):
                start_point = parameters.target
            elif isinstance(parameters.target, Rectangle):
                start_point = parameters.target.center
            else:
                raise GUIToolError(
                    f"Invalid target type: {type(parameters.target)}",
                    self.name,
                    "INVALID_TARGET"
                )
            
            # 计算滑动距离
            distance = parameters.distance or 200  # 默认200像素
            
            # 根据方向计算结束点
            if parameters.direction == 'up':
                end_point = Coordinate(start_point.x, start_point.y - distance)
            elif parameters.direction == 'down':
                end_point = Coordinate(start_point.x, start_point.y + distance)
            elif parameters.direction == 'left':
                end_point = Coordinate(start_point.x - distance, start_point.y)
            elif parameters.direction == 'right':
                end_point = Coordinate(start_point.x + distance, start_point.y)
            elif parameters.direction == 'custom':
                # 自定义方向需要额外参数
                if 'end_point' in parameters.custom_params:
                    end_point_data = parameters.custom_params['end_point']
                    end_point = Coordinate(end_point_data['x'], end_point_data['y'])
                else:
                    raise GUIToolError(
                        "Custom direction requires end_point in custom_params",
                        self.name,
                        "MISSING_END_POINT"
                    )
            else:
                raise GUIToolError(
                    f"Unknown direction: {parameters.direction}",
                    self.name,
                    "UNKNOWN_DIRECTION"
                )
            
            return start_point, end_point
            
        except Exception as e:
            if isinstance(e, GUIToolError):
                raise
            raise GUIToolError(
                f"Error calculating swipe points: {e}",
                self.name,
                "CALCULATION_ERROR"
            )
    
    async def _perform_swipe(
        self,
        start_point: Coordinate,
        end_point: Coordinate,
        duration: float,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """执行实际的滑动操作"""
        try:
            logger.info(
                f"Performing swipe from ({start_point.x}, {start_point.y}) "
                f"to ({end_point.x}, {end_point.y}) in {duration}s"
            )
            
            # 这里应该集成实际的滑动实现
            # 例如通过ADB、iOS自动化工具等
            
            # 模拟滑动延迟
            await asyncio.sleep(duration)
            
            # 模拟成功
            return True
            
        except Exception as e:
            logger.error(f"Error performing swipe: {e}")
            return False
    
    async def _validate_swipe_result(
        self,
        start_point: Coordinate,
        end_point: Coordinate,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证滑动结果"""
        try:
            # 这里可以实现滑动后的验证逻辑
            # 例如检查页面滚动、内容变化等
            
            return {
                'validated': True,
                'validation_time': get_iso_timestamp(),
                'details': 'Swipe validation passed'
            }
            
        except Exception as e:
            return {
                'validated': False,
                'validation_time': get_iso_timestamp(),
                'error': str(e)
            }
    
    async def _take_screenshot(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """截图"""
        try:
            timestamp = int(time.time())
            screenshot_path = f"/tmp/swipe_screenshot_{timestamp}.png"
            
            # 模拟截图
            if Image:
                img = Image.new('RGB', (1080, 1920), color='lightblue')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"Swipe Screenshot at {timestamp}", fill='black')
                img.save(screenshot_path)
                
                return screenshot_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None


class TextInputTool(GUITool):
    """文本输入工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="TextInputTool",
            description="Input text on mobile devices",
            tool_type=ToolType.BASIC,
            **kwargs
        )
    
    async def validate_gui_parameters(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证文本输入参数"""
        try:
            if not parameters.target:
                logger.error("Input target is required")
                return False
            
            if not parameters.text:
                logger.error("Input text is required")
                return False
            
            if isinstance(parameters.text, str) and len(parameters.text) > 10000:
                logger.error("Input text too long (max 10000 characters)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute_gui_tool(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行文本输入操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 找到输入目标
            input_target = await self._find_input_target(
                parameters.target, context
            )
            
            if not input_target:
                raise GUIToolError(
                    f"Input target not found: {parameters.target}",
                    self.name,
                    "TARGET_NOT_FOUND"
                )
            
            # 执行前等待
            if parameters.wait_before:
                await asyncio.sleep(parameters.wait_before)
            
            # 清空现有文本（如果需要）
            if parameters.custom_params.get('clear_before', False):
                await self._clear_input_field(input_target, context)
            
            # 输入文本
            success = await self._perform_text_input(
                input_target,
                parameters.text,
                context
            )
            
            # 执行后等待
            if parameters.wait_after:
                await asyncio.sleep(parameters.wait_after)
            
            # 验证输入结果
            validation_results = None
            if parameters.validate:
                validation_results = await self._validate_input_result(
                    input_target, parameters.text, context
                )
            
            # 截图
            screenshot_path = None
            if parameters.screenshot or self.enable_screenshot:
                screenshot_path = await self._take_screenshot(context)
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'input_target': str(input_target),
                    'text': parameters.text,
                    'text_length': len(parameters.text)
                },
                validation_results=validation_results,
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'INPUT_ERROR')
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.FAILED,
                success=False,
                start_time=start_time,
                end_time=end_time,
                error_message=error_msg,
                error_code=error_code
            )
    
    async def _find_input_target(
        self,
        target: Union[Coordinate, str],
        context: Optional[Dict[str, Any]]
    ) -> Optional[Union[Coordinate, str]]:
        """查找输入目标"""
        try:
            if isinstance(target, Coordinate):
                return target
            elif isinstance(target, str):
                # 通过选择器查找输入框
                if context and 'elements' in context:
                    elements = context['elements']
                    for element in elements:
                        if (
                            element.get('id') == target or
                            element.get('text') == target or
                            element.get('class') == target
                        ):
                            # 检查是否是输入框
                            if element.get('type') in ['input', 'textarea', 'edittext']:
                                bounds = element.get('bounds')
                                if bounds:
                                    return Coordinate(
                                        (bounds['left'] + bounds['right']) / 2,
                                        (bounds['top'] + bounds['bottom']) / 2
                                    )
                return target  # 返回原始选择器
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding input target: {e}")
            return None
    
    async def _clear_input_field(
        self,
        target: Union[Coordinate, str],
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """清空输入框"""
        try:
            # 这里应该集成实际的清空实现
            # 例如全选+删除，或者直接清空
            
            logger.info(f"Clearing input field: {target}")
            
            # 模拟清空操作
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing input field: {e}")
            return False
    
    async def _perform_text_input(
        self,
        target: Union[Coordinate, str],
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """执行实际的文本输入"""
        try:
            logger.info(
                f"Inputting text to {target}: '{text[:50]}{'...' if len(text) > 50 else ''}'"
            )
            
            # 这里应该集成实际的文本输入实现
            # 例如通过ADB input text、iOS自动化工具等
            
            # 模拟输入延迟（根据文本长度）
            input_delay = min(len(text) * 0.01, 2.0)  # 最多2秒
            await asyncio.sleep(input_delay)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing text input: {e}")
            return False
    
    async def _validate_input_result(
        self,
        target: Union[Coordinate, str],
        expected_text: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证输入结果"""
        try:
            # 这里可以实现输入后的验证逻辑
            # 例如读取输入框内容，检查是否与期望一致
            
            return {
                'validated': True,
                'validation_time': get_iso_timestamp(),
                'expected_text': expected_text,
                'actual_text': expected_text,  # 模拟相同
                'match': True
            }
            
        except Exception as e:
            return {
                'validated': False,
                'validation_time': get_iso_timestamp(),
                'error': str(e)
            }
    
    async def _take_screenshot(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """截图"""
        try:
            timestamp = int(time.time())
            screenshot_path = f"/tmp/input_screenshot_{timestamp}.png"
            
            # 模拟截图
            if Image:
                img = Image.new('RGB', (1080, 1920), color='lightgreen')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"Input Screenshot at {timestamp}", fill='black')
                img.save(screenshot_path)
                
                return screenshot_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None


class KeyPressTool(GUITool):
    """按键工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="KeyPressTool",
            description="Press keys on mobile devices",
            tool_type=ToolType.BASIC,
            **kwargs
        )
    
    async def validate_gui_parameters(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证按键参数"""
        try:
            if not parameters.text:  # 使用text字段存储按键名称
                logger.error("Key name is required")
                return False
            
            # 验证按键名称
            valid_keys = [
                'home', 'back', 'menu', 'search', 'volume_up', 'volume_down',
                'power', 'enter', 'delete', 'backspace', 'tab', 'escape',
                'space', 'up', 'down', 'left', 'right'
            ]
            
            if parameters.text.lower() not in valid_keys:
                logger.warning(f"Unknown key: {parameters.text}")
                # 不阻止执行，可能是平台特定的按键
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute_gui_tool(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行按键操作"""
        start_time = get_iso_timestamp()
        
        try:
            key_name = parameters.text.lower()
            
            # 执行前等待
            if parameters.wait_before:
                await asyncio.sleep(parameters.wait_before)
            
            # 执行按键
            success = await self._perform_key_press(
                key_name,
                duration=parameters.duration or 0.1,
                context=context
            )
            
            # 执行后等待
            if parameters.wait_after:
                await asyncio.sleep(parameters.wait_after)
            
            # 验证按键结果
            validation_results = None
            if parameters.validate:
                validation_results = await self._validate_key_press_result(
                    key_name, context
                )
            
            # 截图
            screenshot_path = None
            if parameters.screenshot or self.enable_screenshot:
                screenshot_path = await self._take_screenshot(context)
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'key_name': key_name,
                    'duration': parameters.duration or 0.1
                },
                validation_results=validation_results,
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'KEY_PRESS_ERROR')
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.FAILED,
                success=False,
                start_time=start_time,
                end_time=end_time,
                error_message=error_msg,
                error_code=error_code
            )
    
    async def _perform_key_press(
        self,
        key_name: str,
        duration: float,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """执行实际的按键操作"""
        try:
            logger.info(f"Pressing key: {key_name} for {duration}s")
            
            # 这里应该集成实际的按键实现
            # 例如通过ADB input keyevent、iOS自动化工具等
            
            # 模拟按键延迟
            await asyncio.sleep(duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing key press: {e}")
            return False
    
    async def _validate_key_press_result(
        self,
        key_name: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证按键结果"""
        try:
            # 这里可以实现按键后的验证逻辑
            # 例如检查UI状态变化、应用响应等
            
            return {
                'validated': True,
                'validation_time': get_iso_timestamp(),
                'key_name': key_name,
                'details': 'Key press validation passed'
            }
            
        except Exception as e:
            return {
                'validated': False,
                'validation_time': get_iso_timestamp(),
                'error': str(e)
            }
    
    async def _take_screenshot(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """截图"""
        try:
            timestamp = int(time.time())
            screenshot_path = f"/tmp/keypress_screenshot_{timestamp}.png"
            
            # 模拟截图
            if Image:
                img = Image.new('RGB', (1080, 1920), color='lightyellow')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"KeyPress Screenshot at {timestamp}", fill='black')
                img.save(screenshot_path)
                
                return screenshot_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None


class WaitTool(GUITool):
    """等待工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="WaitTool",
            description="Wait for specified duration or condition",
            tool_type=ToolType.BASIC,
            **kwargs
        )
    
    async def validate_gui_parameters(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证等待参数"""
        try:
            if not parameters.duration and not parameters.target:
                logger.error("Either duration or wait condition is required")
                return False
            
            if parameters.duration and parameters.duration < 0:
                logger.error("Invalid duration: must be non-negative")
                return False
            
            if parameters.duration and parameters.duration > 300:  # 5分钟
                logger.warning(f"Long wait duration: {parameters.duration}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute_gui_tool(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行等待操作"""
        start_time = get_iso_timestamp()
        
        try:
            if parameters.duration:
                # 固定时间等待
                success = await self._wait_for_duration(parameters.duration)
                wait_type = "duration"
                wait_value = parameters.duration
            elif parameters.target:
                # 条件等待
                success, actual_wait_time = await self._wait_for_condition(
                    parameters.target,
                    timeout=parameters.timeout or 30.0,
                    context=context
                )
                wait_type = "condition"
                wait_value = actual_wait_time
            else:
                raise GUIToolError(
                    "No wait duration or condition specified",
                    self.name,
                    "INVALID_PARAMETERS"
                )
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'wait_type': wait_type,
                    'wait_value': wait_value,
                    'condition': str(parameters.target) if parameters.target else None
                }
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'WAIT_ERROR')
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.FAILED,
                success=False,
                start_time=start_time,
                end_time=end_time,
                error_message=error_msg,
                error_code=error_code
            )
    
    async def _wait_for_duration(self, duration: float) -> bool:
        """等待指定时间"""
        try:
            logger.info(f"Waiting for {duration} seconds")
            await asyncio.sleep(duration)
            return True
            
        except Exception as e:
            logger.error(f"Error during wait: {e}")
            return False
    
    async def _wait_for_condition(
        self,
        condition: str,
        timeout: float,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """等待条件满足"""
        start_time = time.time()
        
        try:
            logger.info(f"Waiting for condition: {condition} (timeout: {timeout}s)")
            
            while time.time() - start_time < timeout:
                # 检查条件是否满足
                if await self._check_condition(condition, context):
                    actual_wait_time = time.time() - start_time
                    logger.info(f"Condition met after {actual_wait_time:.2f}s")
                    return True, actual_wait_time
                
                # 短暂等待后重新检查
                await asyncio.sleep(0.5)
            
            # 超时
            actual_wait_time = time.time() - start_time
            logger.warning(f"Condition not met after {actual_wait_time:.2f}s (timeout)")
            return False, actual_wait_time
            
        except Exception as e:
            actual_wait_time = time.time() - start_time
            logger.error(f"Error waiting for condition: {e}")
            return False, actual_wait_time
    
    async def _check_condition(
        self,
        condition: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """检查条件是否满足"""
        try:
            # 这里应该实现各种条件检查逻辑
            # 例如元素出现、文本变化、页面加载完成等
            
            # 模拟条件检查
            if condition.startswith('element_'):
                # 检查元素是否存在
                return await self._check_element_exists(condition, context)
            elif condition.startswith('text_'):
                # 检查文本是否出现
                return await self._check_text_exists(condition, context)
            elif condition == 'page_loaded':
                # 检查页面是否加载完成
                return await self._check_page_loaded(context)
            else:
                # 未知条件，返回False
                return False
            
        except Exception as e:
            logger.error(f"Error checking condition: {e}")
            return False
    
    async def _check_element_exists(
        self,
        condition: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """检查元素是否存在"""
        try:
            element_id = condition.replace('element_', '')
            
            if context and 'elements' in context:
                elements = context['elements']
                for element in elements:
                    if element.get('id') == element_id:
                        return True
            
            return False
            
        except Exception:
            return False
    
    async def _check_text_exists(
        self,
        condition: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """检查文本是否存在"""
        try:
            text_to_find = condition.replace('text_', '')
            
            if context and 'page_text' in context:
                page_text = context['page_text']
                return text_to_find.lower() in page_text.lower()
            
            return False
            
        except Exception:
            return False
    
    async def _check_page_loaded(self, context: Optional[Dict[str, Any]]) -> bool:
        """检查页面是否加载完成"""
        try:
            if context and 'page_state' in context:
                return context['page_state'] == 'loaded'
            
            # 模拟页面加载检查
            return True
            
        except Exception:
            return False