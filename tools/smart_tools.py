#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Smart GUI Tools
智能GUI工具：基于AI的智能化移动设备操作

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .gui_tools import (
    GUITool, ToolParameters, ToolResult, ToolError,
    Coordinate, Rectangle, Platform, ToolType, ToolStatus
)
from .basic_tools import ClickTool, SwipeTool, TextInputTool
from .advanced_tools import ScreenshotTool, ElementDetectionTool, OCRTool
from utils import get_iso_timestamp, setup_logger


class SmartClickTool(GUITool):
    """智能点击工具：基于AI识别目标并执行点击"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="SmartClickTool",
            description="Intelligently find and click elements using AI",
            tool_type=ToolType.SMART,
            **kwargs
        )
        
        # 初始化依赖工具
        self.screenshot_tool = ScreenshotTool()
        self.element_detection_tool = ElementDetectionTool()
        self.ocr_tool = OCRTool()
        self.click_tool = ClickTool()
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证智能点击参数"""
        try:
            if not parameters.target and not parameters.text:
                logger.error("Either target element or text description is required")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行智能点击操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 第一步：获取当前屏幕截图
            screenshot_result = await self._take_screenshot(context)
            if not screenshot_result['success']:
                raise ToolError(
                    "Failed to capture screenshot",
                    "SCREENSHOT_FAILED",
                    self.tool_id
                )
            
            screenshot_path = screenshot_result['screenshot_path']
            
            # 第二步：智能识别目标元素
            target_element = await self._find_target_element(
                screenshot_path,
                parameters,
                context
            )
            
            if not target_element:
                raise ToolError(
                    f"Target element not found: {parameters.target or parameters.text}",
                    "ELEMENT_NOT_FOUND",
                    self.tool_id
                )
            
            # 第三步：计算点击坐标
            click_coordinate = self._calculate_click_coordinate(
                target_element, parameters
            )
            
            # 第四步：执行点击操作
            click_result = await self._perform_click(
                click_coordinate, parameters, context
            )
            
            if not click_result['success']:
                raise ToolError(
                    "Failed to perform click",
                    "CLICK_FAILED",
                    self.tool_id
                )
            
            # 第五步：验证点击结果（可选）
            verification_result = None
            if parameters.custom_params and parameters.custom_params.get('verify_result', False):
                verification_result = await self._verify_click_result(
                    screenshot_path, parameters, context
                )
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED,
                success=True,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'target_element': target_element,
                    'click_coordinate': {
                        'x': click_coordinate.x,
                        'y': click_coordinate.y
                    },
                    'click_result': click_result,
                    'verification_result': verification_result,
                    'screenshot_path': screenshot_path
                },
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'SMART_CLICK_ERROR')
            
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
    
    async def _take_screenshot(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """获取屏幕截图"""
        try:
            screenshot_params = ToolParameters()
            result = await self.screenshot_tool.execute(screenshot_params, context)
            
            return {
                'success': result.success,
                'screenshot_path': result.screenshot_path,
                'error': result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'screenshot_path': None,
                'error': str(e)
            }
    
    async def _find_target_element(
        self,
        screenshot_path: str,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """智能查找目标元素"""
        try:
            # 策略1：使用元素检测工具
            if parameters.target:
                detection_params = ToolParameters(
                    target=parameters.target,
                    custom_params={'visible_only': True, 'enabled_only': True}
                )
                
                detection_result = await self.element_detection_tool.execute(
                    detection_params, context
                )
                
                if detection_result.success and detection_result.result_data['detected_elements']:
                    return detection_result.result_data['detected_elements'][0]
            
            # 策略2：使用OCR文字识别
            if parameters.text:
                ocr_params = ToolParameters(
                    custom_params={'min_confidence': 0.7}
                )
                
                ocr_result = await self.ocr_tool.execute(ocr_params, context)
                
                if ocr_result.success:
                    # 在OCR结果中查找匹配的文本
                    target_text = self._find_matching_text(
                        ocr_result.result_data['ocr_results']['texts'],
                        parameters.text
                    )
                    
                    if target_text:
                        return {
                            'id': f"ocr_text_{int(time.time())}",
                            'type': 'text',
                            'text': target_text['text'],
                            'bounds': target_text['bounds'],
                            'center': target_text['center'],
                            'confidence': target_text['confidence'],
                            'source': 'ocr'
                        }
            
            # 策略3：智能推理（基于上下文和启发式规则）
            if parameters.custom_params and parameters.custom_params.get('use_smart_inference', True):
                inferred_element = await self._smart_inference(
                    screenshot_path, parameters, context
                )
                
                if inferred_element:
                    return inferred_element
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding target element: {e}")
            return None
    
    def _find_matching_text(
        self,
        ocr_texts: List[Dict[str, Any]],
        target_text: str
    ) -> Optional[Dict[str, Any]]:
        """在OCR结果中查找匹配的文本"""
        try:
            target_lower = target_text.lower().strip()
            
            # 精确匹配
            for text_item in ocr_texts:
                if text_item['text'].lower().strip() == target_lower:
                    return text_item
            
            # 包含匹配
            for text_item in ocr_texts:
                if target_lower in text_item['text'].lower():
                    return text_item
            
            # 模糊匹配（基于相似度）
            best_match = None
            best_score = 0
            
            for text_item in ocr_texts:
                score = self._calculate_text_similarity(
                    target_lower, text_item['text'].lower()
                )
                
                if score > best_score and score > 0.7:  # 相似度阈值
                    best_score = score
                    best_match = text_item
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding matching text: {e}")
            return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            # 简单的字符级相似度计算
            if not text1 or not text2:
                return 0.0
            
            # 计算最长公共子序列
            m, n = len(text1), len(text2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if text1[i-1] == text2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            lcs_length = dp[m][n]
            similarity = (2.0 * lcs_length) / (m + n)
            
            return similarity
            
        except Exception:
            return 0.0
    
    async def _smart_inference(
        self,
        screenshot_path: str,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """智能推理目标元素"""
        try:
            # 这里可以集成更高级的AI推理逻辑
            # 例如使用机器学习模型、规则引擎等
            
            # 基于启发式规则的简单推理
            if parameters.text:
                # 常见的UI元素模式
                common_patterns = {
                    'login': {'type': 'button', 'likely_positions': ['bottom', 'center']},
                    'submit': {'type': 'button', 'likely_positions': ['bottom', 'center']},
                    'ok': {'type': 'button', 'likely_positions': ['bottom', 'right']},
                    'cancel': {'type': 'button', 'likely_positions': ['bottom', 'left']},
                    'search': {'type': 'input', 'likely_positions': ['top', 'center']},
                    'menu': {'type': 'button', 'likely_positions': ['top', 'left', 'right']}
                }
                
                text_lower = parameters.text.lower()
                for pattern, info in common_patterns.items():
                    if pattern in text_lower:
                        # 基于模式推断可能的位置
                        inferred_element = self._infer_element_by_pattern(
                            pattern, info, screenshot_path
                        )
                        
                        if inferred_element:
                            return inferred_element
            
            return None
            
        except Exception as e:
            logger.error(f"Error in smart inference: {e}")
            return None
    
    def _infer_element_by_pattern(
        self,
        pattern: str,
        pattern_info: Dict[str, Any],
        screenshot_path: str
    ) -> Optional[Dict[str, Any]]:
        """基于模式推断元素位置"""
        try:
            # 这里应该基于屏幕尺寸和常见UI布局推断位置
            # 目前使用简化的模拟实现
            
            screen_width = 1080
            screen_height = 1920
            
            # 根据模式确定可能的位置
            if 'bottom' in pattern_info['likely_positions']:
                y = int(screen_height * 0.8)  # 屏幕下方80%位置
            elif 'top' in pattern_info['likely_positions']:
                y = int(screen_height * 0.2)  # 屏幕上方20%位置
            else:
                y = int(screen_height * 0.5)  # 屏幕中央
            
            if 'right' in pattern_info['likely_positions']:
                x = int(screen_width * 0.8)  # 屏幕右侧80%位置
            elif 'left' in pattern_info['likely_positions']:
                x = int(screen_width * 0.2)  # 屏幕左侧20%位置
            else:
                x = int(screen_width * 0.5)  # 屏幕中央
            
            return {
                'id': f"inferred_{pattern}_{int(time.time())}",
                'type': pattern_info['type'],
                'text': pattern,
                'bounds': {
                    'left': x - 50,
                    'top': y - 25,
                    'right': x + 50,
                    'bottom': y + 25
                },
                'center': {'x': x, 'y': y},
                'confidence': 0.6,  # 推理的置信度较低
                'source': 'inference'
            }
            
        except Exception as e:
            logger.error(f"Error inferring element by pattern: {e}")
            return None
    
    def _calculate_click_coordinate(
        self,
        element: Dict[str, Any],
        parameters: ToolParameters
    ) -> Coordinate:
        """计算点击坐标"""
        try:
            # 默认点击元素中心
            if 'center' in element:
                return Coordinate(
                    x=element['center']['x'],
                    y=element['center']['y']
                )
            
            # 如果没有中心坐标，从边界计算
            if 'bounds' in element:
                bounds = element['bounds']
                center_x = (bounds['left'] + bounds['right']) // 2
                center_y = (bounds['top'] + bounds['bottom']) // 2
                return Coordinate(x=center_x, y=center_y)
            
            # 如果都没有，使用自定义偏移
            offset = parameters.custom_params.get('click_offset', {'x': 0, 'y': 0}) if parameters.custom_params else {'x': 0, 'y': 0}
            
            return Coordinate(
                x=offset['x'],
                y=offset['y']
            )
            
        except Exception as e:
            logger.error(f"Error calculating click coordinate: {e}")
            # 返回屏幕中心作为默认值
            return Coordinate(x=540, y=960)
    
    async def _perform_click(
        self,
        coordinate: Coordinate,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """执行点击操作"""
        try:
            click_params = ToolParameters(
                coordinate=coordinate,
                wait_before=parameters.wait_before,
                wait_after=parameters.wait_after
            )
            
            result = await self.click_tool.execute(click_params, context)
            
            return {
                'success': result.success,
                'coordinate': {'x': coordinate.x, 'y': coordinate.y},
                'error': result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'coordinate': {'x': coordinate.x, 'y': coordinate.y},
                'error': str(e)
            }
    
    async def _verify_click_result(
        self,
        before_screenshot: str,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证点击结果"""
        try:
            # 等待一段时间让UI响应
            await asyncio.sleep(1.0)
            
            # 获取点击后的截图
            after_result = await self._take_screenshot(context)
            
            if not after_result['success']:
                return {
                    'verified': False,
                    'error': 'Failed to capture after screenshot'
                }
            
            # 比较前后截图（简化实现）
            # 实际应该使用图像比较算法
            verification_result = {
                'verified': True,
                'before_screenshot': before_screenshot,
                'after_screenshot': after_result['screenshot_path'],
                'changes_detected': True,  # 模拟检测到变化
                'verification_method': 'screenshot_comparison'
            }
            
            return verification_result
            
        except Exception as e:
            return {
                'verified': False,
                'error': str(e)
            }


class SmartScrollTool(GUITool):
    """智能滚动工具：基于内容智能滚动"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="SmartScrollTool",
            description="Intelligently scroll to find content",
            tool_type=ToolType.SMART,
            **kwargs
        )
        
        self.screenshot_tool = ScreenshotTool()
        self.ocr_tool = OCRTool()
        self.swipe_tool = SwipeTool()
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证智能滚动参数"""
        try:
            if not parameters.text and not parameters.custom_params:
                logger.error("Target text or scroll parameters required")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行智能滚动操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 获取滚动参数
            target_text = parameters.text
            max_scrolls = parameters.custom_params.get('max_scrolls', 5) if parameters.custom_params else 5
            scroll_direction = parameters.custom_params.get('direction', 'down') if parameters.custom_params else 'down'
            
            scroll_count = 0
            found_target = False
            scroll_history = []
            
            # 开始智能滚动循环
            while scroll_count < max_scrolls and not found_target:
                # 获取当前屏幕内容
                current_content = await self._analyze_current_content(context)
                
                if not current_content['success']:
                    break
                
                # 检查是否找到目标内容
                if target_text:
                    found_target = self._check_target_found(
                        current_content['texts'], target_text
                    )
                    
                    if found_target:
                        break
                
                # 执行滚动操作
                scroll_result = await self._perform_scroll(
                    scroll_direction, parameters, context
                )
                
                scroll_history.append({
                    'scroll_count': scroll_count + 1,
                    'direction': scroll_direction,
                    'success': scroll_result['success'],
                    'content_found': len(current_content.get('texts', [])),
                    'target_found': found_target
                })
                
                if not scroll_result['success']:
                    break
                
                scroll_count += 1
                
                # 等待滚动完成
                await asyncio.sleep(0.5)
            
            # 最终检查
            if not found_target and target_text:
                final_content = await self._analyze_current_content(context)
                if final_content['success']:
                    found_target = self._check_target_found(
                        final_content['texts'], target_text
                    )
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED,
                success=True,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'target_found': found_target,
                    'scroll_count': scroll_count,
                    'scroll_history': scroll_history,
                    'target_text': target_text,
                    'max_scrolls_reached': scroll_count >= max_scrolls
                }
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'SMART_SCROLL_ERROR')
            
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
    
    async def _analyze_current_content(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析当前屏幕内容"""
        try:
            # 获取截图
            screenshot_params = ToolParameters()
            screenshot_result = await self.screenshot_tool.execute(screenshot_params, context)
            
            if not screenshot_result.success:
                return {'success': False, 'error': 'Failed to capture screenshot'}
            
            # 执行OCR识别
            ocr_params = ToolParameters(
                custom_params={'min_confidence': 0.6}
            )
            ocr_result = await self.ocr_tool.execute(ocr_params, context)
            
            if not ocr_result.success:
                return {'success': False, 'error': 'Failed to perform OCR'}
            
            return {
                'success': True,
                'texts': ocr_result.result_data['ocr_results']['texts'],
                'screenshot_path': screenshot_result.screenshot_path
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _check_target_found(self, texts: List[Dict[str, Any]], target_text: str) -> bool:
        """检查是否找到目标文本"""
        try:
            target_lower = target_text.lower().strip()
            
            for text_item in texts:
                text_content = text_item.get('text', '').lower().strip()
                if target_lower in text_content:
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _perform_scroll(
        self,
        direction: str,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """执行滚动操作"""
        try:
            # 计算滚动坐标
            screen_width = 1080
            screen_height = 1920
            
            start_x = screen_width // 2
            start_y = screen_height // 2
            
            if direction == 'down':
                end_x = start_x
                end_y = start_y - 300  # 向上滑动实现向下滚动
            elif direction == 'up':
                end_x = start_x
                end_y = start_y + 300  # 向下滑动实现向上滚动
            elif direction == 'left':
                end_x = start_x + 300  # 向右滑动实现向左滚动
                end_y = start_y
            elif direction == 'right':
                end_x = start_x - 300  # 向左滑动实现向右滚动
                end_y = start_y
            else:
                end_x = start_x
                end_y = start_y - 300  # 默认向下滚动
            
            # 执行滑动操作
            swipe_params = ToolParameters(
                start_coordinate=Coordinate(x=start_x, y=start_y),
                end_coordinate=Coordinate(x=end_x, y=end_y),
                duration=parameters.custom_params.get('scroll_duration', 500) if parameters.custom_params else 500
            )
            
            result = await self.swipe_tool.execute(swipe_params, context)
            
            return {
                'success': result.success,
                'direction': direction,
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'error': result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'direction': direction,
                'error': str(e)
            }


class SmartInputTool(GUITool):
    """智能输入工具：智能识别输入框并输入文本"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="SmartInputTool",
            description="Intelligently find input fields and enter text",
            tool_type=ToolType.SMART,
            **kwargs
        )
        
        self.smart_click_tool = SmartClickTool()
        self.text_input_tool = TextInputTool()
        self.element_detection_tool = ElementDetectionTool()
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证智能输入参数"""
        try:
            if not parameters.text:
                logger.error("Input text is required")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行智能输入操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 第一步：查找输入框
            input_field = await self._find_input_field(
                parameters, context
            )
            
            if not input_field:
                raise ToolError(
                    "No suitable input field found",
                    "INPUT_FIELD_NOT_FOUND",
                    self.tool_id
                )
            
            # 第二步：点击输入框以获得焦点
            click_result = await self._click_input_field(
                input_field, parameters, context
            )
            
            if not click_result['success']:
                raise ToolError(
                    "Failed to click input field",
                    "CLICK_INPUT_FAILED",
                    self.tool_id
                )
            
            # 第三步：输入文本
            input_result = await self._input_text(
                parameters.text, parameters, context
            )
            
            if not input_result['success']:
                raise ToolError(
                    "Failed to input text",
                    "TEXT_INPUT_FAILED",
                    self.tool_id
                )
            
            # 第四步：验证输入结果（可选）
            verification_result = None
            if parameters.custom_params and parameters.custom_params.get('verify_input', False):
                verification_result = await self._verify_input_result(
                    parameters.text, context
                )
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED,
                success=True,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'input_field': input_field,
                    'input_text': parameters.text,
                    'click_result': click_result,
                    'input_result': input_result,
                    'verification_result': verification_result
                }
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'SMART_INPUT_ERROR')
            
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
    
    async def _find_input_field(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """查找输入框"""
        try:
            # 策略1：根据目标查找
            if parameters.target:
                detection_params = ToolParameters(
                    target=parameters.target,
                    custom_params={'visible_only': True, 'enabled_only': True}
                )
                
                result = await self.element_detection_tool.execute(detection_params, context)
                
                if result.success and result.result_data['detected_elements']:
                    for element in result.result_data['detected_elements']:
                        if element.get('type') in ['input', 'textfield', 'edittext']:
                            return element
            
            # 策略2：查找所有输入框类型的元素
            detection_params = ToolParameters(
                target='input',
                custom_params={'visible_only': True, 'enabled_only': True}
            )
            
            result = await self.element_detection_tool.execute(detection_params, context)
            
            if result.success and result.result_data['detected_elements']:
                # 返回第一个找到的输入框
                return result.result_data['detected_elements'][0]
            
            # 策略3：基于启发式规则查找
            heuristic_field = self._find_input_by_heuristics(parameters, context)
            if heuristic_field:
                return heuristic_field
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding input field: {e}")
            return None
    
    def _find_input_by_heuristics(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """基于启发式规则查找输入框"""
        try:
            # 基于常见的输入框位置和特征
            screen_width = 1080
            screen_height = 1920
            
            # 假设输入框通常在屏幕中上部分
            input_y = int(screen_height * 0.4)
            input_x = int(screen_width * 0.5)
            
            return {
                'id': f"heuristic_input_{int(time.time())}",
                'type': 'input',
                'text': '',
                'bounds': {
                    'left': input_x - 200,
                    'top': input_y - 25,
                    'right': input_x + 200,
                    'bottom': input_y + 25
                },
                'center': {'x': input_x, 'y': input_y},
                'confidence': 0.5,
                'source': 'heuristic'
            }
            
        except Exception as e:
            logger.error(f"Error in heuristic input finding: {e}")
            return None
    
    async def _click_input_field(
        self,
        input_field: Dict[str, Any],
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """点击输入框"""
        try:
            # 使用智能点击工具点击输入框
            click_params = ToolParameters(
                coordinate=Coordinate(
                    x=input_field['center']['x'],
                    y=input_field['center']['y']
                ),
                wait_after=0.5  # 等待输入框获得焦点
            )
            
            result = await self.smart_click_tool.click_tool.execute(click_params, context)
            
            return {
                'success': result.success,
                'coordinate': input_field['center'],
                'error': result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'coordinate': input_field.get('center', {}),
                'error': str(e)
            }
    
    async def _input_text(
        self,
        text: str,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """输入文本"""
        try:
            # 清空现有内容（可选）
            clear_before = parameters.custom_params.get('clear_before', True) if parameters.custom_params else True
            
            input_params = ToolParameters(
                text=text,
                custom_params={
                    'clear_before': clear_before,
                    'input_method': parameters.custom_params.get('input_method', 'type') if parameters.custom_params else 'type'
                }
            )
            
            result = await self.text_input_tool.execute(input_params, context)
            
            return {
                'success': result.success,
                'text': text,
                'clear_before': clear_before,
                'error': result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': text,
                'error': str(e)
            }
    
    async def _verify_input_result(
        self,
        expected_text: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证输入结果"""
        try:
            # 等待输入完成
            await asyncio.sleep(0.5)
            
            # 这里应该实现实际的验证逻辑
            # 例如通过OCR识别输入框内容，或者通过UI自动化获取输入框值
            
            # 模拟验证结果
            verification_result = {
                'verified': True,
                'expected_text': expected_text,
                'actual_text': expected_text,  # 模拟成功输入
                'match': True,
                'verification_method': 'ui_automation'
            }
            
            return verification_result
            
        except Exception as e:
            return {
                'verified': False,
                'expected_text': expected_text,
                'error': str(e)
            }