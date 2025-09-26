#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Advanced GUI Tools
高级GUI工具：实现智能化的移动设备操作

Author: AgenticX Team
Date: 2025
"""

import asyncio
import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from .gui_tools import (
    GUITool, ToolParameters, ToolResult, ToolError,
    Coordinate, Rectangle, Platform, ToolType, ToolStatus
)
from utils import get_iso_timestamp, setup_logger


class ScreenshotTool(GUITool):
    """截图工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ScreenshotTool",
            description="Capture screenshots of mobile devices",
            tool_type=ToolType.ADVANCED,
            **kwargs
        )
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证截图参数"""
        try:
            # 截图工具通常不需要特殊参数验证
            if parameters.custom_params:
                # 验证自定义参数
                quality = parameters.custom_params.get('quality')
                if quality and (quality < 1 or quality > 100):
                    logger.error("Invalid quality: must be between 1 and 100")
                    return False
                
                format_type = parameters.custom_params.get('format', 'png')
                if format_type not in ['png', 'jpg', 'jpeg', 'bmp']:
                    logger.error(f"Unsupported format: {format_type}")
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
        """执行截图操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 获取截图参数
            quality = parameters.custom_params.get('quality', 90) if parameters.custom_params else 90
            format_type = parameters.custom_params.get('format', 'png') if parameters.custom_params else 'png'
            region = parameters.custom_params.get('region') if parameters.custom_params else None
            
            # 执行前等待
            if parameters.wait_before:
                await asyncio.sleep(parameters.wait_before)
            
            # 执行截图
            screenshot_path, screenshot_info = await self._capture_screenshot(
                quality=quality,
                format_type=format_type,
                region=region,
                context=context
            )
            
            if not screenshot_path:
                raise ToolError(
                    "Failed to capture screenshot",
                    "SCREENSHOT_FAILED",
                    self.tool_id
                )
            
            # 执行后等待
            if parameters.wait_after:
                await asyncio.sleep(parameters.wait_after)
            
            # 分析截图（如果需要）
            analysis_results = None
            if parameters.custom_params and parameters.custom_params.get('analyze', False):
                analysis_results = await self._analyze_screenshot(
                    screenshot_path, context
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
                    'screenshot_path': screenshot_path,
                    'screenshot_info': screenshot_info,
                    'quality': quality,
                    'format': format_type,
                    'analysis_results': analysis_results
                },
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'SCREENSHOT_ERROR')
            
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
    
    async def _capture_screenshot(
        self,
        quality: int,
        format_type: str,
        region: Optional[Dict[str, int]],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """执行实际的截图操作"""
        try:
            timestamp = int(time.time())
            screenshot_path = f"/tmp/screenshot_{timestamp}.{format_type}"
            
            # 这里应该集成实际的截图实现
            # 例如通过ADB screencap、iOS自动化工具等
            
            if Image:
                # 创建模拟截图
                width = 1080
                height = 1920
                
                if region:
                    width = region.get('width', width)
                    height = region.get('height', height)
                
                img = Image.new('RGB', (width, height), color='white')
                draw = ImageDraw.Draw(img)
                
                # 绘制一些模拟内容
                draw.rectangle([50, 50, width-50, 150], fill='lightblue', outline='blue')
                draw.text((60, 80), f"Screenshot at {timestamp}", fill='black')
                
                # 模拟一些UI元素
                draw.rectangle([50, 200, width-50, 250], fill='lightgreen', outline='green')
                draw.text((60, 215), "Button 1", fill='black')
                
                draw.rectangle([50, 300, width-50, 350], fill='lightcoral', outline='red')
                draw.text((60, 315), "Button 2", fill='black')
                
                # 保存截图
                if format_type.lower() in ['jpg', 'jpeg']:
                    img.save(screenshot_path, 'JPEG', quality=quality)
                else:
                    img.save(screenshot_path, format_type.upper())
                
                # 获取截图信息
                screenshot_info = {
                    'width': width,
                    'height': height,
                    'format': format_type,
                    'quality': quality,
                    'file_size': os.path.getsize(screenshot_path) if os.path.exists(screenshot_path) else 0,
                    'timestamp': timestamp
                }
                
                return screenshot_path, screenshot_info
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return None, None
    
    async def _analyze_screenshot(
        self,
        screenshot_path: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析截图内容"""
        try:
            analysis_results = {
                'analyzed_at': get_iso_timestamp(),
                'elements_detected': [],
                'text_detected': [],
                'colors_detected': [],
                'layout_info': {}
            }
            
            if not os.path.exists(screenshot_path):
                return analysis_results
            
            # 基础图像分析
            if Image and cv2 and np:
                # 加载图像
                img = Image.open(screenshot_path)
                img_array = np.array(img)
                
                # 颜色分析
                dominant_colors = self._analyze_colors(img_array)
                analysis_results['colors_detected'] = dominant_colors
                
                # 布局分析
                layout_info = self._analyze_layout(img_array)
                analysis_results['layout_info'] = layout_info
                
                # 模拟元素检测
                elements = self._detect_ui_elements(img_array)
                analysis_results['elements_detected'] = elements
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot: {e}")
            return {
                'analyzed_at': get_iso_timestamp(),
                'error': str(e)
            }
    
    def _analyze_colors(self, img_array) -> List[Dict[str, Any]]:
        """分析图像主要颜色"""
        try:
            # 简化的颜色分析
            colors = []
            
            # 计算平均颜色
            mean_color = np.mean(img_array, axis=(0, 1))
            colors.append({
                'type': 'average',
                'rgb': [int(c) for c in mean_color],
                'hex': '#{:02x}{:02x}{:02x}'.format(*[int(c) for c in mean_color])
            })
            
            return colors
            
        except Exception:
            return []
    
    def _analyze_layout(self, img_array) -> Dict[str, Any]:
        """分析图像布局"""
        try:
            height, width = img_array.shape[:2]
            
            return {
                'width': int(width),
                'height': int(height),
                'aspect_ratio': round(width / height, 2),
                'orientation': 'portrait' if height > width else 'landscape'
            }
            
        except Exception:
            return {}
    
    def _detect_ui_elements(self, img_array) -> List[Dict[str, Any]]:
        """检测UI元素（模拟实现）"""
        try:
            # 这里应该集成实际的UI元素检测算法
            # 例如使用计算机视觉、机器学习模型等
            
            elements = [
                {
                    'type': 'button',
                    'bounds': {'left': 50, 'top': 200, 'right': 1030, 'bottom': 250},
                    'text': 'Button 1',
                    'confidence': 0.95
                },
                {
                    'type': 'button',
                    'bounds': {'left': 50, 'top': 300, 'right': 1030, 'bottom': 350},
                    'text': 'Button 2',
                    'confidence': 0.92
                }
            ]
            
            return elements
            
        except Exception:
            return []


class ElementDetectionTool(GUITool):
    """元素检测工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ElementDetectionTool",
            description="Detect UI elements on mobile devices",
            tool_type=ToolType.ADVANCED,
            **kwargs
        )
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证元素检测参数"""
        try:
            if not parameters.target and not parameters.text:
                logger.error("Either target element or text to find is required")
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
        """执行元素检测操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 获取当前屏幕截图
            screenshot_path = await self._get_current_screenshot(context)
            
            if not screenshot_path:
                raise ToolError(
                    "Failed to get screenshot for element detection",
                    "SCREENSHOT_FAILED",
                    self.tool_id
                )
            
            # 执行元素检测
            detected_elements = await self._detect_elements(
                screenshot_path,
                target=parameters.target,
                text=parameters.text,
                context=context
            )
            
            # 过滤和排序结果
            filtered_elements = self._filter_and_sort_elements(
                detected_elements,
                parameters
            )
            
            success = len(filtered_elements) > 0
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'detected_elements': filtered_elements,
                    'total_elements': len(detected_elements),
                    'filtered_elements': len(filtered_elements),
                    'screenshot_path': screenshot_path
                },
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'DETECTION_ERROR')
            
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
    
    async def _get_current_screenshot(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """获取当前屏幕截图"""
        try:
            # 如果上下文中已有截图，直接使用
            if context and 'current_screenshot' in context:
                return context['current_screenshot']
            
            # 否则创建新的截图
            screenshot_tool = ScreenshotTool()
            screenshot_params = ToolParameters()
            
            result = await screenshot_tool.execute(screenshot_params, context)
            
            if result.success and result.screenshot_path:
                return result.screenshot_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting screenshot: {e}")
            return None
    
    async def _detect_elements(
        self,
        screenshot_path: str,
        target: Optional[str],
        text: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """检测屏幕上的UI元素"""
        try:
            elements = []
            
            # 这里应该集成实际的元素检测实现
            # 例如使用UI自动化框架、计算机视觉、OCR等
            
            # 模拟元素检测结果
            mock_elements = [
                {
                    'id': 'button1',
                    'type': 'button',
                    'text': 'Login',
                    'bounds': {'left': 100, 'top': 500, 'right': 300, 'bottom': 550},
                    'center': {'x': 200, 'y': 525},
                    'visible': True,
                    'enabled': True,
                    'confidence': 0.95
                },
                {
                    'id': 'input1',
                    'type': 'input',
                    'text': '',
                    'placeholder': 'Username',
                    'bounds': {'left': 50, 'top': 300, 'right': 350, 'bottom': 350},
                    'center': {'x': 200, 'y': 325},
                    'visible': True,
                    'enabled': True,
                    'confidence': 0.88
                },
                {
                    'id': 'text1',
                    'type': 'text',
                    'text': 'Welcome to App',
                    'bounds': {'left': 50, 'top': 100, 'right': 350, 'bottom': 150},
                    'center': {'x': 200, 'y': 125},
                    'visible': True,
                    'enabled': True,
                    'confidence': 0.92
                }
            ]
            
            # 根据目标过滤元素
            if target:
                for element in mock_elements:
                    if (
                        element.get('id') == target or
                        element.get('type') == target or
                        target.lower() in element.get('text', '').lower()
                    ):
                        elements.append(element)
            
            # 根据文本过滤元素
            if text:
                for element in mock_elements:
                    if (
                        text.lower() in element.get('text', '').lower() or
                        text.lower() in element.get('placeholder', '').lower()
                    ):
                        if element not in elements:
                            elements.append(element)
            
            # 如果没有指定目标，返回所有元素
            if not target and not text:
                elements = mock_elements
            
            return elements
            
        except Exception as e:
            logger.error(f"Error detecting elements: {e}")
            return []
    
    def _filter_and_sort_elements(
        self,
        elements: List[Dict[str, Any]],
        parameters: ToolParameters
    ) -> List[Dict[str, Any]]:
        """过滤和排序检测到的元素"""
        try:
            filtered = elements.copy()
            
            # 过滤不可见或不可用的元素
            if parameters.custom_params and parameters.custom_params.get('visible_only', True):
                filtered = [e for e in filtered if e.get('visible', True)]
            
            if parameters.custom_params and parameters.custom_params.get('enabled_only', True):
                filtered = [e for e in filtered if e.get('enabled', True)]
            
            # 按置信度排序
            filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # 限制返回数量
            max_results = parameters.custom_params.get('max_results', 10) if parameters.custom_params else 10
            filtered = filtered[:max_results]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering elements: {e}")
            return elements


class OCRTool(GUITool):
    """OCR文字识别工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="OCRTool",
            description="Perform OCR text recognition on mobile devices",
            tool_type=ToolType.ADVANCED,
            **kwargs
        )
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证OCR参数"""
        try:
            # OCR工具可以不需要特殊参数
            if parameters.custom_params:
                language = parameters.custom_params.get('language')
                if language and not isinstance(language, str):
                    logger.error("Invalid language parameter")
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
        """执行OCR识别操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 获取截图或使用指定的图像
            image_path = None
            if parameters.custom_params and 'image_path' in parameters.custom_params:
                image_path = parameters.custom_params['image_path']
            else:
                image_path = await self._get_current_screenshot(context)
            
            if not image_path:
                raise ToolError(
                    "No image available for OCR",
                    "NO_IMAGE",
                    self.tool_id
                )
            
            # 获取OCR参数
            language = parameters.custom_params.get('language', 'en') if parameters.custom_params else 'en'
            region = parameters.custom_params.get('region') if parameters.custom_params else None
            
            # 执行OCR识别
            ocr_results = await self._perform_ocr(
                image_path,
                language=language,
                region=region,
                context=context
            )
            
            # 后处理OCR结果
            processed_results = self._process_ocr_results(
                ocr_results, parameters
            )
            
            success = len(processed_results.get('texts', [])) > 0
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'ocr_results': processed_results,
                    'language': language,
                    'image_path': image_path,
                    'total_texts': len(processed_results.get('texts', []))
                },
                screenshot_path=image_path
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'OCR_ERROR')
            
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
    
    async def _get_current_screenshot(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """获取当前屏幕截图"""
        try:
            if context and 'current_screenshot' in context:
                return context['current_screenshot']
            
            screenshot_tool = ScreenshotTool()
            screenshot_params = ToolParameters()
            
            result = await screenshot_tool.execute(screenshot_params, context)
            
            if result.success and result.screenshot_path:
                return result.screenshot_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting screenshot: {e}")
            return None
    
    async def _perform_ocr(
        self,
        image_path: str,
        language: str,
        region: Optional[Dict[str, int]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """执行OCR文字识别"""
        try:
            # 这里应该集成实际的OCR实现
            # 例如使用Tesseract、PaddleOCR、云端OCR服务等
            
            logger.info(f"Performing OCR on {image_path} with language {language}")
            
            # 模拟OCR结果
            mock_results = {
                'texts': [
                    {
                        'text': 'Welcome to App',
                        'confidence': 0.95,
                        'bounds': {'left': 50, 'top': 100, 'right': 350, 'bottom': 150},
                        'center': {'x': 200, 'y': 125}
                    },
                    {
                        'text': 'Username',
                        'confidence': 0.88,
                        'bounds': {'left': 60, 'top': 280, 'right': 150, 'bottom': 300},
                        'center': {'x': 105, 'y': 290}
                    },
                    {
                        'text': 'Login',
                        'confidence': 0.92,
                        'bounds': {'left': 150, 'top': 520, 'right': 250, 'bottom': 540},
                        'center': {'x': 200, 'y': 530}
                    }
                ],
                'language': language,
                'processing_time': 0.5,
                'image_info': {
                    'width': 1080,
                    'height': 1920,
                    'format': 'png'
                }
            }
            
            # 如果指定了区域，过滤结果
            if region:
                filtered_texts = []
                for text_item in mock_results['texts']:
                    bounds = text_item['bounds']
                    if (
                        bounds['left'] >= region.get('left', 0) and
                        bounds['top'] >= region.get('top', 0) and
                        bounds['right'] <= region.get('right', float('inf')) and
                        bounds['bottom'] <= region.get('bottom', float('inf'))
                    ):
                        filtered_texts.append(text_item)
                mock_results['texts'] = filtered_texts
            
            return mock_results
            
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            return {
                'texts': [],
                'error': str(e),
                'language': language
            }
    
    def _process_ocr_results(
        self,
        ocr_results: Dict[str, Any],
        parameters: ToolParameters
    ) -> Dict[str, Any]:
        """后处理OCR结果"""
        try:
            processed = ocr_results.copy()
            
            # 过滤低置信度的文本
            min_confidence = parameters.custom_params.get('min_confidence', 0.5) if parameters.custom_params else 0.5
            
            filtered_texts = [
                text for text in processed.get('texts', [])
                if text.get('confidence', 0) >= min_confidence
            ]
            
            # 按置信度排序
            filtered_texts.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            processed['texts'] = filtered_texts
            processed['filtered_count'] = len(filtered_texts)
            processed['original_count'] = len(ocr_results.get('texts', []))
            
            # 提取纯文本
            processed['plain_text'] = ' '.join([t['text'] for t in filtered_texts])
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing OCR results: {e}")
            return ocr_results


class ImageComparisonTool(GUITool):
    """图像比较工具"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ImageComparisonTool",
            description="Compare images to detect changes",
            tool_type=ToolType.ADVANCED,
            **kwargs
        )
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证图像比较参数"""
        try:
            if not parameters.custom_params:
                logger.error("Image comparison requires custom parameters")
                return False
            
            if 'reference_image' not in parameters.custom_params:
                logger.error("Reference image is required")
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
        """执行图像比较操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 获取参考图像和当前图像
            reference_image = parameters.custom_params['reference_image']
            current_image = parameters.custom_params.get('current_image')
            
            if not current_image:
                current_image = await self._get_current_screenshot(context)
            
            if not current_image:
                raise ToolError(
                    "No current image available for comparison",
                    "NO_CURRENT_IMAGE",
                    self.tool_id
                )
            
            # 执行图像比较
            comparison_results = await self._compare_images(
                reference_image,
                current_image,
                parameters.custom_params,
                context
            )
            
            success = comparison_results is not None
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED if success else ToolStatus.FAILED,
                success=success,
                start_time=start_time,
                end_time=end_time,
                result_data={
                    'comparison_results': comparison_results,
                    'reference_image': reference_image,
                    'current_image': current_image
                },
                screenshot_path=current_image
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'COMPARISON_ERROR')
            
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
    
    async def _get_current_screenshot(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """获取当前屏幕截图"""
        try:
            if context and 'current_screenshot' in context:
                return context['current_screenshot']
            
            screenshot_tool = ScreenshotTool()
            screenshot_params = ToolParameters()
            
            result = await screenshot_tool.execute(screenshot_params, context)
            
            if result.success and result.screenshot_path:
                return result.screenshot_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting screenshot: {e}")
            return None
    
    async def _compare_images(
        self,
        reference_path: str,
        current_path: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """比较两张图像"""
        try:
            if not os.path.exists(reference_path) or not os.path.exists(current_path):
                raise ToolError(
                    "One or both image files do not exist",
                    "FILE_NOT_FOUND",
                    self.tool_id
                )
            
            # 这里应该集成实际的图像比较实现
            # 例如使用OpenCV、PIL、SSIM等
            
            comparison_results = {
                'similarity_score': 0.85,  # 模拟相似度分数
                'difference_percentage': 15.0,
                'changed_regions': [
                    {
                        'bounds': {'left': 100, 'top': 200, 'right': 300, 'bottom': 250},
                        'change_type': 'color_change',
                        'confidence': 0.9
                    }
                ],
                'comparison_method': params.get('method', 'pixel_diff'),
                'threshold': params.get('threshold', 0.1),
                'processing_time': 0.3
            }
            
            # 根据阈值判断是否有显著变化
            threshold = params.get('threshold', 0.1)
            has_significant_change = comparison_results['difference_percentage'] / 100 > threshold
            
            comparison_results['has_significant_change'] = has_significant_change
            comparison_results['change_detected'] = has_significant_change
            
            return comparison_results
            
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            logger.error(f"Error comparing images: {e}")
            return None