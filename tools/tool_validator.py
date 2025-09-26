#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Tool Validator
工具验证器：负责工具执行前后的验证、质量检查和结果评估

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .gui_tools import (
    GUITool, ToolParameters, ToolResult, ToolError,
    ToolType, ToolStatus, Coordinate, Rectangle
)
from utils import get_iso_timestamp, setup_logger


class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"          # 基础验证
    STANDARD = "standard"    # 标准验证
    STRICT = "strict"        # 严格验证
    COMPREHENSIVE = "comprehensive"  # 全面验证


class ValidationResult(Enum):
    """验证结果"""
    PASSED = "passed"        # 通过
    WARNING = "warning"      # 警告
    FAILED = "failed"        # 失败
    SKIPPED = "skipped"      # 跳过


@dataclass
class ValidationIssue:
    """验证问题"""
    issue_id: str
    severity: str  # error, warning, info
    category: str  # parameter, precondition, postcondition, performance
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    timestamp: str = field(default_factory=get_iso_timestamp)


@dataclass
class ValidationReport:
    """验证报告"""
    validation_id: str
    tool_id: str
    tool_name: str
    validation_type: str  # pre_execution, post_execution, result_quality
    level: ValidationLevel
    result: ValidationResult
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=get_iso_timestamp)
    
    @property
    def has_errors(self) -> bool:
        """是否有错误"""
        return any(issue.severity == "error" for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """是否有警告"""
        return any(issue.severity == "warning" for issue in self.issues)
    
    @property
    def error_count(self) -> int:
        """错误数量"""
        return sum(1 for issue in self.issues if issue.severity == "error")
    
    @property
    def warning_count(self) -> int:
        """警告数量"""
        return sum(1 for issue in self.issues if issue.severity == "warning")


class BaseValidator(ABC):
    """验证器基类"""
    
    def __init__(self, name: str, level: ValidationLevel = ValidationLevel.STANDARD):
        self.name = name
        self.level = level
        self.logger = logger
        self.enabled = True
    
    @abstractmethod
    async def validate(
        self,
        tool: GUITool,
        parameters: Optional[ToolParameters] = None,
        result: Optional[ToolResult] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """执行验证"""
        pass
    
    def create_issue(
        self,
        issue_id: str,
        severity: str,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ) -> ValidationIssue:
        """创建验证问题"""
        return ValidationIssue(
            issue_id=issue_id,
            severity=severity,
            category=category,
            message=message,
            details=details,
            suggestion=suggestion
        )
    
    def create_report(
        self,
        validation_id: str,
        tool: GUITool,
        validation_type: str,
        result: ValidationResult,
        issues: List[ValidationIssue],
        metrics: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0
    ) -> ValidationReport:
        """创建验证报告"""
        return ValidationReport(
            validation_id=validation_id,
            tool_id=tool.tool_id,
            tool_name=tool.name,
            validation_type=validation_type,
            level=self.level,
            result=result,
            issues=issues,
            metrics=metrics or {},
            execution_time=execution_time
        )


class ParameterValidator(BaseValidator):
    """参数验证器"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__("ParameterValidator", level)
        
        # 参数验证规则
        self.validation_rules = {
            ToolType.CLICK: self._validate_click_parameters,
            ToolType.SWIPE: self._validate_swipe_parameters,
            ToolType.TEXT_INPUT: self._validate_text_input_parameters,
            ToolType.KEY_PRESS: self._validate_key_press_parameters,
            ToolType.SCREENSHOT: self._validate_screenshot_parameters,
            ToolType.ELEMENT_DETECTION: self._validate_element_detection_parameters,
            ToolType.OCR: self._validate_ocr_parameters,
            ToolType.IMAGE_COMPARISON: self._validate_image_comparison_parameters
        }
    
    async def validate(
        self,
        tool: GUITool,
        parameters: Optional[ToolParameters] = None,
        result: Optional[ToolResult] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """验证参数"""
        start_time = time.time()
        validation_id = f"param_val_{int(time.time() * 1000)}"
        issues = []
        
        if not parameters:
            issues.append(self.create_issue(
                "MISSING_PARAMETERS",
                "error",
                "parameter",
                "Tool parameters are required but not provided"
            ))
            
            return self.create_report(
                validation_id, tool, "pre_execution",
                ValidationResult.FAILED, issues,
                execution_time=time.time() - start_time
            )
        
        # 基础参数验证
        issues.extend(await self._validate_basic_parameters(parameters))
        
        # 工具特定参数验证
        if tool.tool_type in self.validation_rules:
            tool_issues = await self.validation_rules[tool.tool_type](parameters)
            issues.extend(tool_issues)
        
        # 根据验证级别进行额外检查
        if self.level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            issues.extend(await self._validate_advanced_parameters(tool, parameters))
        
        # 确定验证结果
        if any(issue.severity == "error" for issue in issues):
            result = ValidationResult.FAILED
        elif any(issue.severity == "warning" for issue in issues):
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.PASSED
        
        return self.create_report(
            validation_id, tool, "pre_execution", result, issues,
            execution_time=time.time() - start_time
        )
    
    async def _validate_basic_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """基础参数验证"""
        issues = []
        
        # 检查必需的基础字段
        if not hasattr(parameters, 'data') or not parameters.data:
            issues.append(self.create_issue(
                "EMPTY_PARAMETERS",
                "error",
                "parameter",
                "Parameter data is empty"
            ))
            return issues
        
        # 检查参数类型
        if not isinstance(parameters.data, dict):
            issues.append(self.create_issue(
                "INVALID_PARAMETER_TYPE",
                "error",
                "parameter",
                f"Parameter data must be dict, got {type(parameters.data)}"
            ))
        
        return issues
    
    async def _validate_click_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证点击参数"""
        issues = []
        data = parameters.data
        
        # 检查坐标
        if 'coordinate' not in data:
            issues.append(self.create_issue(
                "MISSING_COORDINATE",
                "error",
                "parameter",
                "Click operation requires coordinate parameter"
            ))
        else:
            coord = data['coordinate']
            if isinstance(coord, dict):
                if 'x' not in coord or 'y' not in coord:
                    issues.append(self.create_issue(
                        "INVALID_COORDINATE",
                        "error",
                        "parameter",
                        "Coordinate must have x and y values"
                    ))
                else:
                    # 检查坐标范围
                    x, y = coord['x'], coord['y']
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        issues.append(self.create_issue(
                            "INVALID_COORDINATE_TYPE",
                            "error",
                            "parameter",
                            "Coordinate x and y must be numbers"
                        ))
                    elif x < 0 or y < 0:
                        issues.append(self.create_issue(
                            "NEGATIVE_COORDINATE",
                            "warning",
                            "parameter",
                            "Coordinate values should not be negative",
                            suggestion="Check if coordinate is within screen bounds"
                        ))
                    elif x > 10000 or y > 10000:  # 假设最大屏幕尺寸
                        issues.append(self.create_issue(
                            "LARGE_COORDINATE",
                            "warning",
                            "parameter",
                            "Coordinate values seem unusually large",
                            suggestion="Verify coordinate is within screen bounds"
                        ))
        
        # 检查点击类型
        if 'click_type' in data:
            click_type = data['click_type']
            valid_types = ['single', 'double', 'long']
            if click_type not in valid_types:
                issues.append(self.create_issue(
                    "INVALID_CLICK_TYPE",
                    "warning",
                    "parameter",
                    f"Unknown click type: {click_type}",
                    details={'valid_types': valid_types},
                    suggestion="Use one of the valid click types"
                ))
        
        return issues
    
    async def _validate_swipe_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证滑动参数"""
        issues = []
        data = parameters.data
        
        # 检查起始和结束坐标
        required_coords = ['start_coordinate', 'end_coordinate']
        for coord_name in required_coords:
            if coord_name not in data:
                issues.append(self.create_issue(
                    f"MISSING_{coord_name.upper()}",
                    "error",
                    "parameter",
                    f"Swipe operation requires {coord_name} parameter"
                ))
            else:
                coord = data[coord_name]
                if not isinstance(coord, dict) or 'x' not in coord or 'y' not in coord:
                    issues.append(self.create_issue(
                        f"INVALID_{coord_name.upper()}",
                        "error",
                        "parameter",
                        f"{coord_name} must have x and y values"
                    ))
        
        # 检查滑动距离
        if 'start_coordinate' in data and 'end_coordinate' in data:
            start = data['start_coordinate']
            end = data['end_coordinate']
            
            if (isinstance(start, dict) and isinstance(end, dict) and
                all(k in start for k in ['x', 'y']) and
                all(k in end for k in ['x', 'y'])):
                
                distance = ((end['x'] - start['x']) ** 2 + (end['y'] - start['y']) ** 2) ** 0.5
                
                if distance < 10:
                    issues.append(self.create_issue(
                        "SHORT_SWIPE_DISTANCE",
                        "warning",
                        "parameter",
                        "Swipe distance is very short, may not be effective",
                        details={'distance': distance},
                        suggestion="Increase swipe distance for better reliability"
                    ))
                elif distance > 2000:
                    issues.append(self.create_issue(
                        "LONG_SWIPE_DISTANCE",
                        "warning",
                        "parameter",
                        "Swipe distance is very long, may be unrealistic",
                        details={'distance': distance},
                        suggestion="Check if swipe distance is appropriate"
                    ))
        
        # 检查滑动持续时间
        if 'duration' in data:
            duration = data['duration']
            if not isinstance(duration, (int, float)):
                issues.append(self.create_issue(
                    "INVALID_DURATION_TYPE",
                    "error",
                    "parameter",
                    "Duration must be a number"
                ))
            elif duration <= 0:
                issues.append(self.create_issue(
                    "INVALID_DURATION_VALUE",
                    "error",
                    "parameter",
                    "Duration must be positive"
                ))
            elif duration > 10000:  # 10秒
                issues.append(self.create_issue(
                    "LONG_DURATION",
                    "warning",
                    "parameter",
                    "Swipe duration is very long",
                    suggestion="Consider shorter duration for better user experience"
                ))
        
        return issues
    
    async def _validate_text_input_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证文本输入参数"""
        issues = []
        data = parameters.data
        
        # 检查文本内容
        if 'text' not in data:
            issues.append(self.create_issue(
                "MISSING_TEXT",
                "error",
                "parameter",
                "Text input operation requires text parameter"
            ))
        else:
            text = data['text']
            if not isinstance(text, str):
                issues.append(self.create_issue(
                    "INVALID_TEXT_TYPE",
                    "error",
                    "parameter",
                    "Text parameter must be a string"
                ))
            elif len(text) == 0:
                issues.append(self.create_issue(
                    "EMPTY_TEXT",
                    "warning",
                    "parameter",
                    "Text input is empty"
                ))
            elif len(text) > 10000:
                issues.append(self.create_issue(
                    "LONG_TEXT",
                    "warning",
                    "parameter",
                    "Text input is very long",
                    details={'length': len(text)},
                    suggestion="Consider breaking long text into smaller chunks"
                ))
        
        # 检查输入方式
        if 'input_method' in data:
            input_method = data['input_method']
            valid_methods = ['type', 'paste', 'replace']
            if input_method not in valid_methods:
                issues.append(self.create_issue(
                    "INVALID_INPUT_METHOD",
                    "warning",
                    "parameter",
                    f"Unknown input method: {input_method}",
                    details={'valid_methods': valid_methods}
                ))
        
        return issues
    
    async def _validate_key_press_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证按键参数"""
        issues = []
        data = parameters.data
        
        # 检查按键
        if 'key' not in data:
            issues.append(self.create_issue(
                "MISSING_KEY",
                "error",
                "parameter",
                "Key press operation requires key parameter"
            ))
        else:
            key = data['key']
            if not isinstance(key, str):
                issues.append(self.create_issue(
                    "INVALID_KEY_TYPE",
                    "error",
                    "parameter",
                    "Key parameter must be a string"
                ))
            elif len(key) == 0:
                issues.append(self.create_issue(
                    "EMPTY_KEY",
                    "error",
                    "parameter",
                    "Key parameter cannot be empty"
                ))
        
        # 检查修饰键
        if 'modifiers' in data:
            modifiers = data['modifiers']
            if not isinstance(modifiers, list):
                issues.append(self.create_issue(
                    "INVALID_MODIFIERS_TYPE",
                    "error",
                    "parameter",
                    "Modifiers must be a list"
                ))
            else:
                valid_modifiers = ['ctrl', 'alt', 'shift', 'cmd', 'meta']
                for modifier in modifiers:
                    if modifier not in valid_modifiers:
                        issues.append(self.create_issue(
                            "INVALID_MODIFIER",
                            "warning",
                            "parameter",
                            f"Unknown modifier: {modifier}",
                            details={'valid_modifiers': valid_modifiers}
                        ))
        
        return issues
    
    async def _validate_screenshot_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证截图参数"""
        issues = []
        data = parameters.data
        
        # 检查区域参数
        if 'region' in data:
            region = data['region']
            if isinstance(region, dict):
                required_fields = ['x', 'y', 'width', 'height']
                for field in required_fields:
                    if field not in region:
                        issues.append(self.create_issue(
                            f"MISSING_REGION_{field.upper()}",
                            "error",
                            "parameter",
                            f"Region parameter missing {field}"
                        ))
                    elif not isinstance(region[field], (int, float)):
                        issues.append(self.create_issue(
                            f"INVALID_REGION_{field.upper()}_TYPE",
                            "error",
                            "parameter",
                            f"Region {field} must be a number"
                        ))
                    elif region[field] < 0:
                        issues.append(self.create_issue(
                            f"NEGATIVE_REGION_{field.upper()}",
                            "warning",
                            "parameter",
                            f"Region {field} should not be negative"
                        ))
        
        # 检查质量参数
        if 'quality' in data:
            quality = data['quality']
            if not isinstance(quality, (int, float)):
                issues.append(self.create_issue(
                    "INVALID_QUALITY_TYPE",
                    "error",
                    "parameter",
                    "Quality must be a number"
                ))
            elif not 0 <= quality <= 100:
                issues.append(self.create_issue(
                    "INVALID_QUALITY_RANGE",
                    "error",
                    "parameter",
                    "Quality must be between 0 and 100"
                ))
        
        return issues
    
    async def _validate_element_detection_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证元素检测参数"""
        issues = []
        data = parameters.data
        
        # 检查检测类型
        if 'detection_type' in data:
            detection_type = data['detection_type']
            valid_types = ['text', 'image', 'color', 'shape']
            if detection_type not in valid_types:
                issues.append(self.create_issue(
                    "INVALID_DETECTION_TYPE",
                    "error",
                    "parameter",
                    f"Unknown detection type: {detection_type}",
                    details={'valid_types': valid_types}
                ))
        
        # 检查目标参数
        if 'target' not in data:
            issues.append(self.create_issue(
                "MISSING_TARGET",
                "error",
                "parameter",
                "Element detection requires target parameter"
            ))
        
        # 检查置信度阈值
        if 'confidence_threshold' in data:
            threshold = data['confidence_threshold']
            if not isinstance(threshold, (int, float)):
                issues.append(self.create_issue(
                    "INVALID_THRESHOLD_TYPE",
                    "error",
                    "parameter",
                    "Confidence threshold must be a number"
                ))
            elif not 0 <= threshold <= 1:
                issues.append(self.create_issue(
                    "INVALID_THRESHOLD_RANGE",
                    "error",
                    "parameter",
                    "Confidence threshold must be between 0 and 1"
                ))
        
        return issues
    
    async def _validate_ocr_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证OCR参数"""
        issues = []
        data = parameters.data
        
        # 检查语言参数
        if 'language' in data:
            language = data['language']
            if not isinstance(language, str):
                issues.append(self.create_issue(
                    "INVALID_LANGUAGE_TYPE",
                    "error",
                    "parameter",
                    "Language must be a string"
                ))
            elif len(language) == 0:
                issues.append(self.create_issue(
                    "EMPTY_LANGUAGE",
                    "warning",
                    "parameter",
                    "Language parameter is empty"
                ))
        
        # 检查OCR模式
        if 'mode' in data:
            mode = data['mode']
            valid_modes = ['text', 'word', 'char', 'block']
            if mode not in valid_modes:
                issues.append(self.create_issue(
                    "INVALID_OCR_MODE",
                    "warning",
                    "parameter",
                    f"Unknown OCR mode: {mode}",
                    details={'valid_modes': valid_modes}
                ))
        
        return issues
    
    async def _validate_image_comparison_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证图像比较参数"""
        issues = []
        data = parameters.data
        
        # 检查比较图像
        required_images = ['reference_image']
        for image_param in required_images:
            if image_param not in data:
                issues.append(self.create_issue(
                    f"MISSING_{image_param.upper()}",
                    "error",
                    "parameter",
                    f"Image comparison requires {image_param} parameter"
                ))
        
        # 检查相似度阈值
        if 'similarity_threshold' in data:
            threshold = data['similarity_threshold']
            if not isinstance(threshold, (int, float)):
                issues.append(self.create_issue(
                    "INVALID_SIMILARITY_THRESHOLD_TYPE",
                    "error",
                    "parameter",
                    "Similarity threshold must be a number"
                ))
            elif not 0 <= threshold <= 1:
                issues.append(self.create_issue(
                    "INVALID_SIMILARITY_THRESHOLD_RANGE",
                    "error",
                    "parameter",
                    "Similarity threshold must be between 0 and 1"
                ))
        
        # 检查比较方法
        if 'comparison_method' in data:
            method = data['comparison_method']
            valid_methods = ['ssim', 'mse', 'histogram', 'template']
            if method not in valid_methods:
                issues.append(self.create_issue(
                    "INVALID_COMPARISON_METHOD",
                    "warning",
                    "parameter",
                    f"Unknown comparison method: {method}",
                    details={'valid_methods': valid_methods}
                ))
        
        return issues
    
    async def _validate_advanced_parameters(self, tool: GUITool, parameters: ToolParameters) -> List[ValidationIssue]:
        """高级参数验证"""
        issues = []
        
        # 检查参数完整性
        if hasattr(tool, 'required_parameters'):
            required_params = getattr(tool, 'required_parameters', [])
            for param in required_params:
                if param not in parameters.data:
                    issues.append(self.create_issue(
                        f"MISSING_REQUIRED_PARAM_{param.upper()}",
                        "error",
                        "parameter",
                        f"Required parameter '{param}' is missing"
                    ))
        
        # 检查参数组合的有效性
        issues.extend(await self._validate_parameter_combinations(tool, parameters))
        
        # 检查性能相关参数
        issues.extend(await self._validate_performance_parameters(parameters))
        
        return issues
    
    async def _validate_parameter_combinations(self, tool: GUITool, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证参数组合"""
        issues = []
        data = parameters.data
        
        # 检查互斥参数
        mutually_exclusive_groups = [
            ['coordinate', 'element_selector'],
            ['text', 'image_template']
        ]
        
        for group in mutually_exclusive_groups:
            present_params = [param for param in group if param in data]
            if len(present_params) > 1:
                issues.append(self.create_issue(
                    "MUTUALLY_EXCLUSIVE_PARAMS",
                    "error",
                    "parameter",
                    f"Parameters {present_params} are mutually exclusive",
                    details={'conflicting_params': present_params}
                ))
        
        # 检查依赖参数
        dependency_rules = {
            'region': ['screenshot_required'],
            'confidence_threshold': ['detection_type'],
            'similarity_threshold': ['reference_image']
        }
        
        for param, dependencies in dependency_rules.items():
            if param in data:
                missing_deps = [dep for dep in dependencies if dep not in data]
                if missing_deps:
                    issues.append(self.create_issue(
                        "MISSING_DEPENDENT_PARAMS",
                        "warning",
                        "parameter",
                        f"Parameter '{param}' requires {missing_deps}",
                        details={'missing_dependencies': missing_deps}
                    ))
        
        return issues
    
    async def _validate_performance_parameters(self, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证性能相关参数"""
        issues = []
        data = parameters.data
        
        # 检查超时设置
        if 'timeout' in data:
            timeout = data['timeout']
            if isinstance(timeout, (int, float)):
                if timeout <= 0:
                    issues.append(self.create_issue(
                        "INVALID_TIMEOUT",
                        "error",
                        "parameter",
                        "Timeout must be positive"
                    ))
                elif timeout > 300:  # 5分钟
                    issues.append(self.create_issue(
                        "LONG_TIMEOUT",
                        "warning",
                        "parameter",
                        "Timeout is very long, may affect performance",
                        suggestion="Consider shorter timeout for better responsiveness"
                    ))
        
        # 检查重试设置
        if 'max_retries' in data:
            max_retries = data['max_retries']
            if isinstance(max_retries, int):
                if max_retries < 0:
                    issues.append(self.create_issue(
                        "INVALID_MAX_RETRIES",
                        "error",
                        "parameter",
                        "Max retries cannot be negative"
                    ))
                elif max_retries > 10:
                    issues.append(self.create_issue(
                        "HIGH_MAX_RETRIES",
                        "warning",
                        "parameter",
                        "Max retries is very high, may affect performance",
                        suggestion="Consider lower retry count"
                    ))
        
        return issues


class ResultValidator(BaseValidator):
    """结果验证器"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__("ResultValidator", level)
    
    async def validate(
        self,
        tool: GUITool,
        parameters: Optional[ToolParameters] = None,
        result: Optional[ToolResult] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """验证执行结果"""
        start_time = time.time()
        validation_id = f"result_val_{int(time.time() * 1000)}"
        issues = []
        
        if not result:
            issues.append(self.create_issue(
                "MISSING_RESULT",
                "error",
                "postcondition",
                "Tool execution result is required but not provided"
            ))
            
            return self.create_report(
                validation_id, tool, "post_execution",
                ValidationResult.FAILED, issues,
                execution_time=time.time() - start_time
            )
        
        # 基础结果验证
        issues.extend(await self._validate_basic_result(result))
        
        # 工具特定结果验证
        issues.extend(await self._validate_tool_specific_result(tool, result))
        
        # 性能验证
        if self.level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            issues.extend(await self._validate_performance_result(result))
        
        # 质量验证
        if self.level == ValidationLevel.COMPREHENSIVE:
            issues.extend(await self._validate_quality_result(tool, result, parameters))
        
        # 确定验证结果
        if any(issue.severity == "error" for issue in issues):
            validation_result = ValidationResult.FAILED
        elif any(issue.severity == "warning" for issue in issues):
            validation_result = ValidationResult.WARNING
        else:
            validation_result = ValidationResult.PASSED
        
        # 计算质量指标
        metrics = await self._calculate_quality_metrics(tool, result, parameters)
        
        return self.create_report(
            validation_id, tool, "post_execution", validation_result, issues,
            metrics=metrics, execution_time=time.time() - start_time
        )
    
    async def _validate_basic_result(self, result: ToolResult) -> List[ValidationIssue]:
        """基础结果验证"""
        issues = []
        
        # 检查必需字段
        required_fields = ['tool_id', 'tool_type', 'status', 'success', 'start_time', 'end_time']
        for field in required_fields:
            if not hasattr(result, field) or getattr(result, field) is None:
                issues.append(self.create_issue(
                    f"MISSING_RESULT_FIELD_{field.upper()}",
                    "error",
                    "postcondition",
                    f"Result missing required field: {field}"
                ))
        
        # 检查状态一致性
        if hasattr(result, 'success') and hasattr(result, 'status'):
            if result.success and result.status == ToolStatus.FAILED:
                issues.append(self.create_issue(
                    "INCONSISTENT_SUCCESS_STATUS",
                    "error",
                    "postcondition",
                    "Result success=True but status=FAILED"
                ))
            elif not result.success and result.status == ToolStatus.COMPLETED:
                issues.append(self.create_issue(
                    "INCONSISTENT_FAILURE_STATUS",
                    "error",
                    "postcondition",
                    "Result success=False but status=COMPLETED"
                ))
        
        # 检查时间戳
        if hasattr(result, 'start_time') and hasattr(result, 'end_time'):
            if result.start_time and result.end_time:
                try:
                    # 简单的时间戳格式检查
                    if result.end_time < result.start_time:
                        issues.append(self.create_issue(
                            "INVALID_TIME_ORDER",
                            "error",
                            "postcondition",
                            "End time is before start time"
                        ))
                except Exception:
                    issues.append(self.create_issue(
                        "INVALID_TIMESTAMP_FORMAT",
                        "warning",
                        "postcondition",
                        "Invalid timestamp format"
                    ))
        
        return issues
    
    async def _validate_tool_specific_result(self, tool: GUITool, result: ToolResult) -> List[ValidationIssue]:
        """工具特定结果验证"""
        issues = []
        
        # 根据工具类型验证特定结果
        if tool.tool_type == ToolType.SCREENSHOT:
            issues.extend(await self._validate_screenshot_result(result))
        elif tool.tool_type == ToolType.ELEMENT_DETECTION:
            issues.extend(await self._validate_detection_result(result))
        elif tool.tool_type == ToolType.OCR:
            issues.extend(await self._validate_ocr_result(result))
        elif tool.tool_type == ToolType.IMAGE_COMPARISON:
            issues.extend(await self._validate_comparison_result(result))
        
        return issues
    
    async def _validate_screenshot_result(self, result: ToolResult) -> List[ValidationIssue]:
        """验证截图结果"""
        issues = []
        
        if result.success and result.data:
            # 检查图像数据
            if 'image_data' not in result.data:
                issues.append(self.create_issue(
                    "MISSING_IMAGE_DATA",
                    "error",
                    "postcondition",
                    "Screenshot result missing image data"
                ))
            
            # 检查图像尺寸
            if 'image_size' in result.data:
                size = result.data['image_size']
                if isinstance(size, dict) and 'width' in size and 'height' in size:
                    if size['width'] <= 0 or size['height'] <= 0:
                        issues.append(self.create_issue(
                            "INVALID_IMAGE_SIZE",
                            "error",
                            "postcondition",
                            "Invalid image dimensions"
                        ))
        
        return issues
    
    async def _validate_detection_result(self, result: ToolResult) -> List[ValidationIssue]:
        """验证检测结果"""
        issues = []
        
        if result.success and result.data:
            # 检查检测到的元素
            if 'elements' in result.data:
                elements = result.data['elements']
                if not isinstance(elements, list):
                    issues.append(self.create_issue(
                        "INVALID_ELEMENTS_TYPE",
                        "error",
                        "postcondition",
                        "Elements must be a list"
                    ))
                else:
                    for i, element in enumerate(elements):
                        if not isinstance(element, dict):
                            issues.append(self.create_issue(
                                f"INVALID_ELEMENT_{i}_TYPE",
                                "error",
                                "postcondition",
                                f"Element {i} must be a dict"
                            ))
                        elif 'confidence' in element:
                            confidence = element['confidence']
                            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                                issues.append(self.create_issue(
                                    f"INVALID_ELEMENT_{i}_CONFIDENCE",
                                    "warning",
                                    "postcondition",
                                    f"Element {i} has invalid confidence value"
                                ))
        
        return issues
    
    async def _validate_ocr_result(self, result: ToolResult) -> List[ValidationIssue]:
        """验证OCR结果"""
        issues = []
        
        if result.success and result.data:
            # 检查识别的文本
            if 'text' in result.data:
                text = result.data['text']
                if not isinstance(text, str):
                    issues.append(self.create_issue(
                        "INVALID_OCR_TEXT_TYPE",
                        "error",
                        "postcondition",
                        "OCR text must be a string"
                    ))
            
            # 检查置信度
            if 'confidence' in result.data:
                confidence = result.data['confidence']
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    issues.append(self.create_issue(
                        "INVALID_OCR_CONFIDENCE",
                        "warning",
                        "postcondition",
                        "OCR confidence value is invalid"
                    ))
                elif confidence < 0.5:
                    issues.append(self.create_issue(
                        "LOW_OCR_CONFIDENCE",
                        "warning",
                        "postcondition",
                        "OCR confidence is low, result may be unreliable",
                        details={'confidence': confidence}
                    ))
        
        return issues
    
    async def _validate_comparison_result(self, result: ToolResult) -> List[ValidationIssue]:
        """验证比较结果"""
        issues = []
        
        if result.success and result.data:
            # 检查相似度
            if 'similarity' in result.data:
                similarity = result.data['similarity']
                if not isinstance(similarity, (int, float)) or not 0 <= similarity <= 1:
                    issues.append(self.create_issue(
                        "INVALID_SIMILARITY_VALUE",
                        "error",
                        "postcondition",
                        "Similarity value must be between 0 and 1"
                    ))
            
            # 检查匹配结果
            if 'is_match' in result.data:
                is_match = result.data['is_match']
                if not isinstance(is_match, bool):
                    issues.append(self.create_issue(
                        "INVALID_MATCH_TYPE",
                        "error",
                        "postcondition",
                        "Match result must be a boolean"
                    ))
        
        return issues
    
    async def _validate_performance_result(self, result: ToolResult) -> List[ValidationIssue]:
        """验证性能结果"""
        issues = []
        
        # 检查执行时间
        if hasattr(result, 'execution_time'):
            exec_time = getattr(result, 'execution_time', None)
            if exec_time is not None:
                if exec_time < 0:
                    issues.append(self.create_issue(
                        "NEGATIVE_EXECUTION_TIME",
                        "error",
                        "performance",
                        "Execution time cannot be negative"
                    ))
                elif exec_time > 60:  # 1分钟
                    issues.append(self.create_issue(
                        "LONG_EXECUTION_TIME",
                        "warning",
                        "performance",
                        "Execution time is very long",
                        details={'execution_time': exec_time},
                        suggestion="Consider optimizing tool performance"
                    ))
        
        # 检查内存使用
        if result.data and 'memory_usage' in result.data:
            memory_usage = result.data['memory_usage']
            if isinstance(memory_usage, (int, float)):
                if memory_usage > 1000:  # 1GB
                    issues.append(self.create_issue(
                        "HIGH_MEMORY_USAGE",
                        "warning",
                        "performance",
                        "Memory usage is high",
                        details={'memory_usage_mb': memory_usage},
                        suggestion="Consider memory optimization"
                    ))
        
        return issues
    
    async def _validate_quality_result(self, tool: GUITool, result: ToolResult, parameters: Optional[ToolParameters]) -> List[ValidationIssue]:
        """验证结果质量"""
        issues = []
        
        # 检查结果完整性
        if result.success and not result.data:
            issues.append(self.create_issue(
                "EMPTY_SUCCESS_RESULT",
                "warning",
                "postcondition",
                "Successful result has no data"
            ))
        
        # 检查错误信息质量
        if not result.success:
            if not result.error_message:
                issues.append(self.create_issue(
                    "MISSING_ERROR_MESSAGE",
                    "warning",
                    "postcondition",
                    "Failed result missing error message"
                ))
            elif len(result.error_message) < 10:
                issues.append(self.create_issue(
                    "BRIEF_ERROR_MESSAGE",
                    "info",
                    "postcondition",
                    "Error message is very brief",
                    suggestion="Consider providing more detailed error information"
                ))
        
        # 检查数据一致性
        if result.data and parameters:
            issues.extend(await self._validate_data_consistency(result, parameters))
        
        return issues
    
    async def _validate_data_consistency(self, result: ToolResult, parameters: ToolParameters) -> List[ValidationIssue]:
        """验证数据一致性"""
        issues = []
        
        # 检查输入输出一致性
        if 'region' in parameters.data and 'image_size' in result.data:
            param_region = parameters.data['region']
            result_size = result.data['image_size']
            
            if (isinstance(param_region, dict) and isinstance(result_size, dict) and
                'width' in param_region and 'height' in param_region and
                'width' in result_size and 'height' in result_size):
                
                if (param_region['width'] != result_size['width'] or
                    param_region['height'] != result_size['height']):
                    issues.append(self.create_issue(
                        "INCONSISTENT_REGION_SIZE",
                        "warning",
                        "postcondition",
                        "Result image size doesn't match requested region",
                        details={
                            'requested': param_region,
                            'actual': result_size
                        }
                    ))
        
        return issues
    
    async def _calculate_quality_metrics(self, tool: GUITool, result: ToolResult, parameters: Optional[ToolParameters]) -> Dict[str, Any]:
        """计算质量指标"""
        metrics = {}
        
        # 基础指标
        metrics['success_rate'] = 1.0 if result.success else 0.0
        
        # 执行时间指标
        if hasattr(result, 'execution_time'):
            exec_time = getattr(result, 'execution_time', 0)
            metrics['execution_time'] = exec_time
            metrics['performance_score'] = max(0, 1 - exec_time / 30)  # 30秒为基准
        
        # 数据质量指标
        if result.data:
            metrics['data_completeness'] = len(result.data) / 10  # 假设10个字段为完整
            
            # 置信度指标
            if 'confidence' in result.data:
                metrics['confidence'] = result.data['confidence']
            
            # 准确性指标（基于工具类型）
            if tool.tool_type == ToolType.OCR and 'text' in result.data:
                text_length = len(result.data['text'])
                metrics['text_length'] = text_length
                metrics['text_quality_score'] = min(1.0, text_length / 100)
            
            elif tool.tool_type == ToolType.ELEMENT_DETECTION and 'elements' in result.data:
                element_count = len(result.data['elements'])
                metrics['element_count'] = element_count
                metrics['detection_score'] = min(1.0, element_count / 5)
        
        # 综合质量分数
        quality_factors = [
            metrics.get('success_rate', 0),
            metrics.get('performance_score', 0),
            metrics.get('data_completeness', 0),
            metrics.get('confidence', 0.5)
        ]
        
        metrics['overall_quality_score'] = sum(quality_factors) / len(quality_factors)
        
        return metrics


class ToolValidator:
    """工具验证器管理器"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.logger = logger
        
        # 验证器实例
        self.parameter_validator = ParameterValidator(level)
        self.result_validator = ResultValidator(level)
        
        # 验证历史
        self.validation_history: List[ValidationReport] = []
        self.max_history_size = 1000
        
        # 统计信息
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'warning_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0
        }
    
    async def validate_pre_execution(
        self,
        tool: GUITool,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """执行前验证"""
        report = await self.parameter_validator.validate(
            tool, parameters, None, context
        )
        
        await self._record_validation(report)
        return report
    
    async def validate_post_execution(
        self,
        tool: GUITool,
        result: ToolResult,
        parameters: Optional[ToolParameters] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """执行后验证"""
        report = await self.result_validator.validate(
            tool, parameters, result, context
        )
        
        await self._record_validation(report)
        return report
    
    async def validate_full_execution(
        self,
        tool: GUITool,
        parameters: ToolParameters,
        result: ToolResult,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ValidationReport, ValidationReport]:
        """完整执行验证"""
        pre_report = await self.validate_pre_execution(tool, parameters, context)
        post_report = await self.validate_post_execution(tool, result, parameters, context)
        
        return pre_report, post_report
    
    async def get_validation_history(
        self,
        tool_id: Optional[str] = None,
        validation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ValidationReport]:
        """获取验证历史"""
        history = self.validation_history.copy()
        
        # 过滤条件
        if tool_id:
            history = [r for r in history if r.tool_id == tool_id]
        
        if validation_type:
            history = [r for r in history if r.validation_type == validation_type]
        
        # 按时间倒序排列
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        return history[:limit]
    
    async def get_validation_stats(self) -> Dict[str, Any]:
        """获取验证统计"""
        return self.stats.copy()
    
    async def get_tool_quality_report(self, tool_id: str) -> Dict[str, Any]:
        """获取工具质量报告"""
        tool_reports = [r for r in self.validation_history if r.tool_id == tool_id]
        
        if not tool_reports:
            return {'error': 'No validation data found for tool'}
        
        # 计算质量指标
        total_reports = len(tool_reports)
        passed_reports = sum(1 for r in tool_reports if r.result == ValidationResult.PASSED)
        warning_reports = sum(1 for r in tool_reports if r.result == ValidationResult.WARNING)
        failed_reports = sum(1 for r in tool_reports if r.result == ValidationResult.FAILED)
        
        # 错误分析
        error_categories = {}
        for report in tool_reports:
            for issue in report.issues:
                if issue.severity == "error":
                    category = issue.category
                    error_categories[category] = error_categories.get(category, 0) + 1
        
        # 性能分析
        execution_times = [r.execution_time for r in tool_reports if r.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # 质量分数
        quality_scores = []
        for report in tool_reports:
            if 'overall_quality_score' in report.metrics:
                quality_scores.append(report.metrics['overall_quality_score'])
        
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'tool_id': tool_id,
            'total_validations': total_reports,
            'success_rate': passed_reports / total_reports,
            'warning_rate': warning_reports / total_reports,
            'failure_rate': failed_reports / total_reports,
            'average_execution_time': avg_execution_time,
            'average_quality_score': avg_quality_score,
            'common_error_categories': error_categories,
            'recent_reports': tool_reports[-10:]  # 最近10次验证
        }
    
    async def _record_validation(self, report: ValidationReport) -> None:
        """记录验证结果"""
        # 添加到历史记录
        self.validation_history.append(report)
        
        # 限制历史记录大小
        if len(self.validation_history) > self.max_history_size:
            self.validation_history = self.validation_history[-self.max_history_size:]
        
        # 更新统计信息
        self.stats['total_validations'] += 1
        
        if report.result == ValidationResult.PASSED:
            self.stats['passed_validations'] += 1
        elif report.result == ValidationResult.WARNING:
            self.stats['warning_validations'] += 1
        elif report.result == ValidationResult.FAILED:
            self.stats['failed_validations'] += 1
        
        # 更新平均验证时间
        total_time = (self.stats['average_validation_time'] * 
                     (self.stats['total_validations'] - 1) + report.execution_time)
        self.stats['average_validation_time'] = total_time / self.stats['total_validations']
        
        logger.debug(f"Validation recorded: {report.validation_id}, result: {report.result.value}")