#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent工具模块

基于AgenticX框架的通用工具函数，提供日志、配置加载、异常处理等功能。

本模块已完全基于AgenticX框架重构：
- 集成AgenticX日志系统
- 支持AgenticX配置管理
- 兼容AgenticX事件系统
- 提供AgenticX组件工具函数

Author: AgenticX Team
Date: 2025
Version: 1.0.0 (基于AgenticX框架重构)
"""

import os
import re
import sys
import yaml
import json
from loguru import logger
import asyncio
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
from functools import wraps


def setup_logger(
    name: str = "agenticx-guiagent",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
):
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
    
    Returns:
        配置好的 loguru logger
    """
    # 移除默认处理器
    logger.remove()
    
    # 设置日志格式
    if format_string is None:
        format_string = (
            "{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - "
            "[{file}:{line}] - {message}"
        )
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format=format_string,
        level=level.upper(),
        colorize=True
    )
    
    # 添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=level.upper(),
            encoding='utf-8',
            rotation="10 MB",
            retention="30 days"
        )
    
    return logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """加载配置文件并递归替换环境变量.
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        def resolve_placeholders(obj):
            if isinstance(obj, dict):
                return {k: resolve_placeholders(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_placeholders(elem) for elem in obj]
            elif isinstance(obj, str):
                placeholder_pattern = re.compile(r'\$\{(.*?)\}')
                
                def replace_match(match):
                    content = match.group(1)
                    if ':-' in content:
                        var_name, default_value = content.split(':-', 1)
                    else:
                        var_name, default_value = content, None

                    env_value = os.getenv(var_name)
                    
                    if env_value is not None:
                        return env_value
                    
                    if default_value is not None:
                        return default_value
                    
                    raise ValueError(
                        f"Configuration error: Environment variable '{var_name}' is not set and no default value is provided."
                    )

                return placeholder_pattern.sub(replace_match, obj)
            
            return obj

        resolved_config = resolve_placeholders(config_data)

        if not isinstance(resolved_config, dict):
            raise ValueError("配置文件内容不是有效的字典格式")
            
        return resolved_config

    except Exception as e:
        raise ValueError(f"配置文件解析或变量替换错误: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """异步重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟倍数
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                        f"{current_delay:.1f}秒后重试"
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def sync_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """同步重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟倍数
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                        f"{current_delay:.1f}秒后重试"
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_iso_timestamp() -> str:
    """获取ISO格式时间戳"""
    return datetime.now().isoformat()


def ensure_directory(path: Union[str, Path]) -> Path:
    """确保目录存在
    
    Args:
        path: 目录路径
    
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """安全的JSON解析
    
    Args:
        json_str: JSON字符串
        default: 解析失败时的默认值
    
    Returns:
        解析结果或默认值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """安全的JSON序列化
    
    Args:
        obj: 要序列化的对象
        default: 序列化失败时的默认值
    
    Returns:
        JSON字符串或默认值
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并字典
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
    
    Returns:
        合并后的字典
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """验证必需字段
    
    Args:
        data: 数据字典
        required_fields: 必需字段列表
    
    Raises:
        ValueError: 缺少必需字段
    """
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"缺少必需字段: {', '.join(missing_fields)}")


class AsyncContextManager:
    """异步上下文管理器基类"""
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def setup(self):
        """设置资源"""
        pass
    
    async def cleanup(self):
        """清理资源"""
        pass


class SingletonMeta(type):
    """单例元类"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def format_exception(e: Exception) -> str:
    """格式化异常信息
    
    Args:
        e: 异常对象
    
    Returns:
        格式化的异常信息
    """
    import traceback
    return f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断字符串
    
    Args:
        s: 原字符串
        max_length: 最大长度
        suffix: 截断后缀
    
    Returns:
        截断后的字符串
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


# ==================== AgenticX集成工具函数 ====================

def create_agenticx_event(event_type: str, data: Dict[str, Any], source: str = "agenticx-guiagent") -> Dict[str, Any]:
    """创建AgenticX事件
    
    Args:
        event_type: 事件类型
        data: 事件数据
        source: 事件源
    
    Returns:
        AgenticX事件字典
    """
    return {
        "type": event_type,
        "data": data,
        "source": source,
        "timestamp": get_iso_timestamp()
    }


def setup_agenticx_logger(name: str = "agenticx-guiagent", **kwargs) -> "logger":
    """设置AgenticX兼容的日志记录器
    
    Args:
        name: 日志记录器名称
        **kwargs: 其他日志配置参数
    
    Returns:
        配置好的日志记录器
    """
    # 添加AgenticX特定的日志格式
    default_format = (
        "{time:YYYY-MM-DD HH:mm:ss} - [AgenticX] - {name} - {level} - "
        "[{file}:{line}] - {message}"
    )
    kwargs.setdefault("format_string", default_format)
    return setup_logger(name, **kwargs)


def validate_agenticx_config(config: Dict[str, Any]) -> bool:
    """验证AgenticX配置
    
    Args:
        config: 配置字典
    
    Returns:
        是否有效
    
    Raises:
        ValueError: 配置无效
    """
    required_sections = ["agenticx", "llm", "agents"]
    validate_required_fields(config, required_sections)
    
    # 验证AgenticX特定配置
    agenticx_config = config.get("agenticx", {})
    if "event_bus" in agenticx_config:
        event_bus_config = agenticx_config["event_bus"]
        if not isinstance(event_bus_config.get("enabled"), bool):
            raise ValueError("AgenticX event_bus.enabled must be boolean")
    
    return True


def create_agenticx_component_config(
    component_name: str,
    event_bus_enabled: bool = True,
    lifecycle_management: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """创建AgenticX组件配置
    
    Args:
        component_name: 组件名称
        event_bus_enabled: 是否启用事件总线
        lifecycle_management: 是否启用生命周期管理
        **kwargs: 其他配置参数
    
    Returns:
        组件配置字典
    """
    config = {
        "name": component_name,
        "event_bus_enabled": event_bus_enabled,
        "lifecycle_management": lifecycle_management,
        "created_at": get_iso_timestamp()
    }
    config.update(kwargs)
    return config


def merge_agenticx_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """合并AgenticX配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
    
    Returns:
        合并后的配置
    """
    merged = merge_dicts(base_config, override_config)
    
    # 特殊处理AgenticX配置节
    if "agenticx" in base_config and "agenticx" in override_config:
        merged["agenticx"] = merge_dicts(
            base_config["agenticx"],
            override_config["agenticx"]
        )
    
    return merged


def extract_agenticx_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """提取AgenticX指标数据
    
    Args:
        data: 原始数据
    
    Returns:
        指标数据
    """
    metrics = {
        "timestamp": get_iso_timestamp(),
        "source": "agenticx-guiagent"
    }
    
    # 提取执行指标
    if "execution_time" in data:
        metrics["execution_time"] = data["execution_time"]
    if "success" in data:
        metrics["success_rate"] = 1.0 if data["success"] else 0.0
    if "error" in data:
        metrics["error_count"] = 1
    
    # 提取工具指标
    if "tool_name" in data:
        metrics["tool_usage"] = {data["tool_name"]: 1}
    
    # 提取智能体指标
    if "agent_id" in data:
        metrics["agent_activity"] = {data["agent_id"]: 1}
    
    return metrics


class AgenticXContextManager(AsyncContextManager):
    """AgenticX异步上下文管理器"""
    
    def __init__(self, component_name: str, event_bus=None, **kwargs):
        super().__init__()
        self.component_name = component_name
        self.event_bus = event_bus
        self.config = kwargs
        self.logger = setup_agenticx_logger(f"agenticx.{component_name}")
    
    async def setup(self):
        """设置AgenticX组件"""
        logger.info(f"Initializing AgenticX component: {self.component_name}")
        
        # 发布组件初始化事件
        if self.event_bus:
            event = create_agenticx_event(
                "component_initialized",
                {"component_name": self.component_name, "config": self.config},
                f"agenticx.{self.component_name}"
            )
            await self.event_bus.publish_async(event)
    
    async def cleanup(self):
        """清理AgenticX组件"""
        logger.info(f"Cleaning up AgenticX component: {self.component_name}")
        
        # 发布组件清理事件
        if self.event_bus:
            event = create_agenticx_event(
                "component_cleanup",
                {"component_name": self.component_name},
                f"agenticx.{self.component_name}"
            )
            await self.event_bus.publish_async(event)


def calculate_similarity(text1: str, text2: str) -> float:
    """计算文本相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
    
    Returns:
        相似度分数 (0-1)
    """
    # 简单的字符级相似度计算
    if not text1 or not text2:
        return 0.0
    
    # 使用Jaccard相似度
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0