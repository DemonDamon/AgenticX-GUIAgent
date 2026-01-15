#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Tool Adapters
工具适配器：为不同平台和设备提供统一的工具接口

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from .gui_tools import (
    GUITool, ToolParameters, ToolResult, ToolError,
    Coordinate, Rectangle, Platform, ToolType, ToolStatus
)
from utils import get_iso_timestamp, setup_logger


class ToolAdapter(ABC):
    """工具适配器基类"""
    
    def __init__(
        self,
        platform: Platform,
        adapter_name: str,
        adapter_version: str = "1.0.0"
    ):
        self.platform = platform
        self.adapter_name = adapter_name
        self.adapter_version = adapter_version
        self.logger = logger
        self.is_initialized = False
        self.capabilities = set()
        self.device_info = {}
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化适配器"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """清理适配器资源"""
        pass
    
    @abstractmethod
    async def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        pass
    
    @abstractmethod
    async def take_screenshot(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """截图"""
        pass
    
    @abstractmethod
    async def click(self, x: int, y: int, **kwargs) -> Dict[str, Any]:
        """点击"""
        pass
    
    @abstractmethod
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """滑动"""
        pass
    
    @abstractmethod
    async def input_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """输入文本"""
        pass
    
    @abstractmethod
    async def press_key(self, key: str, **kwargs) -> Dict[str, Any]:
        """按键"""
        pass
    
    @abstractmethod
    async def get_elements(self, **kwargs) -> Dict[str, Any]:
        """获取UI元素"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """获取适配器能力"""
        return list(self.capabilities)
    
    def supports_capability(self, capability: str) -> bool:
        """检查是否支持某个能力"""
        return capability in self.capabilities


class AndroidAdapter(ToolAdapter):
    """Android设备适配器"""
    
    def __init__(self, device_id: Optional[str] = None):
        super().__init__(
            platform=Platform.ANDROID,
            adapter_name="AndroidAdapter"
        )
        self.device_id = device_id
        self.adb_path = "adb"  # 假设adb在PATH中
        self.capabilities = {
            'screenshot', 'click', 'swipe', 'input_text',
            'press_key', 'get_elements', 'install_app',
            'uninstall_app', 'start_activity', 'stop_activity'
        }
    
    async def initialize(self) -> bool:
        """初始化Android适配器"""
        try:
            # 检查ADB连接
            if not await self._check_adb_connection():
                logger.error("ADB connection failed")
                return False
            
            # 获取设备信息
            self.device_info = await self.get_device_info()
            
            # 检查设备状态
            if not await self._check_device_ready():
                logger.error("Device not ready")
                return False
            
            self.is_initialized = True
            logger.info(f"Android adapter initialized for device: {self.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Android adapter: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理Android适配器"""
        try:
            self.is_initialized = False
            logger.info("Android adapter cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup Android adapter: {e}")
            return False
    
    async def get_device_info(self) -> Dict[str, Any]:
        """获取Android设备信息"""
        try:
            device_info = {}
            
            # 获取设备型号
            model_result = await self._run_adb_command(['shell', 'getprop', 'ro.product.model'])
            if model_result['success']:
                device_info['model'] = model_result['output'].strip()
            
            # 获取Android版本
            version_result = await self._run_adb_command(['shell', 'getprop', 'ro.build.version.release'])
            if version_result['success']:
                device_info['android_version'] = version_result['output'].strip()
            
            # 获取屏幕分辨率
            resolution_result = await self._run_adb_command(['shell', 'wm', 'size'])
            if resolution_result['success']:
                resolution_line = resolution_result['output'].strip()
                if 'Physical size:' in resolution_line:
                    resolution = resolution_line.split('Physical size:')[1].strip()
                    device_info['resolution'] = resolution
            
            # 获取设备ID
            device_info['device_id'] = self.device_id or 'default'
            device_info['platform'] = 'Android'
            
            return device_info
            
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {'platform': 'Android', 'error': str(e)}
    
    async def take_screenshot(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Android截图"""
        try:
            if not save_path:
                timestamp = int(time.time() * 1000)
                save_path = f"/tmp/screenshot_{timestamp}.png"
            
            # 在设备上截图
            device_path = f"/sdcard/screenshot_{int(time.time())}.png"
            
            # 执行截图命令
            screenshot_result = await self._run_adb_command([
                'shell', 'screencap', '-p', device_path
            ])
            
            if not screenshot_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to capture screenshot on device'
                }
            
            # 拉取截图到本地
            pull_result = await self._run_adb_command(['pull', device_path, save_path])
            
            if not pull_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to pull screenshot from device'
                }
            
            # 删除设备上的临时文件
            await self._run_adb_command(['shell', 'rm', device_path])
            
            return {
                'success': True,
                'screenshot_path': save_path,
                'device_path': device_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def click(self, x: int, y: int, **kwargs) -> Dict[str, Any]:
        """Android点击"""
        try:
            result = await self._run_adb_command([
                'shell', 'input', 'tap', str(x), str(y)
            ])
            
            return {
                'success': result['success'],
                'coordinate': {'x': x, 'y': y},
                'error': result.get('error') if not result['success'] else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'coordinate': {'x': x, 'y': y},
                'error': str(e)
            }
    
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """Android滑动"""
        try:
            result = await self._run_adb_command([
                'shell', 'input', 'swipe',
                str(start_x), str(start_y),
                str(end_x), str(end_y),
                str(duration)
            ])
            
            return {
                'success': result['success'],
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'duration': duration,
                'error': result.get('error') if not result['success'] else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'duration': duration,
                'error': str(e)
            }
    
    async def input_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Android文本输入"""
        try:
            # 转义特殊字符
            escaped_text = text.replace(' ', '%s').replace('&', '\\&')
            
            result = await self._run_adb_command([
                'shell', 'input', 'text', escaped_text
            ])
            
            return {
                'success': result['success'],
                'text': text,
                'error': result.get('error') if not result['success'] else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': text,
                'error': str(e)
            }
    
    async def press_key(self, key: str, **kwargs) -> Dict[str, Any]:
        """Android按键"""
        try:
            # Android按键码映射
            key_codes = {
                'back': 'KEYCODE_BACK',
                'home': 'KEYCODE_HOME',
                'menu': 'KEYCODE_MENU',
                'enter': 'KEYCODE_ENTER',
                'delete': 'KEYCODE_DEL',
                'space': 'KEYCODE_SPACE',
                'tab': 'KEYCODE_TAB',
                'escape': 'KEYCODE_ESCAPE'
            }
            
            key_code = key_codes.get(key.lower(), key)
            
            result = await self._run_adb_command([
                'shell', 'input', 'keyevent', key_code
            ])
            
            return {
                'success': result['success'],
                'key': key,
                'key_code': key_code,
                'error': result.get('error') if not result['success'] else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'key': key,
                'error': str(e)
            }
    
    async def get_elements(self, **kwargs) -> Dict[str, Any]:
        """获取Android UI元素"""
        try:
            # 使用uiautomator dump获取UI层次结构
            dump_result = await self._run_adb_command([
                'shell', 'uiautomator', 'dump', '/sdcard/ui_dump.xml'
            ])
            
            if not dump_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to dump UI hierarchy'
                }
            
            # 拉取UI dump文件
            local_dump_path = f"/tmp/ui_dump_{int(time.time())}.xml"
            pull_result = await self._run_adb_command([
                'pull', '/sdcard/ui_dump.xml', local_dump_path
            ])
            
            if not pull_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to pull UI dump file'
                }
            
            # 解析UI元素（简化实现）
            elements = await self._parse_ui_dump(local_dump_path)
            
            # 清理临时文件
            await self._run_adb_command(['shell', 'rm', '/sdcard/ui_dump.xml'])
            
            return {
                'success': True,
                'elements': elements,
                'dump_path': local_dump_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _check_adb_connection(self) -> bool:
        """检查ADB连接"""
        try:
            result = await self._run_adb_command(['devices'])
            return result['success'] and 'device' in result['output']
            
        except Exception:
            return False
    
    async def _check_device_ready(self) -> bool:
        """检查设备是否就绪"""
        try:
            result = await self._run_adb_command(['shell', 'getprop', 'sys.boot_completed'])
            return result['success'] and '1' in result['output']
            
        except Exception:
            return False
    
    async def _run_adb_command(self, args: List[str]) -> Dict[str, Any]:
        """执行ADB命令"""
        try:
            cmd = [self.adb_path]
            
            if self.device_id:
                cmd.extend(['-s', self.device_id])
            
            cmd.extend(args)
            
            # 异步执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode('utf-8') if stdout else '',
                'error': stderr.decode('utf-8') if stderr else '',
                'return_code': process.returncode
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'return_code': -1
            }
    
    async def _parse_ui_dump(self, dump_path: str) -> List[Dict[str, Any]]:
        """解析UI dump文件"""
        try:
            # 这里应该实现XML解析逻辑
            # 简化实现，返回模拟数据
            elements = [
                {
                    'id': 'element_1',
                    'type': 'button',
                    'text': 'Click Me',
                    'bounds': {'left': 100, 'top': 200, 'right': 300, 'bottom': 250},
                    'center': {'x': 200, 'y': 225},
                    'visible': True,
                    'enabled': True
                },
                {
                    'id': 'element_2',
                    'type': 'input',
                    'text': '',
                    'bounds': {'left': 50, 'top': 300, 'right': 350, 'bottom': 350},
                    'center': {'x': 200, 'y': 325},
                    'visible': True,
                    'enabled': True
                }
            ]
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to parse UI dump: {e}")
            return []


class iOSAdapter(ToolAdapter):
    """iOS设备适配器"""
    
    def __init__(self, device_id: Optional[str] = None):
        super().__init__(
            platform=Platform.IOS,
            adapter_name="iOSAdapter"
        )
        self.device_id = device_id
        self.capabilities = {
            'screenshot', 'click', 'swipe', 'input_text',
            'press_key', 'get_elements'
        }
    
    async def initialize(self) -> bool:
        """初始化iOS适配器"""
        try:
            # iOS适配器初始化逻辑
            # 这里应该集成WebDriverAgent或其他iOS自动化工具
            
            self.device_info = await self.get_device_info()
            self.is_initialized = True
            logger.info(f"iOS adapter initialized for device: {self.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize iOS adapter: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理iOS适配器"""
        try:
            self.is_initialized = False
            logger.info("iOS adapter cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup iOS adapter: {e}")
            return False
    
    async def get_device_info(self) -> Dict[str, Any]:
        """获取iOS设备信息"""
        try:
            # 模拟iOS设备信息
            return {
                'platform': 'iOS',
                'device_id': self.device_id or 'simulator',
                'model': 'iPhone 14',
                'ios_version': '16.0',
                'resolution': '1170x2532'
            }
            
        except Exception as e:
            logger.error(f"Failed to get iOS device info: {e}")
            return {'platform': 'iOS', 'error': str(e)}
    
    async def take_screenshot(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """iOS截图"""
        try:
            if not save_path:
                timestamp = int(time.time() * 1000)
                save_path = f"/tmp/ios_screenshot_{timestamp}.png"
            
            # 模拟iOS截图
            # 实际应该使用WebDriverAgent或其他工具
            
            return {
                'success': True,
                'screenshot_path': save_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def click(self, x: int, y: int, **kwargs) -> Dict[str, Any]:
        """iOS点击"""
        try:
            # 模拟iOS点击
            return {
                'success': True,
                'coordinate': {'x': x, 'y': y}
            }
            
        except Exception as e:
            return {
                'success': False,
                'coordinate': {'x': x, 'y': y},
                'error': str(e)
            }
    
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """iOS滑动"""
        try:
            # 模拟iOS滑动
            return {
                'success': True,
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'duration': duration
            }
            
        except Exception as e:
            return {
                'success': False,
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'duration': duration,
                'error': str(e)
            }
    
    async def input_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """iOS文本输入"""
        try:
            # 模拟iOS文本输入
            return {
                'success': True,
                'text': text
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': text,
                'error': str(e)
            }
    
    async def press_key(self, key: str, **kwargs) -> Dict[str, Any]:
        """iOS按键"""
        try:
            # 模拟iOS按键
            return {
                'success': True,
                'key': key
            }
            
        except Exception as e:
            return {
                'success': False,
                'key': key,
                'error': str(e)
            }
    
    async def get_elements(self, **kwargs) -> Dict[str, Any]:
        """获取iOS UI元素"""
        try:
            # 模拟iOS UI元素
            elements = [
                {
                    'id': 'ios_element_1',
                    'type': 'button',
                    'text': 'iOS Button',
                    'bounds': {'left': 100, 'top': 200, 'right': 300, 'bottom': 250},
                    'center': {'x': 200, 'y': 225},
                    'visible': True,
                    'enabled': True
                }
            ]
            
            return {
                'success': True,
                'elements': elements
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class DesktopAdapter(ToolAdapter):
    """桌面平台适配器"""
    
    def __init__(self, platform: Platform = Platform.UNIVERSAL):
        super().__init__(
            platform=platform,
            adapter_name=f"DesktopAdapter_{platform.value}"
        )
        self.capabilities = {
            'screenshot', 'click', 'swipe', 'input_text',
            'press_key', 'get_elements', 'window_management'
        }
    
    async def initialize(self) -> bool:
        """初始化桌面适配器"""
        try:
            self.device_info = await self.get_device_info()
            self.is_initialized = True
            logger.info(f"Desktop adapter initialized for {self.platform.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize desktop adapter: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理桌面适配器"""
        try:
            self.is_initialized = False
            logger.info("Desktop adapter cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup desktop adapter: {e}")
            return False
    
    async def get_device_info(self) -> Dict[str, Any]:
        """获取桌面设备信息"""
        try:
            import platform as platform_module
            
            return {
                'platform': self.platform.value,
                'system': platform_module.system(),
                'release': platform_module.release(),
                'version': platform_module.version(),
                'machine': platform_module.machine(),
                'processor': platform_module.processor()
            }
            
        except Exception as e:
            logger.error(f"Failed to get desktop device info: {e}")
            return {'platform': self.platform.value, 'error': str(e)}
    
    async def take_screenshot(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """桌面截图"""
        try:
            if not save_path:
                timestamp = int(time.time() * 1000)
                save_path = f"/tmp/desktop_screenshot_{timestamp}.png"
            
            # 这里应该使用PIL、pyautogui等库进行截图
            # 模拟实现
            
            return {
                'success': True,
                'screenshot_path': save_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def click(self, x: int, y: int, **kwargs) -> Dict[str, Any]:
        """桌面点击"""
        try:
            # 这里应该使用pyautogui等库进行点击
            # 模拟实现
            
            return {
                'success': True,
                'coordinate': {'x': x, 'y': y}
            }
            
        except Exception as e:
            return {
                'success': False,
                'coordinate': {'x': x, 'y': y},
                'error': str(e)
            }
    
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """桌面拖拽"""
        try:
            # 桌面平台的拖拽操作
            return {
                'success': True,
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'duration': duration
            }
            
        except Exception as e:
            return {
                'success': False,
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'duration': duration,
                'error': str(e)
            }
    
    async def input_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """桌面文本输入"""
        try:
            # 这里应该使用pyautogui等库进行文本输入
            return {
                'success': True,
                'text': text
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': text,
                'error': str(e)
            }
    
    async def press_key(self, key: str, **kwargs) -> Dict[str, Any]:
        """桌面按键"""
        try:
            # 这里应该使用pyautogui等库进行按键
            return {
                'success': True,
                'key': key
            }
            
        except Exception as e:
            return {
                'success': False,
                'key': key,
                'error': str(e)
            }
    
    async def get_elements(self, **kwargs) -> Dict[str, Any]:
        """获取桌面UI元素"""
        try:
            # 桌面UI元素获取（可能需要使用accessibility API）
            elements = []
            
            return {
                'success': True,
                'elements': elements
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class AdapterFactory:
    """适配器工厂"""
    
    _adapters: Dict[Platform, Type[ToolAdapter]] = {
        Platform.ANDROID: AndroidAdapter,
        Platform.IOS: iOSAdapter,
        Platform.UNIVERSAL: DesktopAdapter
    }
    
    @classmethod
    def create_adapter(
        cls,
        platform: Platform,
        **kwargs
    ) -> Optional[ToolAdapter]:
        """创建适配器实例"""
        try:
            adapter_class = cls._adapters.get(platform)
            
            if not adapter_class:
                raise ValueError(f"Unsupported platform: {platform}")
            
            if platform in [Platform.WINDOWS, Platform.MACOS, Platform.LINUX]:
                return adapter_class(platform=platform)
            else:
                return adapter_class(**kwargs)
            
        except Exception as e:
            logger = get_logger("AdapterFactory")
            logger.error(f"Failed to create adapter for {platform}: {e}")
            return None
    
    @classmethod
    def get_supported_platforms(cls) -> List[Platform]:
        """获取支持的平台列表"""
        return list(cls._adapters.keys())
    
    @classmethod
    def register_adapter(
        cls,
        platform: Platform,
        adapter_class: Type[ToolAdapter]
    ) -> None:
        """注册新的适配器"""
        cls._adapters[platform] = adapter_class


class AdaptedGUITool(GUITool):
    """适配器包装的GUI工具"""
    
    def __init__(
        self,
        base_tool: GUITool,
        adapter: ToolAdapter,
        **kwargs
    ):
        super().__init__(
            name=f"Adapted_{base_tool.name}",
            description=f"Adapter-wrapped {base_tool.description}",
            platform=adapter.platform,
            tool_type=base_tool.tool_type,
            **kwargs
        )
        
        self.base_tool = base_tool
        self.adapter = adapter
    
    async def validate(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """验证参数"""
        try:
            # 检查适配器是否已初始化
            if not self.adapter.is_initialized:
                logger.error("Adapter not initialized")
                return False
            
            # 使用基础工具的验证逻辑
            return await self.base_tool.validate(parameters, context)
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def execute(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行工具操作"""
        start_time = get_iso_timestamp()
        
        try:
            # 将工具操作转换为适配器调用
            adapter_result = await self._execute_with_adapter(
                parameters, context
            )
            
            if not adapter_result['success']:
                raise ToolError(
                    adapter_result.get('error', 'Adapter execution failed'),
                    'ADAPTER_EXECUTION_FAILED',
                    self.tool_id
                )
            
            end_time = get_iso_timestamp()
            
            return ToolResult(
                tool_id=self.tool_id,
                tool_type=self.tool_type.value,
                status=ToolStatus.COMPLETED,
                success=True,
                start_time=start_time,
                end_time=end_time,
                result_data=adapter_result
            )
            
        except Exception as e:
            end_time = get_iso_timestamp()
            error_msg = str(e)
            error_code = getattr(e, 'error_code', 'ADAPTED_TOOL_ERROR')
            
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
    
    async def _execute_with_adapter(
        self,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """使用适配器执行操作"""
        try:
            # 根据工具类型调用相应的适配器方法
            if self.base_tool.name == "ClickTool":
                return await self.adapter.click(
                    parameters.coordinate.x,
                    parameters.coordinate.y
                )
            
            elif self.base_tool.name == "SwipeTool":
                return await self.adapter.swipe(
                    parameters.start_coordinate.x,
                    parameters.start_coordinate.y,
                    parameters.end_coordinate.x,
                    parameters.end_coordinate.y,
                    parameters.duration or 500
                )
            
            elif self.base_tool.name == "TextInputTool":
                return await self.adapter.input_text(parameters.text)
            
            elif self.base_tool.name == "KeyPressTool":
                return await self.adapter.press_key(parameters.key)
            
            elif self.base_tool.name == "ScreenshotTool":
                return await self.adapter.take_screenshot()
            
            elif self.base_tool.name == "ElementDetectionTool":
                return await self.adapter.get_elements()
            
            else:
                # 对于其他工具，使用基础工具的执行逻辑
                result = await self.base_tool.execute(parameters, context)
                return {
                    'success': result.success,
                    'result_data': result.result_data,
                    'error': result.error_message if not result.success else None
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }