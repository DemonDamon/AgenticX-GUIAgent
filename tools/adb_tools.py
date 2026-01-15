#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADB工具模块 - 真实的Android设备操作

提供真实的ADB命令执行，包括点击、滑动、输入等操作
"""

import asyncio
import subprocess
from loguru import logger
from typing import Dict, Any, Optional, Tuple
from agenticx.core.tool import BaseTool
from utils import get_iso_timestamp


class ADBClickTool(BaseTool):
    """ADB点击工具 - 真实设备操作"""
    
    name: str = "adb_click"
    description: str = "使用ADB执行真实的设备点击操作"
    
    def __init__(self):
        super().__init__()
        # 使用对象属性而不是字段
        object.__setattr__(self, 'logger', logger)
    
    def _check_adb_connection(self) -> bool:
        """检查ADB连接状态"""
        try:
            result = subprocess.run(
                ["adb", "devices"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                devices = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                connected_devices = [line for line in devices if line.strip() and 'device' in line]
                return len(connected_devices) > 0
            return False
        except Exception as e:
            logger.error(f"检查ADB连接失败: {e}")
            return False
    
    def _execute_adb_command(self, command: list) -> Tuple[bool, str]:
        """执行ADB命令"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "ADB命令执行超时"
        except Exception as e:
            return False, f"ADB命令执行失败: {e}"
    
    def execute(self, coordinates: Dict[str, int], **kwargs) -> Dict[str, Any]:
        """同步执行点击"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建一个任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.aexecute(coordinates, **kwargs))
                    return future.result()
            else:
                return asyncio.run(self.aexecute(coordinates, **kwargs))
        except RuntimeError:
            # 如果没有事件循环，直接运行
            return asyncio.run(self.aexecute(coordinates, **kwargs))
    
    async def aexecute(self, coordinates: Dict[str, int], **kwargs) -> Dict[str, Any]:
        """异步执行点击操作"""
        x, y = coordinates["x"], coordinates["y"]
        
        # 检查ADB连接
        if not self._check_adb_connection():
            logger.warning("ADB设备未连接，使用模拟操作")
            await asyncio.sleep(0.5)
            return {
                "success": True,
                "coordinates": {"x": x, "y": y},
                "method": "simulated",
                "message": "ADB设备未连接，执行模拟点击",
                "timestamp": get_iso_timestamp()
            }
        
        # 执行真实的ADB点击
        command = ["adb", "shell", "input", "tap", str(x), str(y)]
        success, output = self._execute_adb_command(command)
        
        if success:
            logger.info(f"ADB点击成功: ({x}, {y})")
            return {
                "success": True,
                "coordinates": {"x": x, "y": y},
                "method": "adb",
                "message": f"ADB点击执行成功: ({x}, {y})",
                "output": output.strip(),
                "timestamp": get_iso_timestamp()
            }
        else:
            logger.error(f"ADB点击失败: {output}")
            return {
                "success": False,
                "coordinates": {"x": x, "y": y},
                "method": "adb",
                "error": output,
                "message": f"ADB点击失败: {output}",
                "timestamp": get_iso_timestamp()
            }


class ADBSwipeTool(BaseTool):
    """ADB滑动工具 - 真实设备操作"""
    
    name: str = "adb_swipe"
    description: str = "使用ADB执行真实的设备滑动操作"
    
    def __init__(self):
        super().__init__()
        # 使用对象属性而不是字段
        object.__setattr__(self, 'logger', logger)
    
    def _check_adb_connection(self) -> bool:
        """检查ADB连接状态"""
        try:
            result = subprocess.run(
                ["adb", "devices"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                devices = result.stdout.strip().split('\n')[1:]
                connected_devices = [line for line in devices if line.strip() and 'device' in line]
                return len(connected_devices) > 0
            return False
        except Exception as e:
            logger.error(f"检查ADB连接失败: {e}")
            return False
    
    def _execute_adb_command(self, command: list) -> Tuple[bool, str]:
        """执行ADB命令"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "ADB命令执行超时"
        except Exception as e:
            return False, f"ADB命令执行失败: {e}"
    
    def execute(
        self,
        start_coordinates: Dict[str, int],
        end_coordinates: Dict[str, int],
        duration: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """同步执行滑动"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建一个任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.aexecute(start_coordinates, end_coordinates, duration, **kwargs))
                    return future.result()
            else:
                return asyncio.run(self.aexecute(start_coordinates, end_coordinates, duration, **kwargs))
        except RuntimeError:
            # 如果没有事件循环，直接运行
            return asyncio.run(self.aexecute(start_coordinates, end_coordinates, duration, **kwargs))
    
    async def aexecute(
        self,
        start_coordinates: Dict[str, int],
        end_coordinates: Dict[str, int],
        duration: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """异步执行滑动操作"""
        x1, y1 = start_coordinates["x"], start_coordinates["y"]
        x2, y2 = end_coordinates["x"], end_coordinates["y"]
        duration_ms = int(duration * 1000)  # 转换为毫秒
        
        # 检查ADB连接
        if not self._check_adb_connection():
            logger.warning("ADB设备未连接，使用模拟操作")
            await asyncio.sleep(duration)
            return {
                "success": True,
                "start_coordinates": {"x": x1, "y": y1},
                "end_coordinates": {"x": x2, "y": y2},
                "duration": duration,
                "method": "simulated",
                "message": "ADB设备未连接，执行模拟滑动",
                "timestamp": get_iso_timestamp()
            }
        
        # 执行真实的ADB滑动
        command = ["adb", "shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)]
        success, output = self._execute_adb_command(command)
        
        if success:
            logger.info(f"ADB滑动成功: ({x1}, {y1}) -> ({x2}, {y2})")
            return {
                "success": True,
                "start_coordinates": {"x": x1, "y": y1},
                "end_coordinates": {"x": x2, "y": y2},
                "duration": duration,
                "method": "adb",
                "message": f"ADB滑动执行成功: ({x1}, {y1}) -> ({x2}, {y2})",
                "output": output.strip(),
                "timestamp": get_iso_timestamp()
            }
        else:
            logger.error(f"ADB滑动失败: {output}")
            return {
                "success": False,
                "start_coordinates": {"x": x1, "y": y1},
                "end_coordinates": {"x": x2, "y": y2},
                "duration": duration,
                "method": "adb",
                "error": output,
                "message": f"ADB滑动失败: {output}",
                "timestamp": get_iso_timestamp()
            }


class ADBInputTool(BaseTool):
    """ADB输入工具 - 真实设备操作"""
    
    name: str = "adb_input"
    description: str = "使用ADB执行真实的设备文本输入操作"
    
    def __init__(self):
        super().__init__()
        # 使用对象属性而不是字段
        object.__setattr__(self, 'logger', logger)
    
    def _check_adb_connection(self) -> bool:
        """检查ADB连接状态"""
        try:
            result = subprocess.run(
                ["adb", "devices"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                devices = result.stdout.strip().split('\n')[1:]
                connected_devices = [line for line in devices if line.strip() and 'device' in line]
                return len(connected_devices) > 0
            return False
        except Exception as e:
            logger.error(f"检查ADB连接失败: {e}")
            return False
    
    def _execute_adb_command(self, command: list) -> Tuple[bool, str]:
        """执行ADB命令"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "ADB命令执行超时"
        except Exception as e:
            return False, f"ADB命令执行失败: {e}"
    
    def execute(self, text: str, coordinates: Optional[Dict[str, int]] = None, **kwargs) -> Dict[str, Any]:
        """同步执行输入"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建一个任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.aexecute(text, coordinates, **kwargs))
                    return future.result()
            else:
                return asyncio.run(self.aexecute(text, coordinates, **kwargs))
        except RuntimeError:
            # 如果没有事件循环，直接运行
            return asyncio.run(self.aexecute(text, coordinates, **kwargs))
    
    async def aexecute(self, text: str, coordinates: Optional[Dict[str, int]] = None, **kwargs) -> Dict[str, Any]:
        """异步执行输入操作"""
        # 检查ADB连接
        if not self._check_adb_connection():
            logger.warning("ADB设备未连接，使用模拟操作")
            input_time = len(text) * 0.1
            await asyncio.sleep(input_time)
            return {
                "success": True,
                "text": text,
                "coordinates": coordinates,
                "method": "simulated",
                "message": "ADB设备未连接，执行模拟输入",
                "timestamp": get_iso_timestamp()
            }
        
        # 如果提供了坐标，先点击该位置
        if coordinates:
            click_command = ["adb", "shell", "input", "tap", str(coordinates["x"]), str(coordinates["y"])]
            click_success, click_output = self._execute_adb_command(click_command)
            if not click_success:
                logger.error(f"ADB点击输入位置失败: {click_output}")
        
        # 执行真实的ADB文本输入
        # 注意：ADB input text 需要转义特殊字符
        escaped_text = text.replace(' ', '%s').replace('&', '\\&')
        command = ["adb", "shell", "input", "text", escaped_text]
        success, output = self._execute_adb_command(command)
        
        if success:
            logger.info(f"ADB输入成功: {text}")
            return {
                "success": True,
                "text": text,
                "coordinates": coordinates,
                "method": "adb",
                "message": f"ADB输入执行成功: {text}",
                "output": output.strip(),
                "timestamp": get_iso_timestamp()
            }
        else:
            logger.error(f"ADB输入失败: {output}")
            return {
                "success": False,
                "text": text,
                "coordinates": coordinates,
                "method": "adb",
                "error": output,
                "message": f"ADB输入失败: {output}",
                "timestamp": get_iso_timestamp()
            }


class ADBScreenshotTool(BaseTool):
    """ADB截图工具 - 真实设备操作"""
    
    name: str = "adb_screenshot"
    description: str = "使用ADB获取真实设备截图"
    
    def __init__(self):
        super().__init__()
        # 使用对象属性而不是字段
        object.__setattr__(self, 'logger', logger)
    
    def _check_adb_connection(self) -> bool:
        """检查ADB连接状态"""
        try:
            result = subprocess.run(
                ["adb", "devices"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                devices = result.stdout.strip().split('\n')[1:]
                connected_devices = [line for line in devices if line.strip() and 'device' in line]
                return len(connected_devices) > 0
            return False
        except Exception as e:
            logger.error(f"检查ADB连接失败: {e}")
            return False
    
    def _execute_adb_command(self, command: list) -> Tuple[bool, str]:
        """执行ADB命令"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "ADB命令执行超时"
        except Exception as e:
            return False, f"ADB命令执行失败: {e}"
    
    def execute(self, save_path: str = None, **kwargs) -> Dict[str, Any]:
        """同步执行截图"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建一个任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.aexecute(save_path, **kwargs))
                    return future.result()
            else:
                return asyncio.run(self.aexecute(save_path, **kwargs))
        except RuntimeError:
            # 如果没有事件循环，直接运行
            return asyncio.run(self.aexecute(save_path, **kwargs))
    
    async def aexecute(self, save_path: str = None, **kwargs) -> Dict[str, Any]:
        """异步执行截图操作"""
        import os
        from pathlib import Path
        
        # 设置默认保存路径 - 使用新的命名规则
        if not save_path:
            screenshots_dir = Path("./screenshots")
            screenshots_dir.mkdir(exist_ok=True)
            # 使用新的命名规则：智能体名称_时间戳_操作类型.png
            agent_id = kwargs.get('agent_id', 'adb')  # 从参数获取智能体ID，默认为adb
            timestamp = get_iso_timestamp().replace(':', '-')
            screenshot_filename = f"{agent_id}_{timestamp}_screenshot.png"
            save_path = screenshots_dir / screenshot_filename
        
        # 检查ADB连接
        if not self._check_adb_connection():
            logger.warning("ADB设备未连接，无法获取真实截图")
            return {
                "success": False,
                "screenshot_path": None,
                "method": "adb",
                "error": "ADB设备未连接",
                "message": "ADB设备未连接，无法获取截图",
                "timestamp": get_iso_timestamp()
            }
        
        # 执行ADB截图命令
        device_path = "/sdcard/screenshot.png"
        
        # 1. 在设备上截图
        screenshot_command = ["adb", "shell", "screencap", "-p", device_path]
        success, output = self._execute_adb_command(screenshot_command)
        
        if not success:
            logger.error(f"ADB设备截图失败: {output}")
            return {
                "success": False,
                "screenshot_path": None,
                "method": "adb",
                "error": output,
                "message": f"ADB设备截图失败: {output}",
                "timestamp": get_iso_timestamp()
            }
        
        # 2. 将截图拉取到本地
        pull_command = ["adb", "pull", device_path, str(save_path)]
        success, output = self._execute_adb_command(pull_command)
        
        if success:
            # 3. 清理设备上的临时文件
            cleanup_command = ["adb", "shell", "rm", device_path]
            self._execute_adb_command(cleanup_command)
            
            logger.info(f"ADB截图成功: {save_path}")
            return {
                "success": True,
                "screenshot_path": str(save_path),
                "method": "adb",
                "message": f"ADB截图成功: {save_path}",
                "timestamp": get_iso_timestamp()
            }
        else:
            logger.error(f"ADB截图拉取失败: {output}")
            return {
                "success": False,
                "screenshot_path": None,
                "method": "adb",
                "error": output,
                "message": f"ADB截图拉取失败: {output}",
                "timestamp": get_iso_timestamp()
            }