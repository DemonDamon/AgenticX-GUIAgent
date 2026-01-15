#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ExecutorAgent - 动作执行器智能体

负责执行具体的移动设备GUI操作。
"""

import asyncio
import json
from loguru import logger
from rich import print
from rich.json import JSON
from typing import Dict, Any, List, Optional, Tuple
import time

# 使用AgenticX核心组件
from agenticx.core.tool import BaseTool
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from agenticx.llms.base import BaseLLMProvider
from agenticx.memory.component import MemoryComponent

from core.base_agent import BaseAgenticXGUIAgentAgent
from config import AgentConfig
from utils import get_iso_timestamp
from tools.adb_tools import ADBClickTool, ADBSwipeTool, ADBInputTool, ADBScreenshotTool


# 内部task_type与对外原子操作的映射关系
TASK_TYPE_MAPPING = {
    "take_screenshot": "screenshot",
    "click_action": "click",
    "input_text": "type", 
    "swipe_action": "swipe",
    "locate_element": "locate_element",
    "wait": "wait",
    "long_press_action": "long_press",
    "open_app_action": "open_app",
    "system_button_action": "system_button",
    "verify_action": "verify"
}

# 反向映射：对外原子操作到内部task_type
ATOMIC_ACTION_MAPPING = {v: k for k, v in TASK_TYPE_MAPPING.items()}


def get_external_action_name(internal_task_type: str) -> str:
    """将内部task_type转换为对外的原子操作名称"""
    return TASK_TYPE_MAPPING.get(internal_task_type, internal_task_type)


def get_internal_task_type(external_action: str) -> str:
    """将对外的原子操作名称转换为内部task_type"""
    return ATOMIC_ACTION_MAPPING.get(external_action, external_action)


class ElementLocatorTool(BaseTool):
    """元素定位工具"""
    
    name: str = "element_locator"
    description: str = "在移动设备屏幕上定位UI元素"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法"""
        element_description = kwargs.get('element_description', '')
        # 直接返回模拟结果，避免异步调用问题
        return {
            "found": True,
            "element_id": f"element_{hash(element_description) % 10000}",
            "coordinates": {
                "x": 200 + (hash(element_description) % 400),
                "y": 300 + (hash(element_description) % 600),
                "width": 100,
                "height": 50
            },
            "element_type": "button",
            "confidence": 0.95,
            "success": True
        }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """定位元素
        
        Args:
            element_description: 元素描述
            **kwargs: 额外参数
        
        Returns:
            元素位置信息
        """
        element_description = kwargs.get('element_description', '')
        # 模拟元素定位（实际应用中会使用计算机视觉或UI分析）
        await asyncio.sleep(1)  # 模拟定位时间
        
        # 模拟找到元素
        element_info = {
            "found": True,
            "element_id": f"element_{hash(element_description) % 10000}",
            "coordinates": {
                "x": 200 + (hash(element_description) % 400),
                "y": 300 + (hash(element_description) % 600),
                "width": 100,
                "height": 50
            },
            "element_type": "button",  # 简化处理
            "text": element_description,
            "confidence": 0.95,
            "screenshot_path": f"/tmp/screenshot_{int(time.time())}.png",
            "location_time": get_iso_timestamp()
        }
        
        return element_info


class ClickTool(BaseTool):
    """点击工具"""
    
    name: str = "click"
    description: str = "执行点击操作"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法"""
        coordinates = kwargs.get('coordinates', {"x": 0, "y": 0})
        # 直接返回模拟结果，避免异步调用问题
        x, y = coordinates["x"], coordinates["y"]
        return {
            "success": True,
            "coordinates": {"x": x, "y": y},
            "action": "click",
            "timestamp": get_iso_timestamp()
        }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """执行点击
        
        Args:
            coordinates: 点击坐标
            **kwargs: 额外参数
        
        Returns:
            点击结果
        """
        coordinates = kwargs.get('coordinates', {"x": 0, "y": 0})
        x, y = coordinates["x"], coordinates["y"]
        
        # 模拟点击操作
        await asyncio.sleep(0.5)  # 模拟点击时间
        
        click_result = {
            "success": True,
            "coordinates": {"x": x, "y": y},
            "click_time": get_iso_timestamp(),
            "response_time": 0.5,
            "action_type": "click"
        }
        
        return click_result


class InputTool(BaseTool):
    """输入工具"""
    
    name: str = "input"
    description: str = "执行文本输入操作"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法"""
        # 从kwargs中获取参数，以确保兼容性
        text = kwargs.get('text', '')
        coordinates = kwargs.get('coordinates', {"x": 0, "y": 0})
        
        # 直接返回模拟结果，避免异步调用问题
        return {
            "success": True,
            "text": text,
            "coordinates": coordinates,
            "action": "input",
            "timestamp": get_iso_timestamp()
        }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """执行文本输入
        
        Args:
            **kwargs: 额外参数
                text: 输入文本
                coordinates: 输入框坐标
        
        Returns:
            输入结果
        """
        # 从kwargs中获取参数，以确保兼容性
        text = kwargs.get('text', '')
        coordinates = kwargs.get('coordinates', {"x": 0, "y": 0})
        
        # 模拟输入操作
        input_time = len(text) * 0.1  # 根据文本长度计算输入时间
        await asyncio.sleep(input_time)
        
        input_result = {
            "success": True,
            "text": text,
            "coordinates": coordinates,
            "input_time": get_iso_timestamp(),
            "duration": input_time,
            "action_type": "input"
        }
        
        return input_result


class SwipeTool(BaseTool):
    """滑动工具"""
    
    name: str = "swipe"
    description: str = "执行滑动操作"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法"""
        # 从kwargs中获取参数，以确保兼容性
        start_coordinates = kwargs.get('start_coordinates', {"x": 0, "y": 0})
        end_coordinates = kwargs.get('end_coordinates', {"x": 100, "y": 100})
        duration = kwargs.get('duration', 1.0)
        
        # 直接返回模拟结果，避免异步调用问题
        return {
            "success": True,
            "start_coordinates": start_coordinates,
            "end_coordinates": end_coordinates,
            "duration": duration,
            "action": "swipe",
            "timestamp": get_iso_timestamp()
        }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """执行滑动
        
        Args:
            **kwargs: 额外参数
                start_coordinates: 起始坐标
                end_coordinates: 结束坐标
                duration: 滑动持续时间
        
        Returns:
            滑动结果
        """
        # 从kwargs中获取参数，以确保兼容性
        start_coordinates = kwargs.get('start_coordinates', {"x": 0, "y": 0})
        end_coordinates = kwargs.get('end_coordinates', {"x": 100, "y": 100})
        duration = kwargs.get('duration', 1.0)
        
        # 模拟滑动操作
        await asyncio.sleep(duration)
        
        swipe_result = {
            "success": True,
            "start_coordinates": start_coordinates,
            "end_coordinates": end_coordinates,
            "duration": duration,
            "swipe_time": get_iso_timestamp(),
            "action_type": "swipe"
        }
        
        return swipe_result


class ScreenshotTool(BaseTool):
    """截图工具"""
    
    name: str = "screenshot"
    description: str = "获取设备屏幕截图"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法 - 获取真实设备截图"""
        # 使用新的命名规则：智能体名称_时间戳_操作类型.png
        agent_id = kwargs.get('agent_id', 'executor')  # 从参数获取智能体ID，默认为executor
        timestamp = get_iso_timestamp().replace(':', '-')
        screenshot_filename = f"{agent_id}_{timestamp}_screenshot.png"
        screenshot_path = f"./screenshots/{screenshot_filename}"
        
        try:
            import os
            import subprocess
            
            # 确保目录存在
            os.makedirs("./screenshots", exist_ok=True)
            
            # 检查ADB设备连接
            devices_result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            if 'device' not in devices_result.stdout:
                # 如果没有设备连接，创建模拟截图
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (1080, 1920), color='lightcoral')
                draw = ImageDraw.Draw(img)
                draw.text((50, 50), f"无设备连接\n时间: {get_iso_timestamp()}", fill='white')
                img.save(screenshot_path)
                
                return {
                    "success": True,
                    "screenshot_path": screenshot_path,
                    "action": "screenshot",
                    "device_type": "simulated",
                    "timestamp": get_iso_timestamp()
                }
            
            # 使用ADB获取真实设备截图
            device_screenshot_path = "/sdcard/screenshot_temp.png"
            
            # 在设备上截图
            screencap_result = subprocess.run(
                ['adb', 'shell', 'screencap', '-p', device_screenshot_path],
                capture_output=True, text=True
            )
            
            if screencap_result.returncode != 0:
                raise Exception(f"设备截图失败: {screencap_result.stderr}")
            
            # 将截图从设备拉取到本地
            pull_result = subprocess.run(
                ['adb', 'pull', device_screenshot_path, screenshot_path],
                capture_output=True, text=True
            )
            
            if pull_result.returncode != 0:
                raise Exception(f"截图文件拉取失败: {pull_result.stderr}")
            
            # 清理设备上的临时文件
            subprocess.run(['adb', 'shell', 'rm', device_screenshot_path], capture_output=True)
            
            return {
                "success": True,
                "screenshot_path": screenshot_path,
                "action": "screenshot",
                "device_type": "real_device",
                "timestamp": get_iso_timestamp()
            }
            
        except Exception as e:
            # 如果真实截图失败，创建错误提示截图
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (1080, 1920), color='lightcoral')
                draw = ImageDraw.Draw(img)
                draw.text((50, 50), f"截图失败\n错误: {str(e)}\n时间: {get_iso_timestamp()}", fill='white')
                img.save(screenshot_path)
                
                return {
                    "success": False,
                    "error": str(e),
                    "screenshot_path": screenshot_path,
                    "action": "screenshot",
                    "device_type": "error",
                    "timestamp": get_iso_timestamp()
                }
            except:
                return {
                    "success": False,
                    "error": str(e),
                    "action": "screenshot",
                    "timestamp": get_iso_timestamp()
                }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """获取截图
        
        Args:
            **kwargs: 额外参数
        
        Returns:
            截图信息
        """
        # 模拟截图操作
        await asyncio.sleep(0.3)
        
        screenshot_result = {
            "success": True,
            "screenshot_path": f"/tmp/screenshot_{int(time.time())}.png",
            "timestamp": get_iso_timestamp(),
            "resolution": {"width": 1080, "height": 1920},
            "file_size": 1024 * 200  # 200KB
        }
        
        return screenshot_result


class WaitTool(BaseTool):
    """等待工具"""
    
    name: str = "wait"
    description: str = "等待指定时间或条件"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法"""
        # 从kwargs中获取参数，以确保兼容性
        duration = kwargs.get('duration', None)
        condition = kwargs.get('condition', None)
        
        # 直接返回模拟结果，避免异步调用问题
        return {
            "success": True,
            "duration": duration,
            "condition": condition,
            "action": "wait",
            "timestamp": get_iso_timestamp()
        }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """执行等待
        
        Args:
            **kwargs: 额外参数
                duration: 等待时间（秒）
                condition: 等待条件
        
        Returns:
            等待结果
        """
        # 从kwargs中获取参数，以确保兼容性
        duration = kwargs.get('duration', None)
        condition = kwargs.get('condition', None)
        
        start_time = time.time()
        
        if duration:
            await asyncio.sleep(duration)
            actual_duration = time.time() - start_time
        else:
            # 简化处理，默认等待1秒
            await asyncio.sleep(1.0)
            actual_duration = 1.0
        
        wait_result = {
            "success": True,
            "requested_duration": duration,
            "actual_duration": actual_duration,
            "condition": condition,
            "wait_time": get_iso_timestamp()
        }
        
        return wait_result


class ExecutorAgent(BaseAgenticXGUIAgentAgent):
    """动作执行器智能体
    
    负责：
    1. 执行具体的GUI操作（点击、输入、滑动等）
    2. 元素定位和识别
    3. 屏幕截图和状态获取
    4. 操作结果验证
    5. 异常处理和重试
    """
    
    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        agent_id: str = "executor",
        platform = None,
        info_pool = None,
        tool_manager = None,
        agent_config: Optional[AgentConfig] = None,
        memory: Optional[MemoryComponent] = None
    ):
        # 存储额外参数
        self.agent_id = agent_id
        self.platform = platform
        self.info_pool = info_pool
        self.tool_manager = tool_manager
        
        # 反思反馈机制
        self.reflection_feedback: Dict[str, Any] = {}
        self.coordinate_adjustments: Dict[str, List[int]] = {}  # 学习到的坐标调整
        self.execution_strategies: Dict[str, Dict[str, Any]] = {}  # 执行策略优化
        
        # 创建默认配置（如果未提供）
        if agent_config is None:
            agent_config = AgentConfig(
                id=agent_id,
                name="ExecutorAgent",
                role="executor",
                goal="执行具体的操作任务",
                backstory="我是一个执行智能体，负责执行具体的操作任务",
                tools=[]
            )
        
        # 初始化工具 - 集成真实ADB操作
        self.tools = [
            ElementLocatorTool(),
            ADBClickTool(),        # 使用真实ADB点击
            ClickTool(),           # 保留模拟点击作为备用
            ADBInputTool(),        # 使用真实ADB输入
            InputTool(),           # 保留模拟输入作为备用
            ADBSwipeTool(),        # 使用真实ADB滑动
            SwipeTool(),           # 保留模拟滑动作为备用
            ADBScreenshotTool(),   # 使用真实ADB截图
            ScreenshotTool(),      # 保留模拟截图作为备用
            WaitTool()
        ]
        
        super().__init__(agent_config, llm_provider, memory, self.tools, info_pool=info_pool)
        
        # 执行状态
        self.current_action: Optional[Dict[str, Any]] = None
        self.action_history: List[Dict[str, Any]] = []
        self.retry_count: int = 0
        self.max_retries: int = 3
    
    async def _execute_task_impl(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体操作 - 重构版本，分离分析和执行
        
        Args:
            task_context: 任务上下文
        
        Returns:
            执行结果
        """
        task_type = task_context.get("task_type", "unknown")
        task_description = task_context.get("description", "")
        
        logger.info(f"开始执行任务: {task_description} | 任务类型: {task_type}")
        
        try:
            # 初始化可能用到的变量
            analysis_result = None
            
            # 第一阶段：多模态分析（如果需要）
            if task_context.get("use_multimodal_analysis", False):
                analysis_result = await self._analyze_and_get_coordinates(task_context)
                if not analysis_result.get("success"):
                    return analysis_result
                
                # 将分析结果合并到任务上下文中
                if "coordinates" in analysis_result:
                    task_context["coordinates"] = analysis_result["coordinates"]
                if "text" in analysis_result:
                    task_context["text"] = analysis_result["text"]
                
                # 记录分析结果
                logger.info(f"✅ 多模态分析完成: {analysis_result.get('action_plan', {})}")
            
            # 第二阶段：执行具体操作
            logger.info("⚙️ 开始执行阶段...")

            # 如果是多模态分析任务，从分析结果中确定实际的task_type
            if task_type == "multimodal_analysis" and analysis_result and "action_plan" in analysis_result:
                action = analysis_result["action_plan"].get("action")
                if action:
                    # 将外部原子操作名称转换为内部task_type
                    task_type = get_internal_task_type(action)
                    logger.info(f"🧠 多模态分析确定实际操作为: {task_type}")

            if task_type == "locate_element":
                result = await self._locate_element(task_context)
            elif task_type == "click_action":
                result = await self._execute_click(task_context)
            elif task_type == "input_text":
                result = await self._execute_input(task_context)
            elif task_type == "swipe_action":
                result = await self._execute_swipe(task_context)
            elif task_type == "take_screenshot":
                result = await self._take_screenshot(task_context)
            elif task_type == "wait":
                result = await self._execute_wait(task_context)
            elif task_type == "multimodal_analysis":
                # 如果分析后仍然是分析任务（例如，没有得出具体动作），则按原计划执行
                logger.warning("多模态分析未得出具体执行动作，按原计划执行分析...")
                result = await self._analyze_and_execute_with_llm(task_context)
            else:
                result = await self._execute_generic_action(task_context)
            
            # 如果有分析结果，将其合并到最终结果中
            if task_context.get("use_multimodal_analysis", False) and analysis_result is not None:
                result.update({
                    "llm_thought": analysis_result.get("llm_thought", ""),
                    "llm_description": analysis_result.get("llm_description", ""),
                    "llm_action_plan": analysis_result.get("action_plan", {}),
                    "execution_method": "multimodal_llm"
                })
            
            # 记录操作历史（排除内部分析操作）
            action_record = None
            if task_type != "multimodal_analysis":
                action_record = {
                    "task_type": task_type,
                    "task_context": task_context,
                    "result": result,
                    "timestamp": get_iso_timestamp(),
                    "success": True,
                    "expectation": task_context.get("description", "")
                }
                self.action_history.append(action_record)
            
            # 发布执行结果事件（仅对真实操作）
            if task_type != "multimodal_analysis" and action_record is not None:
                logger.info(f"📢 发布『action_result』事件，任务类型: {task_type}")                   
                logger.info(f"📦 动作执行细节: "); print(action_record)
                await self.info_pool.publish(
                    info_type="action_result",
                    data=action_record,
                    source_agent=self.config.id
                )
            
            self.retry_count = 0  # 重置重试计数
            return result
            
        except Exception as e:
            logger.error(f"操作执行失败: {task_type}, 错误: {e}")
            
            # 记录失败（排除内部分析操作）
            action_record = None
            if task_type != "multimodal_analysis":
                action_record = {
                    "task_type": task_type,
                    "task_context": task_context,
                    "error": str(e),
                    "timestamp": get_iso_timestamp(),
                    "success": False,
                    "retry_count": self.retry_count,
                    "expectation": task_context.get("description", "")
                }
                self.action_history.append(action_record)
            
            # 尝试重试
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                logger.info(f"尝试重试操作，第{self.retry_count}次")
                
                # 应用反思反馈进行智能重试
                await self._apply_reflection_feedback_for_retry(task_context)
                
                await asyncio.sleep(1)  # 等待1秒后重试
                return await self._execute_task_impl(task_context)
            
            raise
    
    async def _locate_element(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """定位元素"""
        element_description = task_context.get("element_description", "")
        
        locator_tool = self.get_tool("element_locator")
        if locator_tool is None:
            raise ValueError("ElementLocatorTool not found")
        result = locator_tool.execute(element_description=element_description)
        
        logger.info(f"元素定位完成: {element_description}")
        return result
    
    async def _analyze_and_get_coordinates(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """多模态分析阶段 - 获取操作坐标和参数（增强版本）
        
        Args:
            task_context: 任务上下文
        
        Returns:
            分析结果，包含坐标、文本等操作参数
        """
        description = task_context.get("description", "")
        target_description = task_context.get("target_description", "")
        full_description = f"{description}。目标元素：{target_description}"
        
        try:
            # 调用多模态分析
            analysis_context = {
                "task_type": "multimodal_analysis",
                "description": full_description
            }
            analysis_result = await self._analyze_and_execute_with_llm(analysis_context)
            
            if analysis_result.get("success"):
                # 提取坐标信息
                action_plan = analysis_result.get("llm_action_plan", {})
                result = {
                    "success": True,
                    "action_plan": action_plan,
                    "llm_thought": analysis_result.get("llm_thought", ""),
                    "llm_description": analysis_result.get("llm_description", "")
                }
                
                # 根据动作类型提取相应参数
                if "coordinate" in action_plan:
                    coord = action_plan["coordinate"]
                    if isinstance(coord, list) and len(coord) == 2:
                        # 应用坐标校准
                        calibrated_coord = await self._calibrate_coordinates(
                            coord, analysis_result.get("screenshot_path")
                        )
                        result["coordinates"] = {"x": calibrated_coord[0], "y": calibrated_coord[1]}
                        result["original_coordinates"] = {"x": coord[0], "y": coord[1]}
                
                if "text" in action_plan:
                    result["text"] = action_plan["text"]
                
                if "start_coordinate" in action_plan and "end_coordinate" in action_plan:
                    start_coord = action_plan["start_coordinate"]
                    end_coord = action_plan["end_coordinate"]
                    if isinstance(start_coord, list) and isinstance(end_coord, list):
                        # 应用坐标校准
                        calibrated_start = await self._calibrate_coordinates(
                            start_coord, analysis_result.get("screenshot_path")
                        )
                        calibrated_end = await self._calibrate_coordinates(
                            end_coord, analysis_result.get("screenshot_path")
                        )
                        result["start_coordinates"] = {"x": calibrated_start[0], "y": calibrated_start[1]}
                        result["end_coordinates"] = {"x": calibrated_end[0], "y": calibrated_end[1]}
                
                return result
            else:
                return {
                    "success": False,
                    "message": f"多模态分析失败: {analysis_result.get('message', '未知错误')}"
                }
                
        except Exception as e:
            logger.error(f"多模态分析异常: {e}")
            return {
                "success": False,
                "message": f"多模态分析异常: {str(e)}"
            }
    
    async def _calibrate_coordinates(self, coordinates: List[int], screenshot_path: Optional[str] = None) -> List[int]:
        """坐标校准方法 - 修正截图坐标与设备实际坐标的偏差，集成反思学习
        
        Args:
            coordinates: 原始坐标 [x, y]
            screenshot_path: 截图路径
        
        Returns:
            校准后的坐标 [x, y]
        """
        try:
            # 获取设备实际分辨率
            device_resolution = await self._get_device_resolution()
            
            # 获取截图分辨率
            if screenshot_path:
                screenshot_resolution = self._get_screen_dimensions(screenshot_path)
            else:
                screenshot_resolution = (1084, 2412)  # 默认值
            
            # 计算缩放比例
            scale_x = device_resolution[0] / screenshot_resolution[0]
            scale_y = device_resolution[1] / screenshot_resolution[1]
            
            # 应用缩放
            calibrated_x = int(coordinates[0] * scale_x)
            calibrated_y = int(coordinates[1] * scale_y)
            
            # 应用学习到的坐标调整（基于反思反馈）
            learned_adjustment = self._get_learned_coordinate_adjustment(coordinates)
            offset_x = learned_adjustment[0]
            offset_y = learned_adjustment[1]
            
            final_x = max(0, min(calibrated_x + offset_x, device_resolution[0] - 1))
            final_y = max(0, min(calibrated_y + offset_y, device_resolution[1] - 1))
            
            logger.info(f"🔧 智能坐标校准: {coordinates} -> [{final_x}, {final_y}] (缩放: {scale_x:.2f}, {scale_y:.2f}, 学习偏移: {offset_x}, {offset_y})")
            
            return [final_x, final_y]
            
        except Exception as e:
            logger.warning(f"坐标校准失败，使用原始坐标: {e}")
            return coordinates
    
    async def _get_device_resolution(self) -> Tuple[int, int]:
        """获取设备实际分辨率"""
        try:
            import subprocess
            result = subprocess.run(
                ["adb", "shell", "wm", "size"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                size_line = result.stdout.strip()
                if "Physical size:" in size_line:
                    size_part = size_line.split("Physical size:")[1].strip()
                    width, height = map(int, size_part.split('x'))
                    return (width, height)
                elif "Override size:" in size_line:
                    size_part = size_line.split("Override size:")[1].strip()
                    width, height = map(int, size_part.split('x'))
                    return (width, height)
        except Exception as e:
            logger.warning(f"获取设备分辨率失败: {e}")
        
        # 默认分辨率
        return (1084, 2412)
    
    async def _execute_click(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行点击操作 - 纯执行函数
        
        Args:
            task_context: 必须包含 coordinates 字段（通过分析阶段获得）
        
        Returns:
            点击执行结果
        """
        before_screenshot_result = await self._take_screenshot({})
        before_screenshot_path = before_screenshot_result.get("screenshot_path")

        # 获取坐标（应该已经通过分析阶段确定）
        if "coordinates" not in task_context:
            # 回退到传统元素定位（兼容性处理）
            logger.warning("未找到预分析的坐标，回退到传统元素定位")
            element_result = await self._locate_element(task_context)
            coordinates = element_result["coordinates"]
        else:
            coordinates = task_context["coordinates"]
        
        logger.info(f"🎯 执行点击操作，目标坐标: {coordinates}")
        
        result = {}
        # 优先使用ADB点击
        adb_click_tool = self.get_tool("adb_click")
        if adb_click_tool:
            adb_result = await adb_click_tool.aexecute(coordinates=coordinates)
            if adb_result.get("success"):
                logger.info(f"✅ ADB点击操作完成: {coordinates}")
                result = adb_result
            else:
                logger.warning(f"⚠️ ADB点击失败，回退到模拟点击: {adb_result.get('error')}")
        
        # 回退到模拟点击
        if not result.get("success"):
            click_tool = self.get_tool("click")
            if click_tool is None:
                raise ValueError("ClickTool not found")
            result = click_tool.execute(coordinates=coordinates)
            logger.info(f"✅ 模拟点击操作完成: {coordinates}")

        after_screenshot_result = await self._take_screenshot({})
        after_screenshot_path = after_screenshot_result.get("screenshot_path")
        
        result['before_screenshot'] = before_screenshot_path
        result['after_screenshot'] = after_screenshot_path
        return result
    
    async def _execute_input(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行输入操作"""
        before_screenshot_result = await self._take_screenshot({})
        before_screenshot_path = before_screenshot_result.get("screenshot_path")
        text = task_context.get("text", "")
        
        # 首先定位输入框
        if "coordinates" not in task_context:
            element_result = await self._locate_element(task_context)
            coordinates = element_result["coordinates"]
        else:
            coordinates = task_context["coordinates"]
        
        # 执行输入
        input_tool = self.get_tool("input")
        if input_tool is None:
            raise ValueError("InputTool not found")
        result = input_tool.execute(text=text, coordinates=coordinates)
        
        logger.info(f"输入操作完成: {text}")
        after_screenshot_result = await self._take_screenshot({})
        after_screenshot_path = after_screenshot_result.get("screenshot_path")
        result['before_screenshot'] = before_screenshot_path
        result['after_screenshot'] = after_screenshot_path
        return result
    
    async def _execute_swipe(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行滑动操作"""
        start_coordinates = task_context.get("start_coordinates", {"x": 500, "y": 1000})
        end_coordinates = task_context.get("end_coordinates", {"x": 500, "y": 500})
        duration = task_context.get("duration", 1.0)
        
        # 执行滑动
        swipe_tool = self.get_tool("swipe")
        if swipe_tool is None:
            raise ValueError("SwipeTool not found")
        result = swipe_tool.execute(start_coordinates=start_coordinates, end_coordinates=end_coordinates, duration=duration)
        
        logger.info(f"滑动操作完成: {start_coordinates} -> {end_coordinates}")
        return result
    
    async def _take_screenshot(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """获取截图 - 优先使用ADB"""
        # 优先使用ADB截图
        adb_screenshot_tool = self.get_tool("adb_screenshot")
        if adb_screenshot_tool:
            # 传递智能体ID给ADB截图工具
            agent_id = getattr(self, 'agent_id', 'executor')
            result = await adb_screenshot_tool.aexecute(agent_id=agent_id)
            if result.get("success"):
                logger.info(f"ADB截图完成: {result['screenshot_path']}")
                return result
            else:
                logger.warning(f"ADB截图失败，回退到模拟截图: {result.get('error')}")
        
        # 回退到模拟截图
        screenshot_tool = self.get_tool("screenshot")
        if screenshot_tool is None:
            raise ValueError("ScreenshotTool not found")
        # 传递智能体ID给截图工具
        agent_id = getattr(self, 'agent_id', 'executor')
        result = screenshot_tool.execute(agent_id=agent_id)
        
        logger.info(f"模拟截图完成: {result['screenshot_path']}")
        return result
    
    async def _execute_wait(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行等待"""
        duration = task_context.get("duration", 1.0)
        condition = task_context.get("condition")
        
        wait_tool = self.get_tool("wait")
        if wait_tool is None:
            raise ValueError("WaitTool not found")
        result = wait_tool.execute(duration=duration, condition=condition)
        
        logger.info(f"等待完成: {duration}秒")
        return result
    
    async def _execute_generic_action(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行通用操作 - 使用多模态LLM分析"""
        description = task_context.get("description", "")
        
        # 简单的操作识别
        if "点击" in description or "click" in description.lower():
            return await self._execute_click(task_context)
        elif "输入" in description or "input" in description.lower():
            return await self._execute_input(task_context)
        elif "滑动" in description or "swipe" in description.lower():
            return await self._execute_swipe(task_context)
        elif "截图" in description or "screenshot" in description.lower():
            return await self._take_screenshot(task_context)
        else:
            # 对于复杂任务，使用多模态LLM分析
            return await self._analyze_and_execute_with_llm(task_context)
    
    async def _analyze_and_execute_with_llm(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """使用多模态LLM分析截图并执行操作 - 基于Mobile Agent v3设计精髓"""
        description = task_context.get("description", "")
        
        try:
            # 1. 获取当前屏幕状态
            screenshot_result = await self._take_screenshot(task_context)
            if not screenshot_result.get("success"):
                return screenshot_result
            
            screenshot_path = screenshot_result.get("screenshot_path")
            
            # 2. 构建多模态分析提示词（参考Mobile Agent v3的Executor设计）
            if hasattr(self, 'llm_provider') and self.llm_provider:
                analysis_prompt = self._build_multimodal_analysis_prompt(
                    task_context, description, screenshot_path
                )
                
                # 3. 调用多模态LLM进行分析
                llm_response = await self._invoke_multimodal_llm(
                    analysis_prompt, screenshot_path
                )
                
                # 4. 解析LLM响应并执行操作
                return await self._parse_and_execute_llm_response(
                    llm_response, task_context, screenshot_path
                )
            
            else:
                # 没有LLM提供者，返回截图结果
                return {
                    "success": True,
                    "action": "screenshot_only",
                    "screenshot_path": screenshot_path,
                    "message": "未配置LLM提供者，仅执行截图操作"
                }
                
        except Exception as e:
            logger.error(f"多模态LLM分析执行失败: {e}")
            # 出错时返回截图结果
            return await self._take_screenshot(task_context)
    
    def _get_screen_dimensions(self, screenshot_path: str) -> Tuple[int, int]:
        """获取屏幕尺寸 - PIL优先，ADB回退"""
        # 优先使用PIL从截图获取尺寸
        try:
            from PIL import Image
            with Image.open(screenshot_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            logger.warning(f"PIL获取屏幕尺寸失败: {e}，尝试使用ADB获取")
        
        # 回退：使用ADB获取设备分辨率
        try:
            import subprocess
            result = subprocess.run(
                ["adb", "shell", "wm", "size"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                size_line = result.stdout.strip()
                if "Physical size:" in size_line:
                    size_part = size_line.split("Physical size:")[1].strip()
                    width, height = map(int, size_part.split('x'))
                    logger.info(f"ADB获取屏幕尺寸成功: {width}x{height}")
                    return (width, height)
                elif "Override size:" in size_line:
                    size_part = size_line.split("Override size:")[1].strip()
                    width, height = map(int, size_part.split('x'))
                    logger.info(f"ADB获取屏幕尺寸成功: {width}x{height}")
                    return (width, height)
        except Exception as e:
            logger.warning(f"ADB获取屏幕尺寸失败: {e}")
        
        # 最后回退：使用默认尺寸
        logger.warning("所有方法都失败，使用默认屏幕尺寸")
        return (640, 1400)  # 默认尺寸
    
    def _build_multimodal_analysis_prompt(
        self, 
        task_context: Dict[str, Any], 
        description: str,
        screenshot_path: Optional[str] = None
    ) -> str:
        """构建多模态分析提示词 - 参考Mobile Agent v3的Executor提示词设计"""
        
        # 获取屏幕尺寸
        if screenshot_path:
            screen_width, screen_height = self._get_screen_dimensions(screenshot_path)
        else:
            screen_width, screen_height = 640, 1400
        
        # 获取历史操作信息
        action_history = self.get_action_history()
        recent_actions = action_history[-5:] if action_history else []
        
        prompt = "你是一个专业的移动设备操作专家，能够分析截图并决定下一步要执行的操作。\n\n"
        
        prompt += "### 当前任务 ###\n"
        prompt += f"{description}\n\n"
        
        # 添加操作历史（如果有）
        if recent_actions:
            prompt += "### 最近操作历史 ###\n"
            prompt += "你之前执行的操作及其结果：\n"
            for i, action_record in enumerate(recent_actions, 1):
                task_type = get_external_action_name(action_record.get('task_type', 'unknown'))
                success = action_record.get('success', False)
                timestamp = action_record.get('timestamp', '')
                outcome = "成功" if success else "失败"
                prompt += f"{i}. 操作: {task_type} | 结果: {outcome} | 时间: {timestamp}\n"
            prompt += "\n"
        else:
            prompt += "### 最近操作历史 ###\n"
            prompt += "暂无之前的操作记录。\n\n"
        
        prompt += "---\n"
        prompt += "请仔细分析提供的截图，并决定下一步要执行的操作。"
        prompt += "你必须从可用的原子操作中选择。\n\n"
        
        prompt += "#### 可用的原子操作 ####\n"
        prompt += "原子操作函数列表如下：\n"
        prompt += "- screenshot(): 获取屏幕截图\n"
        prompt += "- locate_element(description): 定位UI元素\n"
        prompt += "- click(coordinate): 点击操作\n"
        prompt += "- long_press(coordinate): 长按操作\n"
        prompt += "- type(text): 文本输入\n"
        prompt += "- swipe(start_coordinate, end_coordinate): 滑动操作\n"
        prompt += "- open_app(app_name): 打开应用\n"
        prompt += "- system_button(button): 系统按键（back/home/enter）\n"
        prompt += "- wait(duration): 等待\n"
        prompt += "- verify(condition): 验证结果\n\n"
        
        prompt += "---\n"
        prompt += "重要指导原则：\n"
        prompt += "1. 仔细分析截图以识别UI元素及其位置\n"
        prompt += "2. 根据当前任务选择最合适的操作\n"
        prompt += "3. 为点击和滑动操作提供精确的坐标\n"
        prompt += "4. 不要重复之前失败的操作\n"
        prompt += "5. 考虑当前状态和朝向目标的进展\n\n"
        
        prompt += "#### 重要：当前截图分辨率信息 ####\n"
        prompt += f"📱 当前截图尺寸：{screen_width} x {screen_height} 像素\n"
        prompt += f"📐 图片宽度：{screen_width}px（X轴最大值）\n"
        prompt += f"📐 图片高度：{screen_height}px（Y轴最大值）\n\n"
        
        prompt += "#### 坐标系统说明 ####\n"
        prompt += "屏幕坐标系统：\n"
        prompt += "- 坐标原点(0,0)位于截图左上角\n"
        prompt += f"- X轴：从左到右递增，有效范围 0 到 {screen_width-1}\n"
        prompt += f"- Y轴：从上到下递增，有效范围 0 到 {screen_height-1}\n"
        prompt += f"屏幕底部应用栏：Y坐标通常在 {screen_height-50} 到 {screen_height-10} 之间\n"
        prompt += f"屏幕中央区域：X坐标约 {screen_width//2}，Y坐标约 {screen_height//2}\n"
        prompt += "- 状态栏区域：Y坐标通常在 0 到 100 之间\n"
        prompt += f"⚠️ 关键提醒：所有坐标必须在图片范围内！X < {screen_width}, Y < {screen_height}\n\n"
        

        prompt += "#### 坐标计算辅助公式 ####\n"
        prompt += "如果目标位于底部应用栏，你可以使用以下公式辅助计算坐标：\n"
        prompt += "1.  首先，从左到右确定目标的**位置序号**（从1开始）。\n"
        prompt += "2.  然后，使用以下公式计算坐标：\n"
        prompt += f"   - `margin = {screen_width} * 0.1` (屏幕左右边距)\n"
        prompt += f"   - `available_width = {screen_width} - 2 * margin`\n"
        prompt += f"   - `icon_spacing = available_width / 4` (5个图标有4个间距)\n"
        prompt += f"   - `x = margin + (位置序号 - 1) * icon_spacing`\n"
        prompt += f"   - `y = {screen_height} - 180` (估算的底部Y坐标)\n"
        prompt += "你的任务是在思考过程中，明确指出你使用的**位置序号**，并展示计算过程。\n\n"
        prompt += "在生成坐标时，请务必：\n"
        prompt += f"1. 确保 X 坐标在 0 到 {screen_width-1} 之间\n"
        prompt += f"2. 确保 Y 坐标在 0 到 {screen_height-1} 之间\n"
        prompt += "3. 仔细观察截图中元素的实际位置\n"
        prompt += "4. 点击坐标应该在目标元素的中心位置\n"
        prompt += "5. 如果不确定精确位置，请选择元素的视觉中心点\n\n"
        
        prompt += "请按以下格式提供输出：\n"
        prompt += "### 思考过程 ###\n"
        prompt += "对当前屏幕状态和任务的分析，包括目标元素的位置观察和坐标推理过程。\n"
        prompt += "⚠️ 重要：在思考过程中确定的坐标必须与最终操作中的坐标完全一致！\n\n"
        
        prompt += "### 操作 ###\n"
        prompt += "以有效的JSON格式提供你的决策。示例：\n"
        prompt += "- {\"action\": \"click\", \"coordinate\": [x, y], \"target\": \"目标元素描述\"}\n"
        prompt += "- {\"action\": \"type\", \"text\": \"要输入的文本\"}\n"
        prompt += "- {\"action\": \"swipe\", \"start_coordinate\": [x1, y1], \"end_coordinate\": [x2, y2]}\n"
        prompt += "- {\"action\": \"locate_element\", \"description\": \"元素描述\"}\n"
        prompt += "- {\"action\": \"verify\", \"condition\": \"验证条件\"}\n"
        prompt += "⚠️ 坐标一致性要求：JSON中的coordinate值必须与思考过程中分析的坐标完全相同！\n\n"
        
        prompt += "### 描述 ###\n"
        prompt += "对所选操作和预期结果的简要描述，包括为什么选择这个坐标位置。\n"
        
        return prompt
    
    async def _invoke_multimodal_llm(
        self, 
        prompt: str, 
        screenshot_path: Optional[str] = None
    ) -> Any:
        """调用多模态LLM进行分析"""
        try:
            # 打印提示词内容（用于调试）
            logger.info(f"发送给executor的指令: \n"); print(prompt)
            
            # 构建多模态消息
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            image_base64 = ""
            
            # 如果有截图路径，添加图片到消息中
            if screenshot_path:
                import base64
                with open(screenshot_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                )
            
            logger.info(f"📸 截图文件: {screenshot_path}")
            if screenshot_path:
                logger.info(f"🔗 图片大小: {len(image_base64)} 字符 (base64编码)")
            
            # 调用LLM
            # logger.info("🚀 正在调用多模态LLM...")
            if self.llm_provider is None:
                raise ValueError("LLM provider is not configured")
            response = await self.llm_provider.ainvoke(messages)
            # logger.info("✅ 多模态LLM调用成功")
            return response
            
        except Exception as e:
            logger.error(f"多模态LLM调用失败: {e}")
            raise
    
    async def take_screenshot(self) -> str:
        """
        Public method to take a screenshot.
        
        Returns:
            The path to the screenshot file.
        """
        logger.info("Taking screenshot via public method.")
        result = await self._take_screenshot({})
        if result.get("success"):
            return result.get("screenshot_path", "")
        return ""

    async def _parse_and_execute_llm_response(
        self, 
        llm_response: Any, 
        task_context: Dict[str, Any], 
        screenshot_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """解析LLM响应并执行相应操作 - 增强版本"""
        try:
            import json
            import re
            
            response_content = llm_response.content
            
            # # 添加详细的调试日志
            # logger.info(f"🔍 LLM原始响应内容 (前500字符): {response_content[:500]}...")
            # logger.info(f"📏 LLM响应总长度: {len(response_content)} 字符")
            
            # 尝试多种解析策略
            thought, action_plan, description = self._extract_response_components(response_content)
            
            if action_plan is None:
                # 如果标准解析失败，尝试智能解析
                action_plan = self._intelligent_action_extraction(response_content)
            
            if action_plan is None:
                raise ValueError("无法从LLM响应中提取有效的动作信息")
            
            # 记录成功解析的结果
            logger.info("✅ LLM分析思考:"); print(thought)
            logger.info("✅ LLM分析提取的动作:"); print(action_plan)
            logger.info(f"✅ LLM动作描述: {description}")
            
            # 执行相应操作
            return await self._execute_llm_planned_action(
                action_plan, task_context, thought, description, screenshot_path or ""
            )
            
        except Exception as e:
            logger.error(f"LLM响应解析失败: {e}")
            return {
                "success": False,
                "action": "analysis_error",
                "llm_analysis": getattr(llm_response, "content", str(llm_response)),
                "screenshot_path": screenshot_path,
                "error": str(e),
                "message": "LLM响应解析失败"
            }
    
    def _extract_response_components(self, response_content: str) -> Tuple[str, Optional[Dict], str]:
        """提取响应的各个组件：思考过程、动作计划、描述"""
        import json
        import re
        
        thought = ""
        action_plan = None
        description = ""
        
        try:
            # 方法1: 标准格式解析 (中文标题)
            thought_match = re.search(r'### 思考过程 ###\s*(.+?)\s*### 操作 ###', response_content, re.DOTALL)
            if not thought_match:
                # 方法2: 英文标题
                thought_match = re.search(r'### Thought ###\s*(.+?)\s*### Action ###', response_content, re.DOTALL)
            
            thought = thought_match.group(1).strip() if thought_match else ""
            
            # 提取动作JSON (中文标题)
            action_match = re.search(r'### 操作 ###\s*(.+?)\s*### 描述 ###', response_content, re.DOTALL)
            if not action_match:
                # 英文标题
                action_match = re.search(r'### Action ###\s*(.+?)\s*### Description ###', response_content, re.DOTALL)
            if not action_match:
                # 没有描述部分的情况
                action_match = re.search(r'### 操作 ###\s*(.+?)$', response_content, re.DOTALL)
            if not action_match:
                action_match = re.search(r'### Action ###\s*(.+?)$', response_content, re.DOTALL)
            
            if action_match:
                action_str = action_match.group(1).strip()
                action_plan = self._parse_json_from_text(action_str)
            
            # 提取描述 (中文标题)
            desc_match = re.search(r'### 描述 ###\s*(.+?)$', response_content, re.DOTALL)
            if not desc_match:
                # 英文标题
                desc_match = re.search(r'### Description ###\s*(.+?)$', response_content, re.DOTALL)
            
            description = desc_match.group(1).strip() if desc_match else ""
            
        except Exception as e:
            logger.warning(f"标准格式解析失败: {e}")
        
        return thought, action_plan, description
    
    def _parse_json_from_text(self, text: str) -> Optional[Dict]:
        """从文本中解析JSON，支持多种格式"""
        import json
        
        # 清理文本
        text = text.strip()
        
        # 移除代码块标记
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()
        
        # 移除换行符
        text = text.replace('\n', '').strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试修复常见的JSON格式问题
            try:
                # 修复单引号问题
                text = text.replace("'", '"')
                return json.loads(text)
            except json.JSONDecodeError:
                return None
    
    def _intelligent_action_extraction(self, response_content: str) -> Optional[Dict]:
        """智能提取动作信息，处理各种非标准格式"""
        import json
        import re
        
        logger.info("🤖 尝试智能解析LLM响应...")
        
        # 策略1: 查找任何JSON对象
        json_patterns = [
            r'\{[^{}]*"action"[^{}]*\}',  # 包含action字段的JSON
            r'\{[^{}]*"操作"[^{}]*\}',    # 包含中文操作字段的JSON
            r'\{[^{}]*\}',               # 任何JSON对象
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_content, re.DOTALL)
            for match in matches:
                action_plan = self._parse_json_from_text(match)
                if action_plan and ('action' in action_plan or '操作' in action_plan):
                    logger.info(f"✅ 智能解析成功: {action_plan}")
                    return action_plan
        
        # 策略2: 基于关键词推断动作
        action_keywords = {
            '点击': 'click',
            '输入': 'type', 
            '滑动': 'swipe',
            '等待': 'wait',
            '打开': 'open_app',
            'click': 'click',
            'type': 'type',
            'swipe': 'swipe',
            'wait': 'wait',
            'open': 'open_app'
        }
        
        for keyword, action in action_keywords.items():
            if keyword in response_content.lower():
                # 尝试提取坐标
                coord_match = re.search(r'\[(\d+),\s*(\d+)\]', response_content)
                if coord_match:
                    x, y = int(coord_match.group(1)), int(coord_match.group(2))
                    action_plan = {"action": action, "coordinate": [x, y]}
                    logger.info(f"✅ 关键词推断成功: {action_plan}")
                    return action_plan
                else:
                    # 没有坐标的情况
                    action_plan = {"action": action}
                    logger.info(f"✅ 关键词推断成功(无坐标): {action_plan}")
                    return action_plan
        
        # 策略3: 默认截图操作
        logger.warning("⚠️ 所有解析策略都失败，返回默认截图操作")
        return {"action": "screenshot", "reason": "无法解析LLM响应，执行截图操作"}
        
        return None
    
    def _draw_action_marker(self, screenshot_path: str, coordinates: Dict[str, int], action_type: str = "click") -> str:
        """在截图上绘制操作标记"""
        try:
            from PIL import Image, ImageDraw
            import os
            import re
            
            # 打开截图
            img = Image.open(screenshot_path)
            draw = ImageDraw.Draw(img, 'RGBA')
            
            x, y = coordinates['x'], coordinates['y']
            
            # 根据操作类型选择颜色和样式
            if action_type == "click":
                # 紫色半透明圆圈
                color = (128, 0, 128, 100)  # 紫色，透明度100
                radius = 25
            elif action_type == "long_press":
                # 深紫色半透明圆圈
                color = (75, 0, 130, 120)  # 深紫色，透明度120
                radius = 30
            elif action_type == "swipe":
                # 蓝色半透明圆圈
                color = (0, 100, 255, 100)  # 蓝色，透明度100
                radius = 20
            else:
                # 默认绿色
                color = (0, 255, 0, 100)
                radius = 25
            
            # 绘制圆圈
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline=(255, 255, 255, 150),  # 白色边框
                width=2
            )
            
            # 绘制中心点
            center_radius = 3
            draw.ellipse(
                [x - center_radius, y - center_radius, x + center_radius, y + center_radius],
                fill=(255, 255, 255, 200),  # 白色中心点
                outline=(0, 0, 0, 255),     # 黑色边框
                width=1
            )
            
            # 生成新的文件名 - 包含智能体名称和操作类型
            base_name = os.path.splitext(screenshot_path)[0]
            
            # 提取时间戳部分
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)', base_name)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
            else:
                # 如果没有找到时间戳，使用当前时间
                timestamp = get_iso_timestamp().replace(':', '-')
            
            # 获取智能体ID，默认为executor
            agent_id = getattr(self, 'agent_id', 'executor')
            
            # 生成新文件名：智能体名称_时间戳_操作类型.png
            directory = os.path.dirname(screenshot_path)
            marked_filename = f"{agent_id}_{timestamp}_{action_type}.png"
            marked_path = os.path.join(directory, marked_filename)
            
            # 保存标记后的截图
            img.save(marked_path)
            
            logger.info(f"✅ 操作标记已添加: {marked_path} (坐标: {x}, {y})")
            return marked_path
            
        except Exception as e:
            logger.warning(f"⚠️ 添加操作标记失败: {e}")
            return screenshot_path  # 返回原始截图路径
    
    async def _execute_llm_planned_action(
        self, 
        action_plan: Dict[str, Any], 
        task_context: Dict[str, Any],
        thought: str,
        description: str,
        screenshot_path: str
    ) -> Dict[str, Any]:
        """执行LLM规划的动作"""
        action_type = action_plan.get("action")
        
        try:
            if action_type == "screenshot":
                # 执行截图操作
                result = await self._take_screenshot(task_context)
            
            elif action_type == "locate_element":
                # 执行元素定位操作
                element_description = action_plan.get("description", "")
                locate_context = {
                    **task_context,
                    "element_description": element_description
                }
                result = await self._locate_element(locate_context)
            
            elif action_type == "click":
                coordinates = action_plan.get("coordinate", [540, 960])
                if isinstance(coordinates, list) and len(coordinates) == 2:
                    # 先进行坐标校准
                    calibrated_coord = await self._calibrate_coordinates(
                        coordinates, screenshot_path
                    )
                    
                    click_context = {
                        **task_context,
                        "coordinates": {"x": calibrated_coord[0], "y": calibrated_coord[1]},
                        "target": action_plan.get("target", "LLM指定位置")
                    }
                    result = await self._execute_click(click_context)
                    
                    # 在截图上添加点击标记 - 使用原始坐标（截图坐标系）
                    if result.get("success", False) and screenshot_path:
                        marked_path = self._draw_action_marker(
                            screenshot_path, 
                            {"x": coordinates[0], "y": coordinates[1]}, 
                            "click"
                        )
                        result["marked_screenshot_path"] = marked_path
                        result["original_coordinates"] = {"x": coordinates[0], "y": coordinates[1]}
                        result["calibrated_coordinates"] = {"x": calibrated_coord[0], "y": calibrated_coord[1]}
                        
                else:
                    raise ValueError(f"无效的点击坐标: {coordinates}")
            
            elif action_type == "long_press":
                coordinates = action_plan.get("coordinate", [540, 960])
                if isinstance(coordinates, list) and len(coordinates) == 2:
                    # 先进行坐标校准
                    calibrated_coord = await self._calibrate_coordinates(
                        coordinates, screenshot_path
                    )
                    
                    # 长按操作（可以复用点击逻辑，增加持续时间）
                    click_context = {
                        **task_context,
                        "coordinates": {"x": calibrated_coord[0], "y": calibrated_coord[1]},
                        "target": action_plan.get("target", "LLM指定位置"),
                        "duration": 2.0  # 长按2秒
                    }
                    result = await self._execute_click(click_context)
                    
                    # 在截图上添加长按标记 - 使用原始坐标（截图坐标系）
                    if result.get("success", False) and screenshot_path:
                        marked_path = self._draw_action_marker(
                            screenshot_path, 
                            {"x": coordinates[0], "y": coordinates[1]}, 
                            "long_press"
                        )
                        result["marked_screenshot_path"] = marked_path
                        result["original_coordinates"] = {"x": coordinates[0], "y": coordinates[1]}
                        result["calibrated_coordinates"] = {"x": calibrated_coord[0], "y": calibrated_coord[1]}
                        
                else:
                    raise ValueError(f"无效的长按坐标: {coordinates}")
            
            elif action_type == "type":
                text = action_plan.get("text", "")
                input_context = {
                    **task_context,
                    "text": text
                }
                result = await self._execute_input(input_context)
            
            elif action_type == "swipe":
                start_coord = action_plan.get("start_coordinate", [500, 1000])
                end_coord = action_plan.get("end_coordinate", [500, 500])
                swipe_context = {
                    **task_context,
                    "start_coordinates": {"x": start_coord[0], "y": start_coord[1]},
                    "end_coordinates": {"x": end_coord[0], "y": end_coord[1]},
                    "duration": action_plan.get("duration", 1.0)
                }
                result = await self._execute_swipe(swipe_context)
            
            elif action_type == "system_button":
                button = action_plan.get("button", "back")
                # 模拟系统按键操作
                result = {
                    "success": True,
                    "action": "system_button",
                    "button": button,
                    "timestamp": get_iso_timestamp(),
                    "message": f"执行系统按键: {button}"
                }
            
            elif action_type == "open_app":
                app_name = action_plan.get("app_name", action_plan.get("text", ""))
                # 模拟打开应用操作
                result = {
                    "success": True,
                    "action": "open_app",
                    "app_name": app_name,
                    "timestamp": get_iso_timestamp(),
                    "message": f"打开应用: {app_name}"
                }
            
            elif action_type == "wait":
                duration = action_plan.get("duration", 2)
                wait_context = {
                    **task_context,
                    "duration": duration
                }
                result = await self._execute_wait(wait_context)
            
            elif action_type == "verify":
                condition = action_plan.get("condition", "")
                # 执行验证操作
                verify_result = await self.verify_action_result(
                    {"condition": condition},
                    {"expected": condition}
                )
                result = {
                    "success": verify_result.get("success", True),
                    "action": "verify",
                    "condition": condition,
                    "verification_result": verify_result,
                    "timestamp": get_iso_timestamp(),
                    "message": f"验证条件: {condition}"
                }
            
            else:
                raise ValueError(f"不支持的动作类型: {action_type}")
            
            # 增强结果信息
            result.update({
                "llm_thought": thought,
                "llm_description": description,
                "llm_action_plan": action_plan,
                "screenshot_path": screenshot_path,
                "execution_method": "multimodal_llm"
            })
            
            # 如果有标记后的截图，确保传递给用户
            if result.get("marked_screenshot_path"):
                logger.info(f"📍 操作标记截图: {result['marked_screenshot_path']}")
            
            return result
            
        except Exception as e:
            logger.error(f"执行LLM规划动作失败: {e}")
            return {
                "success": False,
                "action": "execution_error",
                "llm_thought": thought,
                "llm_description": description,
                "llm_action_plan": action_plan,
                "screenshot_path": screenshot_path,
                "error": str(e),
                "message": f"执行LLM规划的{action_type}动作失败"
             }
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """获取操作历史"""
        return self.action_history.copy()
    
    def get_current_action(self) -> Optional[Dict[str, Any]]:
        """获取当前操作"""
        return self.current_action
    
    def clear_action_history(self) -> None:
        """清空操作历史"""
        self.action_history.clear()
        logger.info("操作历史已清空")
    
    async def verify_action_result(
        self,
        action: Dict[str, Any],
        expected_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """验证操作结果
        
        Args:
            action: 执行的操作
            expected_result: 期望结果
        
        Returns:
            验证结果
        """
        # 获取当前屏幕状态
        screenshot_result = await self._take_screenshot({})
        
        verification_result = {
            "action": action,
            "screenshot": screenshot_result,
            "verification_time": get_iso_timestamp(),
            "success": True,  # 简化处理，实际应该进行图像分析
            "confidence": 0.9
        }
        
        # 发布验证结果事件
        verification_event = Event(
            type="action_verification",
            data=verification_result,
            agent_id=self.config.id
        )
        await self.info_pool.publish_async(verification_event)
        
        return verification_result
    
    def _handle_reflection_feedback(self, event: Event) -> None:
        """处理反思反馈事件"""
        try:
            reflection_data = event.data.get("reflection_record", {})
            result = reflection_data.get("result", {})
            
            # 提取改进建议
            if result.get("success") and not result.get("operation_success", True):
                improvement_suggestions = result.get("improvement_suggestions", "")
                if improvement_suggestions:
                    self._apply_improvement_suggestions(improvement_suggestions, reflection_data)
                    logger.info(f"📝 应用反思改进建议: {improvement_suggestions[:100]}...")
            
            # 学习坐标调整
            multimodal_analysis = result.get("multimodal_analysis")
            if multimodal_analysis and "坐标" in str(multimodal_analysis):
                self._learn_coordinate_adjustment(multimodal_analysis, reflection_data)
            
        except Exception as e:
            logger.error(f"❌ 处理反思反馈失败: {e}")
    
    def _handle_improvement_suggestion(self, event: Event) -> None:
        """处理改进建议事件"""
        try:
            suggestion_data = event.data
            suggestion_type = suggestion_data.get("type", "general")
            suggestion_content = suggestion_data.get("content", "")
            
            if suggestion_type == "coordinate_adjustment":
                coordinates = suggestion_data.get("coordinates", [0, 0])
                adjustment = suggestion_data.get("adjustment", [0, 0])
                self._store_coordinate_adjustment(coordinates, adjustment)
                
            elif suggestion_type == "execution_strategy":
                task_type = suggestion_data.get("task_type", "unknown")
                strategy = suggestion_data.get("strategy", {})
                self._update_execution_strategy(task_type, strategy)
            
            logger.info(f"💡 应用改进建议: {suggestion_type} - {suggestion_content[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ 处理改进建议失败: {e}")
    
    def _apply_improvement_suggestions(self, suggestions: str, reflection_data: Dict[str, Any]) -> None:
        """应用改进建议"""
        try:
            # 解析建议内容
            if "坐标" in suggestions or "偏移" in suggestions:
                # 坐标相关建议
                task_context = reflection_data.get("task_context", {})
                action_info = task_context.get("action_info", {})
                if "coordinates" in action_info:
                    original_coords = action_info["coordinates"]
                    # 简单的启发式调整
                    if "向上" in suggestions:
                        adjustment = [0, -10]
                    elif "向下" in suggestions:
                        adjustment = [0, 10]
                    elif "向左" in suggestions:
                        adjustment = [-10, 0]
                    elif "向右" in suggestions:
                        adjustment = [10, 0]
                    else:
                        adjustment = [0, 0]
                    
                    self._store_coordinate_adjustment([original_coords["x"], original_coords["y"]], adjustment)
            
            # 存储到反思反馈中
            self.reflection_feedback[get_iso_timestamp()] = {
                "suggestions": suggestions,
                "reflection_data": reflection_data,
                "applied": True
            }
            
        except Exception as e:
            logger.warning(f"应用改进建议失败: {e}")
    
    def _learn_coordinate_adjustment(self, multimodal_analysis: Dict[str, Any], reflection_data: Dict[str, Any]) -> None:
        """学习坐标调整"""
        try:
            # 从多模态分析中提取坐标调整信息
            analysis_text = str(multimodal_analysis)
            
            # 简单的模式匹配来提取调整建议
            task_context = reflection_data.get("task_context", {})
            action_info = task_context.get("action_info", {})
            
            if "coordinates" in action_info:
                original_coords = [action_info["coordinates"]["x"], action_info["coordinates"]["y"]]
                
                # 基于分析结果推断调整
                if "偏上" in analysis_text or "太高" in analysis_text:
                    adjustment = [0, 10]  # 向下调整
                elif "偏下" in analysis_text or "太低" in analysis_text:
                    adjustment = [0, -10]  # 向上调整
                elif "偏左" in analysis_text:
                    adjustment = [10, 0]  # 向右调整
                elif "偏右" in analysis_text:
                    adjustment = [-10, 0]  # 向左调整
                else:
                    adjustment = [0, 0]  # 无调整
                
                if adjustment != [0, 0]:
                    self._store_coordinate_adjustment(original_coords, adjustment)
                    logger.info(f"🎯 学习坐标调整: {original_coords} -> 偏移 {adjustment}")
            
        except Exception as e:
            logger.warning(f"学习坐标调整失败: {e}")
    
    def _store_coordinate_adjustment(self, coordinates: List[int], adjustment: List[int]) -> None:
        """存储坐标调整"""
        coord_key = f"{coordinates[0]}_{coordinates[1]}"
        if coord_key not in self.coordinate_adjustments:
            self.coordinate_adjustments[coord_key] = adjustment
        else:
            # 平均化多次调整
            existing = self.coordinate_adjustments[coord_key]
            self.coordinate_adjustments[coord_key] = [
                int((existing[0] + adjustment[0]) / 2),
                int((existing[1] + adjustment[1]) / 2)
            ]
    
    def _get_learned_coordinate_adjustment(self, coordinates: List[int]) -> List[int]:
        """获取学习到的坐标调整"""
        coord_key = f"{coordinates[0]}_{coordinates[1]}"
        
        # 精确匹配
        if coord_key in self.coordinate_adjustments:
            return self.coordinate_adjustments[coord_key]
        
        # 区域匹配（在附近区域查找类似的调整）
        for stored_key, adjustment in self.coordinate_adjustments.items():
            try:
                stored_x, stored_y = map(int, stored_key.split('_'))
                if abs(stored_x - coordinates[0]) <= 50 and abs(stored_y - coordinates[1]) <= 50:
                    return adjustment
            except:
                continue
        
        # 默认无调整
        return [0, 0]
    
    def _update_execution_strategy(self, task_type: str, strategy: Dict[str, Any]) -> None:
        """更新执行策略"""
        self.execution_strategies[task_type] = strategy
        logger.info(f"📋 更新执行策略: {task_type} -> {strategy}")
    
    async def _apply_reflection_feedback_for_retry(self, task_context: Dict[str, Any]) -> None:
        """在重试时应用反思反馈"""
        try:
            task_type = task_context.get("task_type", "unknown")
            
            # 应用执行策略调整
            if task_type in self.execution_strategies:
                strategy = self.execution_strategies[task_type]
                logger.info(f"🔄 重试时应用策略: {strategy}")
                
                # 根据策略调整任务上下文
                if "timeout" in strategy:
                    task_context["timeout"] = strategy["timeout"]
                if "retry_delay" in strategy:
                    await asyncio.sleep(strategy["retry_delay"])
            
            # 应用最新的反思反馈
            if self.reflection_feedback:
                latest_feedback = list(self.reflection_feedback.values())[-1]
                suggestions = latest_feedback.get("suggestions", "")
                
                if "等待" in suggestions or "延迟" in suggestions:
                    await asyncio.sleep(2)  # 额外等待
                    logger.info("⏱️ 基于反思建议增加等待时间")
            
        except Exception as e:
            logger.warning(f"应用反思反馈失败: {e}")
    
    def get_reflection_feedback_summary(self) -> Dict[str, Any]:
        """获取反思反馈摘要"""
        return {
            "total_feedback_count": len(self.reflection_feedback),
            "coordinate_adjustments_count": len(self.coordinate_adjustments),
            "execution_strategies_count": len(self.execution_strategies),
            "recent_feedback": list(self.reflection_feedback.values())[-5:] if self.reflection_feedback else []
        }
    
    def clear_reflection_feedback(self) -> None:
        """清空反思反馈"""
        self.reflection_feedback.clear()
        self.coordinate_adjustments.clear()
        self.execution_strategies.clear()
        logger.info("🧹 反思反馈数据已清空")