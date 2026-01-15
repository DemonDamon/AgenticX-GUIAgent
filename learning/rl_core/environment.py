#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M6: RL环境抽象 - 移动GUI强化学习环境

基于OpenAI Gym和Ray RLlib接口设计，提供标准的RL环境抽象。
将移动GUI操作任务转化为标准的强化学习环境。

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import torch
from PIL import Image

import gymnasium as gym


@dataclass
class GUIAction:
    """GUI操作动作"""
    action_type: str  # click, swipe, input, scroll, etc.
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    direction: Optional[str] = None  # up, down, left, right
    duration: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ActionSpace(gym.Space):
    """GUI动作空间定义"""
    
    def __init__(self, screen_width: int = 1080, screen_height: int = 1920):
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 定义可用的动作类型
        self.action_types = [
            "click", "double_click", "long_press",
            "swipe", "scroll", "pinch", "zoom",
            "input", "key_press", "back", "home"
        ]
        
        # 动作空间维度：[action_type, x, y, text_length, duration]
        self.shape = (5,)
        self.dtype = np.float32
    
    def sample(self) -> GUIAction:
        """随机采样有效动作"""
        action_type = np.random.choice(self.action_types)
        
        # 根据动作类型生成相应参数
        if action_type in ["click", "double_click", "long_press"]:
            coordinates = (
                np.random.randint(0, self.screen_width),
                np.random.randint(0, self.screen_height)
            )
            return GUIAction(
                action_type=action_type,
                coordinates=coordinates,
                duration=np.random.uniform(0.1, 2.0) if action_type == "long_press" else None
            )
        
        elif action_type == "swipe":
            start_x = np.random.randint(0, self.screen_width)
            start_y = np.random.randint(0, self.screen_height)
            end_x = np.random.randint(0, self.screen_width)
            end_y = np.random.randint(0, self.screen_height)
            
            return GUIAction(
                action_type=action_type,
                coordinates=(start_x, start_y),
                metadata={"end_coordinates": (end_x, end_y)},
                duration=np.random.uniform(0.2, 1.0)
            )
        
        elif action_type == "input":
            # 生成随机文本
            text_length = np.random.randint(1, 50)
            text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789 '), text_length))
            
            return GUIAction(
                action_type=action_type,
                text=text
            )
        
        else:
            return GUIAction(action_type=action_type)
    
    def contains(self, action: GUIAction) -> bool:
        """检查动作有效性"""
        if not isinstance(action, GUIAction):
            return False
        
        if action.action_type not in self.action_types:
            return False
        
        # 检查坐标是否在屏幕范围内
        if action.coordinates:
            x, y = action.coordinates
            if not (0 <= x <= self.screen_width and 0 <= y <= self.screen_height):
                return False
        
        return True
    
    def to_tensor(self, action: GUIAction) -> torch.Tensor:
        """动作转换为张量表示"""
        # 编码动作类型
        action_type_idx = self.action_types.index(action.action_type)
        
        # 编码坐标（归一化）
        x = action.coordinates[0] / self.screen_width if action.coordinates else 0.0
        y = action.coordinates[1] / self.screen_height if action.coordinates else 0.0
        
        # 编码文本长度（归一化）
        text_length = len(action.text) / 100.0 if action.text else 0.0
        
        # 编码持续时间（归一化）
        duration = action.duration / 5.0 if action.duration else 0.0
        
        return torch.tensor([
            action_type_idx / len(self.action_types),
            x, y, text_length, duration
        ], dtype=torch.float32)


class StateSpace(gym.Space):
    """多模态状态空间定义"""
    
    def __init__(self, 
                 image_shape: Tuple[int, int, int] = (224, 224, 3),
                 text_max_length: int = 512,
                 action_history_length: int = 10):
        super().__init__()
        self.image_shape = image_shape
        self.text_max_length = text_max_length
        self.action_history_length = action_history_length
        
        # 状态空间包含：图像 + 文本嵌入 + 动作历史
        self.shape = {
            "image": image_shape,
            "text": (text_max_length,),
            "action_history": (action_history_length, 5)  # 5维动作表示
        }
    
    def sample(self) -> torch.Tensor:
        """随机采样状态"""
        # 生成随机图像
        image = torch.rand(self.image_shape)
        
        # 生成随机文本嵌入
        text = torch.rand(self.text_max_length)
        
        # 生成随机动作历史
        action_history = torch.rand(self.action_history_length, 5)
        
        return {
            "image": image,
            "text": text,
            "action_history": action_history
        }
    
    def contains(self, state: Dict[str, torch.Tensor]) -> bool:
        """检查状态有效性"""
        if not isinstance(state, dict):
            return False
        
        required_keys = ["image", "text", "action_history"]
        if not all(key in state for key in required_keys):
            return False
        
        # 检查各组件形状
        if state["image"].shape != self.image_shape:
            return False
        
        if state["text"].shape != (self.text_max_length,):
            return False
        
        if state["action_history"].shape != (self.action_history_length, 5):
            return False
        
        return True
    
    def normalize(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """状态标准化"""
        normalized_state = {}
        
        # 图像标准化 (0-1)
        normalized_state["image"] = torch.clamp(state["image"], 0, 1)
        
        # 文本嵌入标准化
        text = state["text"]
        if text.std() > 0:
            normalized_state["text"] = (text - text.mean()) / text.std()
        else:
            normalized_state["text"] = text
        
        # 动作历史标准化
        action_history = state["action_history"]
        normalized_state["action_history"] = torch.clamp(action_history, 0, 1)
        
        return normalized_state


class MobileGUIEnvironment(gym.Env):
    """移动GUI强化学习环境"""
    
    def __init__(self, 
                 screen_width: int = 1080,
                 screen_height: int = 1920,
                 max_episode_steps: int = 100,
                 reward_config: Optional[Dict] = None):
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config or {}
        
        # 定义动作和状态空间
        self.action_space = ActionSpace(screen_width, screen_height)
        self.observation_space = StateSpace()
        
        # 环境状态
        self.current_state = None
        self.step_count = 0
        self.episode_count = 0
        self.done = False
        
        # 历史记录
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        
        # 日志
        self.logger = logger
        
        # 初始化环境
        self._initialize_environment()
    
    def _initialize_environment(self):
        """初始化环境"""
        logger.info(f"初始化移动GUI环境 - 屏幕尺寸: {self.screen_width}x{self.screen_height}")
        
        # 这里可以集成真实的移动设备连接
        # 例如：ADB连接、模拟器连接等
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """重置环境并返回初始状态"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.step_count = 0
        self.done = False
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        
        # 获取初始状态
        self.current_state = self._get_current_state()
        
        info = {
            "episode": self.episode_count,
            "step": self.step_count,
            "timestamp": datetime.now().isoformat()
        }
        
        self.episode_count += 1
        logger.info(f"环境重置 - Episode {self.episode_count}")
        
        return self.current_state, info
    
    def step(self, action: GUIAction) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict]:
        """执行动作并返回转移结果"""
        if self.done:
            raise RuntimeError("环境已结束，请先调用reset()")
        
        # 验证动作
        if not self.action_space.contains(action):
            raise ValueError(f"无效动作: {action}")
        
        # 执行动作
        execution_result = self._execute_action(action)
        
        # 获取新状态
        next_state = self._get_current_state()
        
        # 计算奖励
        reward = self._calculate_reward(self.current_state, action, next_state, execution_result)
        
        # 检查是否结束
        self.step_count += 1
        terminated = self._is_terminated(next_state, execution_result)
        truncated = self.step_count >= self.max_episode_steps
        self.done = terminated or truncated
        
        # 更新历史
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.state_history.append(self.current_state)
        
        # 更新当前状态
        self.current_state = next_state
        
        # 构建信息字典
        info = {
            "step": self.step_count,
            "execution_result": execution_result,
            "terminated": terminated,
            "truncated": truncated,
            "timestamp": datetime.now().isoformat()
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _execute_action(self, action: GUIAction) -> Dict[str, Any]:
        """执行GUI动作"""
        # 这里应该集成真实的GUI操作
        # 例如：通过ADB执行点击、滑动等操作
        
        execution_result = {
            "success": True,
            "action_type": action.action_type,
            "coordinates": action.coordinates,
            "execution_time": np.random.uniform(0.1, 0.5),
            "error_message": None
        }
        
        # 模拟执行失败的情况
        if np.random.random() < 0.05:  # 5%失败率
            execution_result["success"] = False
            execution_result["error_message"] = "动作执行失败"
        
        logger.debug(f"执行动作: {action.action_type} - 结果: {execution_result['success']}")
        
        return execution_result
    
    def _get_current_state(self) -> Dict[str, torch.Tensor]:
        """获取当前环境状态"""
        # 这里应该获取真实的屏幕截图和UI信息
        # 目前使用模拟数据
        
        # 模拟屏幕截图
        image = torch.rand(224, 224, 3)
        
        # 模拟文本信息（任务描述等）
        text = torch.rand(512)
        
        # 动作历史（最近10个动作）
        action_history = torch.zeros(10, 5)
        if self.action_history:
            recent_actions = self.action_history[-10:]
            for i, action in enumerate(recent_actions):
                action_tensor = self.action_space.to_tensor(action)
                action_history[i] = action_tensor
        
        state = {
            "image": image,
            "text": text,
            "action_history": action_history
        }
        
        return self.observation_space.normalize(state)
    
    def _calculate_reward(self, 
                         state: Dict[str, torch.Tensor], 
                         action: GUIAction, 
                         next_state: Dict[str, torch.Tensor],
                         execution_result: Dict[str, Any]) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 基础执行奖励
        if execution_result["success"]:
            reward += 1.0
        else:
            reward -= 0.5
        
        # 效率奖励（执行时间越短越好）
        execution_time = execution_result.get("execution_time", 0.5)
        efficiency_reward = max(0, 1.0 - execution_time)
        reward += efficiency_reward * 0.2
        
        # 进度奖励（模拟任务进度）
        progress_reward = np.random.uniform(0, 0.5)
        reward += progress_reward
        
        return reward
    
    def _is_terminated(self, state: Dict[str, torch.Tensor], execution_result: Dict[str, Any]) -> bool:
        """检查是否达到终止条件"""
        # 这里应该根据具体任务判断是否完成
        # 目前使用随机终止条件
        
        # 如果连续失败多次，则终止
        if len(self.reward_history) >= 3:
            recent_rewards = self.reward_history[-3:]
            if all(r < 0 for r in recent_rewards):
                return True
        
        # 随机终止（模拟任务完成）
        if np.random.random() < 0.02:  # 2%概率终止
            return True
        
        return False
    
    def get_multimodal_state(self) -> Dict[str, torch.Tensor]:
        """获取多模态状态表示"""
        return self.current_state if self.current_state is not None else self._get_current_state()
    
    def render(self, mode: str = 'human') -> Any:
        """环境可视化"""
        if mode == 'human':
            print(f"Step: {self.step_count}, Episode: {self.episode_count}")
            print(f"Action History: {len(self.action_history)} actions")
            print(f"Reward History: {self.reward_history[-5:] if self.reward_history else []}")
            return None
        
        elif mode == 'rgb_array':
            # 返回当前屏幕截图
            if self.current_state:
                return self.current_state["image"].numpy()
            return np.zeros((224, 224, 3))
        
        else:
            raise ValueError(f"不支持的渲染模式: {mode}")
    
    def close(self):
        """关闭环境资源"""
        logger.info("关闭移动GUI环境")
        # 这里应该关闭设备连接等资源
        pass


class EnvironmentWrapper(gym.Wrapper):
    """环境包装器基类"""
    
    def __init__(self, env: MobileGUIEnvironment):
        super().__init__(env)
        self.logger = logger
    
    @staticmethod
    def wrap_with_logging(env: MobileGUIEnvironment) -> 'LoggingWrapper':
        """添加日志记录功能"""
        return LoggingWrapper(env)
    
    @staticmethod
    def wrap_with_safety(env: MobileGUIEnvironment) -> 'SafetyWrapper':
        """添加安全检查功能"""
        return SafetyWrapper(env)


class LoggingWrapper(EnvironmentWrapper):
    """日志记录包装器"""
    
    def __init__(self, env: MobileGUIEnvironment):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def reset(self, **kwargs):
        if hasattr(self.env, 'reward_history') and self.env.reward_history:
            episode_reward = sum(self.env.reward_history)
            episode_length = len(self.env.reward_history)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            logger.info(f"Episode结束 - 总奖励: {episode_reward:.2f}, 步数: {episode_length}")
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        result = self.env.step(action)
        state, reward, terminated, truncated, info = result
        
        logger.debug(f"Step {self.env.step_count}: Action={action.action_type}, Reward={reward:.2f}")
        
        return result


class SafetyWrapper(EnvironmentWrapper):
    """安全检查包装器"""
    
    def __init__(self, env: MobileGUIEnvironment, max_failed_actions: int = 5):
        super().__init__(env)
        self.max_failed_actions = max_failed_actions
        self.failed_action_count = 0
    
    def step(self, action):
        # 安全检查：避免危险操作
        if self._is_dangerous_action(action):
            logger.warning(f"阻止危险操作: {action.action_type}")
            # 返回无操作的结果
            return self.env.current_state, -1.0, False, False, {"safety_blocked": True}
        
        result = self.env.step(action)
        state, reward, terminated, truncated, info = result
        
        # 跟踪失败的动作
        if not info.get("execution_result", {}).get("success", True):
            self.failed_action_count += 1
            if self.failed_action_count >= self.max_failed_actions:
                logger.warning(f"连续失败动作过多，强制终止episode")
                terminated = True
                info["safety_terminated"] = True
        else:
            self.failed_action_count = 0
        
        return state, reward, terminated, truncated, info
    
    def _is_dangerous_action(self, action: GUIAction) -> bool:
        """检查是否为危险操作"""
        # 定义危险操作列表
        dangerous_actions = ["factory_reset", "delete_all", "format"]
        
        if action.action_type in dangerous_actions:
            return True
        
        # 检查是否点击系统关键区域
        if action.coordinates and action.action_type == "click":
            x, y = action.coordinates
            # 避免点击状态栏等系统区域
            if y < 50:  # 状态栏区域
                return True
        
        return False
    
    def reset(self, **kwargs):
        self.failed_action_count = 0
        return self.env.reset(**kwargs)