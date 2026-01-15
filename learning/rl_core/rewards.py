#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M10: 奖励函数设计 - 多维度奖励计算

基于Multi-objective RL、Reward Shaping、Intrinsic Motivation等设计理念，
提供任务完成度、执行效率、用户体验等多维度奖励计算。

Author: AgenticX Team
Date: 2025
"""

from loguru import logger
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, UTC

import numpy as np
import torch
from scipy import stats


@dataclass
class RewardComponents:
    """奖励组件"""
    task_completion: float = 0.0
    efficiency: float = 0.0
    user_experience: float = 0.0
    learning_progress: float = 0.0
    safety: float = 0.0
    exploration: float = 0.0
    
    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """计算加权总奖励"""
        if weights is None:
            weights = {
                'task_completion': 0.4,
                'efficiency': 0.2,
                'user_experience': 0.2,
                'learning_progress': 0.1,
                'safety': 0.05,
                'exploration': 0.05
            }
        
        return (
            self.task_completion * weights.get('task_completion', 0.4) +
            self.efficiency * weights.get('efficiency', 0.2) +
            self.user_experience * weights.get('user_experience', 0.2) +
            self.learning_progress * weights.get('learning_progress', 0.1) +
            self.safety * weights.get('safety', 0.05) +
            self.exploration * weights.get('exploration', 0.05)
        )
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'task_completion': self.task_completion,
            'efficiency': self.efficiency,
            'user_experience': self.user_experience,
            'learning_progress': self.learning_progress,
            'safety': self.safety,
            'exploration': self.exploration
        }


class BaseReward(ABC):
    """奖励计算基类"""
    
    def __init__(self, weight: float = 1.0, normalize: bool = True):
        self.weight = weight
        self.normalize = normalize
        self.logger = logger
    
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """计算奖励"""
        pass
    
    def _normalize_reward(self, reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """归一化奖励到指定范围"""
        if not self.normalize:
            return reward
        
        return np.clip(reward, min_val, max_val)


class TaskCompletionReward(BaseReward):
    """任务完成度奖励"""
    
    def __init__(self, 
                 completion_bonus: float = 10.0,
                 progress_weight: float = 1.0,
                 failure_penalty: float = -5.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.completion_bonus = completion_bonus
        self.progress_weight = progress_weight
        self.failure_penalty = failure_penalty
        
        # 进度历史（用于计算进度变化）
        self.progress_history = []
    
    def calculate_completion_reward(self, 
                                  task_progress: float, 
                                  task_success: bool, 
                                  execution_efficiency: float) -> float:
        """计算完成奖励"""
        reward = 0.0
        
        # 基础进度奖励
        progress_reward = task_progress * self.progress_weight
        reward += progress_reward
        
        # 完成奖励
        if task_success:
            # 根据效率调整完成奖励
            efficiency_multiplier = min(2.0, max(0.5, execution_efficiency))
            completion_reward = self.completion_bonus * efficiency_multiplier
            reward += completion_reward
            
            logger.debug(f"任务完成奖励: {completion_reward:.2f} (效率倍数: {efficiency_multiplier:.2f})")
        
        # 失败惩罚
        elif task_progress < 0.1:  # 几乎没有进度
            reward += self.failure_penalty
        
        return self._normalize_reward(reward)
    
    def progressive_reward(self, 
                          subtask_completions: List[bool], 
                          task_complexity: float) -> float:
        """渐进式奖励"""
        if not subtask_completions:
            return 0.0
        
        # 计算完成率
        completion_rate = sum(subtask_completions) / len(subtask_completions)
        
        # 基础渐进奖励
        base_reward = completion_rate * self.progress_weight
        
        # 复杂度调整
        complexity_multiplier = 1.0 + (task_complexity - 0.5) * 0.5
        adjusted_reward = base_reward * complexity_multiplier
        
        # 连续完成奖励
        consecutive_bonus = self._calculate_consecutive_bonus(subtask_completions)
        
        total_reward = adjusted_reward + consecutive_bonus
        
        return self._normalize_reward(total_reward)
    
    def _calculate_consecutive_bonus(self, completions: List[bool]) -> float:
        """计算连续完成奖励"""
        if not completions:
            return 0.0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for completed in completions:
            if completed:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # 连续完成奖励（指数增长）
        consecutive_bonus = 0.1 * (1.2 ** max_consecutive - 1) if max_consecutive > 1 else 0.0
        
        return consecutive_bonus
    
    def calculate(self, **kwargs) -> float:
        """计算奖励"""
        task_progress = kwargs.get('task_progress', 0.0)
        task_success = kwargs.get('task_success', False)
        execution_efficiency = kwargs.get('execution_efficiency', 1.0)
        subtask_completions = kwargs.get('subtask_completions', [])
        task_complexity = kwargs.get('task_complexity', 0.5)
        
        # 主要完成奖励
        completion_reward = self.calculate_completion_reward(
            task_progress, task_success, execution_efficiency
        )
        
        # 渐进式奖励
        progressive_reward = self.progressive_reward(
            subtask_completions, task_complexity
        )
        
        total_reward = (completion_reward + progressive_reward) * self.weight
        
        return self._normalize_reward(total_reward)


class EfficiencyReward(BaseReward):
    """执行效率奖励"""
    
    def __init__(self, 
                 time_weight: float = 0.5,
                 step_weight: float = 0.3,
                 resource_weight: float = 0.2,
                 optimal_time_baseline: float = 30.0,  # 秒
                 optimal_step_baseline: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.time_weight = time_weight
        self.step_weight = step_weight
        self.resource_weight = resource_weight
        self.optimal_time_baseline = optimal_time_baseline
        self.optimal_step_baseline = optimal_step_baseline
        
        # 性能历史（用于自适应基线）
        self.performance_history = {
            'execution_times': [],
            'step_counts': [],
            'resource_usage': []
        }
    
    def calculate_efficiency_reward(self, 
                                  execution_time: float, 
                                  step_count: int, 
                                  optimal_baseline: Dict) -> float:
        """计算效率奖励"""
        reward = 0.0
        
        # 时间效率奖励
        optimal_time = optimal_baseline.get('time', self.optimal_time_baseline)
        time_efficiency = max(0, optimal_time - execution_time) / optimal_time
        time_reward = time_efficiency * self.time_weight
        reward += time_reward
        
        # 步数效率奖励
        optimal_steps = optimal_baseline.get('steps', self.optimal_step_baseline)
        step_efficiency = max(0, optimal_steps - step_count) / optimal_steps
        step_reward = step_efficiency * self.step_weight
        reward += step_reward
        
        # 资源使用效率
        resource_usage = optimal_baseline.get('resource_usage', 0.5)
        resource_efficiency = max(0, 1.0 - resource_usage)
        resource_reward = resource_efficiency * self.resource_weight
        reward += resource_reward
        
        # 更新历史
        self.performance_history['execution_times'].append(execution_time)
        self.performance_history['step_counts'].append(step_count)
        self.performance_history['resource_usage'].append(resource_usage)
        
        # 限制历史长度
        for key in self.performance_history:
            if len(self.performance_history[key]) > 100:
                self.performance_history[key] = self.performance_history[key][-100:]
        
        return self._normalize_reward(reward)
    
    def adaptive_efficiency_baseline(self, performance_history: List[float]) -> float:
        """自适应效率基线"""
        if len(performance_history) < 5:
            return self.optimal_time_baseline
        
        # 使用移动平均和标准差
        recent_performance = performance_history[-20:]  # 最近20次
        mean_performance = np.mean(recent_performance)
        std_performance = np.std(recent_performance)
        
        # 基线设为均值减去一个标准差（鼓励超越平均水平）
        adaptive_baseline = mean_performance - std_performance
        
        # 确保基线在合理范围内
        adaptive_baseline = max(5.0, min(120.0, adaptive_baseline))
        
        return adaptive_baseline
    
    def calculate_speed_bonus(self, execution_time: float, baseline_time: float) -> float:
        """计算速度奖励"""
        if execution_time <= 0 or baseline_time <= 0:
            return 0.0
        
        speed_ratio = baseline_time / execution_time
        
        if speed_ratio > 1.0:  # 比基线快
            # 对数奖励，避免过度奖励极快的执行
            speed_bonus = math.log(speed_ratio) * 0.5
        else:  # 比基线慢
            speed_bonus = -(1.0 - speed_ratio) * 0.5
        
        return self._normalize_reward(speed_bonus, -1.0, 1.0)
    
    def calculate(self, **kwargs) -> float:
        """计算奖励"""
        execution_time = kwargs.get('execution_time', 0.0)
        step_count = kwargs.get('step_count', 0)
        optimal_baseline = kwargs.get('optimal_baseline', {})
        
        # 主要效率奖励
        efficiency_reward = self.calculate_efficiency_reward(
            execution_time, step_count, optimal_baseline
        )
        
        # 速度奖励
        baseline_time = optimal_baseline.get('time', self.optimal_time_baseline)
        speed_bonus = self.calculate_speed_bonus(execution_time, baseline_time)
        
        total_reward = (efficiency_reward + speed_bonus) * self.weight
        
        return self._normalize_reward(total_reward)


class UserExperienceReward(BaseReward):
    """用户体验奖励"""
    
    def __init__(self, 
                 smoothness_weight: float = 0.4,
                 responsiveness_weight: float = 0.3,
                 accuracy_weight: float = 0.3,
                 error_penalty: float = -0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.smoothness_weight = smoothness_weight
        self.responsiveness_weight = responsiveness_weight
        self.accuracy_weight = accuracy_weight
        self.error_penalty = error_penalty
        
        # UX指标历史
        self.ux_history = {
            'smoothness_scores': [],
            'response_times': [],
            'error_counts': []
        }
    
    def calculate_ux_reward(self, 
                           user_feedback: Dict, 
                           interaction_smoothness: float, 
                           error_count: int) -> float:
        """计算UX奖励"""
        reward = 0.0
        
        # 交互流畅度奖励
        smoothness_reward = interaction_smoothness * self.smoothness_weight
        reward += smoothness_reward
        
        # 响应性奖励
        response_time = user_feedback.get('response_time', 1.0)
        responsiveness = max(0, 1.0 - response_time / 5.0)  # 5秒为基准
        responsiveness_reward = responsiveness * self.responsiveness_weight
        reward += responsiveness_reward
        
        # 准确性奖励
        accuracy = user_feedback.get('accuracy', 0.8)
        accuracy_reward = accuracy * self.accuracy_weight
        reward += accuracy_reward
        
        # 错误惩罚
        error_penalty = error_count * self.error_penalty
        reward += error_penalty
        
        # 更新历史
        self.ux_history['smoothness_scores'].append(interaction_smoothness)
        self.ux_history['response_times'].append(response_time)
        self.ux_history['error_counts'].append(error_count)
        
        return self._normalize_reward(reward)
    
    def implicit_ux_signals(self, 
                           action_sequence: List[Dict], 
                           timing_data: List[float]) -> float:
        """隐式UX信号"""
        if not action_sequence or not timing_data:
            return 0.0
        
        reward = 0.0
        
        # 动作序列流畅度
        sequence_smoothness = self._calculate_sequence_smoothness(action_sequence)
        reward += sequence_smoothness * 0.3
        
        # 时间一致性
        timing_consistency = self._calculate_timing_consistency(timing_data)
        reward += timing_consistency * 0.3
        
        # 动作合理性
        action_reasonableness = self._calculate_action_reasonableness(action_sequence)
        reward += action_reasonableness * 0.4
        
        return self._normalize_reward(reward)
    
    def _calculate_sequence_smoothness(self, action_sequence: List[Dict]) -> float:
        """计算动作序列流畅度"""
        if len(action_sequence) < 2:
            return 1.0
        
        smoothness_scores = []
        
        for i in range(1, len(action_sequence)):
            prev_action = action_sequence[i-1]
            curr_action = action_sequence[i]
            
            # 计算动作间的相似性/连续性
            similarity = self._calculate_action_similarity(prev_action, curr_action)
            smoothness_scores.append(similarity)
        
        return np.mean(smoothness_scores) if smoothness_scores else 1.0
    
    def _calculate_timing_consistency(self, timing_data: List[float]) -> float:
        """计算时间一致性"""
        if len(timing_data) < 2:
            return 1.0
        
        # 计算时间间隔的变异系数
        intervals = [timing_data[i] - timing_data[i-1] for i in range(1, len(timing_data))]
        
        if not intervals:
            return 1.0
        
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 1.0
        
        # 变异系数越小，一致性越好
        cv = std_interval / mean_interval
        consistency = max(0, 1.0 - cv)
        
        return consistency
    
    def _calculate_action_similarity(self, action1: Dict, action2: Dict) -> float:
        """计算动作相似性"""
        similarity = 0.0
        
        # 动作类型相似性
        if action1.get('type') == action2.get('type'):
            similarity += 0.5
        
        # 坐标相似性（如果有）
        if 'coordinates' in action1 and 'coordinates' in action2:
            coord1 = action1['coordinates']
            coord2 = action2['coordinates']
            
            if coord1 and coord2:
                distance = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                # 归一化距离（假设屏幕尺寸为1080x1920）
                normalized_distance = distance / math.sqrt(1080**2 + 1920**2)
                coord_similarity = max(0, 1.0 - normalized_distance)
                similarity += coord_similarity * 0.3
        
        # 时间间隔相似性
        time1 = action1.get('timestamp', 0)
        time2 = action2.get('timestamp', 0)
        if time1 and time2:
            time_diff = abs(time2 - time1)
            time_similarity = max(0, 1.0 - time_diff / 5.0)  # 5秒为基准
            similarity += time_similarity * 0.2
        
        return min(1.0, similarity)
    
    def _calculate_action_reasonableness(self, action_sequence: List[Dict]) -> float:
        """计算动作合理性"""
        if not action_sequence:
            return 1.0
        
        reasonableness_scores = []
        
        for action in action_sequence:
            score = 1.0
            
            # 检查动作类型合理性
            action_type = action.get('type', '')
            if action_type in ['click', 'tap', 'input']:
                score += 0.1  # 常见操作加分
            elif action_type in ['factory_reset', 'delete_all']:
                score -= 0.5  # 危险操作减分
            
            # 检查坐标合理性
            coordinates = action.get('coordinates')
            if coordinates:
                x, y = coordinates
                # 检查是否在屏幕范围内
                if 0 <= x <= 1080 and 0 <= y <= 1920:
                    score += 0.1
                else:
                    score -= 0.3
            
            reasonableness_scores.append(max(0, score))
        
        return np.mean(reasonableness_scores) if reasonableness_scores else 1.0
    
    def calculate(self, **kwargs) -> float:
        """计算奖励"""
        user_feedback = kwargs.get('user_feedback', {})
        interaction_smoothness = kwargs.get('interaction_smoothness', 0.8)
        error_count = kwargs.get('error_count', 0)
        action_sequence = kwargs.get('action_sequence', [])
        timing_data = kwargs.get('timing_data', [])
        
        # 显式UX奖励
        explicit_ux_reward = self.calculate_ux_reward(
            user_feedback, interaction_smoothness, error_count
        )
        
        # 隐式UX信号
        implicit_ux_reward = self.implicit_ux_signals(
            action_sequence, timing_data
        )
        
        total_reward = (explicit_ux_reward + implicit_ux_reward * 0.3) * self.weight
        
        return self._normalize_reward(total_reward)


class LearningProgressReward(BaseReward):
    """学习进步奖励"""
    
    def __init__(self, 
                 improvement_weight: float = 0.6,
                 exploration_weight: float = 0.4,
                 novelty_bonus: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.improvement_weight = improvement_weight
        self.exploration_weight = exploration_weight
        self.novelty_bonus = novelty_bonus
        
        # 学习历史
        self.learning_history = {
            'performance_scores': [],
            'exploration_counts': {},
            'novelty_scores': []
        }
    
    def calculate_learning_reward(self, 
                                performance_improvement: float, 
                                exploration_bonus: float) -> float:
        """基于学习进步计算奖励"""
        reward = 0.0
        
        # 性能改进奖励
        improvement_reward = performance_improvement * self.improvement_weight
        reward += improvement_reward
        
        # 探索奖励
        exploration_reward = exploration_bonus * self.exploration_weight
        reward += exploration_reward
        
        return self._normalize_reward(reward)
    
    def curiosity_driven_reward(self, state_novelty: float) -> float:
        """基于好奇心驱动的内在奖励"""
        # 新颖性奖励
        novelty_reward = state_novelty * self.novelty_bonus
        
        # 更新新颖性历史
        self.learning_history['novelty_scores'].append(state_novelty)
        
        return self._normalize_reward(novelty_reward)
    
    def calculate_state_novelty(self, current_state: torch.Tensor, state_history: List[torch.Tensor]) -> float:
        """计算状态新颖性"""
        if not state_history:
            return 1.0  # 第一个状态总是新颖的
        
        # 计算与历史状态的最小距离
        min_distance = float('inf')
        
        for hist_state in state_history[-50:]:  # 只考虑最近50个状态
            distance = torch.norm(current_state - hist_state).item()
            min_distance = min(min_distance, distance)
        
        # 将距离转换为新颖性分数
        novelty = min(1.0, min_distance / 10.0)  # 假设距离10为完全新颖
        
        return novelty
    
    def calculate(self, **kwargs) -> float:
        """计算奖励"""
        performance_improvement = kwargs.get('performance_improvement', 0.0)
        exploration_bonus = kwargs.get('exploration_bonus', 0.0)
        state_novelty = kwargs.get('state_novelty', 0.0)
        
        # 学习进步奖励
        learning_reward = self.calculate_learning_reward(
            performance_improvement, exploration_bonus
        )
        
        # 好奇心奖励
        curiosity_reward = self.curiosity_driven_reward(state_novelty)
        
        total_reward = (learning_reward + curiosity_reward) * self.weight
        
        return self._normalize_reward(total_reward)


class RewardCalculator:
    """奖励计算器主类"""
    
    def __init__(self, 
                 reward_weights: Optional[Dict[str, float]] = None,
                 enable_reward_shaping: bool = True,
                 enable_adaptive_weights: bool = True):
        self.reward_weights = reward_weights or {
            'task_completion': 0.4,
            'efficiency': 0.2,
            'user_experience': 0.2,
            'learning_progress': 0.1,
            'safety': 0.05,
            'exploration': 0.05
        }
        self.enable_reward_shaping = enable_reward_shaping
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # 初始化各奖励组件
        self.task_completion_reward = TaskCompletionReward()
        self.efficiency_reward = EfficiencyReward()
        self.user_experience_reward = UserExperienceReward()
        self.learning_progress_reward = LearningProgressReward()
        
        # 奖励历史（用于自适应权重）
        self.reward_history = []
        self.component_history = {
            'task_completion': [],
            'efficiency': [],
            'user_experience': [],
            'learning_progress': []
        }
        
        self.logger = logger
    
    def calculate_reward(self, 
                       state: torch.Tensor, 
                       action: torch.Tensor, 
                       next_state: torch.Tensor, 
                       context: Dict) -> float:
        """计算综合奖励"""
        components = RewardComponents()
        
        # 任务完成奖励
        components.task_completion = self.task_completion_reward.calculate(**context)
        
        # 效率奖励
        components.efficiency = self.efficiency_reward.calculate(**context)
        
        # 用户体验奖励
        components.user_experience = self.user_experience_reward.calculate(**context)
        
        # 学习进步奖励
        components.learning_progress = self.learning_progress_reward.calculate(**context)
        
        # 安全奖励
        components.safety = self._calculate_safety_reward(state, action, next_state, context)
        
        # 探索奖励
        components.exploration = self._calculate_exploration_reward(state, action, context)
        
        # 更新组件历史
        for component_name, value in components.to_dict().items():
            if component_name in self.component_history:
                self.component_history[component_name].append(value)
                # 限制历史长度
                if len(self.component_history[component_name]) > 1000:
                    self.component_history[component_name] = self.component_history[component_name][-1000:]
        
        # 自适应权重调整
        if self.enable_adaptive_weights:
            self._update_adaptive_weights()
        
        # 计算总奖励
        total_reward = components.total(self.reward_weights)
        
        # 奖励塑形
        if self.enable_reward_shaping:
            total_reward = self._apply_reward_shaping(total_reward, components, context)
        
        # 更新奖励历史
        self.reward_history.append(total_reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        return total_reward
    
    def _calculate_safety_reward(self, 
                               state: torch.Tensor, 
                               action: torch.Tensor, 
                               next_state: torch.Tensor, 
                               context: Dict) -> float:
        """计算安全奖励"""
        safety_reward = 0.0
        
        # 检查是否有安全违规
        safety_violations = context.get('safety_violations', [])
        if safety_violations:
            safety_reward -= len(safety_violations) * 0.5
        
        # 检查动作安全性
        action_safety = context.get('action_safety_score', 1.0)
        safety_reward += (action_safety - 0.5) * 0.5
        
        return np.clip(safety_reward, -1.0, 1.0)
    
    def _calculate_exploration_reward(self, 
                                    state: torch.Tensor, 
                                    action: torch.Tensor, 
                                    context: Dict) -> float:
        """计算探索奖励"""
        exploration_reward = 0.0
        
        # 状态新颖性
        state_novelty = context.get('state_novelty', 0.0)
        exploration_reward += state_novelty * 0.3
        
        # 动作多样性
        action_diversity = context.get('action_diversity', 0.0)
        exploration_reward += action_diversity * 0.2
        
        return np.clip(exploration_reward, 0.0, 1.0)
    
    def _apply_reward_shaping(self, 
                            base_reward: float, 
                            components: RewardComponents, 
                            context: Dict) -> float:
        """应用奖励塑形"""
        shaped_reward = base_reward
        
        # 时间衰减
        time_factor = context.get('time_factor', 1.0)
        shaped_reward *= time_factor
        
        # 难度调整
        difficulty = context.get('difficulty', 0.5)
        difficulty_multiplier = 0.5 + difficulty
        shaped_reward *= difficulty_multiplier
        
        # 连续成功奖励
        success_streak = context.get('success_streak', 0)
        if success_streak > 0:
            streak_bonus = min(0.5, success_streak * 0.1)
            shaped_reward += streak_bonus
        
        return shaped_reward
    
    def _update_adaptive_weights(self):
        """更新自适应权重"""
        if len(self.reward_history) < 50:  # 需要足够的历史数据
            return
        
        # 计算各组件的方差
        component_variances = {}
        for component_name, history in self.component_history.items():
            if len(history) >= 10:
                component_variances[component_name] = np.var(history[-50:])
        
        if not component_variances:
            return
        
        # 根据方差调整权重（方差大的组件权重增加）
        total_variance = sum(component_variances.values())
        if total_variance > 0:
            for component_name, variance in component_variances.items():
                if component_name in self.reward_weights:
                    # 平滑调整权重
                    target_weight = variance / total_variance
                    current_weight = self.reward_weights[component_name]
                    self.reward_weights[component_name] = 0.9 * current_weight + 0.1 * target_weight
        
        # 确保权重和为1
        total_weight = sum(self.reward_weights.values())
        if total_weight > 0:
            for key in self.reward_weights:
                self.reward_weights[key] /= total_weight
    
    def decompose_reward(self, reward: float) -> Dict[str, float]:
        """分解奖励为各个组成部分"""
        # 这是一个简化的分解，实际应用中需要更复杂的逻辑
        return {
            component: reward * weight 
            for component, weight in self.reward_weights.items()
        }
    
    def normalize_reward(self, reward: float, reward_history: List[float]) -> float:
        """奖励标准化"""
        if len(reward_history) < 10:
            return reward
        
        # 使用Z-score标准化
        mean_reward = np.mean(reward_history)
        std_reward = np.std(reward_history)
        
        if std_reward > 0:
            normalized_reward = (reward - mean_reward) / std_reward
            # 限制在合理范围内
            normalized_reward = np.clip(normalized_reward, -3.0, 3.0)
        else:
            normalized_reward = reward
        
        return normalized_reward
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """获取奖励统计信息"""
        stats = {
            'total_rewards': len(self.reward_history),
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'reward_std': np.std(self.reward_history) if self.reward_history else 0.0,
            'current_weights': self.reward_weights.copy(),
            'component_stats': {}
        }
        
        for component_name, history in self.component_history.items():
            if history:
                stats['component_stats'][component_name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history)
                }
        
        return stats


# 工具函数
def create_reward_calculator(config: Dict[str, Any]) -> RewardCalculator:
    """创建奖励计算器"""
    return RewardCalculator(
        reward_weights=config.get('reward_weights'),
        enable_reward_shaping=config.get('enable_reward_shaping', True),
        enable_adaptive_weights=config.get('enable_adaptive_weights', True)
    )


def evaluate_reward_function(reward_calculator: RewardCalculator, 
                           test_scenarios: List[Dict]) -> Dict[str, float]:
    """评估奖励函数"""
    results = {
        'total_scenarios': len(test_scenarios),
        'average_reward': 0.0,
        'reward_variance': 0.0,
        'component_analysis': {}
    }
    
    rewards = []
    
    for scenario in test_scenarios:
        state = scenario.get('state', torch.zeros(10))
        action = scenario.get('action', torch.zeros(5))
        next_state = scenario.get('next_state', torch.zeros(10))
        context = scenario.get('context', {})
        
        reward = reward_calculator.calculate_reward(state, action, next_state, context)
        rewards.append(reward)
    
    if rewards:
        results['average_reward'] = np.mean(rewards)
        results['reward_variance'] = np.var(rewards)
    
    return results