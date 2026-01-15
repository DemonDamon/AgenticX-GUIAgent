#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M11: 在线学习更新 - 实时策略优化

基于A3C、PPO、SAC等在线更新机制，提供实时策略优化和网络参数更新。
支持多智能体协调更新、自适应学习率、梯度裁剪等高级功能。

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.distributions import Categorical, Normal

from .experience import Experience
from .policies import BasePolicyNetwork


@dataclass
class UpdateConfig:
    """更新配置"""
    learning_rate: float = 3e-4
    batch_size: int = 32
    update_frequency: int = 4
    gradient_clip_norm: float = 0.5
    target_update_frequency: int = 100
    use_double_q: bool = True
    use_dueling: bool = True
    optimizer_type: str = "adam"  # adam, adamw, rmsprop
    scheduler_type: Optional[str] = "cosine"  # step, cosine, plateau
    
    # PPO特定参数
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # SAC特定参数
    sac_temperature: float = 0.2
    sac_target_entropy: Optional[float] = None
    sac_tau: float = 0.005
    
    # 自适应参数
    adaptive_lr: bool = True
    lr_decay_factor: float = 0.99
    min_lr: float = 1e-6


@dataclass
class UpdateMetrics:
    """更新指标"""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    total_loss: float = 0.0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    update_time: float = 0.0
    
    # PPO特定指标
    ppo_ratio_mean: float = 0.0
    ppo_ratio_std: float = 0.0
    ppo_clipped_fraction: float = 0.0
    
    # SAC特定指标
    sac_q1_loss: float = 0.0
    sac_q2_loss: float = 0.0
    sac_temperature_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'entropy_loss': self.entropy_loss,
            'total_loss': self.total_loss,
            'gradient_norm': self.gradient_norm,
            'learning_rate': self.learning_rate,
            'update_time': self.update_time,
            'ppo_ratio_mean': self.ppo_ratio_mean,
            'ppo_ratio_std': self.ppo_ratio_std,
            'ppo_clipped_fraction': self.ppo_clipped_fraction,
            'sac_q1_loss': self.sac_q1_loss,
            'sac_q2_loss': self.sac_q2_loss,
            'sac_temperature_loss': self.sac_temperature_loss
        }


class OnlinePolicyUpdater(ABC):
    """在线策略更新器基类"""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.update_count = 0
        self.total_update_time = 0.0
        
        # 更新历史
        self.update_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
        # 自适应学习率
        self.current_lr = config.learning_rate
        
        self.logger = logger
    
    @abstractmethod
    def update_policy(self, 
                     policy: BasePolicyNetwork, 
                     experiences: List[Experience], 
                     update_config: Dict) -> Dict[str, float]:
        """更新策略"""
        pass
    
    def multi_agent_update(self, 
                          policies: Dict[str, BasePolicyNetwork], 
                          shared_experiences: List[Experience]) -> Dict[str, Dict]:
        """多智能体更新"""
        update_results = {}
        
        # 按智能体分组经验
        agent_experiences = defaultdict(list)
        for exp in shared_experiences:
            agent_experiences[exp.agent_id].append(exp)
        
        # 并行更新各智能体策略
        for agent_id, policy in policies.items():
            if agent_id in agent_experiences:
                experiences = agent_experiences[agent_id]
                if len(experiences) >= self.config.batch_size:
                    try:
                        result = self.update_policy(policy, experiences, {})
                        update_results[agent_id] = result
                    except Exception as e:
                        logger.error(f"更新智能体 {agent_id} 策略失败: {e}")
                        update_results[agent_id] = {'error': str(e)}
        
        return update_results
    
    def adaptive_learning_rate(self, 
                              performance_metrics: Dict[str, float], 
                              base_lr: float) -> float:
        """自适应学习率"""
        if not self.config.adaptive_lr:
            return base_lr
        
        # 基于性能指标调整学习率
        performance_score = performance_metrics.get('average_reward', 0.0)
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) < 10:
            return base_lr
        
        # 计算性能趋势
        recent_performance = list(self.performance_history)[-10:]
        early_performance = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else recent_performance
        
        recent_avg = np.mean(recent_performance)
        early_avg = np.mean(early_performance)
        
        # 如果性能在提升，保持或略微增加学习率
        if recent_avg > early_avg:
            lr_multiplier = 1.0
        else:
            # 如果性能下降，降低学习率
            lr_multiplier = self.config.lr_decay_factor
        
        self.current_lr = max(
            self.config.min_lr,
            self.current_lr * lr_multiplier
        )
        
        return self.current_lr
    
    def _create_optimizer(self, policy: BasePolicyNetwork) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer_type == "adam":
            return Adam(policy.parameters(), lr=self.current_lr)
        elif self.config.optimizer_type == "adamw":
            return AdamW(policy.parameters(), lr=self.current_lr, weight_decay=1e-4)
        elif self.config.optimizer_type == "rmsprop":
            return RMSprop(policy.parameters(), lr=self.current_lr)
        else:
            raise ValueError(f"不支持的优化器类型: {self.config.optimizer_type}")
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.config.scheduler_type == "step":
            return StepLR(optimizer, step_size=1000, gamma=0.9)
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=10000)
        elif self.config.scheduler_type == "plateau":
            return ReduceLROnPlateau(optimizer, mode='max', patience=100, factor=0.5)
        else:
            return None
    
    def _clip_gradients(self, policy: BasePolicyNetwork) -> float:
        """梯度裁剪"""
        if self.config.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), 
                self.config.gradient_clip_norm
            )
            return grad_norm.item()
        return 0.0
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        if not self.update_history:
            return {}
        
        recent_metrics = list(self.update_history)[-100:]  # 最近100次更新
        
        stats = {
            'total_updates': self.update_count,
            'average_update_time': self.total_update_time / max(1, self.update_count),
            'current_learning_rate': self.current_lr,
            'recent_metrics': {}
        }
        
        # 计算各指标的统计信息
        if recent_metrics:
            for key in recent_metrics[0].to_dict().keys():
                values = [m.to_dict()[key] for m in recent_metrics if key in m.to_dict()]
                if values:
                    stats['recent_metrics'][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        return stats


class PPOUpdater(OnlinePolicyUpdater):
    """PPO更新策略"""
    
    def __init__(self, config: UpdateConfig):
        super().__init__(config)
        self.old_log_probs = None
        self.advantages = None
        self.returns = None
    
    def update_policy(self, 
                     policy: BasePolicyNetwork, 
                     experiences: List[Experience], 
                     update_config: Dict) -> Dict[str, float]:
        """更新策略"""
        start_time = time.time()
        
        # 准备数据
        states, actions, rewards, next_states, dones = self._prepare_batch(experiences)
        
        # 计算优势和回报
        advantages, returns = self._compute_advantages_and_returns(
            policy, states, rewards, next_states, dones
        )
        
        # 获取旧的动作概率
        with torch.no_grad():
            old_log_probs = self._get_log_probs(policy, states, actions)
        
        # 创建优化器
        optimizer = self._create_optimizer(policy)
        scheduler = self._create_scheduler(optimizer)
        
        total_metrics = UpdateMetrics()
        
        # PPO多轮更新
        for epoch in range(self.config.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 计算PPO损失
                metrics = self._compute_ppo_loss(
                    policy, batch_states, batch_actions, 
                    batch_advantages, batch_returns, batch_old_log_probs
                )
                
                # 反向传播
                optimizer.zero_grad()
                metrics.total_loss.backward()
                
                # 梯度裁剪
                grad_norm = self._clip_gradients(policy)
                metrics.gradient_norm = grad_norm
                
                optimizer.step()
                
                # 累积指标
                total_metrics.policy_loss += metrics.policy_loss
                total_metrics.value_loss += metrics.value_loss
                total_metrics.entropy_loss += metrics.entropy_loss
                total_metrics.total_loss += metrics.total_loss.item()
                total_metrics.gradient_norm += grad_norm
                total_metrics.ppo_ratio_mean += metrics.ppo_ratio_mean
                total_metrics.ppo_ratio_std += metrics.ppo_ratio_std
                total_metrics.ppo_clipped_fraction += metrics.ppo_clipped_fraction
        
        # 平均化指标
        num_updates = self.config.ppo_epochs * math.ceil(len(states) / self.config.batch_size)
        for attr in ['policy_loss', 'value_loss', 'entropy_loss', 'total_loss', 
                    'gradient_norm', 'ppo_ratio_mean', 'ppo_ratio_std', 'ppo_clipped_fraction']:
            setattr(total_metrics, attr, getattr(total_metrics, attr) / num_updates)
        
        # 更新学习率
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(total_metrics.total_loss)
            else:
                scheduler.step()
        
        total_metrics.learning_rate = optimizer.param_groups[0]['lr']
        total_metrics.update_time = time.time() - start_time
        
        # 更新统计
        self.update_count += 1
        self.total_update_time += total_metrics.update_time
        self.update_history.append(total_metrics)
        
        return total_metrics.to_dict()
    
    def _prepare_batch(self, experiences: List[Experience]) -> Tuple[torch.Tensor, ...]:
        """准备批次数据"""
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def _compute_advantages_and_returns(self, 
                                       policy: BasePolicyNetwork,
                                       states: torch.Tensor,
                                       rewards: torch.Tensor,
                                       next_states: torch.Tensor,
                                       dones: torch.Tensor,
                                       gamma: float = 0.99,
                                       gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算优势和回报（使用GAE）"""
        with torch.no_grad():
            values = policy.compute_value(states).squeeze()
            next_values = policy.compute_value(next_states).squeeze()
            
            # 计算TD误差
            td_targets = rewards + gamma * next_values * (~dones)
            td_errors = td_targets - values
            
            # GAE计算优势
            advantages = torch.zeros_like(rewards)
            advantage = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_advantage = 0
                else:
                    next_advantage = advantages[t + 1]
                
                advantages[t] = td_errors[t] + gamma * gae_lambda * next_advantage * (~dones[t])
            
            # 计算回报
            returns = advantages + values
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _get_log_probs(self, 
                      policy: BasePolicyNetwork, 
                      states: torch.Tensor, 
                      actions: torch.Tensor) -> torch.Tensor:
        """获取动作的对数概率"""
        action_probs = policy.get_action_probabilities(states)
        
        # 假设actions是动作索引
        if actions.dim() == 1:
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        else:
            log_probs = torch.log(action_probs.gather(1, actions.long()) + 1e-8)
        
        return log_probs.squeeze()
    
    def _compute_ppo_loss(self, 
                         policy: BasePolicyNetwork,
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         advantages: torch.Tensor,
                         returns: torch.Tensor,
                         old_log_probs: torch.Tensor) -> UpdateMetrics:
        """计算PPO损失"""
        # 获取当前策略的动作概率
        current_log_probs = self._get_log_probs(policy, states, actions)
        
        # 计算概率比率
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO裁剪损失
        clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio)
        policy_loss1 = ratio * advantages
        policy_loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        # 价值函数损失
        values = policy.compute_value(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # 熵损失（鼓励探索）
        action_probs = policy.get_action_probabilities(states)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
        
        # 计算指标
        metrics = UpdateMetrics()
        metrics.policy_loss = policy_loss.item()
        metrics.value_loss = value_loss.item()
        metrics.entropy_loss = entropy_loss.item()
        metrics.total_loss = total_loss
        metrics.ppo_ratio_mean = ratio.mean().item()
        metrics.ppo_ratio_std = ratio.std().item()
        metrics.ppo_clipped_fraction = ((ratio < 1 - self.config.ppo_clip_ratio) | 
                                       (ratio > 1 + self.config.ppo_clip_ratio)).float().mean().item()
        
        return metrics
    
    def compute_ppo_loss(self, 
                        policy: BasePolicyNetwork, 
                        experiences: List[Experience], 
                        clip_ratio: float = 0.2) -> torch.Tensor:
        """计算PPO损失（简化接口）"""
        # 临时更新配置
        old_clip_ratio = self.config.ppo_clip_ratio
        self.config.ppo_clip_ratio = clip_ratio
        
        try:
            result = self.update_policy(policy, experiences, {})
            return torch.tensor(result['total_loss'])
        finally:
            self.config.ppo_clip_ratio = old_clip_ratio
    
    def update_value_function(self, 
                             value_network: nn.Module, 
                             experiences: List[Experience]) -> float:
        """更新价值函数"""
        states, _, rewards, next_states, dones = self._prepare_batch(experiences)
        
        # 计算目标值
        with torch.no_grad():
            next_values = value_network(next_states).squeeze()
            targets = rewards + 0.99 * next_values * (~dones)
        
        # 计算当前值
        current_values = value_network(states).squeeze()
        
        # 计算损失
        value_loss = F.mse_loss(current_values, targets)
        
        return value_loss.item()


class SACUpdater(OnlinePolicyUpdater):
    """SAC更新策略"""
    
    def __init__(self, config: UpdateConfig):
        super().__init__(config)
        self.target_entropy = config.sac_target_entropy
        self.log_temperature = torch.log(torch.tensor(config.sac_temperature))
        self.log_temperature.requires_grad = True
    
    def update_policy(self, 
                     policy: BasePolicyNetwork, 
                     experiences: List[Experience], 
                     update_config: Dict) -> Dict[str, float]:
        """更新策略"""
        start_time = time.time()
        
        # 准备数据
        states, actions, rewards, next_states, dones = self._prepare_batch(experiences)
        
        # 创建优化器
        policy_optimizer = self._create_optimizer(policy)
        
        # 计算SAC损失
        metrics = self._compute_sac_loss(policy, states, actions, rewards, next_states, dones)
        
        # 更新策略
        policy_optimizer.zero_grad()
        metrics.total_loss.backward()
        
        # 梯度裁剪
        grad_norm = self._clip_gradients(policy)
        metrics.gradient_norm = grad_norm
        
        policy_optimizer.step()
        
        metrics.learning_rate = policy_optimizer.param_groups[0]['lr']
        metrics.update_time = time.time() - start_time
        
        # 更新统计
        self.update_count += 1
        self.total_update_time += metrics.update_time
        self.update_history.append(metrics)
        
        return metrics.to_dict()
    
    def _prepare_batch(self, experiences: List[Experience]) -> Tuple[torch.Tensor, ...]:
        """准备批次数据"""
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def _compute_sac_loss(self, 
                         policy: BasePolicyNetwork,
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_states: torch.Tensor,
                         dones: torch.Tensor) -> UpdateMetrics:
        """计算SAC损失"""
        # 获取当前策略的动作概率
        action_probs = policy.get_action_probabilities(states)
        
        # 计算策略损失（最大化熵正则化的期望回报）
        log_probs = torch.log(action_probs + 1e-8)
        entropy = -(action_probs * log_probs).sum(dim=1)
        
        # 简化的SAC损失（实际实现需要Q网络）
        policy_loss = -(entropy * torch.exp(self.log_temperature)).mean()
        
        # 价值函数损失
        values = policy.compute_value(states).squeeze()
        with torch.no_grad():
            next_values = policy.compute_value(next_states).squeeze()
            targets = rewards + 0.99 * next_values * (~dones)
        
        value_loss = F.mse_loss(values, targets)
        
        # 温度参数损失
        if self.target_entropy is None:
            self.target_entropy = -float(action_probs.shape[1])  # 启发式目标熵
        
        temperature_loss = -(self.log_temperature * (entropy.detach() + self.target_entropy)).mean()
        
        # 总损失
        total_loss = policy_loss + value_loss + temperature_loss
        
        # 构建指标
        metrics = UpdateMetrics()
        metrics.policy_loss = policy_loss.item()
        metrics.value_loss = value_loss.item()
        metrics.entropy_loss = -entropy.mean().item()
        metrics.total_loss = total_loss
        metrics.sac_temperature_loss = temperature_loss.item()
        
        return metrics
    
    def compute_sac_loss(self, 
                        policy: BasePolicyNetwork, 
                        experiences: List[Experience], 
                        temperature: float) -> torch.Tensor:
        """计算SAC损失（简化接口）"""
        # 临时更新温度
        old_temperature = self.log_temperature
        self.log_temperature = torch.log(torch.tensor(temperature))
        self.log_temperature.requires_grad = True
        
        try:
            result = self.update_policy(policy, experiences, {})
            return torch.tensor(result['total_loss'])
        finally:
            self.log_temperature = old_temperature
    
    def update_q_networks(self, 
                         q_networks: List[nn.Module], 
                         experiences: List[Experience]) -> List[float]:
        """更新Q网络"""
        states, actions, rewards, next_states, dones = self._prepare_batch(experiences)
        
        losses = []
        
        for q_network in q_networks:
            # 计算当前Q值
            current_q = q_network(torch.cat([states, actions], dim=1))
            
            # 计算目标Q值
            with torch.no_grad():
                next_q = q_network(torch.cat([next_states, actions], dim=1))
                target_q = rewards.unsqueeze(1) + 0.99 * next_q * (~dones).unsqueeze(1)
            
            # 计算损失
            q_loss = F.mse_loss(current_q, target_q)
            losses.append(q_loss.item())
        
        return losses


class A3CUpdater(OnlinePolicyUpdater):
    """A3C更新策略"""
    
    def __init__(self, config: UpdateConfig):
        super().__init__(config)
        self.global_step = 0
    
    def update_policy(self, 
                     policy: BasePolicyNetwork, 
                     experiences: List[Experience], 
                     update_config: Dict) -> Dict[str, float]:
        """更新策略"""
        start_time = time.time()
        
        # 准备数据
        states, actions, rewards, next_states, dones = self._prepare_batch(experiences)
        
        # 计算优势
        advantages = self._compute_advantages(policy, states, rewards, next_states, dones)
        
        # 创建优化器
        optimizer = self._create_optimizer(policy)
        
        # 计算A3C损失
        metrics = self._compute_a3c_loss(policy, states, actions, advantages)
        
        # 更新策略
        optimizer.zero_grad()
        metrics.total_loss.backward()
        
        # 梯度裁剪
        grad_norm = self._clip_gradients(policy)
        metrics.gradient_norm = grad_norm
        
        optimizer.step()
        
        metrics.learning_rate = optimizer.param_groups[0]['lr']
        metrics.update_time = time.time() - start_time
        
        # 更新统计
        self.update_count += 1
        self.total_update_time += metrics.update_time
        self.update_history.append(metrics)
        self.global_step += len(experiences)
        
        return metrics.to_dict()
    
    def _prepare_batch(self, experiences: List[Experience]) -> Tuple[torch.Tensor, ...]:
        """准备批次数据"""
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def _compute_advantages(self, 
                           policy: BasePolicyNetwork,
                           states: torch.Tensor,
                           rewards: torch.Tensor,
                           next_states: torch.Tensor,
                           dones: torch.Tensor,
                           gamma: float = 0.99) -> torch.Tensor:
        """计算优势"""
        with torch.no_grad():
            values = policy.compute_value(states).squeeze()
            next_values = policy.compute_value(next_states).squeeze()
            
            # 计算折扣回报
            returns = torch.zeros_like(rewards)
            running_return = 0
            
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_return = 0
                running_return = rewards[t] + gamma * running_return
                returns[t] = running_return
            
            # 计算优势
            advantages = returns - values
        
        return advantages
    
    def _compute_a3c_loss(self, 
                         policy: BasePolicyNetwork,
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         advantages: torch.Tensor) -> UpdateMetrics:
        """计算A3C损失"""
        # 获取动作概率
        action_probs = policy.get_action_probabilities(states)
        
        # 计算策略损失
        log_probs = torch.log(action_probs.gather(1, actions.long().unsqueeze(1)) + 1e-8)
        policy_loss = -(log_probs.squeeze() * advantages.detach()).mean()
        
        # 计算价值损失
        values = policy.compute_value(states).squeeze()
        value_targets = advantages + values.detach()
        value_loss = F.mse_loss(values, value_targets)
        
        # 计算熵损失
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
        
        # 构建指标
        metrics = UpdateMetrics()
        metrics.policy_loss = policy_loss.item()
        metrics.value_loss = value_loss.item()
        metrics.entropy_loss = entropy_loss.item()
        metrics.total_loss = total_loss
        
        return metrics


# 工具函数
def create_updater(algorithm: str, config: UpdateConfig) -> OnlinePolicyUpdater:
    """创建更新器"""
    if algorithm.lower() == "ppo":
        return PPOUpdater(config)
    elif algorithm.lower() == "sac":
        return SACUpdater(config)
    elif algorithm.lower() == "a3c":
        return A3CUpdater(config)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")


def create_update_config(**kwargs) -> UpdateConfig:
    """创建更新配置"""
    return UpdateConfig(**kwargs)


async def async_policy_update(updater: OnlinePolicyUpdater,
                             policy: BasePolicyNetwork,
                             experiences: List[Experience]) -> Dict[str, float]:
    """异步策略更新"""
    loop = asyncio.get_event_loop()
    
    # 在线程池中执行更新
    result = await loop.run_in_executor(
        None, 
        updater.update_policy, 
        policy, 
        experiences, 
        {}
    )
    
    return result


def evaluate_update_performance(updater: OnlinePolicyUpdater,
                              policy: BasePolicyNetwork,
                              test_experiences: List[Experience],
                              num_trials: int = 10) -> Dict[str, float]:
    """评估更新性能"""
    update_times = []
    loss_values = []
    
    for _ in range(num_trials):
        # 复制策略以避免影响原始策略
        test_policy = type(policy)(
            state_dim=policy.state_dim,
            action_dim=policy.action_dim
        )
        test_policy.load_state_dict(policy.state_dict())
        
        # 执行更新
        start_time = time.time()
        result = updater.update_policy(test_policy, test_experiences, {})
        update_time = time.time() - start_time
        
        update_times.append(update_time)
        loss_values.append(result.get('total_loss', 0.0))
    
    return {
        'average_update_time': np.mean(update_times),
        'update_time_std': np.std(update_times),
        'average_loss': np.mean(loss_values),
        'loss_std': np.std(loss_values),
        'min_update_time': np.min(update_times),
        'max_update_time': np.max(update_times)
    }