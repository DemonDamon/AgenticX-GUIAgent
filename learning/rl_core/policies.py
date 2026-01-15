#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M8: 策略网络架构 - Agent专用策略网络

基于Actor-Critic、PPO、SAC等策略网络设计，为四个Agent提供专用的策略网络和价值函数。
每个Agent都有针对其特定任务优化的策略网络架构。

Author: AgenticX Team
Date: 2025
"""

from loguru import logger
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

from .environment import GUIAction


@dataclass
class SubTask:
    """子任务定义"""
    task_id: str
    description: str
    priority: float
    estimated_steps: int
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningPattern:
    """学习模式"""
    pattern_id: str
    pattern_type: str  # success, failure, efficiency, etc.
    context: Dict[str, Any]
    frequency: int
    confidence: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningInsight:
    """学习洞察"""
    insight_id: str
    insight_type: str
    description: str
    importance: float
    actionable: bool
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePolicyNetwork(nn.Module, ABC):
    """策略网络基类"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 activation: str = "relu",
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # 激活函数
        self.activation = self._get_activation(activation)
        
        # 构建网络层
        self.feature_extractor = self._build_feature_extractor(
            state_dim, hidden_dims, dropout, use_layer_norm
        )
        
        # 策略头（输出动作概率）
        self.policy_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 价值头（输出状态价值）
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        
        # 优势头（用于A3C等算法）
        self.advantage_head = nn.Linear(hidden_dims[-1], action_dim)
        
        self.logger = logger
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())
    
    def _build_feature_extractor(self, 
                                input_dim: int, 
                                hidden_dims: List[int],
                                dropout: float,
                                use_layer_norm: bool) -> nn.Module:
        """构建特征提取器"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """选择动作"""
        features = self.feature_extractor(state)
        action_logits = self.policy_head(features)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            log_prob = F.log_softmax(action_logits, dim=-1).gather(1, action.unsqueeze(-1))
        else:
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """获取动作概率"""
        features = self.feature_extractor(state)
        action_logits = self.policy_head(features)
        return F.softmax(action_logits, dim=-1)
    
    def compute_value(self, state: torch.Tensor) -> torch.Tensor:
        """计算状态价值"""
        features = self.feature_extractor(state)
        return self.value_head(features)
    
    def compute_advantage(self, state: torch.Tensor) -> torch.Tensor:
        """计算优势函数"""
        features = self.feature_extractor(state)
        advantages = self.advantage_head(features)
        
        # 减去平均值以减少方差
        advantages = advantages - advantages.mean(dim=-1, keepdim=True)
        
        return advantages
    
    def update_parameters(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """更新网络参数"""
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        optimizer.step()
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        pass


class ManagerPolicyNetwork(BasePolicyNetwork):
    """Manager Agent策略网络 - 专注于任务分解和协调"""
    
    def __init__(self,
                 state_dim: int,
                 max_subtasks: int = 10,
                 task_embedding_dim: int = 256,
                 **kwargs):
        # Manager的动作空间：任务分解决策
        action_dim = max_subtasks * 3  # 每个子任务：[是否创建, 优先级, 预估步数]
        
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.max_subtasks = max_subtasks
        self.task_embedding_dim = task_embedding_dim
        
        # 任务分解专用网络
        self.task_decomposer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], task_embedding_dim),
            nn.ReLU(),
            nn.Linear(task_embedding_dim, max_subtasks * task_embedding_dim)
        )
        
        # 协调策略网络
        self.coordination_network = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4种协调策略
        )
        
        # 复杂度估计网络
        self.complexity_estimator = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1之间的复杂度分数
        )
    
    def decompose_task(self, state: torch.Tensor, task: str) -> List[SubTask]:
        """任务分解策略"""
        features = self.feature_extractor(state)
        
        # 任务分解
        task_features = self.task_decomposer(features)
        task_features = task_features.view(-1, self.max_subtasks, self.task_embedding_dim)
        
        subtasks = []
        
        for i in range(self.max_subtasks):
            task_feature = task_features[0, i]  # 取第一个batch
            
            # 判断是否创建子任务
            create_prob = torch.sigmoid(task_feature[0])
            if create_prob > 0.5:
                # 计算优先级和预估步数
                priority = torch.sigmoid(task_feature[1]).item()
                estimated_steps = int(torch.relu(task_feature[2]).item() * 20) + 1
                
                subtask = SubTask(
                    task_id=f"subtask_{i}",
                    description=f"子任务{i}: {task}",
                    priority=priority,
                    estimated_steps=estimated_steps
                )
                subtasks.append(subtask)
        
        return subtasks
    
    def select_coordination_strategy(self, state: torch.Tensor, agents: List[str]) -> str:
        """选择协调策略"""
        features = self.feature_extractor(state)
        coordination_logits = self.coordination_network(features)
        
        strategy_idx = torch.argmax(coordination_logits, dim=-1).item()
        
        strategies = ["sequential", "parallel", "hierarchical", "adaptive"]
        return strategies[strategy_idx]
    
    def estimate_task_complexity(self, state: torch.Tensor, task: str) -> float:
        """估计任务复杂度"""
        features = self.feature_extractor(state)
        complexity = self.complexity_estimator(features)
        
        return complexity.item()
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        
        # 策略输出
        action_logits = self.policy_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 价值输出
        value = self.value_head(features)
        
        # 协调策略
        coordination_logits = self.coordination_network(features)
        
        # 复杂度估计
        complexity = self.complexity_estimator(features)
        
        return {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "value": value,
            "coordination_logits": coordination_logits,
            "complexity": complexity
        }


class ExecutorPolicyNetwork(BasePolicyNetwork):
    """Executor Agent策略网络 - 专注于GUI操作执行"""
    
    def __init__(self,
                 state_dim: int,
                 screen_width: int = 1080,
                 screen_height: int = 1920,
                 **kwargs):
        # Executor的动作空间：GUI操作类型 + 坐标 + 参数
        action_dim = 11 + 2 + 3  # 11种操作类型 + x,y坐标 + 3个参数
        
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # GUI操作类型网络
        self.action_type_head = nn.Linear(self.hidden_dims[-1], 11)
        
        # 坐标预测网络
        self.coordinate_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()  # 输出0-1之间的归一化坐标
        )
        
        # 操作参数网络（持续时间、力度等）
        self.parameter_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        
        # 成功概率预测网络
        self.success_predictor = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def select_gui_action(self, state: torch.Tensor) -> GUIAction:
        """选择GUI操作"""
        features = self.feature_extractor(state)
        
        # 选择操作类型
        action_type_logits = self.action_type_head(features)
        action_type_idx = torch.argmax(action_type_logits, dim=-1).item()
        
        action_types = [
            "click", "double_click", "long_press", "swipe", "scroll",
            "pinch", "zoom", "input", "key_press", "back", "home"
        ]
        action_type = action_types[action_type_idx]
        
        # 预测坐标
        normalized_coords = self.coordinate_head(features)
        x = int(normalized_coords[0, 0].item() * self.screen_width)
        y = int(normalized_coords[0, 1].item() * self.screen_height)
        
        # 预测参数
        parameters = self.parameter_head(features)
        duration = parameters[0, 0].item() * 5.0  # 0-5秒
        force = parameters[0, 1].item()  # 0-1
        text_length = int(parameters[0, 2].item() * 50)  # 0-50字符
        
        # 构建GUI动作
        gui_action = GUIAction(
            action_type=action_type,
            coordinates=(x, y) if action_type in ["click", "double_click", "long_press", "swipe"] else None,
            duration=duration if action_type in ["long_press", "swipe"] else None,
            text="sample_text"[:text_length] if action_type == "input" else None
        )
        
        return gui_action
    
    def predict_action_coordinates(self, state: torch.Tensor, action_type: str) -> Tuple[int, int]:
        """预测操作坐标"""
        features = self.feature_extractor(state)
        normalized_coords = self.coordinate_head(features)
        
        x = int(normalized_coords[0, 0].item() * self.screen_width)
        y = int(normalized_coords[0, 1].item() * self.screen_height)
        
        return x, y
    
    def estimate_action_success_probability(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """估计动作成功概率"""
        features = self.feature_extractor(state)
        
        # 将状态特征和动作特征拼接
        combined_features = torch.cat([features, action], dim=-1)
        
        success_prob = self.success_predictor(combined_features)
        
        return success_prob.item()
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        
        # 动作类型
        action_type_logits = self.action_type_head(features)
        action_type_probs = F.softmax(action_type_logits, dim=-1)
        
        # 坐标
        coordinates = self.coordinate_head(features)
        
        # 参数
        parameters = self.parameter_head(features)
        
        # 价值
        value = self.value_head(features)
        
        return {
            "action_type_logits": action_type_logits,
            "action_type_probs": action_type_probs,
            "coordinates": coordinates,
            "parameters": parameters,
            "value": value
        }


class ReflectorPolicyNetwork(BasePolicyNetwork):
    """ActionReflector Agent策略网络 - 专注于质量评估和改进建议"""
    
    def __init__(self,
                 state_dim: int,
                 num_quality_metrics: int = 5,
                 num_failure_modes: int = 10,
                 **kwargs):
        # Reflector的动作空间：质量评分 + 改进建议
        action_dim = num_quality_metrics + num_failure_modes
        
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.num_quality_metrics = num_quality_metrics
        self.num_failure_modes = num_failure_modes
        
        # 质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, num_quality_metrics),
            nn.Sigmoid()  # 输出0-1之间的质量分数
        )
        
        # 改进建议生成网络
        self.improvement_generator = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 失败模式预测网络
        self.failure_predictor = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, num_failure_modes),
            nn.Sigmoid()
        )
        
        # 状态差异编码器
        self.state_diff_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, 256),  # before + after states
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def assess_action_quality(self, 
                             before_state: torch.Tensor, 
                             after_state: torch.Tensor, 
                             action: torch.Tensor) -> float:
        """评估动作质量"""
        # 编码状态差异
        state_diff = torch.cat([before_state, after_state], dim=-1)
        diff_features = self.state_diff_encoder(state_diff)
        
        # 结合动作信息
        combined_features = torch.cat([diff_features, action], dim=-1)
        
        # 通过特征提取器
        features = self.feature_extractor(combined_features)
        
        # 质量评估
        quality_scores = self.quality_assessor(features)
        
        # 计算综合质量分数
        overall_quality = torch.mean(quality_scores).item()
        
        return overall_quality
    
    def generate_improvement_suggestions(self, 
                                       state: torch.Tensor, 
                                       quality_score: float) -> List[str]:
        """生成改进建议"""
        features = self.feature_extractor(state)
        improvement_features = self.improvement_generator(features)
        
        # 基于质量分数和特征生成建议
        suggestions = []
        
        if quality_score < 0.3:
            suggestions.extend([
                "考虑重新选择操作类型",
                "检查目标元素是否可见",
                "调整操作时机"
            ])
        elif quality_score < 0.6:
            suggestions.extend([
                "优化操作精度",
                "考虑添加等待时间"
            ])
        else:
            suggestions.append("操作质量良好，继续保持")
        
        return suggestions
    
    def predict_failure_modes(self, state: torch.Tensor, action: torch.Tensor) -> List[str]:
        """预测失败模式"""
        features = self.feature_extractor(state)
        failure_probs = self.failure_predictor(features)
        
        failure_modes = [
            "元素不可点击", "网络延迟", "界面变化", "权限不足", "操作超时",
            "坐标偏移", "元素被遮挡", "应用崩溃", "系统繁忙", "用户干预"
        ]
        
        predicted_failures = []
        for i, prob in enumerate(failure_probs[0]):
            if prob > 0.5:
                predicted_failures.append(failure_modes[i])
        
        return predicted_failures
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        
        # 质量评估
        quality_scores = self.quality_assessor(features)
        
        # 改进建议特征
        improvement_features = self.improvement_generator(features)
        
        # 失败模式预测
        failure_probs = self.failure_predictor(features)
        
        # 价值
        value = self.value_head(features)
        
        return {
            "quality_scores": quality_scores,
            "improvement_features": improvement_features,
            "failure_probs": failure_probs,
            "value": value
        }


class NotetakerPolicyNetwork(BasePolicyNetwork):
    """Notetaker Agent策略网络 - 专注于学习模式提取和知识贡献"""
    
    def __init__(self,
                 state_dim: int,
                 num_pattern_types: int = 8,
                 num_insight_types: int = 6,
                 pattern_embedding_dim: int = 128,
                 **kwargs):
        # Notetaker的动作空间：模式识别 + 洞察生成 + 知识贡献策略
        action_dim = num_pattern_types + num_insight_types + 4
        
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.num_pattern_types = num_pattern_types
        self.num_insight_types = num_insight_types
        self.pattern_embedding_dim = pattern_embedding_dim
        
        # 模式提取网络
        self.pattern_extractor = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Linear(256, pattern_embedding_dim),
            nn.ReLU(),
            nn.Linear(pattern_embedding_dim, num_pattern_types),
            nn.Sigmoid()
        )
        
        # 洞察生成网络
        self.insight_generator = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_insight_types),
            nn.Sigmoid()
        )
        
        # 知识贡献策略网络
        self.contribution_strategy = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4种贡献策略
            nn.Softmax(dim=-1)
        )
        
        # 优先级排序网络
        self.priority_ranker = nn.Sequential(
            nn.Linear(pattern_embedding_dim + num_insight_types, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def extract_learning_patterns(self, experiences: List[Dict]) -> List[LearningPattern]:
        """提取学习模式"""
        if not experiences:
            return []
        
        # 将经验转换为张量
        experience_tensors = []
        for exp in experiences:
            # 简化的经验编码
            exp_tensor = torch.tensor([
                exp.get("reward", 0.0),
                exp.get("success", 0.0),
                exp.get("execution_time", 0.0),
                exp.get("step_count", 0.0)
            ], dtype=torch.float32)
            experience_tensors.append(exp_tensor)
        
        if not experience_tensors:
            return []
        
        # 聚合经验特征
        aggregated_features = torch.mean(torch.stack(experience_tensors), dim=0)
        
        # 扩展到网络输入维度
        if aggregated_features.shape[0] < self.state_dim:
            padding = torch.zeros(self.state_dim - aggregated_features.shape[0])
            aggregated_features = torch.cat([aggregated_features, padding])
        else:
            aggregated_features = aggregated_features[:self.state_dim]
        
        # 提取模式
        features = self.feature_extractor(aggregated_features.unsqueeze(0))
        pattern_probs = self.pattern_extractor(features)
        
        pattern_types = [
            "success_pattern", "failure_pattern", "efficiency_pattern", "timing_pattern",
            "sequence_pattern", "context_pattern", "user_pattern", "system_pattern"
        ]
        
        patterns = []
        for i, prob in enumerate(pattern_probs[0]):
            if prob > 0.5:
                pattern = LearningPattern(
                    pattern_id=f"pattern_{i}_{len(experiences)}",
                    pattern_type=pattern_types[i],
                    context={"experience_count": len(experiences)},
                    frequency=int(prob.item() * 10),
                    confidence=prob.item()
                )
                patterns.append(pattern)
        
        return patterns
    
    def prioritize_learning_insights(self, insights: List[LearningInsight]) -> List[LearningInsight]:
        """优先级排序学习洞察"""
        if not insights:
            return []
        
        prioritized_insights = []
        
        for insight in insights:
            # 简化的洞察编码
            insight_features = torch.tensor([
                insight.importance,
                float(insight.actionable),
                len(insight.description) / 100.0,  # 描述长度归一化
                hash(insight.insight_type) % 1000 / 1000.0  # 类型哈希归一化
            ], dtype=torch.float32)
            
            # 扩展到所需维度
            if insight_features.shape[0] < self.pattern_embedding_dim + self.num_insight_types:
                padding_size = self.pattern_embedding_dim + self.num_insight_types - insight_features.shape[0]
                padding = torch.zeros(padding_size)
                insight_features = torch.cat([insight_features, padding])
            
            # 计算优先级
            priority_score = self.priority_ranker(insight_features.unsqueeze(0))
            
            # 更新洞察的重要性
            insight.importance = priority_score.item()
            prioritized_insights.append(insight)
        
        # 按重要性排序
        prioritized_insights.sort(key=lambda x: x.importance, reverse=True)
        
        return prioritized_insights
    
    def select_knowledge_contribution_strategy(self, context: Dict) -> str:
        """选择知识贡献策略"""
        # 将上下文转换为特征
        context_features = torch.zeros(self.state_dim)
        
        # 简化的上下文编码
        if "success_rate" in context:
            context_features[0] = context["success_rate"]
        if "learning_progress" in context:
            context_features[1] = context["learning_progress"]
        
        features = self.feature_extractor(context_features.unsqueeze(0))
        strategy_probs = self.contribution_strategy(features)
        
        strategy_idx = torch.argmax(strategy_probs, dim=-1).item()
        
        strategies = ["immediate", "batch", "selective", "adaptive"]
        return strategies[strategy_idx]
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        
        # 模式提取
        pattern_probs = self.pattern_extractor(features)
        
        # 洞察生成
        insight_probs = self.insight_generator(features)
        
        # 贡献策略
        contribution_probs = self.contribution_strategy(features)
        
        # 价值
        value = self.value_head(features)
        
        return {
            "pattern_probs": pattern_probs,
            "insight_probs": insight_probs,
            "contribution_probs": contribution_probs,
            "value": value
        }


# 工具函数
def create_policy_network(agent_type: str, 
                         state_dim: int, 
                         config: Optional[Dict] = None) -> BasePolicyNetwork:
    """创建策略网络"""
    config = config or {}
    
    if agent_type == "manager":
        return ManagerPolicyNetwork(state_dim, **config)
    elif agent_type == "executor":
        return ExecutorPolicyNetwork(state_dim, **config)
    elif agent_type == "reflector":
        return ReflectorPolicyNetwork(state_dim, **config)
    elif agent_type == "notetaker":
        return NotetakerPolicyNetwork(state_dim, **config)
    else:
        raise ValueError(f"未知的智能体类型: {agent_type}")


def load_pretrained_policy(agent_type: str, 
                          checkpoint_path: str, 
                          state_dim: int,
                          config: Optional[Dict] = None) -> BasePolicyNetwork:
    """加载预训练策略"""
    policy = create_policy_network(agent_type, state_dim, config)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        policy.load_state_dict(checkpoint['policy_state_dict'])
        logger.info(f"成功加载预训练策略: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"加载预训练策略失败: {e}，使用随机初始化")
    
    return policy


def save_policy_checkpoint(policy: BasePolicyNetwork, 
                          checkpoint_path: str, 
                          metadata: Optional[Dict] = None):
    """保存策略检查点"""
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'policy_class': policy.__class__.__name__,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"策略检查点已保存: {checkpoint_path}")