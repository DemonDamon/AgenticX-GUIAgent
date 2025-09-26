#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent RL Core Module - 强化学习核心组件

基于PRD设计的RL核心模块，提供：
- M6: RL环境抽象 (environment)
- M7: 多模态状态编码 (state)
- M8: 策略网络架构 (policies)
- M9: 经验管理系统 (experience)
- M10: 奖励函数设计 (rewards)
- M11: 在线学习更新 (updates)
- M12: 监控部署系统 (deployment)

Author: AgenticX Team
Date: 2025
"""

from .environment import (
    MobileGUIEnvironment,
    ActionSpace,
    StateSpace,
    EnvironmentWrapper
)

from .state import (
    MultimodalStateEncoder,
    VisionEncoder,
    TextEncoder,
    ActionHistoryEncoder
)

from .policies import (
    BasePolicyNetwork,
    ManagerPolicyNetwork,
    ExecutorPolicyNetwork,
    ReflectorPolicyNetwork,
    NotetakerPolicyNetwork
)

from .experience import (
    Experience,
    ExperienceReplayBuffer,
    ExperienceSharingHub
)

from .rewards import (
    RewardCalculator,
    TaskCompletionReward,
    EfficiencyReward,
    UserExperienceReward
)

from .updates import (
    OnlinePolicyUpdater,
    PPOUpdater,
    SACUpdater
)

from .deployment import (
    LearningMonitor,
    PolicyDeployment,
    SafetyGuard
)

__all__ = [
    # Environment
    "MobileGUIEnvironment",
    "ActionSpace",
    "StateSpace",
    "EnvironmentWrapper",
    
    # State Encoding
    "MultimodalStateEncoder",
    "VisionEncoder",
    "TextEncoder",
    "ActionHistoryEncoder",
    
    # Policies
    "BasePolicyNetwork",
    "ManagerPolicyNetwork",
    "ExecutorPolicyNetwork",
    "ReflectorPolicyNetwork",
    "NotetakerPolicyNetwork",
    
    # Experience
    "Experience",
    "ExperienceReplayBuffer",
    "ExperienceSharingHub",
    
    # Rewards
    "RewardCalculator",
    "TaskCompletionReward",
    "EfficiencyReward",
    "UserExperienceReward",
    
    # Updates
    "OnlinePolicyUpdater",
    "PPOUpdater",
    "SACUpdater",
    
    # Deployment
    "LearningMonitor",
    "PolicyDeployment",
    "SafetyGuard"
]

__version__ = "1.0.0"
__author__ = "AgenticX Team"
__description__ = "AgenticX-GUIAgent强化学习核心模块 - 提供完整的RL基础设施"