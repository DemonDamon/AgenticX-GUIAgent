#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Learning Module - RL增强的学习模块
学习模块：五阶段学习引擎 + 强化学习能力

重构说明：
- 基于AgenticX的Component基类重构所有学习组件
- 使用AgenticX的事件系统进行组件间通信
- 集成AgenticX的工作流引擎进行学习流程编排
- 遵循AgenticX的生命周期管理和配置架构
- 支持异步操作和并发学习任务
- 新增强化学习能力和知识协作机制

Author: AgenticX Team
Date: 2025
"""

# 核心学习引擎
from .learning_engine import (
    LearningEngine,
    LearningEngineStatus,
    LearningTrigger,
    LearningConfiguration,
    LearningResult
)

# RL增强学习引擎
from .rl_enhanced_learning_engine import (
    RLEnhancedLearningEngine,
    RLConfiguration,
    RLLearningMode,
    RLLearningResult,
    create_rl_enhanced_learning_engine,
    create_rl_configuration
)

# 学习协调器
from .learning_coordinator import (
    LearningCoordinator,
    LearningMode,
    LearningPhase,
    LearningTask,
    LearningTaskStatus,
    LearningTaskPriority
)

# 五阶段学习组件
from .prior_knowledge import (
    PriorKnowledgeRetriever,
    KnowledgeType,
    KnowledgeSource,
    KnowledgeRetrievalResult
)

from .guided_explorer import (
    GuidedExplorer,
    ExplorationStrategy,
    ExplorationAction,
    ExplorationResult
)

from .task_synthesizer import (
    TaskSynthesizer,
    SynthesisStrategy,
    SynthesisResult,
    TaskComplexity
)

from .usage_optimizer import (
    UsageOptimizer,
    OptimizationStrategy,
    OptimizationResult,
    OptimizationType
)

from .edge_handler import (
    EdgeHandler,
    EdgeCaseType,
    EdgeHandlingStrategy,
    EdgeHandlingResult
)

# 知识学习协作
from .knowledge_integration import (
    KnowledgeLearningBridge,
    LearningKnowledgeAdapter,
    IntegrationType,
    SyncStrategy,
    IntegrationConfig,
    SyncResult,
    create_knowledge_learning_bridge,
    create_integration_config
)

# RL核心组件（选择性导出）
try:
    from .rl_core import (
        # 环境
        MobileGUIEnvironment,
        GUIAction,
        ActionSpace,
        StateSpace,
        
        # 状态编码
        MultimodalStateEncoder,
        create_multimodal_encoder,
        
        # 策略网络
        BasePolicyNetwork,
        ManagerPolicyNetwork,
        ExecutorPolicyNetwork,
        ReflectorPolicyNetwork,
        NotetakerPolicyNetwork,
        create_policy_network,
        
        # 经验管理
        Experience,
        ExperienceReplayBuffer,
        ExperienceSharingHub,
        create_experience_buffer,
        create_sharing_hub,
        
        # 奖励计算
        RewardCalculator,
        RewardComponents,
        create_reward_calculator,
        
        # 策略更新
        OnlinePolicyUpdater,
        PPOUpdater,
        SACUpdater,
        UpdateConfig,
        create_updater,
        create_update_config,
        
        # 部署监控
        LearningMonitor,
        PolicyDeployment,
        SafetyGuard,
        create_learning_monitor,
        create_policy_deployment,
        create_safety_guard
    )
    RL_CORE_AVAILABLE = True
except ImportError:
    RL_CORE_AVAILABLE = False

# 导出列表
__all__ = [
    # 核心引擎
    "LearningEngine",
    "LearningEngineStatus",
    "LearningTrigger",
    "LearningConfiguration",
    "LearningResult",
    
    # RL增强引擎
    "RLEnhancedLearningEngine",
    "RLConfiguration",
    "RLLearningMode",
    "RLLearningResult",
    "create_rl_enhanced_learning_engine",
    "create_rl_configuration",
    
    # 协调器
    "LearningCoordinator",
    "LearningMode",
    "LearningPhase",
    "LearningTask",
    "LearningTaskStatus",
    "LearningTaskPriority",
    
    # 阶段1：先验知识检索
    "PriorKnowledgeRetriever",
    "KnowledgeType",
    "KnowledgeSource",
    "KnowledgeRetrievalResult",
    
    # 阶段2：引导探索
    "GuidedExplorer",
    "ExplorationStrategy",
    "ExplorationAction",
    "ExplorationResult",
    
    # 阶段3：任务合成
    "TaskSynthesizer",
    "SynthesisStrategy",
    "SynthesisResult",
    "TaskComplexity",
    
    # 阶段4：使用优化
    "UsageOptimizer",
    "OptimizationStrategy",
    "OptimizationResult",
    "OptimizationType",
    
    # 阶段5：边缘处理
    "EdgeHandler",
    "EdgeCaseType",
    "EdgeHandlingStrategy",
    "EdgeHandlingResult",
    
    # 知识学习协作
    "KnowledgeLearningBridge",
    "LearningKnowledgeAdapter",
    "IntegrationType",
    "SyncStrategy",
    "IntegrationConfig",
    "SyncResult",
    "create_knowledge_learning_bridge",
    "create_integration_config"
]

# 条件性添加RL核心组件
if RL_CORE_AVAILABLE:
    __all__.extend([
        # RL环境
        "MobileGUIEnvironment",
        "GUIAction",
        "ActionSpace",
        "StateSpace",
        
        # RL状态编码
        "MultimodalStateEncoder",
        "create_multimodal_encoder",
        
        # RL策略网络
        "BasePolicyNetwork",
        "ManagerPolicyNetwork",
        "ExecutorPolicyNetwork",
        "ReflectorPolicyNetwork",
        "NotetakerPolicyNetwork",
        "create_policy_network",
        
        # RL经验管理
        "Experience",
        "ExperienceReplayBuffer",
        "ExperienceSharingHub",
        "create_experience_buffer",
        "create_sharing_hub",
        
        # RL奖励计算
        "RewardCalculator",
        "RewardComponents",
        "create_reward_calculator",
        
        # RL策略更新
        "OnlinePolicyUpdater",
        "PPOUpdater",
        "SACUpdater",
        "UpdateConfig",
        "create_updater",
        "create_update_config",
        
        # RL部署监控
        "LearningMonitor",
        "PolicyDeployment",
        "SafetyGuard",
        "create_learning_monitor",
        "create_policy_deployment",
        "create_safety_guard"
    ])

# 版本信息
__version__ = "3.0.0"
__author__ = "AgenticX Team"
__description__ = "AgenticX-GUIAgent学习模块 - RL增强的五阶段学习引擎，支持强化学习和知识协作"