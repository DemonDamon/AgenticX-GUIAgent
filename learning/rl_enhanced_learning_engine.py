#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Enhanced Learning Engine - 强化学习增强的学习引擎

基于现有LearningEngine，集成强化学习能力，实现真正的在线学习。
保持向后兼容性，支持渐进式RL能力启用。

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from uuid import uuid4

import torch
import numpy as np

# 导入现有组件
from .learning_engine import (
    LearningEngine, LearningEngineStatus, LearningTrigger, 
    LearningConfiguration, LearningResult
)
from .learning_coordinator import LearningCoordinator, LearningMode, LearningPhase

# 导入RL核心组件
from .rl_core.environment import MobileGUIEnvironment, GUIAction
from .rl_core.state import MultimodalStateEncoder, create_multimodal_encoder
from .rl_core.policies import (
    BasePolicyNetwork, ManagerPolicyNetwork, ExecutorPolicyNetwork,
    ReflectorPolicyNetwork, NotetakerPolicyNetwork, create_policy_network
)
from .rl_core.experience import (
    Experience, ExperienceReplayBuffer, ExperienceSharingHub,
    create_experience_buffer, create_sharing_hub
)
from .rl_core.rewards import RewardCalculator, create_reward_calculator
from .rl_core.updates import (
    OnlinePolicyUpdater, PPOUpdater, SACUpdater, 
    create_updater, UpdateConfig, create_update_config
)
from .rl_core.deployment import (
    LearningMonitor, PolicyDeployment, SafetyGuard,
    create_learning_monitor, create_policy_deployment, create_safety_guard
)

from agenticx.core.component import Component


class RLLearningMode(Enum):
    """RL学习模式"""
    TRADITIONAL = "traditional"  # 传统学习模式
    RL_ONLY = "rl_only"  # 纯RL模式
    HYBRID = "hybrid"  # 混合模式
    ADAPTIVE = "adaptive"  # 自适应模式


@dataclass
class RLConfiguration:
    """RL配置"""
    # RL启用配置
    rl_enabled: bool = False
    rl_mode: RLLearningMode = RLLearningMode.TRADITIONAL
    
    # 环境配置
    environment_config: Dict[str, Any] = field(default_factory=lambda: {
        'screen_width': 1080,
        'screen_height': 1920,
        'max_episode_steps': 100
    })
    
    # 状态编码配置
    state_encoder_config: Dict[str, Any] = field(default_factory=lambda: {
        'fusion_dim': 768,
        'num_fusion_layers': 3,
        'dropout': 0.1
    })
    
    # 策略网络配置
    policy_config: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dims': [512, 256, 128],
        'activation': 'relu',
        'dropout': 0.1
    })
    
    # 经验管理配置
    experience_config: Dict[str, Any] = field(default_factory=lambda: {
        'capacity': 100000,
        'alpha': 0.6,
        'beta': 0.4,
        'enable_prioritized': True
    })
    
    # 奖励配置
    reward_config: Dict[str, Any] = field(default_factory=lambda: {
        'reward_weights': {
            'task_completion': 0.4,
            'efficiency': 0.2,
            'user_experience': 0.2,
            'learning_progress': 0.1,
            'safety': 0.05,
            'exploration': 0.05
        },
        'enable_reward_shaping': True,
        'enable_adaptive_weights': True
    })
    
    # 更新配置
    update_config: Dict[str, Any] = field(default_factory=lambda: {
        'algorithm': 'ppo',
        'learning_rate': 3e-4,
        'batch_size': 32,
        'update_frequency': 4,
        'ppo_epochs': 4,
        'ppo_clip_ratio': 0.2
    })
    
    # 部署配置
    deployment_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_monitoring': True,
        'enable_safety_guard': True,
        'log_dir': './logs/rl_learning',
        'deployment_dir': './deployments',
        'backup_dir': './backups'
    })
    
    # 知识协作配置
    knowledge_integration_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_knowledge_integration': True,
        'knowledge_weight': 0.3,
        'strategy_weight': 0.7
    })


@dataclass
class RLLearningResult(LearningResult):
    """RL学习结果"""
    # RL特定指标
    rl_enabled: bool = False
    policy_updates: int = 0
    average_reward: float = 0.0
    exploration_rate: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    experience_collected: int = 0
    safety_violations: int = 0
    
    # 多智能体指标
    agent_performance: Dict[str, float] = field(default_factory=dict)
    coordination_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = {
            'session_id': self.session_id,
            'trigger': self.trigger.value,
            'phases_executed': self.phases_executed,
            'success': self.success,
            'duration': self.duration,
            'knowledge_gained': self.knowledge_gained,
            'patterns_discovered': self.patterns_discovered,
            'optimizations_applied': self.optimizations_applied,
            'errors_handled': self.errors_handled,
            'performance_improvement': self.performance_improvement,
            'timestamp': self.timestamp,
            'details': self.details,
            'errors': self.errors
        }
        
        # 添加RL特定指标
        rl_dict = {
            'rl_enabled': self.rl_enabled,
            'policy_updates': self.policy_updates,
            'average_reward': self.average_reward,
            'exploration_rate': self.exploration_rate,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'experience_collected': self.experience_collected,
            'safety_violations': self.safety_violations,
            'agent_performance': self.agent_performance,
            'coordination_score': self.coordination_score
        }
        
        base_dict.update(rl_dict)
        return base_dict


class RLEnhancedLearningEngine(LearningEngine):
    """RL增强的学习引擎
    
    在现有LearningEngine基础上集成强化学习能力，支持：
    1. 传统学习模式（向后兼容）
    2. 纯RL学习模式
    3. 混合学习模式
    4. 自适应学习模式
    """
    
    def __init__(self,
                 info_pool=None,
                 config: Optional[LearningConfiguration] = None,
                 rl_config: Optional[RLConfiguration] = None,
                 knowledge_manager=None):
        # 初始化基础学习引擎
        super().__init__(info_pool, config)
        
        # RL配置
        self.rl_config = rl_config or RLConfiguration()
        self.knowledge_manager = knowledge_manager
        
        # RL组件（延迟初始化）
        self.rl_components = None
        self.rl_initialized = False
        
        # RL状态
        self.rl_mode = self.rl_config.rl_mode
        self.current_episode = 0
        self.total_steps = 0
        
        # 性能跟踪
        self.rl_performance_history = deque(maxlen=1000)
        self.agent_performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # 知识协作桥接器
        self.knowledge_bridge = None
        
        self.logger = logger
    
    async def initialize(self) -> bool:
        """初始化RL增强学习引擎"""
        try:
            # 首先初始化基础学习引擎
            if not await super().initialize():
                return False
            
            # 如果启用RL，初始化RL组件
            if self.rl_config.rl_enabled:
                await self.enable_rl_mode()
            
            logger.info(f"RL增强学习引擎初始化完成 - RL模式: {self.rl_mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"RL增强学习引擎初始化失败: {e}")
            return False
    
    async def enable_rl_mode(self) -> bool:
        """启用RL模式"""
        try:
            if self.rl_initialized:
                logger.info("RL模式已启用")
                return True
            
            logger.info("正在初始化RL组件...")
            
            # 初始化RL核心组件
            self.rl_components = await self._initialize_rl_components()
            
            # 初始化知识协作桥接器
            if (self.knowledge_manager and 
                self.rl_config.knowledge_integration_config.get('enable_knowledge_integration', True)):
                self.knowledge_bridge = await self._initialize_knowledge_bridge()
            
            # 设置RL事件处理
            self._setup_rl_event_handlers()
            
            self.rl_initialized = True
            self.rl_config.rl_enabled = True
            
            logger.info("RL模式已成功启用")
            return True
            
        except Exception as e:
            logger.error(f"启用RL模式失败: {e}")
            return False
    
    async def _initialize_rl_components(self) -> Dict[str, Any]:
        """初始化RL核心组件"""
        components = {}
        
        # 1. 环境
        components['environment'] = MobileGUIEnvironment(
            **self.rl_config.environment_config
        )
        
        # 2. 状态编码器
        components['state_encoder'] = create_multimodal_encoder(
            self.rl_config.state_encoder_config
        )
        
        # 3. 策略网络（为四个Agent创建）
        state_dim = components['state_encoder'].get_state_embedding_dim()
        
        components['policies'] = {
            'manager': create_policy_network('manager', state_dim, self.rl_config.policy_config),
            'executor': create_policy_network('executor', state_dim, self.rl_config.policy_config),
            'reflector': create_policy_network('reflector', state_dim, self.rl_config.policy_config),
            'notetaker': create_policy_network('notetaker', state_dim, self.rl_config.policy_config)
        }
        
        # 4. 经验管理
        components['experience_buffer'] = create_experience_buffer(
            self.rl_config.experience_config
        )
        components['sharing_hub'] = create_sharing_hub(
            self.rl_config.experience_config
        )
        
        # 5. 奖励计算器
        components['reward_calculator'] = create_reward_calculator(
            self.rl_config.reward_config
        )
        
        # 6. 策略更新器
        update_config = create_update_config(**self.rl_config.update_config)
        components['policy_updater'] = create_updater(
            self.rl_config.update_config['algorithm'],
            update_config
        )
        
        # 7. 监控和部署
        if self.rl_config.deployment_config.get('enable_monitoring', True):
            components['monitor'] = create_learning_monitor(
                self.rl_config.deployment_config
            )
        
        components['deployment'] = create_policy_deployment(
            self.rl_config.deployment_config
        )
        
        if self.rl_config.deployment_config.get('enable_safety_guard', True):
            components['safety_guard'] = create_safety_guard(
                self.rl_config.deployment_config
            )
        
        return components
    
    async def _initialize_knowledge_bridge(self):
        """初始化知识协作桥接器"""
        # 这里应该导入并初始化知识学习协作模块
        # 暂时返回None，等待M13模块实现
        return None
    
    def _setup_rl_event_handlers(self):
        """设置RL事件处理器"""
        if self.info_pool:
            # 监听RL特定事件
            self.info_pool.subscribe(
                "rl_experience_collected",
                self._handle_rl_experience
            )
            
            self.info_pool.subscribe(
                "rl_policy_update",
                self._handle_policy_update
            )
    
    async def hybrid_learning_session(self, 
                                    trigger: LearningTrigger, 
                                    context: Dict) -> RLLearningResult:
        """混合学习会话（传统学习 + RL学习）"""
        session_id = str(uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"开始混合学习会话: {session_id}")
            
            # 创建结果对象
            result = RLLearningResult(
                session_id=session_id,
                trigger=trigger,
                phases_executed=[],
                success=False,
                duration=0.0,
                knowledge_gained=0,
                patterns_discovered=0,
                optimizations_applied=0,
                errors_handled=0,
                performance_improvement=0.0,
                timestamp=datetime.now().isoformat(),
                rl_enabled=self.rl_initialized
            )
            
            # 根据RL模式执行不同的学习策略
            if self.rl_mode == RLLearningMode.TRADITIONAL:
                # 纯传统学习
                traditional_result = await self._execute_traditional_learning(context)
                result = self._merge_traditional_result(result, traditional_result)
                
            elif self.rl_mode == RLLearningMode.RL_ONLY and self.rl_initialized:
                # 纯RL学习
                rl_result = await self._execute_rl_learning(context)
                result = self._merge_rl_result(result, rl_result)
                
            elif self.rl_mode == RLLearningMode.HYBRID and self.rl_initialized:
                # 混合学习
                traditional_result = await self._execute_traditional_learning(context)
                rl_result = await self._execute_rl_learning(context)
                
                result = self._merge_traditional_result(result, traditional_result)
                result = self._merge_rl_result(result, rl_result)
                
                # 协调传统学习和RL学习的结果
                result = await self._coordinate_hybrid_results(result, traditional_result, rl_result)
                
            elif self.rl_mode == RLLearningMode.ADAPTIVE:
                # 自适应学习
                result = await self._execute_adaptive_learning(context)
            
            # 计算总体性能改进
            result.performance_improvement = await self._calculate_performance_improvement(result)
            
            # 更新统计信息
            await self._update_learning_statistics(result)
            
            result.success = True
            result.duration = time.time() - start_time
            
            logger.info(f"混合学习会话完成: {session_id}, 耗时: {result.duration:.2f}s")
            
        except Exception as e:
            result.success = False
            result.duration = time.time() - start_time
            result.errors.append(str(e))
            logger.error(f"混合学习会话失败: {session_id}, 错误: {e}")
        
        # 记录学习历史
        self.learning_history.append(result)
        
        return result
    
    async def _execute_traditional_learning(self, context: Dict) -> LearningResult:
        """执行传统学习"""
        # 调用父类的学习方法
        return await super().trigger_learning(
            trigger=LearningTrigger.MANUAL,
            description="传统学习组件",
            context=context
        )
    
    async def _execute_rl_learning(self, context: Dict) -> Dict[str, Any]:
        """执行RL学习"""
        if not self.rl_initialized:
            raise RuntimeError("RL组件未初始化")
        
        rl_result = {
            'policy_updates': 0,
            'average_reward': 0.0,
            'exploration_rate': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'experience_collected': 0,
            'safety_violations': 0,
            'agent_performance': {},
            'coordination_score': 0.0
        }
        
        try:
            # 1. 收集经验
            experiences = await self._collect_experiences(context)
            rl_result['experience_collected'] = len(experiences)
            
            # 2. 更新策略
            if len(experiences) >= self.rl_config.update_config.get('batch_size', 32):
                update_results = await self._update_policies(experiences)
                rl_result.update(update_results)
            
            # 3. 评估性能
            performance = await self._evaluate_agent_performance()
            rl_result['agent_performance'] = performance
            
            # 4. 计算协调分数
            rl_result['coordination_score'] = await self._calculate_coordination_score(performance)
            
            # 5. 安全检查
            if 'safety_guard' in self.rl_components:
                safety_stats = self.rl_components['safety_guard'].get_safety_statistics()
                rl_result['safety_violations'] = safety_stats.get('total_violations', 0)
            
        except Exception as e:
            logger.error(f"RL学习执行失败: {e}")
            raise
        
        return rl_result
    
    async def _execute_adaptive_learning(self, context: Dict) -> RLLearningResult:
        """执行自适应学习"""
        # 根据当前性能和环境动态选择学习策略
        performance_score = await self._get_current_performance_score()
        
        if performance_score < 0.5:
            # 性能较差，使用RL学习
            if self.rl_initialized:
                rl_result = await self._execute_rl_learning(context)
                result = RLLearningResult(
                    session_id=str(uuid4()),
                    trigger=LearningTrigger.AUTOMATIC,
                    phases_executed=['rl_learning'],
                    success=True,
                    duration=0.0,
                    knowledge_gained=0,
                    patterns_discovered=0,
                    optimizations_applied=rl_result.get('policy_updates', 0),
                    errors_handled=0,
                    performance_improvement=0.0,
                    timestamp=datetime.now().isoformat(),
                    rl_enabled=True
                )
                return self._merge_rl_result(result, rl_result)
        
        # 性能较好，使用传统学习
        traditional_result = await self._execute_traditional_learning(context)
        result = RLLearningResult(
            session_id=traditional_result.session_id,
            trigger=traditional_result.trigger,
            phases_executed=traditional_result.phases_executed,
            success=traditional_result.success,
            duration=traditional_result.duration,
            knowledge_gained=traditional_result.knowledge_gained,
            patterns_discovered=traditional_result.patterns_discovered,
            optimizations_applied=traditional_result.optimizations_applied,
            errors_handled=traditional_result.errors_handled,
            performance_improvement=traditional_result.performance_improvement,
            timestamp=traditional_result.timestamp,
            rl_enabled=False
        )
        
        return result
    
    async def _collect_experiences(self, context: Dict) -> List[Experience]:
        """收集经验数据"""
        experiences = []
        
        # 这里应该从实际的Agent执行中收集经验
        # 简化实现：生成模拟经验
        for i in range(10):  # 模拟收集10个经验
            experience = Experience(
                state=torch.randn(768),  # 模拟状态
                action=torch.randint(0, 10, (1,)),  # 模拟动作
                reward=np.random.uniform(-1, 1),  # 模拟奖励
                next_state=torch.randn(768),  # 模拟下一状态
                done=np.random.random() < 0.1,  # 10%概率结束
                agent_id=np.random.choice(['manager', 'executor', 'reflector', 'notetaker']),
                timestamp=datetime.now()
            )
            experiences.append(experience)
        
        # 添加到经验缓冲区
        for exp in experiences:
            self.rl_components['experience_buffer'].add(exp)
        
        return experiences
    
    async def _update_policies(self, experiences: List[Experience]) -> Dict[str, Any]:
        """更新策略网络"""
        update_results = {
            'policy_updates': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }
        
        try:
            # 按智能体分组经验
            agent_experiences = defaultdict(list)
            for exp in experiences:
                agent_experiences[exp.agent_id].append(exp)
            
            # 更新各智能体策略
            total_updates = 0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            
            for agent_id, agent_exps in agent_experiences.items():
                if len(agent_exps) >= 4:  # 最小批次大小
                    policy = self.rl_components['policies'][agent_id]
                    result = self.rl_components['policy_updater'].update_policy(
                        policy, agent_exps, {}
                    )
                    
                    total_updates += 1
                    total_policy_loss += result.get('policy_loss', 0.0)
                    total_value_loss += result.get('value_loss', 0.0)
            
            if total_updates > 0:
                update_results['policy_updates'] = total_updates
                update_results['policy_loss'] = total_policy_loss / total_updates
                update_results['value_loss'] = total_value_loss / total_updates
            
        except Exception as e:
            logger.error(f"策略更新失败: {e}")
        
        return update_results
    
    async def _evaluate_agent_performance(self) -> Dict[str, float]:
        """评估智能体性能"""
        performance = {}
        
        for agent_id in ['manager', 'executor', 'reflector', 'notetaker']:
            # 简化的性能评估
            recent_rewards = []
            
            # 从经验缓冲区获取最近的奖励
            all_experiences = self.rl_components['experience_buffer'].get_all_experiences()
            agent_experiences = [exp for exp in all_experiences if exp.agent_id == agent_id]
            
            if agent_experiences:
                recent_rewards = [exp.reward for exp in agent_experiences[-10:]]  # 最近10个
                performance[agent_id] = np.mean(recent_rewards)
            else:
                performance[agent_id] = 0.0
        
        return performance
    
    async def _calculate_coordination_score(self, agent_performance: Dict[str, float]) -> float:
        """计算协调分数"""
        if not agent_performance:
            return 0.0
        
        # 简化的协调分数计算
        performances = list(agent_performance.values())
        mean_performance = np.mean(performances)
        std_performance = np.std(performances)
        
        # 协调分数：平均性能高且方差小表示协调好
        coordination_score = mean_performance * (1.0 - min(1.0, std_performance))
        
        return max(0.0, coordination_score)
    
    async def _get_current_performance_score(self) -> float:
        """获取当前性能分数"""
        if not self.rl_performance_history:
            return 0.5  # 默认中等性能
        
        recent_scores = list(self.rl_performance_history)[-10:]
        return np.mean(recent_scores)
    
    def _merge_traditional_result(self, 
                                 rl_result: RLLearningResult, 
                                 traditional_result: LearningResult) -> RLLearningResult:
        """合并传统学习结果"""
        rl_result.phases_executed.extend(traditional_result.phases_executed)
        rl_result.knowledge_gained += traditional_result.knowledge_gained
        rl_result.patterns_discovered += traditional_result.patterns_discovered
        rl_result.optimizations_applied += traditional_result.optimizations_applied
        rl_result.errors_handled += traditional_result.errors_handled
        rl_result.errors.extend(traditional_result.errors)
        
        return rl_result
    
    def _merge_rl_result(self, 
                        rl_result: RLLearningResult, 
                        rl_data: Dict[str, Any]) -> RLLearningResult:
        """合并RL学习结果"""
        rl_result.policy_updates = rl_data.get('policy_updates', 0)
        rl_result.average_reward = rl_data.get('average_reward', 0.0)
        rl_result.exploration_rate = rl_data.get('exploration_rate', 0.0)
        rl_result.policy_loss = rl_data.get('policy_loss', 0.0)
        rl_result.value_loss = rl_data.get('value_loss', 0.0)
        rl_result.experience_collected = rl_data.get('experience_collected', 0)
        rl_result.safety_violations = rl_data.get('safety_violations', 0)
        rl_result.agent_performance = rl_data.get('agent_performance', {})
        rl_result.coordination_score = rl_data.get('coordination_score', 0.0)
        
        rl_result.phases_executed.append('rl_learning')
        
        return rl_result
    
    async def _coordinate_hybrid_results(self, 
                                       result: RLLearningResult,
                                       traditional_result: LearningResult,
                                       rl_result: Dict[str, Any]) -> RLLearningResult:
        """协调混合学习结果"""
        # 计算权重
        knowledge_weight = self.rl_config.knowledge_integration_config.get('knowledge_weight', 0.3)
        strategy_weight = self.rl_config.knowledge_integration_config.get('strategy_weight', 0.7)
        
        # 加权合并性能改进
        traditional_improvement = traditional_result.performance_improvement
        rl_improvement = rl_result.get('average_reward', 0.0)
        
        result.performance_improvement = (
            traditional_improvement * knowledge_weight + 
            rl_improvement * strategy_weight
        )
        
        return result
    
    async def _calculate_performance_improvement(self, result: RLLearningResult) -> float:
        """计算性能改进"""
        if not self.rl_performance_history:
            return result.performance_improvement
        
        # 计算相对于历史平均的改进
        historical_avg = np.mean(list(self.rl_performance_history))
        current_performance = result.average_reward if result.rl_enabled else result.performance_improvement
        
        improvement = (current_performance - historical_avg) / max(abs(historical_avg), 0.1)
        
        return improvement
    
    async def _update_learning_statistics(self, result: RLLearningResult):
        """更新学习统计信息"""
        # 更新基础统计
        self.engine_stats['total_learning_sessions'] += 1
        if result.success:
            self.engine_stats['successful_sessions'] += 1
        else:
            self.engine_stats['failed_sessions'] += 1
        
        self.engine_stats['total_learning_time'] += result.duration
        self.engine_stats['knowledge_items_learned'] += result.knowledge_gained
        self.engine_stats['patterns_discovered'] += result.patterns_discovered
        self.engine_stats['optimizations_applied'] += result.optimizations_applied
        
        # 更新RL特定统计
        if result.rl_enabled:
            self.rl_performance_history.append(result.average_reward)
            
            for agent_id, performance in result.agent_performance.items():
                self.agent_performance_history[agent_id].append(performance)
        
        # 更新平均会话时长
        total_sessions = self.engine_stats['total_learning_sessions']
        if total_sessions > 0:
            self.engine_stats['avg_session_duration'] = (
                self.engine_stats['total_learning_time'] / total_sessions
            )
        
        self.engine_stats['last_learning_timestamp'] = result.timestamp
    
    async def _handle_rl_experience(self, event_data: Dict[str, Any]):
        """处理RL经验事件"""
        try:
            if not self.rl_initialized:
                return
            
            # 处理新收集的经验
            experience_data = event_data.get('experience')
            if experience_data:
                # 这里可以进行经验的预处理、过滤等
                pass
                
        except Exception as e:
            logger.error(f"处理RL经验事件失败: {e}")
    
    async def _handle_policy_update(self, event_data: Dict[str, Any]):
        """处理策略更新事件"""
        try:
            if not self.rl_initialized:
                return
            
            # 处理策略更新结果
            update_result = event_data.get('update_result')
            if update_result:
                # 可以进行策略部署、监控等操作
                if 'monitor' in self.rl_components:
                    await self.rl_components['monitor'].monitor_learning_progress(
                        update_result, self.rl_components['policies']
                    )
                
        except Exception as e:
            logger.error(f"处理策略更新事件失败: {e}")
    
    def get_rl_statistics(self) -> Dict[str, Any]:
        """获取RL统计信息"""
        stats = {
            'rl_enabled': self.rl_initialized,
            'rl_mode': self.rl_mode.value,
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
            'performance_history_length': len(self.rl_performance_history),
            'agent_performance_history': {
                agent_id: len(history) 
                for agent_id, history in self.agent_performance_history.items()
            }
        }
        
        if self.rl_initialized and self.rl_components:
            # 添加组件统计
            if 'experience_buffer' in self.rl_components:
                stats['experience_buffer'] = self.rl_components['experience_buffer'].get_buffer_statistics()
            
            if 'sharing_hub' in self.rl_components:
                stats['sharing_hub'] = self.rl_components['sharing_hub'].get_sharing_statistics()
            
            if 'safety_guard' in self.rl_components:
                stats['safety'] = self.rl_components['safety_guard'].get_safety_statistics()
        
        return stats
    
    async def trigger_learning(self, 
                             trigger: LearningTrigger, 
                             description: str = "", 
                             context: Optional[Dict] = None) -> RLLearningResult:
        """触发学习（重写以支持RL）"""
        context = context or {}
        
        # 根据RL模式选择学习方式
        if self.rl_initialized and self.rl_mode != RLLearningMode.TRADITIONAL:
            return await self.hybrid_learning_session(trigger, context)
        else:
            # 传统学习模式
            traditional_result = await super().trigger_learning(trigger, description, context)
            
            # 转换为RLLearningResult
            rl_result = RLLearningResult(
                session_id=traditional_result.session_id,
                trigger=traditional_result.trigger,
                phases_executed=traditional_result.phases_executed,
                success=traditional_result.success,
                duration=traditional_result.duration,
                knowledge_gained=traditional_result.knowledge_gained,
                patterns_discovered=traditional_result.patterns_discovered,
                optimizations_applied=traditional_result.optimizations_applied,
                errors_handled=traditional_result.errors_handled,
                performance_improvement=traditional_result.performance_improvement,
                timestamp=traditional_result.timestamp,
                details=traditional_result.details,
                errors=traditional_result.errors,
                rl_enabled=False
            )
            
            return rl_result
    
    async def shutdown(self):
        """关闭RL增强学习引擎"""
        try:
            # 关闭RL组件
            if self.rl_initialized and self.rl_components:
                if 'monitor' in self.rl_components:
                    self.rl_components['monitor'].close()
                
                if 'environment' in self.rl_components:
                    self.rl_components['environment'].close()
            
            # 关闭基础学习引擎
            await super().shutdown()
            
            logger.info("RL增强学习引擎已关闭")
            
        except Exception as e:
            logger.error(f"关闭RL增强学习引擎失败: {e}")


# 工具函数
def create_rl_enhanced_learning_engine(
    info_pool=None,
    config: Optional[LearningConfiguration] = None,
    rl_config: Optional[RLConfiguration] = None,
    knowledge_manager=None
) -> RLEnhancedLearningEngine:
    """创建RL增强学习引擎"""
    return RLEnhancedLearningEngine(
        info_pool=info_pool,
        config=config,
        rl_config=rl_config,
        knowledge_manager=knowledge_manager
    )


def create_rl_configuration(**kwargs) -> RLConfiguration:
    """创建RL配置"""
    return RLConfiguration(**kwargs)