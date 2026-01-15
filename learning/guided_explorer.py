#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Guided Explorer - 引导探索器 (基于AgenticX框架重构)

学习引擎第二阶段：基于先验知识进行引导式探索。

重构说明：
- 基于AgenticX的Component基类重构
- 使用AgenticX的事件系统进行探索结果通知
- 集成AgenticX的内存组件进行探索历史管理
- 遵循AgenticX的异步执行和错误处理模式
"""

import asyncio
from loguru import logger
import json
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass
from enum import Enum
import numpy as np

from agenticx.core.component import Component

from core.info_pool import InfoPool, InfoType, InfoPriority
from utils import get_iso_timestamp
from .prior_knowledge import KnowledgeMatch, PriorKnowledgeRetriever


class ExplorationStrategy(Enum):
    """探索策略"""
    GREEDY = "greedy"  # 贪心策略
    EPSILON_GREEDY = "epsilon_greedy"  # ε-贪心策略
    UCB = "ucb"  # 上置信界策略
    THOMPSON_SAMPLING = "thompson_sampling"  # 汤普森采样
    GUIDED_RANDOM = "guided_random"  # 引导随机


@dataclass
class ExplorationAction:
    """探索动作"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    expected_reward: float
    confidence: float
    exploration_value: float
    knowledge_source: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ExplorationResult:
    """探索结果"""
    action: ExplorationAction
    success: bool
    actual_reward: float
    execution_time: float
    error_message: Optional[str] = None
    observations: Dict[str, Any] = None
    learned_insights: List[str] = None


@dataclass
class ExplorationContext:
    """探索上下文"""
    task_description: str
    current_state: Dict[str, Any]
    available_actions: List[str]
    constraints: Dict[str, Any]
    exploration_budget: int
    strategy: ExplorationStrategy
    agent_id: str
    prior_knowledge: List[KnowledgeMatch]
    exploration_history: List[ExplorationResult]


class GuidedExplorer(Component):
    """引导探索器
    
    负责：
    1. 基于先验知识的探索策略制定
    2. 动作空间的智能采样
    3. 探索-利用平衡
    4. 探索结果评估
    5. 探索经验积累
    """
    
    def __init__(self, info_pool: InfoPool, prior_knowledge_retriever: PriorKnowledgeRetriever):
        super().__init__()
        self.info_pool = info_pool
        self.prior_knowledge_retriever = prior_knowledge_retriever
        self.logger = logger
        
        # 探索统计
        self.exploration_stats = {
            "total_explorations": 0,
            "successful_explorations": 0,
            "average_reward": 0.0,
            "strategy_performance": {strategy.value: {"count": 0, "success_rate": 0.0, "avg_reward": 0.0} for strategy in ExplorationStrategy},
            "action_performance": {},
            "exploration_history": []
        }
        
        # 策略参数
        self.strategy_params = {
            ExplorationStrategy.EPSILON_GREEDY: {"epsilon": 0.1, "decay_rate": 0.995},
            ExplorationStrategy.UCB: {"c": 1.414},  # 探索参数
            ExplorationStrategy.THOMPSON_SAMPLING: {"alpha": 1.0, "beta": 1.0},
            ExplorationStrategy.GUIDED_RANDOM: {"knowledge_weight": 0.7}
        }
        
        # 动作价值估计
        self.action_values = {}
        self.action_counts = {}
        
        # 探索记忆
        self.exploration_memory = []
        self.max_memory_size = 1000
    
    async def explore(
        self,
        exploration_context: ExplorationContext
    ) -> List[ExplorationResult]:
        """执行引导探索
        
        Args:
            exploration_context: 探索上下文
        
        Returns:
            探索结果列表
        """
        logger.info(f"开始引导探索: {exploration_context.task_description}")
        
        exploration_results = []
        
        try:
            # 生成探索计划
            exploration_plan = await self._generate_exploration_plan(exploration_context)
            
            # 执行探索
            for i, action in enumerate(exploration_plan):
                if i >= exploration_context.exploration_budget:
                    break
                
                logger.info(f"执行探索动作 {i+1}/{len(exploration_plan)}: {action.action_type}")
                
                # 执行动作
                result = await self._execute_exploration_action(action, exploration_context)
                exploration_results.append(result)
                
                # 更新动作价值
                self._update_action_value(action, result)
                
                # 更新探索上下文
                exploration_context.exploration_history.append(result)
                
                # 如果成功且满足条件，可以提前结束
                if result.success and self._should_stop_exploration(exploration_results, exploration_context):
                    logger.info("探索提前结束：找到满意解")
                    break
            
            # 分析探索结果
            insights = await self._analyze_exploration_results(exploration_results, exploration_context)
            
            # 更新统计信息
            self._update_exploration_stats(exploration_results, exploration_context)
            
            # 发布探索结果
            self.info_pool.publish(
                InfoType.LEARNING_UPDATE,
                {
                    "stage": "guided_exploration",
                    "agent_id": exploration_context.agent_id,
                    "task_description": exploration_context.task_description,
                    "exploration_count": len(exploration_results),
                    "success_count": sum(1 for r in exploration_results if r.success),
                    "insights": insights,
                    "strategy": exploration_context.strategy.value
                },
                source_agent="GuidedExplorer",
                priority=InfoPriority.NORMAL
            )
            
            logger.info(f"探索完成，执行了{len(exploration_results)}个动作，成功{sum(1 for r in exploration_results if r.success)}个")
            return exploration_results
            
        except Exception as e:
            logger.error(f"引导探索失败: {e}")
            return exploration_results
    
    async def _generate_exploration_plan(
        self,
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """生成探索计划"""
        # 基于先验知识生成候选动作
        candidate_actions = await self._generate_candidate_actions(context)
        
        # 根据策略选择动作
        selected_actions = await self._select_actions_by_strategy(
            candidate_actions, context
        )
        
        return selected_actions
    
    async def _generate_candidate_actions(
        self,
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """生成候选动作"""
        candidate_actions = []
        
        # 1. 基于先验知识的动作
        knowledge_actions = self._generate_knowledge_based_actions(context)
        candidate_actions.extend(knowledge_actions)
        
        # 2. 基于可用动作的随机探索
        random_actions = self._generate_random_actions(context)
        candidate_actions.extend(random_actions)
        
        # 3. 基于历史成功经验的动作
        experience_actions = self._generate_experience_based_actions(context)
        candidate_actions.extend(experience_actions)
        
        # 去重和排序
        unique_actions = self._deduplicate_actions(candidate_actions)
        
        return unique_actions
    
    def _generate_knowledge_based_actions(
        self,
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """基于先验知识生成动作"""
        actions = []
        
        for knowledge in context.prior_knowledge:
            content = knowledge.content
            
            # 从知识中提取动作信息
            if "steps" in content:
                for i, step in enumerate(content["steps"]):
                    action = ExplorationAction(
                        action_id=f"knowledge_{knowledge.knowledge_id}_{i}",
                        action_type=step.get("action_type", "unknown"),
                        parameters=step.get("parameters", {}),
                        expected_reward=knowledge.relevance_score * knowledge.confidence_score,
                        confidence=knowledge.confidence_score,
                        exploration_value=self._calculate_exploration_value(
                            step.get("action_type", "unknown"), context
                        ),
                        knowledge_source=knowledge.knowledge_id,
                        metadata={
                            "source": "prior_knowledge",
                            "knowledge_type": knowledge.knowledge_type,
                            "step_index": i
                        }
                    )
                    actions.append(action)
            
            elif "action_type" in content:
                action = ExplorationAction(
                    action_id=f"knowledge_{knowledge.knowledge_id}",
                    action_type=content["action_type"],
                    parameters=content.get("parameters", {}),
                    expected_reward=knowledge.relevance_score * knowledge.confidence_score,
                    confidence=knowledge.confidence_score,
                    exploration_value=self._calculate_exploration_value(
                        content["action_type"], context
                    ),
                    knowledge_source=knowledge.knowledge_id,
                    metadata={
                        "source": "prior_knowledge",
                        "knowledge_type": knowledge.knowledge_type
                    }
                )
                actions.append(action)
        
        return actions
    
    def _generate_random_actions(
        self,
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """生成随机探索动作"""
        actions = []
        
        # 为每个可用动作类型生成随机参数
        for action_type in context.available_actions:
            # 生成多个随机参数组合
            for i in range(3):  # 每种动作类型生成3个随机变体
                random_params = self._generate_random_parameters(
                    action_type, context.current_state
                )
                
                action = ExplorationAction(
                    action_id=f"random_{action_type}_{i}",
                    action_type=action_type,
                    parameters=random_params,
                    expected_reward=0.3,  # 随机动作的默认期望奖励
                    confidence=0.2,
                    exploration_value=self._calculate_exploration_value(action_type, context),
                    metadata={
                        "source": "random_exploration",
                        "variant": i
                    }
                )
                actions.append(action)
        
        return actions
    
    def _generate_experience_based_actions(
        self,
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """基于历史经验生成动作"""
        actions = []
        
        # 从探索记忆中找到成功的动作
        successful_actions = [
            result.action for result in self.exploration_memory
            if result.success and result.actual_reward > 0.5
        ]
        
        # 选择最近的成功动作
        recent_successful = successful_actions[-10:]  # 最近10个成功动作
        
        for i, base_action in enumerate(recent_successful):
            # 创建基于成功经验的变体
            action = ExplorationAction(
                action_id=f"experience_{base_action.action_id}_{i}",
                action_type=base_action.action_type,
                parameters=self._mutate_parameters(base_action.parameters),
                expected_reward=base_action.expected_reward * 0.8,  # 稍微降低期望
                confidence=base_action.confidence * 0.9,
                exploration_value=self._calculate_exploration_value(
                    base_action.action_type, context
                ),
                metadata={
                    "source": "experience_based",
                    "base_action": base_action.action_id
                }
            )
            actions.append(action)
        
        return actions
    
    def _calculate_exploration_value(
        self,
        action_type: str,
        context: ExplorationContext
    ) -> float:
        """计算探索价值"""
        # 基于动作类型的基础价值
        base_values = {
            "click": 0.8,
            "input": 0.7,
            "swipe": 0.6,
            "wait": 0.3,
            "screenshot": 0.4,
            "locate": 0.5
        }
        
        base_value = base_values.get(action_type, 0.5)
        
        # 考虑动作的执行频率（较少执行的动作有更高的探索价值）
        action_count = self.action_counts.get(action_type, 0)
        frequency_bonus = max(0.0, 0.3 - action_count * 0.01)
        
        # 考虑约束条件
        constraint_penalty = 0.0
        if "forbidden_actions" in context.constraints:
            if action_type in context.constraints["forbidden_actions"]:
                constraint_penalty = 0.5
        
        exploration_value = base_value + frequency_bonus - constraint_penalty
        return max(0.0, min(1.0, exploration_value))
    
    def _generate_random_parameters(
        self,
        action_type: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成随机参数"""
        if action_type == "click":
            # 随机点击位置
            screen_width = current_state.get("screen_width", 1080)
            screen_height = current_state.get("screen_height", 1920)
            return {
                "x": random.randint(50, screen_width - 50),
                "y": random.randint(100, screen_height - 100),
                "duration": random.uniform(0.1, 0.5)
            }
        
        elif action_type == "input":
            # 随机输入文本
            sample_texts = ["test", "hello", "123", "sample", "input"]
            return {
                "text": random.choice(sample_texts),
                "clear_first": random.choice([True, False])
            }
        
        elif action_type == "swipe":
            # 随机滑动
            screen_width = current_state.get("screen_width", 1080)
            screen_height = current_state.get("screen_height", 1920)
            directions = ["up", "down", "left", "right"]
            return {
                "direction": random.choice(directions),
                "distance": random.randint(200, 800),
                "duration": random.uniform(0.3, 1.0)
            }
        
        elif action_type == "wait":
            return {
                "duration": random.uniform(0.5, 3.0)
            }
        
        else:
            return {}
    
    def _mutate_parameters(
        self,
        original_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """变异参数"""
        mutated_params = original_params.copy()
        
        for key, value in mutated_params.items():
            if isinstance(value, (int, float)):
                # 数值参数添加噪声
                noise_factor = 0.1
                noise = random.uniform(-noise_factor, noise_factor) * value
                mutated_params[key] = max(0, value + noise)
            elif isinstance(value, str) and key == "text":
                # 文本参数添加变体
                variations = [value, value.upper(), value.lower(), value + "_v2"]
                mutated_params[key] = random.choice(variations)
        
        return mutated_params
    
    def _deduplicate_actions(
        self,
        actions: List[ExplorationAction]
    ) -> List[ExplorationAction]:
        """去重动作"""
        seen_signatures = set()
        unique_actions = []
        
        for action in actions:
            # 创建动作签名
            signature = f"{action.action_type}_{hash(str(sorted(action.parameters.items())))}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_actions.append(action)
        
        # 按期望奖励排序
        unique_actions.sort(key=lambda x: x.expected_reward, reverse=True)
        
        return unique_actions
    
    async def _select_actions_by_strategy(
        self,
        candidate_actions: List[ExplorationAction],
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """根据策略选择动作"""
        strategy = context.strategy
        
        if strategy == ExplorationStrategy.GREEDY:
            return self._greedy_selection(candidate_actions, context)
        elif strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_selection(candidate_actions, context)
        elif strategy == ExplorationStrategy.UCB:
            return self._ucb_selection(candidate_actions, context)
        elif strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling_selection(candidate_actions, context)
        elif strategy == ExplorationStrategy.GUIDED_RANDOM:
            return self._guided_random_selection(candidate_actions, context)
        else:
            return self._greedy_selection(candidate_actions, context)
    
    def _greedy_selection(
        self,
        candidate_actions: List[ExplorationAction],
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """贪心选择"""
        # 按期望奖励排序，选择前N个
        sorted_actions = sorted(
            candidate_actions,
            key=lambda x: x.expected_reward,
            reverse=True
        )
        return sorted_actions[:context.exploration_budget]
    
    def _epsilon_greedy_selection(
        self,
        candidate_actions: List[ExplorationAction],
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """ε-贪心选择"""
        epsilon = self.strategy_params[ExplorationStrategy.EPSILON_GREEDY]["epsilon"]
        selected_actions = []
        
        for _ in range(min(context.exploration_budget, len(candidate_actions))):
            if random.random() < epsilon:
                # 随机选择
                action = random.choice(candidate_actions)
            else:
                # 贪心选择
                action = max(candidate_actions, key=lambda x: x.expected_reward)
            
            selected_actions.append(action)
            candidate_actions.remove(action)
        
        return selected_actions
    
    def _ucb_selection(
        self,
        candidate_actions: List[ExplorationAction],
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """上置信界选择"""
        c = self.strategy_params[ExplorationStrategy.UCB]["c"]
        total_count = sum(self.action_counts.values()) + 1
        
        # 计算UCB值
        for action in candidate_actions:
            action_count = self.action_counts.get(action.action_type, 0) + 1
            avg_reward = self.action_values.get(action.action_type, action.expected_reward)
            
            ucb_value = avg_reward + c * np.sqrt(np.log(total_count) / action_count)
            action.exploration_value = ucb_value
        
        # 按UCB值排序选择
        sorted_actions = sorted(
            candidate_actions,
            key=lambda x: x.exploration_value,
            reverse=True
        )
        
        return sorted_actions[:context.exploration_budget]
    
    def _thompson_sampling_selection(
        self,
        candidate_actions: List[ExplorationAction],
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """汤普森采样选择"""
        alpha = self.strategy_params[ExplorationStrategy.THOMPSON_SAMPLING]["alpha"]
        beta = self.strategy_params[ExplorationStrategy.THOMPSON_SAMPLING]["beta"]
        
        # 为每个动作采样奖励
        for action in candidate_actions:
            # 使用Beta分布采样
            sampled_reward = np.random.beta(alpha, beta)
            action.exploration_value = sampled_reward * action.expected_reward
        
        # 按采样值排序选择
        sorted_actions = sorted(
            candidate_actions,
            key=lambda x: x.exploration_value,
            reverse=True
        )
        
        return sorted_actions[:context.exploration_budget]
    
    def _guided_random_selection(
        self,
        candidate_actions: List[ExplorationAction],
        context: ExplorationContext
    ) -> List[ExplorationAction]:
        """引导随机选择"""
        knowledge_weight = self.strategy_params[ExplorationStrategy.GUIDED_RANDOM]["knowledge_weight"]
        
        # 分离基于知识的动作和随机动作
        knowledge_actions = [
            action for action in candidate_actions
            if action.metadata and action.metadata.get("source") == "prior_knowledge"
        ]
        other_actions = [
            action for action in candidate_actions
            if action not in knowledge_actions
        ]
        
        selected_actions = []
        budget = context.exploration_budget
        
        # 优先选择基于知识的动作
        knowledge_count = int(budget * knowledge_weight)
        if knowledge_actions:
            selected_knowledge = random.sample(
                knowledge_actions,
                min(knowledge_count, len(knowledge_actions))
            )
            selected_actions.extend(selected_knowledge)
        
        # 随机选择其他动作
        remaining_budget = budget - len(selected_actions)
        if remaining_budget > 0 and other_actions:
            selected_others = random.sample(
                other_actions,
                min(remaining_budget, len(other_actions))
            )
            selected_actions.extend(selected_others)
        
        return selected_actions
    
    async def _execute_exploration_action(
        self,
        action: ExplorationAction,
        context: ExplorationContext
    ) -> ExplorationResult:
        """执行探索动作"""
        start_time = datetime.now()
        
        try:
            # 模拟动作执行（实际实现中应该调用真实的执行器）
            success, observations = await self._simulate_action_execution(
                action, context.current_state
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 计算实际奖励
            actual_reward = self._calculate_actual_reward(
                action, success, observations, execution_time
            )
            
            # 提取学习洞察
            learned_insights = self._extract_learning_insights(
                action, success, observations
            )
            
            result = ExplorationResult(
                action=action,
                success=success,
                actual_reward=actual_reward,
                execution_time=execution_time,
                observations=observations,
                learned_insights=learned_insights
            )
            
            # 添加到探索记忆
            self.exploration_memory.append(result)
            if len(self.exploration_memory) > self.max_memory_size:
                self.exploration_memory = self.exploration_memory[-self.max_memory_size:]
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExplorationResult(
                action=action,
                success=False,
                actual_reward=0.0,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            return result
    
    async def _simulate_action_execution(
        self,
        action: ExplorationAction,
        current_state: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """模拟动作执行（实际实现中应该调用真实的执行器）"""
        # 简单的模拟逻辑
        await asyncio.sleep(0.1)  # 模拟执行时间
        
        # 基于动作类型和参数判断成功概率
        success_probability = 0.7  # 基础成功概率
        
        if action.knowledge_source:
            success_probability += 0.2  # 基于知识的动作有更高成功率
        
        if action.confidence > 0.8:
            success_probability += 0.1
        
        success = random.random() < success_probability
        
        observations = {
            "action_executed": True,
            "state_changed": success,
            "execution_method": "simulation",
            "timestamp": get_iso_timestamp()
        }
        
        return success, observations
    
    def _calculate_actual_reward(
        self,
        action: ExplorationAction,
        success: bool,
        observations: Dict[str, Any],
        execution_time: float
    ) -> float:
        """计算实际奖励"""
        if not success:
            return 0.0
        
        # 基础奖励
        base_reward = 0.5
        
        # 成功奖励
        success_reward = 0.3
        
        # 效率奖励（执行时间越短奖励越高）
        efficiency_reward = max(0.0, 0.2 - execution_time * 0.1)
        
        # 知识匹配奖励
        knowledge_reward = 0.0
        if action.knowledge_source:
            knowledge_reward = 0.1
        
        total_reward = base_reward + success_reward + efficiency_reward + knowledge_reward
        return min(1.0, total_reward)
    
    def _extract_learning_insights(
        self,
        action: ExplorationAction,
        success: bool,
        observations: Dict[str, Any]
    ) -> List[str]:
        """提取学习洞察"""
        insights = []
        
        if success:
            insights.append(f"动作 {action.action_type} 执行成功")
            
            if action.knowledge_source:
                insights.append(f"基于知识 {action.knowledge_source} 的动作有效")
            
            if action.confidence > 0.8:
                insights.append("高置信度动作表现良好")
        else:
            insights.append(f"动作 {action.action_type} 执行失败")
            
            if action.expected_reward > 0.7:
                insights.append("高期望奖励动作未达预期")
        
        return insights
    
    def _update_action_value(
        self,
        action: ExplorationAction,
        result: ExplorationResult
    ) -> None:
        """更新动作价值"""
        action_type = action.action_type
        
        # 更新计数
        if action_type not in self.action_counts:
            self.action_counts[action_type] = 0
        self.action_counts[action_type] += 1
        
        # 更新平均奖励
        if action_type not in self.action_values:
            self.action_values[action_type] = result.actual_reward
        else:
            count = self.action_counts[action_type]
            old_avg = self.action_values[action_type]
            self.action_values[action_type] = (old_avg * (count - 1) + result.actual_reward) / count
    
    def _should_stop_exploration(
        self,
        results: List[ExplorationResult],
        context: ExplorationContext
    ) -> bool:
        """判断是否应该停止探索"""
        if len(results) < 3:  # 至少执行3个动作
            return False
        
        # 如果最近3个动作都成功且奖励较高
        recent_results = results[-3:]
        if all(r.success and r.actual_reward > 0.7 for r in recent_results):
            return True
        
        # 如果找到了非常高奖励的动作
        if any(r.actual_reward > 0.9 for r in results):
            return True
        
        return False
    
    async def _analyze_exploration_results(
        self,
        results: List[ExplorationResult],
        context: ExplorationContext
    ) -> List[str]:
        """分析探索结果"""
        insights = []
        
        if not results:
            return insights
        
        # 成功率分析
        success_count = sum(1 for r in results if r.success)
        success_rate = success_count / len(results)
        insights.append(f"探索成功率: {success_rate:.2%}")
        
        # 平均奖励分析
        avg_reward = sum(r.actual_reward for r in results) / len(results)
        insights.append(f"平均奖励: {avg_reward:.3f}")
        
        # 最佳动作分析
        best_result = max(results, key=lambda r: r.actual_reward)
        insights.append(f"最佳动作: {best_result.action.action_type} (奖励: {best_result.actual_reward:.3f})")
        
        # 策略效果分析
        strategy_name = context.strategy.value
        insights.append(f"使用策略: {strategy_name}")
        
        # 知识利用分析
        knowledge_based_results = [
            r for r in results
            if r.action.knowledge_source is not None
        ]
        if knowledge_based_results:
            knowledge_success_rate = sum(1 for r in knowledge_based_results if r.success) / len(knowledge_based_results)
            insights.append(f"基于知识的动作成功率: {knowledge_success_rate:.2%}")
        
        return insights
    
    def _update_exploration_stats(
        self,
        results: List[ExplorationResult],
        context: ExplorationContext
    ) -> None:
        """更新探索统计信息"""
        self.exploration_stats["total_explorations"] += len(results)
        
        success_count = sum(1 for r in results if r.success)
        self.exploration_stats["successful_explorations"] += success_count
        
        # 更新平均奖励
        if results:
            total_reward = sum(r.actual_reward for r in results)
            current_avg = self.exploration_stats["average_reward"]
            total_explorations = self.exploration_stats["total_explorations"]
            
            self.exploration_stats["average_reward"] = (
                (current_avg * (total_explorations - len(results)) + total_reward) / total_explorations
            )
        
        # 更新策略性能
        strategy = context.strategy.value
        strategy_stats = self.exploration_stats["strategy_performance"][strategy]
        strategy_stats["count"] += len(results)
        
        if len(results) > 0:
            strategy_success_rate = success_count / len(results)
            strategy_avg_reward = sum(r.actual_reward for r in results) / len(results)
            
            # 更新策略成功率
            old_count = strategy_stats["count"] - len(results)
            if old_count > 0:
                old_success_rate = strategy_stats["success_rate"]
                strategy_stats["success_rate"] = (
                    (old_success_rate * old_count + strategy_success_rate * len(results)) / strategy_stats["count"]
                )
            else:
                strategy_stats["success_rate"] = strategy_success_rate
            
            # 更新策略平均奖励
            if old_count > 0:
                old_avg_reward = strategy_stats["avg_reward"]
                strategy_stats["avg_reward"] = (
                    (old_avg_reward * old_count + strategy_avg_reward * len(results)) / strategy_stats["count"]
                )
            else:
                strategy_stats["avg_reward"] = strategy_avg_reward
        
        # 更新动作性能
        for result in results:
            action_type = result.action.action_type
            if action_type not in self.exploration_stats["action_performance"]:
                self.exploration_stats["action_performance"][action_type] = {
                    "count": 0,
                    "success_count": 0,
                    "total_reward": 0.0
                }
            
            action_stats = self.exploration_stats["action_performance"][action_type]
            action_stats["count"] += 1
            if result.success:
                action_stats["success_count"] += 1
            action_stats["total_reward"] += result.actual_reward
        
        # 记录探索历史
        exploration_record = {
            "timestamp": get_iso_timestamp(),
            "task_description": context.task_description,
            "agent_id": context.agent_id,
            "strategy": context.strategy.value,
            "exploration_count": len(results),
            "success_count": success_count,
            "avg_reward": sum(r.actual_reward for r in results) / len(results) if results else 0.0
        }
        
        self.exploration_stats["exploration_history"].append(exploration_record)
        
        # 保持最近100条记录
        if len(self.exploration_stats["exploration_history"]) > 100:
            self.exploration_stats["exploration_history"] = self.exploration_stats["exploration_history"][-100:]
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """获取探索统计信息"""
        stats = self.exploration_stats.copy()
        
        # 计算总体成功率
        if stats["total_explorations"] > 0:
            stats["overall_success_rate"] = stats["successful_explorations"] / stats["total_explorations"]
        else:
            stats["overall_success_rate"] = 0.0
        
        # 计算动作性能统计
        for action_type, action_stats in stats["action_performance"].items():
            if action_stats["count"] > 0:
                action_stats["success_rate"] = action_stats["success_count"] / action_stats["count"]
                action_stats["avg_reward"] = action_stats["total_reward"] / action_stats["count"]
            else:
                action_stats["success_rate"] = 0.0
                action_stats["avg_reward"] = 0.0
        
        return stats
    
    async def clear_memory(self) -> None:
        """清理探索记忆"""
        self.exploration_memory.clear()
        self.action_values.clear()
        self.action_counts.clear()
        logger.info("引导探索器记忆已清理")