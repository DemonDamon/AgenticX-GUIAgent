#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task Synthesizer - 任务合成器 (基于AgenticX框架重构)

学习引擎第三阶段：基于探索结果合成新的任务和策略。

重构说明：
- 基于AgenticX的Component基类重构
- 使用AgenticX的事件系统进行任务合成通知
- 集成AgenticX的工作流组件进行任务编排
- 遵循AgenticX的模式识别和策略生成架构
"""

import asyncio
from loguru import logger
import json
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter

from agenticx.core.component import Component

from core.info_pool import InfoPool, InfoType, InfoPriority
from utils import get_iso_timestamp
from .prior_knowledge import KnowledgeMatch
from .guided_explorer import ExplorationResult, ExplorationAction


class SynthesisType(Enum):
    """合成类型"""
    TASK_DECOMPOSITION = "task_decomposition"  # 任务分解
    STRATEGY_COMBINATION = "strategy_combination"  # 策略组合
    PATTERN_EXTRACTION = "pattern_extraction"  # 模式提取
    WORKFLOW_OPTIMIZATION = "workflow_optimization"  # 工作流优化
    ERROR_RECOVERY = "error_recovery"  # 错误恢复


class SynthesisStrategy(Enum):
    """合成策略"""
    GREEDY = "greedy"  # 贪心策略
    OPTIMAL = "optimal"  # 最优策略
    HEURISTIC = "heuristic"  # 启发式策略
    RANDOM = "random"  # 随机策略
    ADAPTIVE = "adaptive"  # 自适应策略
    HYBRID = "hybrid"  # 混合策略


class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"  # 简单
    MEDIUM = "medium"  # 中等
    COMPLEX = "complex"  # 复杂
    VERY_COMPLEX = "very_complex"  # 非常复杂


@dataclass
class SynthesizedTask:
    """合成任务"""
    task_id: str
    task_type: str
    description: str
    steps: List[Dict[str, Any]]
    success_probability: float
    estimated_duration: float
    required_tools: List[str]
    preconditions: Dict[str, Any]
    postconditions: Dict[str, Any]
    synthesis_source: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesizedStrategy:
    """合成策略"""
    strategy_id: str
    strategy_type: str
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    success_rate: float
    efficiency_score: float
    applicability: Dict[str, Any]
    synthesis_source: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesizedPattern:
    """合成模式"""
    pattern_id: str
    pattern_type: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    action_sequence: List[Dict[str, Any]]
    success_indicators: List[str]
    failure_indicators: List[str]
    frequency: int
    reliability: float
    synthesis_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisContext:
    """合成上下文"""
    exploration_results: List[ExplorationResult]
    prior_knowledge: List[KnowledgeMatch]
    task_history: List[Dict[str, Any]]
    agent_id: str
    synthesis_types: List[SynthesisType]
    quality_threshold: float = 0.6
    max_synthesis_count: int = 10


@dataclass
class SynthesisResult:
    """合成结果"""
    synthesis_id: str
    synthesis_type: SynthesisType
    synthesized_tasks: List[SynthesizedTask]
    synthesized_strategies: List[SynthesizedStrategy]
    synthesized_patterns: List[SynthesizedPattern]
    quality_score: float
    confidence: float
    synthesis_time: float
    context: SynthesisContext
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=get_iso_timestamp)


class TaskSynthesizer(Component):
    """任务合成器
    
    负责：
    1. 从探索结果中提取成功模式
    2. 合成新的任务和子任务
    3. 组合和优化执行策略
    4. 生成工作流模板
    5. 创建错误恢复机制
    """
    
    def __init__(self, info_pool: InfoPool):
        super().__init__()
        self.info_pool = info_pool
        self.logger = logger
        
        # 合成统计
        self.synthesis_stats = {
            "total_synthesis": 0,
            "successful_synthesis": 0,
            "synthesis_by_type": {synthesis_type.value: 0 for synthesis_type in SynthesisType},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "synthesis_history": []
        }
        
        # 合成缓存
        self.synthesized_tasks = []
        self.synthesized_strategies = []
        self.synthesized_patterns = []
        
        # 模式识别参数
        self.pattern_recognition_params = {
            "min_frequency": 2,  # 最小出现频率
            "min_success_rate": 0.7,  # 最小成功率
            "similarity_threshold": 0.8,  # 相似度阈值
            "sequence_length_range": (2, 8)  # 序列长度范围
        }
        
        # 质量评估权重
        self.quality_weights = {
            "success_rate": 0.3,
            "efficiency": 0.25,
            "reliability": 0.2,
            "novelty": 0.15,
            "applicability": 0.1
        }
    
    async def synthesize(
        self,
        synthesis_context: SynthesisContext
    ) -> Dict[str, List[Any]]:
        """执行任务合成
        
        Args:
            synthesis_context: 合成上下文
        
        Returns:
            合成结果字典，包含任务、策略和模式
        """
        logger.info(f"开始任务合成，探索结果数量: {len(synthesis_context.exploration_results)}")
        
        synthesis_results = {
            "tasks": [],
            "strategies": [],
            "patterns": []
        }
        
        try:
            # 预处理探索结果
            processed_results = self._preprocess_exploration_results(
                synthesis_context.exploration_results
            )
            
            # 执行不同类型的合成
            for synthesis_type in synthesis_context.synthesis_types:
                logger.info(f"执行 {synthesis_type.value} 合成")
                
                if synthesis_type == SynthesisType.TASK_DECOMPOSITION:
                    tasks = await self._synthesize_tasks(processed_results, synthesis_context)
                    synthesis_results["tasks"].extend(tasks)
                
                elif synthesis_type == SynthesisType.STRATEGY_COMBINATION:
                    strategies = await self._synthesize_strategies(processed_results, synthesis_context)
                    synthesis_results["strategies"].extend(strategies)
                
                elif synthesis_type == SynthesisType.PATTERN_EXTRACTION:
                    patterns = await self._synthesize_patterns(processed_results, synthesis_context)
                    synthesis_results["patterns"].extend(patterns)
                
                elif synthesis_type == SynthesisType.WORKFLOW_OPTIMIZATION:
                    optimized_tasks = await self._synthesize_optimized_workflows(
                        processed_results, synthesis_context
                    )
                    synthesis_results["tasks"].extend(optimized_tasks)
                
                elif synthesis_type == SynthesisType.ERROR_RECOVERY:
                    recovery_strategies = await self._synthesize_error_recovery(
                        processed_results, synthesis_context
                    )
                    synthesis_results["strategies"].extend(recovery_strategies)
            
            # 质量过滤
            synthesis_results = self._filter_by_quality(
                synthesis_results, synthesis_context.quality_threshold
            )
            
            # 去重和排序
            synthesis_results = self._deduplicate_and_sort(synthesis_results)
            
            # 限制数量
            synthesis_results = self._limit_results(
                synthesis_results, synthesis_context.max_synthesis_count
            )
            
            # 更新缓存
            self._update_synthesis_cache(synthesis_results)
            
            # 更新统计信息
            self._update_synthesis_stats(synthesis_results, synthesis_context)
            
            # 发布合成结果
            self.info_pool.publish(
                InfoType.LEARNING_UPDATE,
                {
                    "stage": "task_synthesis",
                    "agent_id": synthesis_context.agent_id,
                    "synthesis_count": {
                        "tasks": len(synthesis_results["tasks"]),
                        "strategies": len(synthesis_results["strategies"]),
                        "patterns": len(synthesis_results["patterns"])
                    },
                    "quality_summary": self._get_quality_summary(synthesis_results)
                },
                source_agent="TaskSynthesizer",
                priority=InfoPriority.NORMAL
            )
            
            total_synthesized = sum(len(results) for results in synthesis_results.values())
            logger.info(f"任务合成完成，共生成 {total_synthesized} 个合成结果")
            
            return synthesis_results
            
        except Exception as e:
            logger.error(f"任务合成失败: {e}")
            return synthesis_results
    
    def _preprocess_exploration_results(
        self,
        exploration_results: List[ExplorationResult]
    ) -> Dict[str, Any]:
        """预处理探索结果"""
        processed = {
            "successful_results": [],
            "failed_results": [],
            "action_sequences": [],
            "performance_metrics": {},
            "error_patterns": [],
            "success_patterns": []
        }
        
        # 分类结果
        for result in exploration_results:
            if result.success:
                processed["successful_results"].append(result)
            else:
                processed["failed_results"].append(result)
        
        # 提取动作序列
        processed["action_sequences"] = self._extract_action_sequences(exploration_results)
        
        # 计算性能指标
        processed["performance_metrics"] = self._calculate_performance_metrics(exploration_results)
        
        # 识别错误模式
        processed["error_patterns"] = self._identify_error_patterns(processed["failed_results"])
        
        # 识别成功模式
        processed["success_patterns"] = self._identify_success_patterns(processed["successful_results"])
        
        return processed
    
    def _extract_action_sequences(
        self,
        exploration_results: List[ExplorationResult]
    ) -> List[List[Dict[str, Any]]]:
        """提取动作序列"""
        sequences = []
        current_sequence = []
        
        for result in exploration_results:
            action_info = {
                "action_type": result.action.action_type,
                "parameters": result.action.parameters,
                "success": result.success,
                "reward": result.actual_reward,
                "execution_time": result.execution_time
            }
            
            current_sequence.append(action_info)
            
            # 如果动作失败或序列过长，结束当前序列
            if not result.success or len(current_sequence) >= 10:
                if len(current_sequence) >= 2:  # 至少包含2个动作
                    sequences.append(current_sequence.copy())
                current_sequence = []
        
        # 添加最后一个序列
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        return sequences
    
    def _calculate_performance_metrics(
        self,
        exploration_results: List[ExplorationResult]
    ) -> Dict[str, Any]:
        """计算性能指标"""
        if not exploration_results:
            return {}
        
        # 基本统计
        total_count = len(exploration_results)
        success_count = sum(1 for r in exploration_results if r.success)
        total_reward = sum(r.actual_reward for r in exploration_results)
        total_time = sum(r.execution_time for r in exploration_results)
        
        # 按动作类型统计
        action_stats = defaultdict(lambda: {"count": 0, "success": 0, "total_reward": 0.0, "total_time": 0.0})
        
        for result in exploration_results:
            action_type = result.action.action_type
            stats = action_stats[action_type]
            stats["count"] += 1
            if result.success:
                stats["success"] += 1
            stats["total_reward"] += result.actual_reward
            stats["total_time"] += result.execution_time
        
        # 计算派生指标
        metrics = {
            "overall_success_rate": success_count / total_count,
            "average_reward": total_reward / total_count,
            "average_execution_time": total_time / total_count,
            "efficiency_score": (total_reward / total_time) if total_time > 0 else 0.0,
            "action_type_performance": {}
        }
        
        for action_type, stats in action_stats.items():
            metrics["action_type_performance"][action_type] = {
                "success_rate": stats["success"] / stats["count"],
                "average_reward": stats["total_reward"] / stats["count"],
                "average_time": stats["total_time"] / stats["count"],
                "efficiency": (stats["total_reward"] / stats["total_time"]) if stats["total_time"] > 0 else 0.0
            }
        
        return metrics
    
    def _identify_error_patterns(
        self,
        failed_results: List[ExplorationResult]
    ) -> List[Dict[str, Any]]:
        """识别错误模式"""
        error_patterns = []
        
        # 按错误类型分组
        error_groups = defaultdict(list)
        
        for result in failed_results:
            error_key = f"{result.action.action_type}_{result.error_message or 'unknown'}"
            error_groups[error_key].append(result)
        
        # 分析每个错误组
        for error_key, results in error_groups.items():
            if len(results) >= self.pattern_recognition_params["min_frequency"]:
                pattern = {
                    "pattern_type": "error",
                    "error_signature": error_key,
                    "frequency": len(results),
                    "common_parameters": self._find_common_parameters([r.action.parameters for r in results]),
                    "context_conditions": self._extract_context_conditions(results),
                    "suggested_fixes": self._suggest_error_fixes(results)
                }
                error_patterns.append(pattern)
        
        return error_patterns
    
    def _identify_success_patterns(
        self,
        successful_results: List[ExplorationResult]
    ) -> List[Dict[str, Any]]:
        """识别成功模式"""
        success_patterns = []
        
        # 按动作类型和奖励范围分组
        success_groups = defaultdict(list)
        
        for result in successful_results:
            reward_range = "high" if result.actual_reward > 0.8 else "medium" if result.actual_reward > 0.5 else "low"
            group_key = f"{result.action.action_type}_{reward_range}"
            success_groups[group_key].append(result)
        
        # 分析每个成功组
        for group_key, results in success_groups.items():
            if len(results) >= self.pattern_recognition_params["min_frequency"]:
                pattern = {
                    "pattern_type": "success",
                    "success_signature": group_key,
                    "frequency": len(results),
                    "average_reward": sum(r.actual_reward for r in results) / len(results),
                    "common_parameters": self._find_common_parameters([r.action.parameters for r in results]),
                    "optimal_conditions": self._extract_optimal_conditions(results),
                    "replication_guide": self._create_replication_guide(results)
                }
                success_patterns.append(pattern)
        
        return success_patterns
    
    def _find_common_parameters(
        self,
        parameter_lists: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """找到共同参数"""
        if not parameter_lists:
            return {}
        
        common_params = {}
        
        # 找到所有参数键的交集
        all_keys = set(parameter_lists[0].keys())
        for params in parameter_lists[1:]:
            all_keys &= set(params.keys())
        
        # 对于每个共同键，分析值的分布
        for key in all_keys:
            values = [params[key] for params in parameter_lists]
            
            if isinstance(values[0], (int, float)):
                # 数值参数：计算统计信息
                common_params[key] = {
                    "type": "numeric",
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2]
                }
            elif isinstance(values[0], str):
                # 字符串参数：找到最常见的值
                value_counts = Counter(values)
                most_common = value_counts.most_common(1)[0]
                if most_common[1] / len(values) > 0.5:  # 超过50%的频率
                    common_params[key] = {
                        "type": "categorical",
                        "most_common": most_common[0],
                        "frequency": most_common[1] / len(values)
                    }
        
        return common_params
    
    def _extract_context_conditions(
        self,
        results: List[ExplorationResult]
    ) -> Dict[str, Any]:
        """提取上下文条件"""
        conditions = {
            "execution_time_range": {
                "min": min(r.execution_time for r in results),
                "max": max(r.execution_time for r in results),
                "avg": sum(r.execution_time for r in results) / len(results)
            },
            "common_observations": self._find_common_observations(results)
        }
        
        return conditions
    
    def _find_common_observations(
        self,
        results: List[ExplorationResult]
    ) -> Dict[str, Any]:
        """找到共同观察"""
        all_observations = []
        for result in results:
            if result.observations:
                all_observations.append(result.observations)
        
        if not all_observations:
            return {}
        
        return self._find_common_parameters(all_observations)
    
    def _suggest_error_fixes(
        self,
        failed_results: List[ExplorationResult]
    ) -> List[str]:
        """建议错误修复方案"""
        fixes = []
        
        # 基于错误类型和参数模式建议修复
        common_params = self._find_common_parameters([r.action.parameters for r in failed_results])
        
        if "x" in common_params and "y" in common_params:
            fixes.append("调整点击坐标范围")
        
        if "duration" in common_params:
            fixes.append("优化执行时间参数")
        
        if "text" in common_params:
            fixes.append("检查输入文本格式")
        
        # 基于执行时间建议
        avg_time = sum(r.execution_time for r in failed_results) / len(failed_results)
        if avg_time > 2.0:
            fixes.append("减少执行超时时间")
        
        return fixes
    
    def _extract_optimal_conditions(
        self,
        successful_results: List[ExplorationResult]
    ) -> Dict[str, Any]:
        """提取最优条件"""
        # 找到奖励最高的结果
        best_results = sorted(successful_results, key=lambda r: r.actual_reward, reverse=True)[:3]
        
        optimal_conditions = {
            "best_parameters": self._find_common_parameters([r.action.parameters for r in best_results]),
            "optimal_timing": {
                "avg_execution_time": sum(r.execution_time for r in best_results) / len(best_results),
                "max_execution_time": max(r.execution_time for r in best_results)
            },
            "success_indicators": self._extract_success_indicators(best_results)
        }
        
        return optimal_conditions
    
    def _extract_success_indicators(
        self,
        results: List[ExplorationResult]
    ) -> List[str]:
        """提取成功指标"""
        indicators = []
        
        # 基于观察结果提取指标
        for result in results:
            if result.observations:
                if result.observations.get("state_changed"):
                    indicators.append("状态成功改变")
                if result.observations.get("action_executed"):
                    indicators.append("动作成功执行")
        
        # 基于奖励范围
        avg_reward = sum(r.actual_reward for r in results) / len(results)
        if avg_reward > 0.8:
            indicators.append("高奖励获得")
        
        return list(set(indicators))  # 去重
    
    def _create_replication_guide(
        self,
        successful_results: List[ExplorationResult]
    ) -> Dict[str, Any]:
        """创建复制指南"""
        guide = {
            "recommended_parameters": self._find_common_parameters([r.action.parameters for r in successful_results]),
            "execution_tips": [],
            "preconditions": [],
            "expected_outcomes": []
        }
        
        # 执行建议
        avg_time = sum(r.execution_time for r in successful_results) / len(successful_results)
        guide["execution_tips"].append(f"建议执行时间: {avg_time:.2f}秒")
        
        # 预期结果
        avg_reward = sum(r.actual_reward for r in successful_results) / len(successful_results)
        guide["expected_outcomes"].append(f"预期奖励: {avg_reward:.3f}")
        
        return guide
    
    async def _synthesize_tasks(
        self,
        processed_results: Dict[str, Any],
        context: SynthesisContext
    ) -> List[SynthesizedTask]:
        """合成任务"""
        synthesized_tasks = []
        
        # 基于成功的动作序列合成任务
        for sequence in processed_results["action_sequences"]:
            if self._is_successful_sequence(sequence):
                task = self._create_task_from_sequence(sequence, context)
                if task:
                    synthesized_tasks.append(task)
        
        # 基于成功模式合成任务
        for pattern in processed_results["success_patterns"]:
            task = self._create_task_from_pattern(pattern, context)
            if task:
                synthesized_tasks.append(task)
        
        return synthesized_tasks
    
    def _is_successful_sequence(
        self,
        sequence: List[Dict[str, Any]]
    ) -> bool:
        """判断是否为成功序列"""
        if len(sequence) < 2:
            return False
        
        success_count = sum(1 for action in sequence if action["success"])
        success_rate = success_count / len(sequence)
        
        return success_rate >= self.pattern_recognition_params["min_success_rate"]
    
    def _create_task_from_sequence(
        self,
        sequence: List[Dict[str, Any]],
        context: SynthesisContext
    ) -> Optional[SynthesizedTask]:
        """从序列创建任务"""
        try:
            # 生成任务描述
            action_types = [action["action_type"] for action in sequence]
            description = f"执行序列: {' -> '.join(action_types)}"
            
            # 创建步骤
            steps = []
            for i, action in enumerate(sequence):
                step = {
                    "step_id": i + 1,
                    "action_type": action["action_type"],
                    "parameters": action["parameters"],
                    "expected_duration": action["execution_time"],
                    "success_probability": 1.0 if action["success"] else 0.0
                }
                steps.append(step)
            
            # 计算整体指标
            success_probability = sum(action["success"] for action in sequence) / len(sequence)
            estimated_duration = sum(action["execution_time"] for action in sequence)
            
            # 提取所需工具
            required_tools = list(set(action["action_type"] for action in sequence))
            
            task = SynthesizedTask(
                task_id=f"seq_task_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                task_type="sequence",
                description=description,
                steps=steps,
                success_probability=success_probability,
                estimated_duration=estimated_duration,
                required_tools=required_tools,
                preconditions={"sequence_length": len(sequence)},
                postconditions={"expected_success_rate": success_probability},
                synthesis_source="action_sequence",
                confidence=self._calculate_task_confidence(sequence),
                metadata={
                    "original_sequence_length": len(sequence),
                    "synthesis_timestamp": get_iso_timestamp()
                }
            )
            
            return task
            
        except Exception as e:
            logger.warning(f"从序列创建任务失败: {e}")
            return None
    
    def _create_task_from_pattern(
        self,
        pattern: Dict[str, Any],
        context: SynthesisContext
    ) -> Optional[SynthesizedTask]:
        """从模式创建任务"""
        try:
            # 基于模式类型创建不同的任务
            if pattern["pattern_type"] == "success":
                action_type = pattern["success_signature"].split("_")[0]
                
                description = f"执行优化的 {action_type} 操作"
                
                # 创建单步任务
                steps = [{
                    "step_id": 1,
                    "action_type": action_type,
                    "parameters": pattern["common_parameters"],
                    "expected_duration": 1.0,  # 默认时长
                    "success_probability": pattern["average_reward"]
                }]
                
                task = SynthesizedTask(
                    task_id=f"pattern_task_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                    task_type="optimized_single",
                    description=description,
                    steps=steps,
                    success_probability=pattern["average_reward"],
                    estimated_duration=1.0,
                    required_tools=[action_type],
                    preconditions=pattern["optimal_conditions"],
                    postconditions={"expected_reward": pattern["average_reward"]},
                    synthesis_source="success_pattern",
                    confidence=min(1.0, pattern["frequency"] / 10.0),
                    metadata={
                        "pattern_frequency": pattern["frequency"],
                        "synthesis_timestamp": get_iso_timestamp()
                    }
                )
                
                return task
        
        except Exception as e:
            logger.warning(f"从模式创建任务失败: {e}")
            return None
    
    def _calculate_task_confidence(
        self,
        sequence: List[Dict[str, Any]]
    ) -> float:
        """计算任务置信度"""
        # 基于成功率
        success_rate = sum(action["success"] for action in sequence) / len(sequence)
        
        # 基于奖励
        avg_reward = sum(action["reward"] for action in sequence) / len(sequence)
        
        # 基于序列长度（适中长度更可信）
        length_score = 1.0 - abs(len(sequence) - 4) / 10.0  # 4步序列为最优
        length_score = max(0.0, length_score)
        
        confidence = (success_rate * 0.5 + avg_reward * 0.3 + length_score * 0.2)
        return min(1.0, confidence)
    
    async def _synthesize_strategies(
        self,
        processed_results: Dict[str, Any],
        context: SynthesisContext
    ) -> List[SynthesizedStrategy]:
        """合成策略"""
        synthesized_strategies = []
        
        # 基于性能指标合成策略
        performance_metrics = processed_results["performance_metrics"]
        
        for action_type, metrics in performance_metrics.get("action_type_performance", {}).items():
            if metrics["success_rate"] > 0.7:  # 高成功率的动作类型
                strategy = self._create_performance_strategy(action_type, metrics, context)
                if strategy:
                    synthesized_strategies.append(strategy)
        
        # 基于错误模式合成恢复策略
        for error_pattern in processed_results["error_patterns"]:
            strategy = self._create_recovery_strategy(error_pattern, context)
            if strategy:
                synthesized_strategies.append(strategy)
        
        return synthesized_strategies
    
    def _create_performance_strategy(
        self,
        action_type: str,
        metrics: Dict[str, Any],
        context: SynthesisContext
    ) -> Optional[SynthesizedStrategy]:
        """创建性能策略"""
        try:
            strategy = SynthesizedStrategy(
                strategy_id=f"perf_strategy_{action_type}_{get_iso_timestamp()}",
                strategy_type="performance_optimization",
                name=f"{action_type} 性能优化策略",
                description=f"基于历史数据优化 {action_type} 操作的执行策略",
                conditions={
                    "action_type": action_type,
                    "min_success_rate": 0.7
                },
                actions=[
                    {
                        "action": "optimize_parameters",
                        "target_success_rate": metrics["success_rate"],
                        "target_efficiency": metrics["efficiency"]
                    },
                    {
                        "action": "adjust_timing",
                        "optimal_duration": metrics["average_time"]
                    }
                ],
                success_rate=metrics["success_rate"],
                efficiency_score=metrics["efficiency"],
                applicability={"action_types": [action_type]},
                synthesis_source="performance_metrics",
                confidence=min(1.0, metrics["success_rate"]),
                metadata={
                    "original_metrics": metrics,
                    "synthesis_timestamp": get_iso_timestamp()
                }
            )
            
            return strategy
            
        except Exception as e:
            logger.warning(f"创建性能策略失败: {e}")
            return None
    
    def _create_recovery_strategy(
        self,
        error_pattern: Dict[str, Any],
        context: SynthesisContext
    ) -> Optional[SynthesizedStrategy]:
        """创建恢复策略"""
        try:
            strategy = SynthesizedStrategy(
                strategy_id=f"recovery_strategy_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                strategy_type="error_recovery",
                name=f"错误恢复策略: {error_pattern['error_signature']}",
                description=f"针对 {error_pattern['error_signature']} 错误的恢复策略",
                conditions={
                    "error_signature": error_pattern["error_signature"],
                    "min_frequency": error_pattern["frequency"]
                },
                actions=[
                    {"action": "detect_error", "signature": error_pattern["error_signature"]},
                    {"action": "apply_fixes", "fixes": error_pattern["suggested_fixes"]},
                    {"action": "retry_with_modifications", "max_retries": 3}
                ],
                success_rate=0.6,  # 恢复策略的估计成功率
                efficiency_score=0.5,  # 恢复策略的效率较低
                applicability={"error_types": [error_pattern["error_signature"]]},
                synthesis_source="error_pattern",
                confidence=min(1.0, error_pattern["frequency"] / 10.0),
                metadata={
                    "error_frequency": error_pattern["frequency"],
                    "synthesis_timestamp": get_iso_timestamp()
                }
            )
            
            return strategy
            
        except Exception as e:
            logger.warning(f"创建恢复策略失败: {e}")
            return None
    
    async def _synthesize_patterns(
        self,
        processed_results: Dict[str, Any],
        context: SynthesisContext
    ) -> List[SynthesizedPattern]:
        """合成模式"""
        synthesized_patterns = []
        
        # 从成功模式合成
        for success_pattern in processed_results["success_patterns"]:
            pattern = self._create_pattern_from_success(success_pattern, context)
            if pattern:
                synthesized_patterns.append(pattern)
        
        # 从动作序列合成
        frequent_sequences = self._find_frequent_sequences(processed_results["action_sequences"])
        for sequence_pattern in frequent_sequences:
            pattern = self._create_pattern_from_sequence(sequence_pattern, context)
            if pattern:
                synthesized_patterns.append(pattern)
        
        return synthesized_patterns
    
    def _create_pattern_from_success(
        self,
        success_pattern: Dict[str, Any],
        context: SynthesisContext
    ) -> Optional[SynthesizedPattern]:
        """从成功模式创建模式"""
        try:
            pattern = SynthesizedPattern(
                pattern_id=f"success_pattern_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                pattern_type="success_execution",
                name=f"成功执行模式: {success_pattern['success_signature']}",
                description=f"基于 {success_pattern['frequency']} 次成功执行提取的模式",
                trigger_conditions=success_pattern["optimal_conditions"],
                action_sequence=[
                    {
                        "action_type": success_pattern["success_signature"].split("_")[0],
                        "parameters": success_pattern["common_parameters"],
                        "expected_reward": success_pattern["average_reward"]
                    }
                ],
                success_indicators=["高奖励获得", "状态成功改变"],
                failure_indicators=["低奖励", "执行超时"],
                frequency=success_pattern["frequency"],
                reliability=success_pattern["average_reward"],
                synthesis_source="success_pattern",
                metadata={
                    "original_pattern": success_pattern,
                    "synthesis_timestamp": get_iso_timestamp()
                }
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"从成功模式创建模式失败: {e}")
            return None
    
    def _find_frequent_sequences(
        self,
        action_sequences: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """找到频繁序列"""
        # 简化的频繁序列挖掘
        sequence_patterns = defaultdict(int)
        
        for sequence in action_sequences:
            if len(sequence) >= 2:
                # 提取动作类型序列
                action_types = tuple(action["action_type"] for action in sequence)
                sequence_patterns[action_types] += 1
        
        # 过滤频繁序列
        frequent_patterns = []
        for pattern, frequency in sequence_patterns.items():
            if frequency >= self.pattern_recognition_params["min_frequency"]:
                frequent_patterns.append({
                    "pattern": pattern,
                    "frequency": frequency,
                    "length": len(pattern)
                })
        
        return frequent_patterns
    
    def _create_pattern_from_sequence(
        self,
        sequence_pattern: Dict[str, Any],
        context: SynthesisContext
    ) -> Optional[SynthesizedPattern]:
        """从序列模式创建模式"""
        try:
            pattern = SynthesizedPattern(
                pattern_id=f"seq_pattern_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                pattern_type="action_sequence",
                name=f"动作序列模式: {' -> '.join(sequence_pattern['pattern'])}",
                description=f"频繁出现的动作序列，出现 {sequence_pattern['frequency']} 次",
                trigger_conditions={"sequence_context": "general"},
                action_sequence=[
                    {"action_type": action_type, "order": i+1}
                    for i, action_type in enumerate(sequence_pattern["pattern"])
                ],
                success_indicators=["序列完整执行"],
                failure_indicators=["序列中断"],
                frequency=sequence_pattern["frequency"],
                reliability=min(1.0, sequence_pattern["frequency"] / 10.0),
                synthesis_source="frequent_sequence",
                metadata={
                    "sequence_length": sequence_pattern["length"],
                    "synthesis_timestamp": get_iso_timestamp()
                }
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"从序列模式创建模式失败: {e}")
            return None
    
    async def _synthesize_optimized_workflows(
        self,
        processed_results: Dict[str, Any],
        context: SynthesisContext
    ) -> List[SynthesizedTask]:
        """合成优化工作流"""
        optimized_tasks = []
        
        # 基于性能指标优化现有任务
        performance_metrics = processed_results["performance_metrics"]
        
        # 创建优化的复合任务
        if len(processed_results["action_sequences"]) > 0:
            optimized_task = self._create_optimized_composite_task(
                processed_results["action_sequences"],
                performance_metrics,
                context
            )
            if optimized_task:
                optimized_tasks.append(optimized_task)
        
        return optimized_tasks
    
    def _create_optimized_composite_task(
        self,
        action_sequences: List[List[Dict[str, Any]]],
        performance_metrics: Dict[str, Any],
        context: SynthesisContext
    ) -> Optional[SynthesizedTask]:
        """创建优化的复合任务"""
        try:
            # 选择最佳序列
            best_sequences = sorted(
                action_sequences,
                key=lambda seq: sum(action["reward"] for action in seq) / len(seq),
                reverse=True
            )[:3]
            
            if not best_sequences:
                return None
            
            # 合并最佳实践
            optimized_steps = []
            for i, sequence in enumerate(best_sequences):
                for j, action in enumerate(sequence):
                    step = {
                        "step_id": f"{i+1}_{j+1}",
                        "action_type": action["action_type"],
                        "parameters": action["parameters"],
                        "expected_duration": action["execution_time"],
                        "success_probability": 1.0 if action["success"] else 0.0,
                        "optimization_source": f"sequence_{i+1}"
                    }
                    optimized_steps.append(step)
            
            # 计算整体指标
            total_duration = sum(step["expected_duration"] for step in optimized_steps)
            avg_success_prob = sum(step["success_probability"] for step in optimized_steps) / len(optimized_steps)
            
            task = SynthesizedTask(
                task_id=f"optimized_workflow_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                task_type="optimized_workflow",
                description="基于最佳实践优化的复合工作流",
                steps=optimized_steps,
                success_probability=avg_success_prob,
                estimated_duration=total_duration,
                required_tools=list(set(step["action_type"] for step in optimized_steps)),
                preconditions={"optimization_based": True},
                postconditions={"expected_efficiency": performance_metrics.get("efficiency_score", 0.5)},
                synthesis_source="workflow_optimization",
                confidence=min(1.0, len(best_sequences) / 5.0),
                metadata={
                    "source_sequences_count": len(best_sequences),
                    "synthesis_timestamp": get_iso_timestamp()
                }
            )
            
            return task
            
        except Exception as e:
            logger.warning(f"创建优化复合任务失败: {e}")
            return None
    
    async def _synthesize_error_recovery(
        self,
        processed_results: Dict[str, Any],
        context: SynthesisContext
    ) -> List[SynthesizedStrategy]:
        """合成错误恢复策略"""
        recovery_strategies = []
        
        # 为每个错误模式创建恢复策略
        for error_pattern in processed_results["error_patterns"]:
            strategy = self._create_advanced_recovery_strategy(error_pattern, context)
            if strategy:
                recovery_strategies.append(strategy)
        
        return recovery_strategies
    
    def _create_advanced_recovery_strategy(
        self,
        error_pattern: Dict[str, Any],
        context: SynthesisContext
    ) -> Optional[SynthesizedStrategy]:
        """创建高级恢复策略"""
        try:
            # 基于错误频率和类型创建多层恢复策略
            recovery_actions = [
                {"action": "immediate_retry", "max_attempts": 2},
                {"action": "parameter_adjustment", "adjustments": error_pattern["suggested_fixes"]},
                {"action": "alternative_approach", "fallback_methods": ["manual_intervention"]},
                {"action": "error_reporting", "severity": "medium"}
            ]
            
            strategy = SynthesizedStrategy(
                strategy_id=f"advanced_recovery_{get_iso_timestamp()}_{random.randint(1000, 9999)}",
                strategy_type="advanced_error_recovery",
                name=f"高级错误恢复: {error_pattern['error_signature']}",
                description=f"多层次错误恢复策略，针对频繁出现的 {error_pattern['error_signature']} 错误",
                conditions={
                    "error_signature": error_pattern["error_signature"],
                    "frequency_threshold": error_pattern["frequency"],
                    "recovery_enabled": True
                },
                actions=recovery_actions,
                success_rate=0.75,  # 高级恢复策略的估计成功率
                efficiency_score=0.6,
                applicability={"error_types": [error_pattern["error_signature"]]},
                synthesis_source="advanced_error_analysis",
                confidence=min(1.0, error_pattern["frequency"] / 5.0),
                metadata={
                    "error_analysis": error_pattern,
                    "recovery_layers": len(recovery_actions),
                    "synthesis_timestamp": get_iso_timestamp()
                }
            )
            
            return strategy
            
        except Exception as e:
            logger.warning(f"创建高级恢复策略失败: {e}")
            return None
    
    def _filter_by_quality(
        self,
        synthesis_results: Dict[str, List[Any]],
        quality_threshold: float
    ) -> Dict[str, List[Any]]:
        """按质量过滤结果"""
        filtered_results = {"tasks": [], "strategies": [], "patterns": []}
        
        # 过滤任务
        for task in synthesis_results["tasks"]:
            quality_score = self._calculate_task_quality(task)
            if quality_score >= quality_threshold:
                filtered_results["tasks"].append(task)
        
        # 过滤策略
        for strategy in synthesis_results["strategies"]:
            quality_score = self._calculate_strategy_quality(strategy)
            if quality_score >= quality_threshold:
                filtered_results["strategies"].append(strategy)
        
        # 过滤模式
        for pattern in synthesis_results["patterns"]:
            quality_score = self._calculate_pattern_quality(pattern)
            if quality_score >= quality_threshold:
                filtered_results["patterns"].append(pattern)
        
        return filtered_results
    
    def _calculate_task_quality(
        self,
        task: SynthesizedTask
    ) -> float:
        """计算任务质量分数"""
        # 基于多个维度计算质量
        success_score = task.success_probability
        confidence_score = task.confidence
        complexity_score = min(1.0, len(task.steps) / 10.0)  # 适中复杂度更好
        
        quality_score = (
            success_score * self.quality_weights["success_rate"] +
            confidence_score * self.quality_weights["reliability"] +
            complexity_score * self.quality_weights["applicability"]
        )
        
        return min(1.0, quality_score)
    
    def _calculate_strategy_quality(
        self,
        strategy: SynthesizedStrategy
    ) -> float:
        """计算策略质量分数"""
        success_score = strategy.success_rate
        efficiency_score = strategy.efficiency_score
        confidence_score = strategy.confidence
        
        quality_score = (
            success_score * self.quality_weights["success_rate"] +
            efficiency_score * self.quality_weights["efficiency"] +
            confidence_score * self.quality_weights["reliability"]
        )
        
        return min(1.0, quality_score)
    
    def _calculate_pattern_quality(
        self,
        pattern: SynthesizedPattern
    ) -> float:
        """计算模式质量分数"""
        frequency_score = min(1.0, pattern.frequency / 10.0)
        reliability_score = pattern.reliability
        
        quality_score = (
            frequency_score * self.quality_weights["success_rate"] +
            reliability_score * self.quality_weights["reliability"]
        )
        
        return min(1.0, quality_score)
    
    def _deduplicate_and_sort(
        self,
        synthesis_results: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """去重和排序"""
        # 简单的去重逻辑（基于描述）
        deduplicated_results = {"tasks": [], "strategies": [], "patterns": []}
        
        # 去重任务
        seen_task_descriptions = set()
        for task in synthesis_results["tasks"]:
            if task.description not in seen_task_descriptions:
                seen_task_descriptions.add(task.description)
                deduplicated_results["tasks"].append(task)
        
        # 按成功概率排序
        deduplicated_results["tasks"].sort(key=lambda x: x.success_probability, reverse=True)
        
        # 去重策略
        seen_strategy_names = set()
        for strategy in synthesis_results["strategies"]:
            if strategy.name not in seen_strategy_names:
                seen_strategy_names.add(strategy.name)
                deduplicated_results["strategies"].append(strategy)
        
        # 按成功率排序
        deduplicated_results["strategies"].sort(key=lambda x: x.success_rate, reverse=True)
        
        # 去重模式
        seen_pattern_names = set()
        for pattern in synthesis_results["patterns"]:
            if pattern.name not in seen_pattern_names:
                seen_pattern_names.add(pattern.name)
                deduplicated_results["patterns"].append(pattern)
        
        # 按频率排序
        deduplicated_results["patterns"].sort(key=lambda x: x.frequency, reverse=True)
        
        return deduplicated_results
    
    def _limit_results(
        self,
        synthesis_results: Dict[str, List[Any]],
        max_count: int
    ) -> Dict[str, List[Any]]:
        """限制结果数量"""
        limited_results = {}
        
        for result_type, results in synthesis_results.items():
            limited_results[result_type] = results[:max_count]
        
        return limited_results
    
    def _update_synthesis_cache(
        self,
        synthesis_results: Dict[str, List[Any]]
    ) -> None:
        """更新合成缓存"""
        self.synthesized_tasks.extend(synthesis_results["tasks"])
        self.synthesized_strategies.extend(synthesis_results["strategies"])
        self.synthesized_patterns.extend(synthesis_results["patterns"])
        
        # 限制缓存大小
        max_cache_size = 100
        if len(self.synthesized_tasks) > max_cache_size:
            self.synthesized_tasks = self.synthesized_tasks[-max_cache_size:]
        if len(self.synthesized_strategies) > max_cache_size:
            self.synthesized_strategies = self.synthesized_strategies[-max_cache_size:]
        if len(self.synthesized_patterns) > max_cache_size:
            self.synthesized_patterns = self.synthesized_patterns[-max_cache_size:]
    
    def _get_quality_summary(
        self,
        synthesis_results: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """获取质量摘要"""
        summary = {
            "total_count": sum(len(results) for results in synthesis_results.values()),
            "by_type": {}
        }
        
        for result_type, results in synthesis_results.items():
            if results:
                if result_type == "tasks":
                    avg_success_prob = sum(task.success_probability for task in results) / len(results)
                    avg_confidence = sum(task.confidence for task in results) / len(results)
                    summary["by_type"][result_type] = {
                        "count": len(results),
                        "avg_success_probability": avg_success_prob,
                        "avg_confidence": avg_confidence
                    }
                elif result_type == "strategies":
                    avg_success_rate = sum(strategy.success_rate for strategy in results) / len(results)
                    avg_efficiency = sum(strategy.efficiency_score for strategy in results) / len(results)
                    summary["by_type"][result_type] = {
                        "count": len(results),
                        "avg_success_rate": avg_success_rate,
                        "avg_efficiency": avg_efficiency
                    }
                elif result_type == "patterns":
                    avg_frequency = sum(pattern.frequency for pattern in results) / len(results)
                    avg_reliability = sum(pattern.reliability for pattern in results) / len(results)
                    summary["by_type"][result_type] = {
                        "count": len(results),
                        "avg_frequency": avg_frequency,
                        "avg_reliability": avg_reliability
                    }
        
        return summary
    
    def _update_synthesis_stats(
        self,
        synthesis_results: Dict[str, List[Any]],
        context: SynthesisContext
    ) -> None:
        """更新合成统计信息"""
        total_synthesized = sum(len(results) for results in synthesis_results.values())
        
        self.synthesis_stats["total_synthesis"] += total_synthesized
        if total_synthesized > 0:
            self.synthesis_stats["successful_synthesis"] += 1
        
        # 按类型统计
        for synthesis_type in context.synthesis_types:
            self.synthesis_stats["synthesis_by_type"][synthesis_type.value] += 1
        
        # 质量分布统计
        for result_type, results in synthesis_results.items():
            for result in results:
                if result_type == "tasks":
                    quality = self._calculate_task_quality(result)
                elif result_type == "strategies":
                    quality = self._calculate_strategy_quality(result)
                elif result_type == "patterns":
                    quality = self._calculate_pattern_quality(result)
                else:
                    quality = 0.5
                
                if quality >= 0.8:
                    self.synthesis_stats["quality_distribution"]["high"] += 1
                elif quality >= 0.6:
                    self.synthesis_stats["quality_distribution"]["medium"] += 1
                else:
                    self.synthesis_stats["quality_distribution"]["low"] += 1
        
        # 记录合成历史
        synthesis_record = {
            "timestamp": get_iso_timestamp(),
            "agent_id": context.agent_id,
            "synthesis_types": [st.value for st in context.synthesis_types],
            "results_count": {k: len(v) for k, v in synthesis_results.items()},
            "exploration_results_count": len(context.exploration_results)
        }
        
        self.synthesis_stats["synthesis_history"].append(synthesis_record)
        
        # 保持最近50条记录
        if len(self.synthesis_stats["synthesis_history"]) > 50:
            self.synthesis_stats["synthesis_history"] = self.synthesis_stats["synthesis_history"][-50:]
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """获取合成统计信息"""
        stats = self.synthesis_stats.copy()
        
        # 计算成功率
        if stats["total_synthesis"] > 0:
            stats["synthesis_success_rate"] = stats["successful_synthesis"] / stats["total_synthesis"]
        else:
            stats["synthesis_success_rate"] = 0.0
        
        # 添加缓存信息
        stats["cache_info"] = {
            "tasks_cached": len(self.synthesized_tasks),
            "strategies_cached": len(self.synthesized_strategies),
            "patterns_cached": len(self.synthesized_patterns)
        }
        
        return stats
    
    def get_synthesized_tasks(self) -> List[SynthesizedTask]:
        """获取合成的任务"""
        return self.synthesized_tasks.copy()
    
    def get_synthesized_strategies(self) -> List[SynthesizedStrategy]:
        """获取合成的策略"""
        return self.synthesized_strategies.copy()
    
    def get_synthesized_patterns(self) -> List[SynthesizedPattern]:
        """获取合成的模式"""
        return self.synthesized_patterns.copy()
    
    def clear_synthesis_cache(self) -> None:
        """清空合成缓存"""
        self.synthesized_tasks.clear()
        self.synthesized_strategies.clear()
        self.synthesized_patterns.clear()
        logger.info("合成缓存已清空")
    
    async def shutdown(self) -> None:
        """关闭任务合成器"""
        try:
            # 保存合成统计信息
            if hasattr(self, 'info_pool') and self.info_pool:
                await self.info_pool.publish_info(
                    InfoType.LEARNING_UPDATE,
                    {
                        "agent_id": "task_synthesizer",
                        "update_type": "shutdown",
                        "synthesis_stats": self.get_synthesis_stats(),
                        "timestamp": get_iso_timestamp()
                    },
                    InfoPriority.LOW
                )
            
            logger.info("任务合成器已关闭")
            
        except Exception as e:
            logger.error(f"关闭任务合成器时出错: {e}")