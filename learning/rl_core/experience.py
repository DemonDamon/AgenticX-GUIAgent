#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M9: 经验管理系统 - 经验收集、回放、共享

基于DQN Experience Replay和Prioritized Experience Replay设计，
提供多智能体经验管理、优先级采样、经验共享等功能。

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
import pickle
import random
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Experience:
    """经验数据结构"""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    agent_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 优先级相关
    priority: float = 1.0
    td_error: Optional[float] = None
    
    # 多智能体相关
    episode_id: Optional[str] = None
    step_id: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # 确保张量在CPU上（便于存储）
        if isinstance(self.state, torch.Tensor) and self.state.device != torch.device('cpu'):
            self.state = self.state.cpu()
        if isinstance(self.next_state, torch.Tensor) and self.next_state.device != torch.device('cpu'):
            self.next_state = self.next_state.cpu()
        if isinstance(self.action, torch.Tensor) and self.action.device != torch.device('cpu'):
            self.action = self.action.cpu()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'state': self.state.numpy() if isinstance(self.state, torch.Tensor) else self.state,
            'action': self.action.numpy() if isinstance(self.action, torch.Tensor) else self.action,
            'reward': self.reward,
            'next_state': self.next_state.numpy() if isinstance(self.next_state, torch.Tensor) else self.next_state,
            'done': self.done,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'priority': self.priority,
            'td_error': self.td_error,
            'episode_id': self.episode_id,
            'step_id': self.step_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """从字典创建经验"""
        return cls(
            state=torch.tensor(data['state']) if isinstance(data['state'], np.ndarray) else data['state'],
            action=torch.tensor(data['action']) if isinstance(data['action'], np.ndarray) else data['action'],
            reward=data['reward'],
            next_state=torch.tensor(data['next_state']) if isinstance(data['next_state'], np.ndarray) else data['next_state'],
            done=data['done'],
            agent_id=data['agent_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            priority=data.get('priority', 1.0),
            td_error=data.get('td_error'),
            episode_id=data.get('episode_id'),
            step_id=data.get('step_id', 0)
        )


class SumTree:
    """用于优先级采样的Sum Tree数据结构"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """检索叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """获取总优先级"""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """添加数据"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """根据优先级获取数据"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self,
                 capacity: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 enable_prioritized: bool = True,
                 save_path: Optional[str] = None):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # 防止零优先级
        self.enable_prioritized = enable_prioritized
        self.save_path = save_path
        
        # 存储结构
        if enable_prioritized:
            self.tree = SumTree(capacity)
        else:
            self.buffer = deque(maxlen=capacity)
        
        # 统计信息
        self.total_added = 0
        self.agent_stats = defaultdict(int)
        self.reward_stats = {'total': 0.0, 'count': 0, 'max': float('-inf'), 'min': float('inf')}
        
        # 线程安全
        self.lock = threading.RLock()
        
        self.logger = logger
    
    def add(self, experience: Experience) -> None:
        """添加经验"""
        with self.lock:
            # 计算初始优先级
            if self.enable_prioritized:
                max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
                priority = max_priority if experience.priority is None else experience.priority
                self.tree.add(priority, experience)
            else:
                self.buffer.append(experience)
            
            # 更新统计信息
            self.total_added += 1
            self.agent_stats[experience.agent_id] += 1
            
            self.reward_stats['total'] += experience.reward
            self.reward_stats['count'] += 1
            self.reward_stats['max'] = max(self.reward_stats['max'], experience.reward)
            self.reward_stats['min'] = min(self.reward_stats['min'], experience.reward)
            
            # 定期保存
            if self.save_path and self.total_added % 1000 == 0:
                self._save_buffer()
    
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样"""
        with self.lock:
            if self.enable_prioritized:
                return self._prioritized_sample(batch_size)[0]
            else:
                if len(self.buffer) < batch_size:
                    return list(self.buffer)
                return random.sample(list(self.buffer), batch_size)
    
    def prioritized_sample(self, batch_size: int, alpha: Optional[float] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """优先级采样"""
        if not self.enable_prioritized:
            experiences = self.sample(batch_size)
            weights = np.ones(len(experiences))
            indices = np.arange(len(experiences))
            return experiences, weights, indices
        
        return self._prioritized_sample(batch_size, alpha)
    
    def _prioritized_sample(self, batch_size: int, alpha: Optional[float] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """内部优先级采样实现"""
        if alpha is None:
            alpha = self.alpha
        
        experiences = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            idx, priority, experience = self.tree.get(s)
            
            if experience is not None:
                experiences.append(experience)
                indices.append(idx)
                priorities.append(priority)
        
        # 计算重要性采样权重
        if len(priorities) > 0:
            sampling_probabilities = np.array(priorities) / self.tree.total()
            weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
            weights /= weights.max()  # 归一化
        else:
            weights = np.ones(len(experiences))
        
        return experiences, weights, np.array(indices)
    
    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """更新优先级"""
        if not self.enable_prioritized:
            return
        
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                priority = (abs(td_error) + self.epsilon) ** self.alpha
                self.tree.update(idx, priority)
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        with self.lock:
            size = self.tree.n_entries if self.enable_prioritized else len(self.buffer)
            
            avg_reward = self.reward_stats['total'] / max(1, self.reward_stats['count'])
            
            return {
                'size': size,
                'capacity': self.capacity,
                'total_added': self.total_added,
                'agent_stats': dict(self.agent_stats),
                'reward_stats': {
                    'average': avg_reward,
                    'max': self.reward_stats['max'],
                    'min': self.reward_stats['min'],
                    'total': self.reward_stats['total']
                },
                'prioritized_enabled': self.enable_prioritized,
                'beta': self.beta if self.enable_prioritized else None
            }
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            if self.enable_prioritized:
                self.tree = SumTree(self.capacity)
            else:
                self.buffer.clear()
            
            self.total_added = 0
            self.agent_stats.clear()
            self.reward_stats = {'total': 0.0, 'count': 0, 'max': float('-inf'), 'min': float('inf')}
    
    def _save_buffer(self):
        """保存缓冲区到磁盘"""
        if not self.save_path:
            return
        
        try:
            save_data = {
                'experiences': self.get_all_experiences()[:1000],  # 只保存最近1000个
                'stats': self.get_buffer_statistics(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"缓冲区已保存到: {self.save_path}")
        except Exception as e:
            logger.error(f"保存缓冲区失败: {e}")
    
    def load_buffer(self, load_path: str):
        """从磁盘加载缓冲区"""
        try:
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)
            
            experiences = save_data.get('experiences', [])
            for exp in experiences:
                if isinstance(exp, dict):
                    exp = Experience.from_dict(exp)
                self.add(exp)
            
            logger.info(f"从 {load_path} 加载了 {len(experiences)} 个经验")
        except Exception as e:
            logger.error(f"加载缓冲区失败: {e}")
    
    def get_all_experiences(self) -> List[Experience]:
        """获取所有经验"""
        with self.lock:
            if self.enable_prioritized:
                return [exp for exp in self.tree.data[:self.tree.n_entries] if exp is not None]
            else:
                return list(self.buffer)
    
    def __len__(self) -> int:
        with self.lock:
            return self.tree.n_entries if self.enable_prioritized else len(self.buffer)


class ExperienceSharingHub:
    """经验共享中心"""
    
    def __init__(self,
                 max_shared_experiences: int = 10000,
                 sharing_strategies: Optional[List[str]] = None,
                 relevance_threshold: float = 0.5):
        self.max_shared_experiences = max_shared_experiences
        self.sharing_strategies = sharing_strategies or ['similarity', 'diversity', 'performance']
        self.relevance_threshold = relevance_threshold
        
        # 共享经验存储
        self.shared_experiences = defaultdict(deque)  # agent_id -> experiences
        self.global_experience_pool = deque(maxlen=max_shared_experiences)
        
        # 共享统计
        self.sharing_stats = defaultdict(lambda: {'sent': 0, 'received': 0})
        
        # 相关性计算缓存
        self.relevance_cache = {}
        
        # 线程安全
        self.lock = threading.RLock()
        
        self.logger = logger
    
    def share_experience(self, 
                        source_agent: str, 
                        target_agents: List[str], 
                        experience: Experience) -> None:
        """共享经验"""
        with self.lock:
            # 添加到全局池
            self.global_experience_pool.append(experience)
            
            # 分享给目标智能体
            for target_agent in target_agents:
                if target_agent != source_agent:
                    self.shared_experiences[target_agent].append(experience)
                    
                    # 限制每个智能体的共享经验数量
                    if len(self.shared_experiences[target_agent]) > self.max_shared_experiences // 4:
                        self.shared_experiences[target_agent].popleft()
            
            # 更新统计
            self.sharing_stats[source_agent]['sent'] += len(target_agents)
            for target_agent in target_agents:
                if target_agent != source_agent:
                    self.sharing_stats[target_agent]['received'] += 1
            
            logger.debug(f"经验已从 {source_agent} 共享给 {target_agents}")
    
    def aggregate_multi_agent_experiences(self, 
                                        agent_experiences: Dict[str, List[Experience]]) -> List[Experience]:
        """聚合多智能体经验"""
        with self.lock:
            aggregated = []
            
            # 收集所有经验
            all_experiences = []
            for agent_id, experiences in agent_experiences.items():
                for exp in experiences:
                    exp.metadata['original_agent'] = agent_id
                    all_experiences.append(exp)
            
            # 按时间排序
            all_experiences.sort(key=lambda x: x.timestamp)
            
            # 应用聚合策略
            aggregated = self._apply_aggregation_strategy(all_experiences)
            
            return aggregated
    
    def _apply_aggregation_strategy(self, experiences: List[Experience]) -> List[Experience]:
        """应用聚合策略"""
        if not experiences:
            return []
        
        # 简单策略：保留所有经验，但添加聚合元数据
        for i, exp in enumerate(experiences):
            exp.metadata['aggregation_index'] = i
            exp.metadata['total_experiences'] = len(experiences)
        
        return experiences
    
    def filter_relevant_experiences(self, 
                                  experiences: List[Experience], 
                                  context: Dict) -> List[Experience]:
        """过滤相关经验"""
        if not experiences:
            return []
        
        relevant_experiences = []
        
        for exp in experiences:
            relevance_score = self._calculate_relevance(exp, context)
            
            if relevance_score >= self.relevance_threshold:
                exp.metadata['relevance_score'] = relevance_score
                relevant_experiences.append(exp)
        
        # 按相关性排序
        relevant_experiences.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        
        return relevant_experiences
    
    def _calculate_relevance(self, experience: Experience, context: Dict) -> float:
        """计算经验相关性"""
        # 创建缓存键
        cache_key = (id(experience), str(sorted(context.items())))
        
        if cache_key in self.relevance_cache:
            return self.relevance_cache[cache_key]
        
        relevance_score = 0.0
        
        # 智能体类型相关性
        if 'agent_type' in context and experience.agent_id == context['agent_type']:
            relevance_score += 0.3
        
        # 任务类型相关性
        if 'task_type' in context and 'task_type' in experience.metadata:
            if experience.metadata['task_type'] == context['task_type']:
                relevance_score += 0.3
        
        # 奖励相关性
        if 'target_reward_range' in context:
            min_reward, max_reward = context['target_reward_range']
            if min_reward <= experience.reward <= max_reward:
                relevance_score += 0.2
        
        # 时间相关性
        if 'time_window' in context:
            time_diff = abs((datetime.now() - experience.timestamp).total_seconds())
            if time_diff <= context['time_window']:
                relevance_score += 0.2
        
        # 缓存结果
        self.relevance_cache[cache_key] = relevance_score
        
        # 限制缓存大小
        if len(self.relevance_cache) > 10000:
            # 删除最旧的一半缓存
            keys_to_remove = list(self.relevance_cache.keys())[:5000]
            for key in keys_to_remove:
                del self.relevance_cache[key]
        
        return relevance_score
    
    def get_shared_experiences(self, agent_id: str, max_count: Optional[int] = None) -> List[Experience]:
        """获取共享给特定智能体的经验"""
        with self.lock:
            experiences = list(self.shared_experiences[agent_id])
            
            if max_count and len(experiences) > max_count:
                # 返回最新的经验
                experiences = experiences[-max_count:]
            
            return experiences
    
    def get_global_experiences(self, max_count: Optional[int] = None) -> List[Experience]:
        """获取全局经验池"""
        with self.lock:
            experiences = list(self.global_experience_pool)
            
            if max_count and len(experiences) > max_count:
                experiences = experiences[-max_count:]
            
            return experiences
    
    def get_sharing_statistics(self) -> Dict[str, Any]:
        """获取共享统计信息"""
        with self.lock:
            return {
                'sharing_stats': dict(self.sharing_stats),
                'global_pool_size': len(self.global_experience_pool),
                'shared_experiences_per_agent': {
                    agent_id: len(experiences) 
                    for agent_id, experiences in self.shared_experiences.items()
                },
                'total_shared_experiences': sum(
                    len(experiences) for experiences in self.shared_experiences.values()
                ),
                'cache_size': len(self.relevance_cache)
            }
    
    def clear_shared_experiences(self, agent_id: Optional[str] = None):
        """清空共享经验"""
        with self.lock:
            if agent_id:
                self.shared_experiences[agent_id].clear()
                self.sharing_stats[agent_id] = {'sent': 0, 'received': 0}
            else:
                self.shared_experiences.clear()
                self.global_experience_pool.clear()
                self.sharing_stats.clear()
                self.relevance_cache.clear()


class ExperienceDataset(Dataset):
    """经验数据集（用于批量训练）"""
    
    def __init__(self, experiences: List[Experience]):
        self.experiences = experiences
    
    def __len__(self) -> int:
        return len(self.experiences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        exp = self.experiences[idx]
        
        return {
            'state': exp.state,
            'action': exp.action,
            'reward': torch.tensor(exp.reward, dtype=torch.float32),
            'next_state': exp.next_state,
            'done': torch.tensor(exp.done, dtype=torch.bool),
            'agent_id': exp.agent_id,
            'priority': torch.tensor(exp.priority, dtype=torch.float32)
        }


# 工具函数
def create_experience_buffer(config: Dict[str, Any]) -> ExperienceReplayBuffer:
    """创建经验缓冲区"""
    return ExperienceReplayBuffer(
        capacity=config.get('capacity', 100000),
        alpha=config.get('alpha', 0.6),
        beta=config.get('beta', 0.4),
        beta_increment=config.get('beta_increment', 0.001),
        epsilon=config.get('epsilon', 1e-6),
        enable_prioritized=config.get('enable_prioritized', True),
        save_path=config.get('save_path')
    )


def create_sharing_hub(config: Dict[str, Any]) -> ExperienceSharingHub:
    """创建经验共享中心"""
    return ExperienceSharingHub(
        max_shared_experiences=config.get('max_shared_experiences', 10000),
        sharing_strategies=config.get('sharing_strategies'),
        relevance_threshold=config.get('relevance_threshold', 0.5)
    )


def create_experience_dataloader(experiences: List[Experience], 
                               batch_size: int = 32, 
                               shuffle: bool = True) -> DataLoader:
    """创建经验数据加载器"""
    dataset = ExperienceDataset(experiences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


async def collect_experiences_async(agents: List[Any], 
                                  environment: Any, 
                                  num_episodes: int = 10) -> List[Experience]:
    """异步收集经验"""
    experiences = []
    
    for episode in range(num_episodes):
        state = environment.reset()
        done = False
        step = 0
        
        while not done:
            for agent in agents:
                # 智能体选择动作
                action = await agent.select_action_async(state)
                
                # 执行动作
                next_state, reward, done, info = environment.step(action)
                
                # 创建经验
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    agent_id=agent.agent_id,
                    timestamp=datetime.now(),
                    episode_id=f"episode_{episode}",
                    step_id=step
                )
                
                experiences.append(experience)
                
                state = next_state
                step += 1
                
                if done:
                    break
    
    return experiences