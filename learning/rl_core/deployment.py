#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M12: 监控部署系统 - MLOps和安全保障

基于MLOps、Model Deployment、A/B Testing等最佳实践，
提供学习过程监控、策略部署管理、安全保障机制。

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
from loguru import logger
import os
import pickle
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .policies import BasePolicyNetwork
from .experience import Experience


class DeploymentStatus(Enum):
    """部署状态"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    TESTING = "testing"
    FAILED = "failed"
    ROLLBACK = "rollback"
    RETIRED = "retired"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DeploymentConfig:
    """部署配置"""
    deployment_id: str
    policy_path: str
    target_agents: List[str]
    rollout_percentage: float = 100.0
    canary_percentage: float = 10.0
    monitoring_duration: int = 3600  # 秒
    auto_rollback: bool = True
    performance_threshold: float = 0.8
    safety_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringMetrics:
    """监控指标"""
    timestamp: datetime
    agent_id: str
    policy_version: str
    
    # 性能指标
    success_rate: float = 0.0
    average_reward: float = 0.0
    execution_time: float = 0.0
    error_rate: float = 0.0
    
    # 资源指标
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # 业务指标
    task_completion_rate: float = 0.0
    user_satisfaction: float = 0.0
    
    # 安全指标
    safety_violations: int = 0
    anomaly_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'policy_version': self.policy_version,
            'success_rate': self.success_rate,
            'average_reward': self.average_reward,
            'execution_time': self.execution_time,
            'error_rate': self.error_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'task_completion_rate': self.task_completion_rate,
            'user_satisfaction': self.user_satisfaction,
            'safety_violations': self.safety_violations,
            'anomaly_score': self.anomaly_score
        }


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    agent_id: Optional[str] = None
    policy_version: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'policy_version': self.policy_version,
            'metrics': self.metrics,
            'resolved': self.resolved
        }


class LearningMonitor:
    """学习监控器"""
    
    def __init__(self,
                 log_dir: str = "./logs",
                 metrics_retention_days: int = 30,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 enable_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_retention_days = metrics_retention_days
        self.alert_thresholds = alert_thresholds or {
            'success_rate_min': 0.7,
            'error_rate_max': 0.1,
            'cpu_usage_max': 0.8,
            'memory_usage_max': 0.8,
            'anomaly_score_max': 0.8
        }
        
        # 监控数据存储
        self.metrics_history = defaultdict(deque)  # agent_id -> metrics
        self.alerts = deque(maxlen=1000)
        self.performance_baselines = {}
        
        # TensorBoard支持
        self.enable_tensorboard = enable_tensorboard
        if enable_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        else:
            self.tb_writer = None
        
        # 监控状态
        self.monitoring_active = True
        self.last_cleanup = datetime.now()
        
        self.logger = logger
        
        # 启动后台监控任务
        self._start_background_tasks()
    
    def monitor_learning_progress(self, 
                                 metrics: Dict[str, float], 
                                 policies: Dict[str, BasePolicyNetwork]) -> None:
        """监控学习进度"""
        timestamp = datetime.now()
        
        for agent_id, policy in policies.items():
            # 创建监控指标
            monitoring_metrics = MonitoringMetrics(
                timestamp=timestamp,
                agent_id=agent_id,
                policy_version=self._get_policy_version(policy),
                success_rate=metrics.get(f'{agent_id}_success_rate', 0.0),
                average_reward=metrics.get(f'{agent_id}_avg_reward', 0.0),
                execution_time=metrics.get(f'{agent_id}_exec_time', 0.0),
                error_rate=metrics.get(f'{agent_id}_error_rate', 0.0),
                cpu_usage=metrics.get('cpu_usage', 0.0),
                memory_usage=metrics.get('memory_usage', 0.0),
                gpu_usage=metrics.get('gpu_usage', 0.0),
                task_completion_rate=metrics.get(f'{agent_id}_completion_rate', 0.0),
                user_satisfaction=metrics.get(f'{agent_id}_satisfaction', 0.0),
                safety_violations=int(metrics.get(f'{agent_id}_safety_violations', 0)),
                anomaly_score=metrics.get(f'{agent_id}_anomaly_score', 0.0)
            )
            
            # 存储指标
            self.metrics_history[agent_id].append(monitoring_metrics)
            
            # 限制历史长度
            if len(self.metrics_history[agent_id]) > 10000:
                self.metrics_history[agent_id].popleft()
            
            # 检查告警
            self._check_alerts(monitoring_metrics)
            
            # 记录到TensorBoard
            if self.tb_writer:
                self._log_to_tensorboard(monitoring_metrics)
            
            # 保存指标到文件
            self._save_metrics_to_file(monitoring_metrics)
    
    def detect_performance_degradation(self, 
                                     performance_history: List[float], 
                                     threshold: float = 0.1) -> bool:
        """检测性能退化"""
        if len(performance_history) < 10:
            return False
        
        # 计算最近和历史性能
        recent_performance = np.mean(performance_history[-5:])
        historical_performance = np.mean(performance_history[-20:-5])
        
        # 检测显著下降
        if historical_performance > 0:
            degradation = (historical_performance - recent_performance) / historical_performance
            
            if degradation > threshold:
                logger.warning(f"检测到性能退化: {degradation:.2%}")
                
                # 生成告警
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    level=AlertLevel.WARNING,
                    message=f"性能退化检测: 下降 {degradation:.2%}",
                    timestamp=datetime.now(),
                    metrics={'degradation': degradation}
                )
                self.alerts.append(alert)
                
                return True
        
        return False
    
    def generate_learning_dashboard(self) -> Dict[str, Any]:
        """生成学习仪表板"""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'agents': {},
            'alerts': {
                'total': len(self.alerts),
                'critical': len([a for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.resolved]),
                'warnings': len([a for a in self.alerts if a.level == AlertLevel.WARNING and not a.resolved]),
                'recent': [a.to_dict() for a in list(self.alerts)[-10:]]
            },
            'system_health': self._calculate_system_health()
        }
        
        # 为每个智能体生成统计信息
        for agent_id, metrics_list in self.metrics_history.items():
            if not metrics_list:
                continue
            
            recent_metrics = list(metrics_list)[-100:]  # 最近100个指标
            
            agent_stats = {
                'total_metrics': len(metrics_list),
                'current_policy_version': recent_metrics[-1].policy_version if recent_metrics else 'unknown',
                'performance': {
                    'success_rate': {
                        'current': recent_metrics[-1].success_rate if recent_metrics else 0.0,
                        'average': np.mean([m.success_rate for m in recent_metrics]),
                        'trend': self._calculate_trend([m.success_rate for m in recent_metrics[-10:]])
                    },
                    'average_reward': {
                        'current': recent_metrics[-1].average_reward if recent_metrics else 0.0,
                        'average': np.mean([m.average_reward for m in recent_metrics]),
                        'trend': self._calculate_trend([m.average_reward for m in recent_metrics[-10:]])
                    },
                    'error_rate': {
                        'current': recent_metrics[-1].error_rate if recent_metrics else 0.0,
                        'average': np.mean([m.error_rate for m in recent_metrics]),
                        'trend': self._calculate_trend([m.error_rate for m in recent_metrics[-10:]])
                    }
                },
                'resources': {
                    'cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
                    'memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
                    'gpu_usage': np.mean([m.gpu_usage for m in recent_metrics])
                },
                'safety': {
                    'total_violations': sum([m.safety_violations for m in recent_metrics]),
                    'anomaly_score': np.mean([m.anomaly_score for m in recent_metrics])
                }
            }
            
            dashboard['agents'][agent_id] = agent_stats
        
        return dashboard
    
    def _get_policy_version(self, policy: BasePolicyNetwork) -> str:
        """获取策略版本"""
        # 简化的版本计算（实际应用中可能需要更复杂的版本管理）
        param_hash = hash(str(policy.state_dict()))
        return f"v{abs(param_hash) % 10000}"
    
    def _check_alerts(self, metrics: MonitoringMetrics):
        """检查告警条件"""
        alerts_to_generate = []
        
        # 成功率告警
        if metrics.success_rate < self.alert_thresholds['success_rate_min']:
            alerts_to_generate.append((
                AlertLevel.WARNING,
                f"智能体 {metrics.agent_id} 成功率过低: {metrics.success_rate:.2%}"
            ))
        
        # 错误率告警
        if metrics.error_rate > self.alert_thresholds['error_rate_max']:
            alerts_to_generate.append((
                AlertLevel.ERROR,
                f"智能体 {metrics.agent_id} 错误率过高: {metrics.error_rate:.2%}"
            ))
        
        # 资源使用告警
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage_max']:
            alerts_to_generate.append((
                AlertLevel.WARNING,
                f"CPU使用率过高: {metrics.cpu_usage:.2%}"
            ))
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage_max']:
            alerts_to_generate.append((
                AlertLevel.WARNING,
                f"内存使用率过高: {metrics.memory_usage:.2%}"
            ))
        
        # 异常分数告警
        if metrics.anomaly_score > self.alert_thresholds['anomaly_score_max']:
            alerts_to_generate.append((
                AlertLevel.ERROR,
                f"智能体 {metrics.agent_id} 异常分数过高: {metrics.anomaly_score:.2f}"
            ))
        
        # 安全违规告警
        if metrics.safety_violations > 0:
            level = AlertLevel.CRITICAL if metrics.safety_violations > 5 else AlertLevel.WARNING
            alerts_to_generate.append((
                level,
                f"智能体 {metrics.agent_id} 安全违规: {metrics.safety_violations} 次"
            ))
        
        # 生成告警
        for level, message in alerts_to_generate:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                level=level,
                message=message,
                timestamp=metrics.timestamp,
                agent_id=metrics.agent_id,
                policy_version=metrics.policy_version,
                metrics=metrics.to_dict()
            )
            self.alerts.append(alert)
            
            # 记录日志
            if level == AlertLevel.CRITICAL:
                logger.critical(message)
            elif level == AlertLevel.ERROR:
                logger.error(message)
            elif level == AlertLevel.WARNING:
                logger.warning(message)
            else:
                logger.info(message)
    
    def _log_to_tensorboard(self, metrics: MonitoringMetrics):
        """记录到TensorBoard"""
        if not self.tb_writer:
            return
        
        step = int(metrics.timestamp.timestamp())
        prefix = f"{metrics.agent_id}/"
        
        # 性能指标
        self.tb_writer.add_scalar(f"{prefix}success_rate", metrics.success_rate, step)
        self.tb_writer.add_scalar(f"{prefix}average_reward", metrics.average_reward, step)
        self.tb_writer.add_scalar(f"{prefix}execution_time", metrics.execution_time, step)
        self.tb_writer.add_scalar(f"{prefix}error_rate", metrics.error_rate, step)
        
        # 资源指标
        self.tb_writer.add_scalar("system/cpu_usage", metrics.cpu_usage, step)
        self.tb_writer.add_scalar("system/memory_usage", metrics.memory_usage, step)
        self.tb_writer.add_scalar("system/gpu_usage", metrics.gpu_usage, step)
        
        # 业务指标
        self.tb_writer.add_scalar(f"{prefix}task_completion_rate", metrics.task_completion_rate, step)
        self.tb_writer.add_scalar(f"{prefix}user_satisfaction", metrics.user_satisfaction, step)
        
        # 安全指标
        self.tb_writer.add_scalar(f"{prefix}safety_violations", metrics.safety_violations, step)
        self.tb_writer.add_scalar(f"{prefix}anomaly_score", metrics.anomaly_score, step)
    
    def _save_metrics_to_file(self, metrics: MonitoringMetrics):
        """保存指标到文件"""
        date_str = metrics.timestamp.strftime("%Y-%m-%d")
        metrics_file = self.log_dir / "metrics" / f"{date_str}.jsonl"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """计算系统健康状况"""
        if not self.metrics_history:
            return {'status': 'unknown', 'score': 0.0}
        
        # 收集最近的指标
        recent_metrics = []
        for agent_metrics in self.metrics_history.values():
            if agent_metrics:
                recent_metrics.extend(list(agent_metrics)[-10:])
        
        if not recent_metrics:
            return {'status': 'unknown', 'score': 0.0}
        
        # 计算健康分数
        success_rate_score = np.mean([m.success_rate for m in recent_metrics])
        error_rate_score = 1.0 - np.mean([m.error_rate for m in recent_metrics])
        resource_score = 1.0 - max(
            np.mean([m.cpu_usage for m in recent_metrics]),
            np.mean([m.memory_usage for m in recent_metrics])
        )
        safety_score = 1.0 - min(1.0, np.mean([m.anomaly_score for m in recent_metrics]))
        
        overall_score = (success_rate_score + error_rate_score + resource_score + safety_score) / 4
        
        # 确定状态
        if overall_score >= 0.8:
            status = 'healthy'
        elif overall_score >= 0.6:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': overall_score,
            'components': {
                'performance': success_rate_score,
                'reliability': error_rate_score,
                'resources': resource_score,
                'safety': safety_score
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 简单的线性趋势计算
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 这里可以启动定期清理、数据聚合等后台任务
        pass
    
    def cleanup_old_data(self):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)
        
        # 清理内存中的旧指标
        for agent_id in self.metrics_history:
            metrics_list = self.metrics_history[agent_id]
            while metrics_list and metrics_list[0].timestamp < cutoff_date:
                metrics_list.popleft()
        
        # 清理旧的日志文件
        metrics_dir = self.log_dir / "metrics"
        if metrics_dir.exists():
            for file_path in metrics_dir.glob("*.jsonl"):
                try:
                    file_date = datetime.strptime(file_path.stem, "%Y-%m-%d")
                    if file_date < cutoff_date:
                        file_path.unlink()
                except ValueError:
                    continue
        
        self.last_cleanup = datetime.now()
        logger.info(f"清理了 {cutoff_date} 之前的旧数据")
    
    def close(self):
        """关闭监控器"""
        self.monitoring_active = False
        if self.tb_writer:
            self.tb_writer.close()


class PolicyDeployment:
    """策略部署管理"""
    
    def __init__(self,
                 deployment_dir: str = "./deployments",
                 backup_dir: str = "./backups",
                 max_concurrent_deployments: int = 3):
        self.deployment_dir = Path(deployment_dir)
        self.backup_dir = Path(backup_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_deployments = max_concurrent_deployments
        
        # 部署状态跟踪
        self.active_deployments = {}  # deployment_id -> DeploymentConfig
        self.deployment_history = deque(maxlen=1000)
        self.policy_versions = {}  # agent_id -> policy_version
        
        self.logger = logger
    
    def deploy_policy(self, 
                     policy: BasePolicyNetwork, 
                     deployment_config: Dict, 
                     safety_checks: List[Callable] = None) -> str:
        """部署策略"""
        deployment_id = str(uuid.uuid4())
        
        try:
            # 检查并发部署限制
            if len(self.active_deployments) >= self.max_concurrent_deployments:
                raise RuntimeError(f"超过最大并发部署数量: {self.max_concurrent_deployments}")
            
            # 创建部署配置
            config = DeploymentConfig(
                deployment_id=deployment_id,
                policy_path=str(self.deployment_dir / f"{deployment_id}.pth"),
                target_agents=deployment_config.get('target_agents', []),
                rollout_percentage=deployment_config.get('rollout_percentage', 100.0),
                canary_percentage=deployment_config.get('canary_percentage', 10.0),
                monitoring_duration=deployment_config.get('monitoring_duration', 3600),
                auto_rollback=deployment_config.get('auto_rollback', True),
                performance_threshold=deployment_config.get('performance_threshold', 0.8),
                safety_checks=deployment_config.get('safety_checks', []),
                metadata=deployment_config.get('metadata', {})
            )
            
            # 执行安全检查
            if safety_checks:
                for check in safety_checks:
                    if not check(policy):
                        raise RuntimeError(f"安全检查失败: {check.__name__}")
            
            # 备份当前策略
            self._backup_current_policies(config.target_agents)
            
            # 保存新策略
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'policy_class': policy.__class__.__name__,
                'deployment_config': config.__dict__,
                'timestamp': datetime.now().isoformat()
            }, config.policy_path)
            
            # 记录部署
            self.active_deployments[deployment_id] = config
            
            # 开始部署过程
            self._start_deployment_process(deployment_id, config)
            
            logger.info(f"策略部署已启动: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"策略部署失败: {e}")
            # 清理失败的部署
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
            raise
    
    def rollback_policy(self, 
                       deployment_id: str, 
                       backup_policy: Optional[BasePolicyNetwork] = None) -> bool:
        """回滚策略"""
        try:
            if deployment_id not in self.active_deployments:
                logger.warning(f"部署 {deployment_id} 不存在或已完成")
                return False
            
            config = self.active_deployments[deployment_id]
            
            # 查找备份策略
            if backup_policy is None:
                backup_policy = self._load_backup_policy(config.target_agents[0])
            
            if backup_policy is None:
                logger.error(f"无法找到备份策略进行回滚")
                return False
            
            # 执行回滚
            backup_path = self.deployment_dir / f"{deployment_id}_rollback.pth"
            torch.save({
                'policy_state_dict': backup_policy.state_dict(),
                'policy_class': backup_policy.__class__.__name__,
                'rollback_from': deployment_id,
                'timestamp': datetime.now().isoformat()
            }, backup_path)
            
            # 更新部署状态
            config.metadata['status'] = DeploymentStatus.ROLLBACK.value
            config.metadata['rollback_time'] = datetime.now().isoformat()
            
            # 记录回滚历史
            self.deployment_history.append({
                'deployment_id': deployment_id,
                'action': 'rollback',
                'timestamp': datetime.now().isoformat(),
                'reason': 'manual_rollback'
            })
            
            logger.info(f"策略已回滚: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"策略回滚失败: {e}")
            return False
    
    def a_b_test_policies(self, 
                         policy_a: BasePolicyNetwork, 
                         policy_b: BasePolicyNetwork, 
                         test_config: Dict) -> Dict:
        """A/B测试策略"""
        test_id = str(uuid.uuid4())
        
        try:
            # 部署策略A（对照组）
            deployment_a_config = {
                'target_agents': test_config.get('target_agents', []),
                'rollout_percentage': test_config.get('control_percentage', 50.0),
                'metadata': {
                    'test_id': test_id,
                    'test_group': 'control',
                    'test_type': 'a_b_test'
                }
            }
            deployment_a_id = self.deploy_policy(policy_a, deployment_a_config)
            
            # 部署策略B（实验组）
            deployment_b_config = {
                'target_agents': test_config.get('target_agents', []),
                'rollout_percentage': test_config.get('treatment_percentage', 50.0),
                'metadata': {
                    'test_id': test_id,
                    'test_group': 'treatment',
                    'test_type': 'a_b_test'
                }
            }
            deployment_b_id = self.deploy_policy(policy_b, deployment_b_config)
            
            # 创建A/B测试记录
            ab_test_record = {
                'test_id': test_id,
                'deployment_a_id': deployment_a_id,
                'deployment_b_id': deployment_b_id,
                'start_time': datetime.now().isoformat(),
                'duration': test_config.get('duration', 3600),
                'metrics_to_track': test_config.get('metrics', ['success_rate', 'average_reward']),
                'status': 'running'
            }
            
            # 保存A/B测试配置
            ab_test_file = self.deployment_dir / f"ab_test_{test_id}.json"
            with open(ab_test_file, 'w') as f:
                json.dump(ab_test_record, f, indent=2)
            
            logger.info(f"A/B测试已启动: {test_id}")
            return ab_test_record
            
        except Exception as e:
            logger.error(f"A/B测试启动失败: {e}")
            raise
    
    def _backup_current_policies(self, target_agents: List[str]):
        """备份当前策略"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for agent_id in target_agents:
            if agent_id in self.policy_versions:
                current_version = self.policy_versions[agent_id]
                backup_path = self.backup_dir / f"{agent_id}_{timestamp}_{current_version}.pth"
                
                # 这里应该从实际运行环境中获取当前策略
                # 简化实现：创建备份记录
                backup_record = {
                    'agent_id': agent_id,
                    'policy_version': current_version,
                    'backup_time': datetime.now().isoformat(),
                    'backup_path': str(backup_path)
                }
                
                backup_record_file = self.backup_dir / f"{agent_id}_{timestamp}_record.json"
                with open(backup_record_file, 'w') as f:
                    json.dump(backup_record, f, indent=2)
    
    def _load_backup_policy(self, agent_id: str) -> Optional[BasePolicyNetwork]:
        """加载备份策略"""
        # 查找最新的备份
        backup_files = list(self.backup_dir.glob(f"{agent_id}_*_record.json"))
        if not backup_files:
            return None
        
        # 按时间排序，获取最新备份
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_backup = backup_files[0]
        
        try:
            with open(latest_backup, 'r') as f:
                backup_record = json.load(f)
            
            backup_path = backup_record['backup_path']
            if os.path.exists(backup_path):
                # 这里应该根据实际策略类加载
                # 简化实现：返回None
                return None
        except Exception as e:
            logger.error(f"加载备份策略失败: {e}")
        
        return None
    
    def _start_deployment_process(self, deployment_id: str, config: DeploymentConfig):
        """启动部署过程"""
        # 这里应该启动实际的部署过程
        # 包括金丝雀部署、监控、自动回滚等
        
        # 记录部署开始
        self.deployment_history.append({
            'deployment_id': deployment_id,
            'action': 'deploy_start',
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__
        })
        
        # 模拟部署过程
        config.metadata['status'] = DeploymentStatus.DEPLOYING.value
        config.metadata['deploy_start_time'] = datetime.now().isoformat()
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """获取部署状态"""
        if deployment_id not in self.active_deployments:
            return None
        
        config = self.active_deployments[deployment_id]
        return {
            'deployment_id': deployment_id,
            'status': config.metadata.get('status', DeploymentStatus.PENDING.value),
            'target_agents': config.target_agents,
            'rollout_percentage': config.rollout_percentage,
            'start_time': config.metadata.get('deploy_start_time'),
            'monitoring_duration': config.monitoring_duration,
            'auto_rollback': config.auto_rollback
        }
    
    def list_active_deployments(self) -> List[Dict[str, Any]]:
        """列出活跃部署"""
        return [self.get_deployment_status(dep_id) for dep_id in self.active_deployments.keys()]


class SafetyGuard:
    """安全保障机制"""
    
    def __init__(self,
                 safety_config: Optional[Dict[str, Any]] = None):
        self.safety_config = safety_config or {
            'max_action_frequency': 10,  # 每秒最大动作数
            'forbidden_actions': ['factory_reset', 'delete_all'],
            'safe_zones': [(0, 0, 100, 50)],  # 安全区域坐标
            'anomaly_threshold': 0.8,
            'emergency_stop_threshold': 0.9
        }
        
        # 安全状态跟踪
        self.action_history = defaultdict(deque)  # agent_id -> actions
        self.anomaly_scores = defaultdict(deque)  # agent_id -> scores
        self.safety_violations = defaultdict(int)
        self.emergency_stops = set()
        
        self.logger = logger
    
    def validate_policy_safety(self, 
                              policy: BasePolicyNetwork, 
                              safety_tests: List[Callable]) -> bool:
        """验证策略安全性"""
        try:
            # 执行所有安全测试
            for test in safety_tests:
                if not test(policy):
                    logger.warning(f"策略安全测试失败: {test.__name__}")
                    return False
            
            # 检查策略参数异常
            if self._check_policy_parameters(policy):
                return False
            
            # 检查策略行为
            if self._check_policy_behavior(policy):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"策略安全验证失败: {e}")
            return False
    
    def emergency_stop(self, reason: str, affected_agents: List[str] = None):
        """紧急停止"""
        timestamp = datetime.now()
        
        if affected_agents:
            for agent_id in affected_agents:
                self.emergency_stops.add(agent_id)
        else:
            # 全局紧急停止
            self.emergency_stops.add('*')
        
        # 记录紧急停止事件
        emergency_record = {
            'timestamp': timestamp.isoformat(),
            'reason': reason,
            'affected_agents': affected_agents or ['*'],
            'action': 'emergency_stop'
        }
        
        # 保存紧急停止记录
        emergency_file = Path("./logs/emergency_stops.jsonl")
        emergency_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(emergency_file, 'a') as f:
            f.write(json.dumps(emergency_record) + '\n')
        
        logger.critical(f"紧急停止已激活: {reason}")
    
    def safe_exploration_bounds(self, 
                               action_space: Any, 
                               safety_constraints: Dict) -> Any:
        """设置安全探索边界"""
        # 这里应该根据具体的动作空间类型实现安全边界
        # 简化实现：返回原始动作空间
        
        safe_bounds = {
            'coordinate_bounds': safety_constraints.get('coordinate_bounds', (0, 0, 1080, 1920)),
            'action_frequency_limit': safety_constraints.get('action_frequency_limit', 10),
            'forbidden_areas': safety_constraints.get('forbidden_areas', []),
            'allowed_actions': safety_constraints.get('allowed_actions', None)
        }
        
        logger.info(f"安全探索边界已设置: {safe_bounds}")
        return action_space  # 简化实现
    
    def check_action_safety(self, agent_id: str, action: Dict[str, Any]) -> bool:
        """检查动作安全性"""
        # 检查是否在紧急停止状态
        if agent_id in self.emergency_stops or '*' in self.emergency_stops:
            logger.warning(f"智能体 {agent_id} 处于紧急停止状态")
            return False
        
        # 检查动作频率
        current_time = time.time()
        self.action_history[agent_id].append(current_time)
        
        # 清理旧的动作记录（1秒前）
        while (self.action_history[agent_id] and 
               current_time - self.action_history[agent_id][0] > 1.0):
            self.action_history[agent_id].popleft()
        
        # 检查频率限制
        if len(self.action_history[agent_id]) > self.safety_config['max_action_frequency']:
            self.safety_violations[agent_id] += 1
            logger.warning(f"智能体 {agent_id} 动作频率过高")
            return False
        
        # 检查禁止动作
        action_type = action.get('type', '')
        if action_type in self.safety_config['forbidden_actions']:
            self.safety_violations[agent_id] += 1
            logger.warning(f"智能体 {agent_id} 尝试执行禁止动作: {action_type}")
            return False
        
        # 检查坐标安全
        coordinates = action.get('coordinates')
        if coordinates and not self._check_coordinate_safety(coordinates):
            self.safety_violations[agent_id] += 1
            logger.warning(f"智能体 {agent_id} 尝试在不安全区域操作: {coordinates}")
            return False
        
        return True
    
    def _check_policy_parameters(self, policy: BasePolicyNetwork) -> bool:
        """检查策略参数异常"""
        try:
            for name, param in policy.named_parameters():
                # 检查NaN或Inf
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"策略参数异常: {name} 包含NaN或Inf")
                    return True
                
                # 检查参数范围
                if param.abs().max() > 100:  # 参数过大
                    logger.warning(f"策略参数可能过大: {name} max={param.abs().max()}")
            
            return False
            
        except Exception as e:
            logger.error(f"检查策略参数失败: {e}")
            return True
    
    def _check_policy_behavior(self, policy: BasePolicyNetwork) -> bool:
        """检查策略行为"""
        try:
            # 生成测试输入
            test_state = torch.randn(1, policy.state_dim)
            
            # 检查策略输出
            with torch.no_grad():
                output = policy(test_state)
                
                # 检查输出是否包含NaN或Inf
                for key, value in output.items():
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        logger.error(f"策略输出异常: {key} 包含NaN或Inf")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查策略行为失败: {e}")
            return True
    
    def _check_coordinate_safety(self, coordinates: Tuple[int, int]) -> bool:
        """检查坐标安全性"""
        x, y = coordinates
        
        # 检查是否在安全区域内
        for safe_zone in self.safety_config['safe_zones']:
            x1, y1, x2, y2 = safe_zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        
        # 检查是否在屏幕范围内
        if 0 <= x <= 1080 and 0 <= y <= 1920:
            return True
        
        return False
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """获取安全统计信息"""
        return {
            'total_violations': sum(self.safety_violations.values()),
            'violations_by_agent': dict(self.safety_violations),
            'emergency_stops': list(self.emergency_stops),
            'active_agents': len(self.action_history),
            'safety_config': self.safety_config
        }


# 工具函数
def create_learning_monitor(config: Dict[str, Any]) -> LearningMonitor:
    """创建学习监控器"""
    return LearningMonitor(
        log_dir=config.get('log_dir', './logs'),
        metrics_retention_days=config.get('metrics_retention_days', 30),
        alert_thresholds=config.get('alert_thresholds'),
        enable_tensorboard=config.get('enable_tensorboard', True)
    )


def create_policy_deployment(config: Dict[str, Any]) -> PolicyDeployment:
    """创建策略部署管理器"""
    return PolicyDeployment(
        deployment_dir=config.get('deployment_dir', './deployments'),
        backup_dir=config.get('backup_dir', './backups'),
        max_concurrent_deployments=config.get('max_concurrent_deployments', 3)
    )


def create_safety_guard(config: Dict[str, Any]) -> SafetyGuard:
    """创建安全保障机制"""
    return SafetyGuard(
        safety_config=config.get('safety_config')
    )


# 预定义的安全检查函数
def check_policy_gradient_norm(policy: BasePolicyNetwork) -> bool:
    """检查策略梯度范数"""
    total_norm = 0
    for p in policy.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    return total_norm < 10.0  # 梯度范数阈值


def check_policy_output_range(policy: BasePolicyNetwork) -> bool:
    """检查策略输出范围"""
    test_input = torch.randn(1, policy.state_dim)
    
    with torch.no_grad():
        output = policy(test_input)
        
        # 检查动作概率是否在合理范围内
        if 'action_probs' in output:
            probs = output['action_probs']
            if (probs < 0).any() or (probs > 1).any():
                return False
            if abs(probs.sum() - 1.0) > 0.1:
                return False
    
    return True


def check_policy_determinism(policy: BasePolicyNetwork) -> bool:
    """检查策略确定性"""
    test_input = torch.randn(1, policy.state_dim)
    
    with torch.no_grad():
        output1 = policy(test_input)
        output2 = policy(test_input)
        
        # 相同输入应该产生相同输出
        for key in output1.keys():
            if key in output2:
                if not torch.allclose(output1[key], output2[key], atol=1e-6):
                    return False
    
    return True