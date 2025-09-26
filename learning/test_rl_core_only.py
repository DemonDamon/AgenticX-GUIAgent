#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Core Components Test Suite

测试RL核心组件的功能，不依赖外部模块。

Author: AgenticX Team
Date: 2025
"""

import asyncio
from loguru import logger
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

# 导入RL核心组件
try:
    from learning.rl_core.environment import (
        MobileGUIEnvironment, GUIAction, ActionSpace, StateSpace
    )
    from learning.rl_core.state import (
        MultimodalStateEncoder, create_multimodal_encoder
    )
    from learning.rl_core.policies import (
        create_policy_network, ManagerPolicyNetwork, ExecutorPolicyNetwork
    )
    from learning.rl_core.experience import (
        Experience, create_experience_buffer, create_sharing_hub
    )
    from learning.rl_core.rewards import (
        RewardCalculator, create_reward_calculator
    )
    from learning.rl_core.updates import (
        create_updater, create_update_config
    )
    from learning.rl_core.deployment import (
        create_learning_monitor, create_policy_deployment, create_safety_guard
    )
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"导入失败: {e}")
    IMPORTS_SUCCESSFUL = False


class RLCoreTester:
    """RL核心组件测试器"""
    
    def __init__(self):
        self.logger = logger
        self.setup_logging()
        
        # 测试结果
        self.test_results = {}
    
    def setup_logging(self):
        """设置日志"""
        # logging.basicConfig replaced with logurus - %(name)s - %(levelname)s - %(message)s'
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
            level="INFO"
        )
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("\n=== RL核心组件测试开始 ===")
        
        if not IMPORTS_SUCCESSFUL:
            print("❌ 导入失败，跳过测试")
            return False
        
        tests = [
            self.test_environment,
            self.test_state_encoder,
            self.test_policy_networks,
            self.test_experience_management,
            self.test_reward_calculation,
            self.test_policy_updates,
            self.test_deployment_components
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for test in tests:
            try:
                print(f"\n--- 运行测试: {test.__name__} ---")
                result = await test()
                self.test_results[test.__name__] = result
                
                if result:
                    print(f"✅ {test.__name__} 通过")
                    passed_tests += 1
                else:
                    print(f"❌ {test.__name__} 失败")
                    
            except Exception as e:
                print(f"❌ {test.__name__} 异常: {e}")
                self.test_results[test.__name__] = False
        
        print(f"\n=== 测试完成: {passed_tests}/{total_tests} 通过 ===")
        return passed_tests == total_tests
    
    async def test_environment(self) -> bool:
        """测试环境组件"""
        try:
            # 创建环境
            env = MobileGUIEnvironment(
                screen_width=1080,
                screen_height=1920,
                max_episode_steps=10
            )
            
            # 测试重置
            state, info = env.reset()
            assert state is not None
            assert 'episode' in info
            
            # 测试动作空间
            action_space = env.action_space
            sample_action = action_space.sample()
            assert action_space.contains(sample_action)
            
            # 测试步进
            next_state, reward, terminated, truncated, step_info = env.step(sample_action)
            assert next_state is not None
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            
            # 测试渲染
            env.render(mode='human')
            
            # 关闭环境
            env.close()
            
            print("  ✓ 环境创建、重置、步进、渲染测试通过")
            return True
            
        except Exception as e:
            print(f"  ✗ 环境测试失败: {e}")
            return False
    
    async def test_state_encoder(self) -> bool:
        """测试状态编码器"""
        try:
            # 创建状态编码器
            encoder = create_multimodal_encoder({
                'fusion_dim': 256,
                'num_fusion_layers': 2
            })
            
            # 测试编码
            screenshot = torch.rand(3, 224, 224)
            task_text = "点击登录按钮"
            action_history = torch.rand(10, 5)
            
            inputs = {
                'screenshot': screenshot,
                'task_text': task_text,
                'action_history': action_history
            }
            
            encoded_state = encoder(inputs)
            assert encoded_state is not None
            assert encoded_state.shape[0] == 1  # batch size
            assert encoded_state.shape[1] == 256  # fusion_dim
            
            print(f"  ✓ 状态编码测试通过，输出形状: {encoded_state.shape}")
            return True
            
        except Exception as e:
            print(f"  ✗ 状态编码器测试失败: {e}")
            return False
    
    async def test_policy_networks(self) -> bool:
        """测试策略网络"""
        try:
            state_dim = 256
            
            # 测试各种策略网络
            policies = {
                'manager': create_policy_network('manager', state_dim),
                'executor': create_policy_network('executor', state_dim),
                'reflector': create_policy_network('reflector', state_dim),
                'notetaker': create_policy_network('notetaker', state_dim)
            }
            
            test_state = torch.rand(1, state_dim)
            
            for agent_type, policy in policies.items():
                # 测试前向传播
                output = policy(test_state)
                assert isinstance(output, dict)
                
                # 测试动作选择
                action, log_prob = policy.select_action(test_state)
                assert action is not None
                assert log_prob is not None
                
                # 测试价值计算
                value = policy.compute_value(test_state)
                assert value is not None
                
                print(f"  ✓ {agent_type} 策略网络测试通过")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 策略网络测试失败: {e}")
            return False
    
    async def test_experience_management(self) -> bool:
        """测试经验管理"""
        try:
            # 创建经验缓冲区
            buffer = create_experience_buffer({
                'capacity': 1000,
                'enable_prioritized': True
            })
            
            # 创建共享中心
            sharing_hub = create_sharing_hub({
                'max_shared_experiences': 500
            })
            
            # 创建测试经验
            experiences = []
            for i in range(10):
                exp = Experience(
                    state=torch.rand(256),
                    action=torch.randint(0, 10, (1,)),
                    reward=np.random.uniform(-1, 1),
                    next_state=torch.rand(256),
                    done=i == 9,
                    agent_id=f"agent_{i % 4}",
                    timestamp=datetime.now()
                )
                experiences.append(exp)
                buffer.add(exp)
            
            # 测试采样
            sampled = buffer.sample(5)
            assert len(sampled) == 5
            
            # 测试优先级采样
            sampled_with_weights, weights, indices = buffer.prioritized_sample(3)
            assert len(sampled_with_weights) == 3
            assert len(weights) == 3
            
            # 测试经验共享
            sharing_hub.share_experience(
                'agent_0', ['agent_1', 'agent_2'], experiences[0]
            )
            
            shared_exps = sharing_hub.get_shared_experiences('agent_1')
            assert len(shared_exps) > 0
            
            print(f"  ✓ 经验管理测试通过，缓冲区大小: {len(buffer)}")
            return True
            
        except Exception as e:
            print(f"  ✗ 经验管理测试失败: {e}")
            return False
    
    async def test_reward_calculation(self) -> bool:
        """测试奖励计算"""
        try:
            # 创建奖励计算器
            calculator = create_reward_calculator({
                'enable_reward_shaping': True,
                'enable_adaptive_weights': True
            })
            
            # 测试奖励计算
            state = torch.rand(256)
            action = torch.rand(10)
            next_state = torch.rand(256)
            
            context = {
                'task_progress': 0.8,
                'task_success': True,
                'execution_efficiency': 0.9,
                'execution_time': 2.5,
                'step_count': 15,
                'user_feedback': {'accuracy': 0.85, 'response_time': 1.2},
                'interaction_smoothness': 0.9,
                'error_count': 0
            }
            
            reward = calculator.calculate_reward(state, action, next_state, context)
            assert isinstance(reward, (int, float))
            assert -10 <= reward <= 10  # 合理的奖励范围
            
            # 测试奖励分解
            decomposed = calculator.decompose_reward(reward)
            assert isinstance(decomposed, dict)
            
            # 测试统计信息
            stats = calculator.get_reward_statistics()
            assert 'total_rewards' in stats
            
            print(f"  ✓ 奖励计算测试通过，奖励值: {reward:.3f}")
            return True
            
        except Exception as e:
            print(f"  ✗ 奖励计算测试失败: {e}")
            return False
    
    async def test_policy_updates(self) -> bool:
        """测试策略更新"""
        try:
            # 创建更新器
            update_config = create_update_config(
                algorithm='ppo',
                learning_rate=3e-4,
                batch_size=4
            )
            updater = create_updater('ppo', update_config)
            
            # 创建策略
            policy = create_policy_network('manager', 256)
            
            # 创建经验
            experiences = []
            for i in range(8):  # 足够的批次大小
                exp = Experience(
                    state=torch.rand(256),
                    action=torch.randint(0, 10, (1,)),
                    reward=np.random.uniform(-1, 1),
                    next_state=torch.rand(256),
                    done=False,
                    agent_id="manager",
                    timestamp=datetime.now()
                )
                experiences.append(exp)
            
            # 测试策略更新
            update_result = updater.update_policy(policy, experiences, {})
            assert isinstance(update_result, dict)
            assert 'total_loss' in update_result
            assert 'policy_loss' in update_result
            
            # 测试更新统计
            stats = updater.get_update_statistics()
            assert 'total_updates' in stats
            
            print(f"  ✓ 策略更新测试通过，损失: {update_result['total_loss']:.3f}")
            return True
            
        except Exception as e:
            print(f"  ✗ 策略更新测试失败: {e}")
            return False
    
    async def test_deployment_components(self) -> bool:
        """测试部署组件"""
        try:
            # 创建监控器
            monitor = create_learning_monitor({
                'log_dir': './test_logs',
                'enable_tensorboard': False
            })
            
            # 创建部署管理器
            deployment = create_policy_deployment({
                'deployment_dir': './test_deployments',
                'backup_dir': './test_backups'
            })
            
            # 创建安全保护
            safety_guard = create_safety_guard({})
            
            # 测试监控
            metrics = {
                'manager_success_rate': 0.8,
                'manager_avg_reward': 0.5,
                'cpu_usage': 0.3,
                'memory_usage': 0.4
            }
            
            policies = {
                'manager': create_policy_network('manager', 256)
            }
            
            monitor.monitor_learning_progress(metrics, policies)
            
            # 测试安全检查
            test_action = {
                'type': 'click',
                'coordinates': (500, 1000)
            }
            
            is_safe = safety_guard.check_action_safety('test_agent', test_action)
            assert isinstance(is_safe, bool)
            
            # 测试统计信息
            safety_stats = safety_guard.get_safety_statistics()
            assert isinstance(safety_stats, dict)
            
            print("  ✓ 部署组件测试通过")
            return True
            
        except Exception as e:
            print(f"  ✗ 部署组件测试失败: {e}")
            return False
    
    def print_test_summary(self):
        """打印测试摘要"""
        print("\n=== 测试结果摘要 ===")
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r)
        
        print(f"\n总计: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            print("🎉 所有RL核心组件测试通过！")
        else:
            print("⚠️  部分测试失败，请检查相关组件。")


async def main():
    """主测试函数"""
    tester = RLCoreTester()
    
    try:
        success = await tester.run_all_tests()
        tester.print_test_summary()
        
        return success
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return False
    except Exception as e:
        print(f"\n测试过程中发生异常: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(main())
    sys.exit(0 if success else 1)