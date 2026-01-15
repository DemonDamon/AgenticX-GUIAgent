#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多模态智能体集成协作功能

验证ActionReflectorAgent和NotetakerAgent的协作能力，
以及基于AgenticX框架的事件驱动架构
"""

import asyncio
from loguru import logger
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载.env文件
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"已加载环境变量文件: {env_path}")
except ImportError:
    print("未安装python-dotenv，跳过.env文件加载")

# 导入必要的模块
from agenticx.llms.bailian_provider import BailianProvider
from agenticx.core.event_bus import EventBus
from agenticx.core.event import Event
from agents.action_reflector_agent import ActionReflectorAgent
from agents.notetaker_agent import NotetakerAgent
from config import AgentConfig
from utils import setup_logger

# 设置日志
logger = setup_logger("test_multimodal_integration", level="INFO")

class IntegrationTestCoordinator:
    """集成测试协调器"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.reflector_agent = None
        self.notetaker_agent = None
        self.test_results = []
        self.event_log = []
        
        # 订阅所有事件进行监控
        self.event_bus.subscribe("multimodal_reflection_result", self._log_event)
        self.event_bus.subscribe("knowledge_update", self._log_event)
        self.event_bus.subscribe("action_result", self._log_event)
    
    def _log_event(self, event):
        """记录事件"""
        self.event_log.append({
            "type": event.type,
            "timestamp": event.timestamp,
            "agent_id": getattr(event, 'agent_id', 'unknown'),
            "data_keys": list(event.data.keys()) if hasattr(event, 'data') else []
        })
        logger.info(f"📡 事件记录: {event.type} from {getattr(event, 'agent_id', 'unknown')}")
    
    async def initialize_agents(self):
        """初始化智能体"""
        logger.info("🚀 初始化多模态智能体集成环境")
        
        # 初始化LLM提供者
        api_key = os.getenv('BAILIAN_API_KEY')
        if api_key:
            llm_provider = BailianProvider(
                api_key=api_key,
                model="qwen-vl-max",
                temperature=0.3
            )
            logger.info("🤖 百炼多模态LLM提供者初始化成功")
        else:
            logger.warning("⚠️ 未设置BAILIAN_API_KEY，将使用模拟模式")
            llm_provider = None
        
        # 创建ActionReflectorAgent
        reflector_config = AgentConfig(
            id="integration_reflector",
            name="IntegrationActionReflectorAgent",
            role="action_reflector",
            goal="在集成环境中进行多模态动作反思分析",
            backstory="我是集成测试中的反思智能体，负责分析操作效果并与知识记录智能体协作。"
        )
        
        self.reflector_agent = ActionReflectorAgent(
            llm_provider=llm_provider,
            agent_id="integration_reflector",
            info_pool=self.event_bus,
            agent_config=reflector_config
        )
        
        # 创建NotetakerAgent
        notetaker_config = AgentConfig(
            id="integration_notetaker",
            name="IntegrationNotetakerAgent",
            role="notetaker",
            goal="在集成环境中进行智能知识管理",
            backstory="我是集成测试中的知识记录智能体，负责捕获反思结果并管理知识库。"
        )
        
        self.notetaker_agent = NotetakerAgent(
            llm_provider=llm_provider,
            agent_id="integration_notetaker",
            info_pool=self.event_bus,
            agent_config=notetaker_config
        )
        
        logger.info("✅ 智能体初始化完成")
    
    async def test_reflection_to_knowledge_flow(self):
        """测试反思到知识的流程"""
        logger.info("\n🔄 测试反思分析到知识捕获的完整流程")
        
        # 模拟反思分析任务
        reflection_task = {
            "analysis_type": "multimodal_reflection",
            "before_screenshot": "./screenshots/test_before.png",
            "after_screenshot": "./screenshots/test_after.png",
            "action_info": {
                "task_type": "click_action",
                "target": "音乐播放按钮",
                "coordinates": [400, 800],
                "description": "点击播放按钮开始播放音乐"
            },
            "expectation": "音乐开始播放，界面显示播放状态"
        }
        
        try:
            # 执行反思分析
            logger.info("🔍 执行反思分析...")
            reflection_result = await self.reflector_agent._execute_task_impl(reflection_task)
            
            if reflection_result.get('success'):
                logger.info(f"✅ 反思分析成功: 操作{'成功' if reflection_result.get('operation_success') else '失败'}")
                
                # 基于反思结果创建知识
                knowledge_data = self._create_knowledge_from_reflection(reflection_result)
                
                # 执行知识捕获
                logger.info("📝 基于反思结果捕获知识...")
                knowledge_task = {
                    "task_type": "capture",
                    "knowledge_data": knowledge_data
                }
                
                knowledge_result = await self.notetaker_agent._execute_task_impl(knowledge_task)
                
                if knowledge_result.get('success'):
                    logger.info(f"✅ 知识捕获成功: {knowledge_result.get('knowledge_id')}")
                    
                    # 记录测试结果
                    self.test_results.append({
                        "test_name": "reflection_to_knowledge_flow",
                        "success": True,
                        "reflection_result": reflection_result,
                        "knowledge_result": knowledge_result
                    })
                else:
                    logger.error(f"❌ 知识捕获失败: {knowledge_result.get('error')}")
                    self.test_results.append({
                        "test_name": "reflection_to_knowledge_flow",
                        "success": False,
                        "error": "knowledge_capture_failed"
                    })
            else:
                logger.error(f"❌ 反思分析失败: {reflection_result.get('error')}")
                self.test_results.append({
                    "test_name": "reflection_to_knowledge_flow",
                    "success": False,
                    "error": "reflection_failed"
                })
        
        except Exception as e:
            logger.error(f"❌ 流程测试异常: {e}")
            self.test_results.append({
                "test_name": "reflection_to_knowledge_flow",
                "success": False,
                "error": str(e)
            })
    
    def _create_knowledge_from_reflection(self, reflection_result):
        """基于反思结果创建知识数据"""
        operation_success = reflection_result.get('operation_success', False)
        outcome = reflection_result.get('outcome', 'unknown')
        
        if operation_success:
            # 成功操作 -> 最佳实践知识
            return {
                "type": "best_practice",
                "content": {
                    "area": "click_action",
                    "practices": [
                        "确保目标元素可见且可交互",
                        "使用精确的坐标定位",
                        "验证操作后的状态变化"
                    ],
                    "success_indicators": reflection_result.get('comparison_analysis', ''),
                    "applicable_scenarios": "所有点击操作",
                    "confidence": 0.9
                },
                "source": "ActionReflectorAgent"
            }
        else:
            # 失败操作 -> 错误解决方案知识
            return {
                "type": "error_solution",
                "content": {
                    "error_type": f"click_action_{outcome}",
                    "error_description": reflection_result.get('error_analysis', ''),
                    "solutions": reflection_result.get('improvement_suggestions', '').split('\n') if reflection_result.get('improvement_suggestions') else [],
                    "prevention": [
                        "预先验证元素状态",
                        "使用多模态分析确认目标"
                    ],
                    "confidence": 0.8
                },
                "source": "ActionReflectorAgent"
            }
    
    async def test_knowledge_query_and_application(self):
        """测试知识查询和应用"""
        logger.info("\n🔍 测试知识查询和应用流程")
        
        try:
            # 查询点击操作相关的最佳实践
            logger.info("📚 查询点击操作最佳实践...")
            best_practices = await self.notetaker_agent.get_best_practices("click_action")
            
            if best_practices:
                logger.info(f"✅ 找到{len(best_practices)}条最佳实践")
                for i, practice in enumerate(best_practices[:3], 1):
                    logger.info(f"  {i}. {practice.get('title', 'N/A')}")
                
                # 模拟应用最佳实践进行新的反思分析
                logger.info("🎯 应用最佳实践进行新的反思分析...")
                enhanced_reflection_task = {
                    "analysis_type": "multimodal_reflection",
                    "before_screenshot": "./screenshots/enhanced_before.png",
                    "after_screenshot": "./screenshots/enhanced_after.png",
                    "action_info": {
                        "task_type": "click_action",
                        "target": "设置按钮",
                        "description": "应用最佳实践点击设置按钮",
                        "applied_practices": [p.get('title', '') for p in best_practices[:2]]
                    },
                    "expectation": "成功打开设置页面"
                }
                
                enhanced_result = await self.reflector_agent._execute_task_impl(enhanced_reflection_task)
                
                if enhanced_result.get('success'):
                    logger.info(f"✅ 增强反思分析成功: 操作{'成功' if enhanced_result.get('operation_success') else '失败'}")
                    
                    self.test_results.append({
                        "test_name": "knowledge_query_and_application",
                        "success": True,
                        "best_practices_found": len(best_practices),
                        "enhanced_result": enhanced_result
                    })
                else:
                    logger.error(f"❌ 增强反思分析失败: {enhanced_result.get('error')}")
            else:
                logger.warning("⚠️ 未找到相关最佳实践")
                self.test_results.append({
                    "test_name": "knowledge_query_and_application",
                    "success": False,
                    "error": "no_best_practices_found"
                })
        
        except Exception as e:
            logger.error(f"❌ 知识查询应用测试异常: {e}")
            self.test_results.append({
                "test_name": "knowledge_query_and_application",
                "success": False,
                "error": str(e)
            })
    
    async def test_event_driven_collaboration(self):
        """测试事件驱动的协作"""
        logger.info("\n📡 测试事件驱动的智能体协作")
        
        try:
            # 清空事件日志
            self.event_log.clear()
            
            # 发布模拟的动作结果事件
            action_event = Event(
                type="action_result",
                data={
                    "action_record": {
                        "task_type": "click_action",
                        "target": "搜索按钮",
                        "success": True,
                        "expected_result": "打开搜索界面"
                    }
                },
                agent_id="test_executor"
            )
            
            logger.info("📤 发布动作结果事件...")
            await self.event_bus.publish_async(action_event)
            
            # 等待事件处理
            await asyncio.sleep(2)
            
            # 发布反思结果事件
            reflection_event = Event(
                type="multimodal_reflection_result",
                data={
                    "reflection_record": {
                        "analysis_type": "multimodal_reflection",
                        "result": {
                            "operation_success": True,
                            "outcome": "A",
                            "improvement_suggestions": "继续保持当前操作策略"
                        }
                    }
                },
                agent_id="integration_reflector"
            )
            
            logger.info("📤 发布反思结果事件...")
            await self.event_bus.publish_async(reflection_event)
            
            # 等待事件处理
            await asyncio.sleep(2)
            
            # 检查事件日志
            logger.info(f"📊 事件处理统计: 共处理{len(self.event_log)}个事件")
            
            event_types = [event['type'] for event in self.event_log]
            for event_type in set(event_types):
                count = event_types.count(event_type)
                logger.info(f"  📋 {event_type}: {count}次")
            
            self.test_results.append({
                "test_name": "event_driven_collaboration",
                "success": True,
                "events_processed": len(self.event_log),
                "event_types": list(set(event_types))
            })
        
        except Exception as e:
            logger.error(f"❌ 事件驱动协作测试异常: {e}")
            self.test_results.append({
                "test_name": "event_driven_collaboration",
                "success": False,
                "error": str(e)
            })
    
    async def test_knowledge_evolution(self):
        """测试知识演化"""
        logger.info("\n🌱 测试知识演化和学习")
        
        try:
            # 生成知识库摘要
            logger.info("📊 生成知识库摘要...")
            summary_task = {
                "task_type": "summary"
            }
            
            summary_result = await self.notetaker_agent._execute_task_impl(summary_task)
            
            if summary_result.get('success'):
                summary = summary_result.get('summary', {})
                logger.info(f"✅ 知识库摘要生成成功")
                logger.info(f"📝 总计项目: {summary.get('total_items', 0)}")
                logger.info(f"📂 类型分布: {summary.get('by_type', {})}")
                logger.info(f"⭐ 重要性分布: {summary.get('by_importance', {})}")
                
                # 组织知识链接
                logger.info("🔗 组织知识关联...")
                link_task = {
                    "task_type": "organize",
                    "organization_type": "link"
                }
                
                link_result = await self.notetaker_agent._execute_task_impl(link_task)
                
                if link_result.get('success'):
                    links_created = link_result.get('links_created', 0)
                    logger.info(f"✅ 知识关联成功: 创建{links_created}个链接")
                    
                    self.test_results.append({
                        "test_name": "knowledge_evolution",
                        "success": True,
                        "summary": summary,
                        "links_created": links_created
                    })
                else:
                    logger.error(f"❌ 知识关联失败: {link_result.get('error')}")
            else:
                logger.error(f"❌ 知识库摘要生成失败: {summary_result.get('error')}")
                self.test_results.append({
                    "test_name": "knowledge_evolution",
                    "success": False,
                    "error": "summary_generation_failed"
                })
        
        except Exception as e:
            logger.error(f"❌ 知识演化测试异常: {e}")
            self.test_results.append({
                "test_name": "knowledge_evolution",
                "success": False,
                "error": str(e)
            })
    
    def generate_test_report(self):
        """生成测试报告"""
        logger.info("\n📋 生成集成测试报告")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get('success', False))
        
        logger.info(f"{'='*60}")
        logger.info(f"🎯 多模态智能体集成测试报告")
        logger.info(f"{'='*60}")
        logger.info(f"📊 测试统计: {successful_tests}/{total_tests} 成功")
        logger.info(f"📈 成功率: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "📈 成功率: 0%")
        
        logger.info(f"\n📝 详细结果:")
        for i, result in enumerate(self.test_results, 1):
            status = "✅" if result.get('success') else "❌"
            test_name = result.get('test_name', 'unknown')
            logger.info(f"  {i}. {status} {test_name}")
            
            if not result.get('success') and 'error' in result:
                logger.info(f"     💥 错误: {result['error']}")
        
        logger.info(f"\n📡 事件处理统计:")
        logger.info(f"  📨 总事件数: {len(self.event_log)}")
        
        if self.event_log:
            event_types = [event['type'] for event in self.event_log]
            for event_type in set(event_types):
                count = event_types.count(event_type)
                logger.info(f"  📋 {event_type}: {count}次")
        
        # 智能体状态统计
        if self.reflector_agent:
            reflection_history = self.reflector_agent.get_reflection_history()
            logger.info(f"\n🔍 反思智能体统计:")
            logger.info(f"  📊 反思次数: {len(reflection_history)}")
            logger.info(f"  📈 成功率: {self.reflector_agent.get_success_rate():.2%}")
            
            model_stats = self.reflector_agent.get_model_usage_stats()
            if model_stats:
                logger.info(f"  🤖 模型使用: {model_stats}")
        
        if self.notetaker_agent:
            recent_captures = self.notetaker_agent.get_recent_captures()
            query_history = self.notetaker_agent.get_query_history()
            logger.info(f"\n📝 知识智能体统计:")
            logger.info(f"  📚 知识捕获: {len(recent_captures)}次")
            logger.info(f"  🔍 知识查询: {len(query_history)}次")
        
        logger.info(f"\n{'='*60}")
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests/total_tests if total_tests > 0 else 0,
            "test_results": self.test_results,
            "event_log": self.event_log
        }

async def main():
    """主函数"""
    logger.info("🚀 开始多模态智能体集成测试")
    
    coordinator = IntegrationTestCoordinator()
    
    try:
        # 初始化智能体
        await coordinator.initialize_agents()
        
        # 执行集成测试
        await coordinator.test_reflection_to_knowledge_flow()
        await coordinator.test_knowledge_query_and_application()
        await coordinator.test_event_driven_collaboration()
        await coordinator.test_knowledge_evolution()
        
        # 生成测试报告
        report = coordinator.generate_test_report()
        
        logger.info("\n🎊 多模态智能体集成测试完成！")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ 集成测试失败: {e}")
        logger.error("\n💡 请检查:")
        logger.error("   1. 是否在.env文件中设置了 BAILIAN_API_KEY")
        logger.error("   2. API Key是否有效且有足够额度")
        logger.error("   3. 网络连接是否可以访问百炼API服务")
        logger.error("   4. 是否已安装所需依赖")
        logger.error("   5. 智能体模块是否正确导入")
        
        import traceback
        traceback.print_exc()
        
        return None

if __name__ == "__main__":
    asyncio.run(main())