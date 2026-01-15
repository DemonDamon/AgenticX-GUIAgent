#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多模态ExecutorAgent

验证基于AgenticX框架的真正多模态LLM驱动的ExecutorAgent实现
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
from agents.executor_agent import ExecutorAgent
from config import AgentConfig
from utils import setup_logger

# 设置日志
logger = setup_logger("test_multimodal_executor", level="INFO")

async def test_multimodal_executor():
    """测试多模态ExecutorAgent功能"""
    
    logger.info("🚀 开始测试多模态ExecutorAgent")
    
    # 初始化LLM提供者（使用百炼多模态模型）
    try:
        # 从环境变量获取百炼API密钥
        api_key = os.getenv('BAILIAN_API_KEY')
        if not api_key:
            logger.warning("🔄 未设置BAILIAN_API_KEY环境变量，将测试无LLM模式")
            llm_provider = None
        else:
            llm_provider = BailianProvider(
                api_key=api_key,
                model="qwen-vl-max",  # 使用多模态模型
                temperature=0.3
            )
            logger.info(f"🤖 百炼多模态LLM提供者初始化成功，模型: qwen-vl-max")
    except Exception as e:
        logger.error(f"❌ LLM提供者初始化失败: {e}")
        llm_provider = None
    
    # 初始化事件总线
    event_bus = EventBus()
    
    # 创建ExecutorAgent配置
    agent_config = AgentConfig(
        id="test_executor",
        name="TestExecutorAgent",
        role="executor",
        goal="测试多模态LLM驱动的移动设备操作执行",
        backstory="我是一个测试用的执行智能体，能够使用多模态LLM分析截图并执行相应操作。"
    )
    
    # 初始化ExecutorAgent
    executor_agent = ExecutorAgent(
        llm_provider=llm_provider,
        agent_id="test_executor",
        event_bus=event_bus,
        agent_config=agent_config
    )
    
    logger.info("ExecutorAgent初始化完成")
    
    # 测试用例 - 使用真正的原子操作序列
    test_cases = [
        {
            "name": "步骤1: 截图获取当前屏幕",
            "task_context": {
                "task_type": "take_screenshot",
                "description": "获取当前手机屏幕截图"
            }
        },
        {
            "name": "步骤2: 点击网易云音乐图标",
            "task_context": {
                "task_type": "click_action",
                "description": "点击网易云音乐应用图标打开应用",
                "target_description": "网易云音乐图标，通常是红色圆形，里面有白色音符",
                "use_multimodal_analysis": True
            }
        },
        {
            "name": "步骤3: 等待应用加载",
            "task_context": {
                "task_type": "wait",
                "duration": 3,
                "description": "等待网易云音乐应用完全加载"
            }
        },
        {
            "name": "步骤4: 点击搜索框",
            "task_context": {
                "task_type": "click_action",
                "description": "点击搜索框准备输入搜索内容",
                "target_description": "搜索框，通常在应用顶部，有放大镜图标或'搜索'字样",
                "use_multimodal_analysis": True
            }
        },
        {
            "name": "步骤5: 输入搜索内容",
            "task_context": {
                "task_type": "input_text",
                "text": "周杰伦 稻香",
                "description": "在搜索框中输入'周杰伦 稻香'"
            }
        },
        {
            "name": "步骤6: 点击搜索按钮",
            "task_context": {
                "task_type": "click_action",
                "description": "点击搜索按钮开始搜索",
                "target_description": "搜索按钮，通常是放大镜图标或'搜索'按钮",
                "use_multimodal_analysis": True
            }
        },
        {
            "name": "步骤7: 等待搜索结果",
            "task_context": {
                "task_type": "wait",
                "duration": 3,
                "description": "等待搜索结果加载"
            }
        },
        {
            "name": "步骤8: 点击播放稻香",
            "task_context": {
                "task_type": "click_action",
                "description": "点击周杰伦《稻香》的播放按钮",
                "target_description": "周杰伦《稻香》歌曲的播放按钮，通常是三角形图标",
                "use_multimodal_analysis": True
            }
        }
    ]
    
    # 执行测试用例
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试用例 {i}: {test_case['name']}")
        logger.info(f"{'='*60}")
        
        try:
            # 执行任务
            result = await executor_agent._execute_task_impl(test_case['task_context'])
            
            # 打印结果，清楚标识数据来源
            execution_method = result.get('execution_method', 'unknown')
            success = result.get('success', False)
            
            if execution_method == 'multimodal_llm':
                logger.info(f"🤖 【多模态LLM执行成功】")
                logger.info(f"🧠 LLM思考过程: {result.get('llm_thought', 'N/A')}")
                logger.info(f"🎯 LLM动作计划: {result.get('llm_action_plan', 'N/A')}")
                logger.info(f"📝 LLM动作描述: {result.get('llm_description', 'N/A')}")
                logger.info(f"✅ 执行状态: {'成功' if success else '失败'}")
            elif success:
                logger.info(f"⚙️ 【传统工具执行成功】")
                logger.info(f"🔧 执行方法: {result.get('action', 'N/A')}")
                logger.info(f"📄 执行结果: {result.get('message', 'N/A')}")
            else:
                logger.error(f"❌ 【执行失败】")
                logger.error(f"💥 错误信息: {result.get('error', 'N/A')}")
                logger.error(f"📄 失败消息: {result.get('message', 'N/A')}")
            
            # 显示截图路径（如果有）
            if result.get('screenshot_path'):
                logger.info(f"📸 截图路径: {result['screenshot_path']}")
            
        except Exception as e:
            logger.error(f"❌ 测试用例执行异常: {e}")
    
    logger.info("\n🎉 所有测试完成")
    
    # 显示操作历史
    action_history = executor_agent.get_action_history()
    if action_history:
        logger.info(f"\n📊 操作历史统计: 共{len(action_history)}个操作")
        success_count = sum(1 for action in action_history if action.get('success', False))
        logger.info(f"✅ 成功操作: {success_count}")
        logger.info(f"❌ 失败操作: {len(action_history) - success_count}")
    
    return executor_agent

async def test_with_real_screenshot():
    """测试使用真实截图的多模态分析"""
    logger.info("\n🖼️ 开始测试真实截图的多模态分析")
    
    # 查找最新的截图文件
    screenshots_dir = Path("./screenshots")
    if screenshots_dir.exists():
        screenshot_files = list(screenshots_dir.glob("*.png"))
        if screenshot_files:
            # 按修改时间排序，获取最新的截图
            latest_screenshot = max(screenshot_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"找到截图文件: {latest_screenshot}")
            
            # 初始化百炼LLM提供者
            api_key = os.getenv('BAILIAN_API_KEY')
            if api_key:
                llm_provider = BailianProvider(
                    api_key=api_key,
                    model="qwen-vl-max",
                    temperature=0.3
                )
                
                # 创建ExecutorAgent
                executor_agent = ExecutorAgent(
                    llm_provider=llm_provider,
                    agent_id="screenshot_test_executor"
                )
                
                # 测试多模态分析
                task_context = {
                    "task_type": "multimodal_analysis",
                    "description": "在手机主屏幕上找到网易云音乐应用图标并点击打开。网易云音乐图标通常是红色圆形，里面有白色音符或'网易云音乐'字样。",
                    "screenshot_path": str(latest_screenshot)
                }
                
                logger.info("测试多模态分析: 分析截图并决定下一步操作")
                result = await executor_agent._analyze_and_execute_with_llm(task_context)
                
                logger.info(f"分析结果: {result}")
            else:
                logger.warning("未设置BAILIAN_API_KEY，跳过真实截图测试")
        else:
            logger.warning("screenshots目录中没有找到截图文件")
    else:
        logger.warning("screenshots目录不存在")

def main():
    """主函数"""
    try:
        # 运行基础测试
        executor_agent = asyncio.run(test_multimodal_executor())
        
        # 运行真实截图测试
        asyncio.run(test_with_real_screenshot())
        
        logger.info("\n🎊 所有测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        logger.error("\n💡 请检查:")
        logger.error("   1. 是否在.env文件中设置了 BAILIAN_API_KEY")
        logger.error("   2. API Key是否有效且有足够额度")
        logger.error("   3. 网络连接是否可以访问百炼API服务")
        logger.error("   4. 是否已安装所需依赖")

if __name__ == "__main__":
    main()