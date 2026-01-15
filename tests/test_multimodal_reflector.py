#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多模态ActionReflectorAgent的反思分析功能

验证基于AgenticX框架的真正多模态LLM驱动的ActionReflectorAgent实现
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
from agents.action_reflector_agent import ActionReflectorAgent
from config import AgentConfig
from utils import setup_logger

# 设置日志
logger = setup_logger("test_multimodal_reflector", level="INFO")

async def test_multimodal_reflection():
    """测试多模态反思分析功能"""
    
    logger.info("🚀 开始测试多模态ActionReflectorAgent")
    
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
    
    # 创建ActionReflectorAgent配置
    agent_config = AgentConfig(
        id="test_reflector",
        name="TestActionReflectorAgent",
        role="action_reflector",
        goal="测试多模态LLM驱动的动作反思分析",
        backstory="我是一个测试用的反思智能体，能够使用多模态LLM分析操作前后的截图变化。"
    )
    
    # 初始化ActionReflectorAgent
    reflector_agent = ActionReflectorAgent(
        llm_provider=llm_provider,
        agent_id="test_reflector",
        event_bus=event_bus,
        agent_config=agent_config
    )
    
    logger.info("ActionReflectorAgent初始化完成")
    
    # 获取实际存在的截图文件
    screenshots_dir = Path("./screenshots")
    available_screenshots = []
    if screenshots_dir.exists():
        available_screenshots = sorted([f for f in screenshots_dir.glob("*.png") if not f.name.endswith("_marked.png")])
    
    if len(available_screenshots) < 2:
        logger.warning("⚠️ 截图文件不足，将使用模拟路径进行测试")
        # 使用模拟路径，但会在测试中失败
        before_screenshot = "./screenshots/before_click.png"
        after_screenshot = "./screenshots/after_click.png"
    else:
        # 使用实际存在的截图文件
        before_screenshot = str(available_screenshots[0])
        after_screenshot = str(available_screenshots[-1])
        logger.info(f"📸 使用实际截图文件: {before_screenshot} -> {after_screenshot}")
    
    # 测试用例 - 使用实际截图文件进行对比分析
    test_cases = [
        {
            "name": "测试1: 点击操作成功分析",
            "task_context": {
                "analysis_type": "multimodal_reflection",
                "before_screenshot": before_screenshot,
                "after_screenshot": after_screenshot,
                "action_info": {
                    "task_type": "click_action",
                    "target": "网易云音乐图标",
                    "coordinates": [320, 1200],
                    "description": "点击网易云音乐应用图标"
                },
                "expectation": "成功打开网易云音乐应用"
            }
        },
        {
            "name": "测试2: 输入操作分析",
            "task_context": {
                "analysis_type": "multimodal_reflection",
                "before_screenshot": before_screenshot,
                "after_screenshot": after_screenshot,
                "action_info": {
                    "task_type": "input_text",
                    "text": "周杰伦 稻香",
                    "target": "搜索框",
                    "description": "在搜索框中输入歌曲名称"
                },
                "expectation": "搜索框中显示输入的文本"
            }
        },
        {
            "name": "测试3: 滑动操作分析",
            "task_context": {
                "analysis_type": "multimodal_reflection",
                "before_screenshot": before_screenshot,
                "after_screenshot": after_screenshot,
                "action_info": {
                    "task_type": "swipe_action",
                    "direction": "up",
                    "distance": 500,
                    "description": "向上滑动页面"
                },
                "expectation": "页面内容向上滚动"
            }
        },
        {
            "name": "测试4: 综合多模态分析",
            "task_context": {
                "analysis_type": "comprehensive_analysis",
                "before_screenshot": before_screenshot,
                "after_screenshot": after_screenshot,
                "action_info": {
                    "task_type": "click_action",
                    "target": "播放按钮",
                    "description": "点击歌曲播放按钮"
                },
                "expectation": "开始播放音乐，界面显示播放状态"
            }
        }
    ]
    
    # 执行测试用例
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试用例 {i}: {test_case['name']}")
        logger.info(f"{'='*60}")
        
        try:
            # 执行反思分析
            result = await reflector_agent._execute_task_impl(test_case['task_context'])
            
            # 打印结果，清楚标识数据来源
            analysis_method = result.get('method', 'unknown')
            success = result.get('success', False)
            operation_success = result.get('operation_success', False)
            
            if analysis_method == 'multimodal_llm_reflection' and success:
                logger.info(f"🤖 【多模态LLM反思分析成功】")
                logger.info(f"🧠 对比分析: {result.get('comparison_analysis', 'N/A')[:100]}...")
                logger.info(f"🎯 成功判断: {result.get('success_judgment', 'N/A')[:100]}...")
                logger.info(f"📝 错误分析: {result.get('error_analysis', 'N/A')[:100]}...")
                logger.info(f"💡 改进建议: {result.get('improvement_suggestions', 'N/A')[:100]}...")
                logger.info(f"✅ 操作结果: {'成功' if operation_success else '失败'} ({result.get('outcome', 'unknown')})")
                logger.info(f"🔧 使用模型: {result.get('model_used', 'unknown')}")
            elif not success:
                logger.error(f"❌ 【反思分析失败】")
                logger.error(f"💥 错误信息: {result.get('error', 'N/A')}")
                attempted_models = result.get('attempted_models', [])
                if attempted_models:
                    logger.error(f"🔄 尝试的模型: {', '.join(attempted_models)}")
            else:
                logger.info(f"⚙️ 【其他分析方法】")
                logger.info(f"📋 分析结果: {result.get('analysis', 'N/A')}")
                logger.info(f"✅ 操作状态: {'成功' if operation_success else '失败'}")
            
            # 显示分析的截图信息
            screenshots_analyzed = result.get('screenshots_analyzed', {})
            if screenshots_analyzed:
                logger.info(f"📸 分析截图: 前-{screenshots_analyzed.get('before', 'N/A')} | 后-{screenshots_analyzed.get('after', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ 测试用例执行异常: {e}")
            import traceback
            traceback.print_exc()
        
        # 等待一下再进行下一个测试
        await asyncio.sleep(1)
    
    logger.info("\n🎉 所有反思分析测试完成")
    
    # 显示反思历史统计
    reflection_history = reflector_agent.get_reflection_history()
    if reflection_history:
        logger.info(f"\n📊 反思分析统计: 共{len(reflection_history)}次分析")
        success_count = sum(1 for reflection in reflection_history if reflection.get('result', {}).get('operation_success', False))
        logger.info(f"✅ 成功判断: {success_count}")
        logger.info(f"❌ 失败判断: {len(reflection_history) - success_count}")
        
        # 模型使用统计
        model_stats = reflector_agent.get_model_usage_stats()
        if model_stats:
            logger.info(f"🤖 模型使用统计: {model_stats}")
        
        # 成功率
        success_rate = reflector_agent.get_success_rate()
        logger.info(f"📈 操作成功率: {success_rate:.2%}")
    
    return reflector_agent

async def test_manual_reflection():
    """测试手动反思分析"""
    logger.info("\n🖼️ 开始测试手动反思分析")
    
    # 查找最新的截图文件
    screenshots_dir = Path("./screenshots")
    if screenshots_dir.exists():
        screenshot_files = list(screenshots_dir.glob("*.png"))
        if len(screenshot_files) >= 2:
            # 按修改时间排序，获取最新的两张截图
            screenshot_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            before_screenshot = str(screenshot_files[1])  # 较早的作为before
            after_screenshot = str(screenshot_files[0])   # 较新的作为after
            
            logger.info(f"找到截图文件: {before_screenshot} -> {after_screenshot}")
            
            # 初始化百炼LLM提供者
            api_key = os.getenv('BAILIAN_API_KEY')
            if api_key:
                llm_provider = BailianProvider(
                    api_key=api_key,
                    model="qwen-vl-max",
                    temperature=0.3
                )
                
                # 创建ActionReflectorAgent
                reflector_agent = ActionReflectorAgent(
                    llm_provider=llm_provider,
                    agent_id="manual_test_reflector"
                )
                
                # 测试手动反思分析
                action_info = {
                    "task_type": "click_action",
                    "description": "根据截图对比分析操作效果",
                    "target": "界面元素"
                }
                expectation = "操作产生预期的界面变化"
                
                logger.info("测试手动反思分析: 对比两张截图的变化")
                
                try:
                    result = await reflector_agent.manual_reflection_analysis(
                        before_screenshot=before_screenshot,
                        after_screenshot=after_screenshot,
                        action_info=action_info,
                        expectation=expectation
                    )
                    
                    logger.info(f"手动反思分析结果: {result.get('operation_success', 'unknown')}")
                    logger.info(f"分析详情: {result.get('comparison_analysis', 'N/A')[:200]}...")
                    
                except Exception as e:
                    logger.error(f"手动反思分析失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning("未设置BAILIAN_API_KEY，跳过手动反思测试")
        else:
            logger.warning("screenshots目录中截图文件不足（需要至少2张）")
    else:
        logger.warning("screenshots目录不存在")

def main():
    """主函数"""
    try:
        # 运行基础测试
        reflector_agent = asyncio.run(test_multimodal_reflection())
        
        # 运行手动反思测试
        asyncio.run(test_manual_reflection())
        
        logger.info("\n🎊 所有反思分析测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        logger.error("\n💡 请检查:")
        logger.error("   1. 是否在.env文件中设置了 BAILIAN_API_KEY")
        logger.error("   2. API Key是否有效且有足够额度")
        logger.error("   3. 网络连接是否可以访问百炼API服务")
        logger.error("   4. 是否已安装所需依赖")
        logger.error("   5. screenshots目录中是否有测试用的截图文件")

if __name__ == "__main__":
    main()