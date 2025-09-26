#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多模态ManagerAgent的任务分解功能
"""

import asyncio
from loguru import logger
import sys
import os
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
from agents.manager_agent import ManagerAgent
from core.info_pool import AgenticXGUIAgentInfoPool
from config import AgentConfig
from utils import setup_logger

# 设置日志
# logging.basicConfig replaced with loguru
logger = logger

async def test_multimodal_task_decomposition():
    """测试多模态任务分解功能"""
    logger.info("开始测试多模态任务分解功能")
    
    # 初始化百炼LLM提供者
    api_key = os.getenv('BAILIAN_API_KEY')
    if not api_key:
        logger.error("❌ 未设置BAILIAN_API_KEY环境变量")
        return
    
    llm_provider = BailianProvider(
        api_key=api_key,
        model="qwen-vl-max",
        temperature=0.3
    )
    
    # 初始化事件总线
    event_bus = EventBus()
    
    # 初始化信息池
    info_pool = AgenticXGUIAgentInfoPool()
    
    # 创建ManagerAgent配置
    manager_config = AgentConfig(
        id="test_manager",
        name="TestManagerAgent",
        role="task_manager",
        goal="测试多模态任务分解功能",
        backstory="我是测试用的Manager智能体，专门用于测试多模态任务分解功能。"
    )
    
    # 创建ManagerAgent
    manager_agent = ManagerAgent(
        agent_config=manager_config,
        llm=llm_provider,
        info_pool=info_pool,
        event_bus=event_bus
    )
    
    logger.info("ManagerAgent初始化完成")
    
    # 测试用例
    test_cases = [
        "打开微信应用",
        "在设置中找到通知选项并打开",
        "发送一条消息给张三",
        "截取当前屏幕截图",
        "向下滑动页面",
        "在应用商店搜索并下载抖音"
    ]
    
    for i, task_description in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试用例 {i}: {task_description}")
        logger.info(f"{'='*60}")
        
        try:
            # 执行任务分解
            result = await manager_agent._decompose_task(task_description)
            
            # 打印结果，清楚标识数据来源
            success = result.get('success', True)
            if success and result.get('model_used'):
                # 成功的LLM分析
                model_name = result.get('model_used', 'Unknown')
                logger.info(f"🤖 【{model_name} 多模态分析成功】 子任务数量: {result.get('total_subtasks', 0)}")
                logger.info(f"🧠 LLM分析结果: {result.get('analysis', 'N/A')}")
                logger.info(f"🎯 成功标准: {result.get('success_criteria', 'N/A')}")
            elif not success:
                # 失败情况
                error_msg = result.get('error', 'Unknown error')
                attempted_models = result.get('attempted_models', [])
                if attempted_models:
                    logger.error(f"❌ 【多模型降级失败】 尝试的模型: {', '.join(attempted_models)}")
                    logger.error(f"💥 错误信息: {error_msg}")
                else:
                    logger.error(f"❌ 【任务分解失败】 错误: {error_msg}")
                logger.info(f"📝 子任务数量: {result.get('total_subtasks', 0)}")
            else:
                # 其他情况（兼容性）
                logger.info(f"⚙️ 【未知方法】 子任务数量: {result.get('total_subtasks', 0)}")
                logger.info(f"📋 分析结果: {result.get('analysis', 'N/A')}")
                logger.info(f"✅ 成功标准: {result.get('success_criteria', 'N/A')}")
            
            logger.info("📝 子任务列表:")
            for j, subtask in enumerate(result.get('subtasks', []), 1):
                logger.info(f"  {j}. [{subtask.get('type', 'unknown')}] {subtask.get('description', '')}")
                if subtask.get('target'):
                    logger.info(f"     🎯 目标: {subtask['target']}")
                logger.info(f"     ⏱️ 优先级: {subtask.get('priority', 'medium')}, 预估时间: {subtask.get('estimated_time', 0)}秒")
            
            if result.get('dependencies'):
                logger.info(f"🔗 依赖关系: {result['dependencies']}")
            
        except Exception as e:
            logger.error(f"任务分解失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 等待一下再进行下一个测试
        await asyncio.sleep(1)
    
    logger.info("\n所有测试完成")

async def test_with_screenshot():
    """测试带截图的任务分解"""
    
    # 检查是否有现有截图
    screenshot_dir = "./screenshots"
    if os.path.exists(screenshot_dir):
        import glob
        screenshot_files = glob.glob(os.path.join(screenshot_dir, "*.png"))
        if screenshot_files:
            latest_screenshot = max(screenshot_files, key=os.path.getctime)
            logger.info(f"找到截图文件: {latest_screenshot}")
            
            # 初始化百炼LLM提供者
            api_key = os.getenv('BAILIAN_API_KEY')
            if api_key:
                llm_provider = BailianProvider(
                    api_key=api_key,
                    model="qwen-vl-max",
                    temperature=0.3
                )
                
                # 创建任务分解工具
                from agents.manager_agent import MultimodalTaskDecompositionTool
                
                tool = MultimodalTaskDecompositionTool(llm_provider=llm_provider)
                
                # 测试多模态分解
                task_description = "根据当前屏幕内容，执行最合适的操作"
                logger.info(f"测试多模态任务分解: {task_description}")
                
                try:
                    result = await tool.aexecute(
                        task_description=task_description,
                        screenshot_path=latest_screenshot
                    )
                    
                    logger.info(f"分解结果: {result}")
                    
                except Exception as e:
                    logger.error(f"多模态分解失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning("未设置API密钥，跳过多模态测试")
        else:
            logger.warning("未找到截图文件")
    else:
        logger.warning("截图目录不存在")

def main():
    """主函数"""
    logger.info("开始测试多模态ManagerAgent")
    
    # 运行基本测试
    asyncio.run(test_multimodal_task_decomposition())
    
    # 运行截图测试
    logger.info("\n开始测试带截图的任务分解")
    asyncio.run(test_with_screenshot())

if __name__ == "__main__":
    main()