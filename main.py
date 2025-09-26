#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent 移动GUI智能体系统主程序

基于AgenticX框架构建的四智能体协作系统，
融合MobileAgent v3架构和五阶段学习方法论。

Author: AgenticX Team
Date: 2025
Version: 1.0.0 (基于AgenticX框架重构)
"""

import asyncio
import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# 添加项目根目录到Python路径# 添加项目路径
# 为了支持从 'agenticx-guiagent' 进行绝对导入，需要将其父目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 加载.env文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"已加载环境变量文件: {env_path}")
    else:
        logger.error(f"环境变量文件不存在: {env_path}")
except ImportError:
    logger.error("未安装python-dotenv，跳过.env文件加载")

# 导入AgenticX核心组件
from agenticx.core.workflow import Workflow
from agenticx.core.event_bus import EventBus
from agenticx.llms import OpenAIProvider
from agenticx.llms.bailian_provider import BailianProvider
from agenticx.memory.component import MemoryComponent
from agenticx.tools.executor import ToolExecutor

# 导入AgenticX-GUIAgent内部的Platform定义
from tools.gui_tools import Platform

# 导入AgenticX-GUIAgent组件
try:
    from agents import ManagerAgent, ExecutorAgent, ActionReflectorAgent, NotetakerAgent
    from core.info_pool import InfoPool
    from tools.gui_tools import GUIToolManager
    from workflows.collaboration import AgentCoordinator
    from config import AgenticXGUIAgentConfig, AgentConfig
    from utils import setup_logger, load_config, validate_agenticx_config
    from learning.learning_engine import LearningEngine
    from evaluation.framework import EvaluationFramework
except ImportError as e:
    import traceback
    print(f"导入错误: {e}")
    traceback.print_exc()
    print("请确保所有必要的模块都已正确安装和配置")
    sys.exit(1)


class AgenticXGUIAgentApp:
    """
    AgenticX-GUIAgent应用程序主类
    
    基于AgenticX框架的四智能体协作系统，
    集成了完整的配置管理、学习引擎、工具管理和评估框架。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化AgenticX-GUIAgent应用程序
        
        Args:
            config_path: 配置文件路径，默认为config.yaml
        """
        self.config_path = config_path or "config.yaml"
        self.config: Optional[AgenticXGUIAgentConfig] = None
        # 配置loguru日志并禁用标准logging
        import logging
        
        # 禁用标准logging的根logger
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # 配置loguru
        logger.remove()  # 移除默认处理器
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )
        logger.add(
            "logs/agenticx-guiagent.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8"
        )
        
        # 拦截标准logging并重定向到loguru
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                # 获取对应的loguru级别
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                
                # 查找调用者
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                
                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
        
        # 设置拦截器
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
        self.logger = logger
        
        # AgenticX核心组件
        self.platform = Platform.ANDROID  # 设置为Android平台
        self.event_bus: Optional[EventBus] = None
        self.llm_provider: Optional[OpenAIProvider] = None
        
        # AgenticX-GUIAgent核心组件
        self.info_pool: Optional[InfoPool] = None
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.tool_manager: Optional[GUIToolManager] = None
        self.evaluation_framework: Optional[EvaluationFramework] = None
        
        # 四个核心智能体
        self.manager_agent: Optional[ManagerAgent] = None
        self.executor_agent: Optional[ExecutorAgent] = None
        self.reflector_agent: Optional[ActionReflectorAgent] = None
        self.notetaker_agent: Optional[NotetakerAgent] = None
            
    async def initialize(self) -> None:
        """
        异步初始化所有组件
        """
        logger.info("开始初始化AgenticX-GUIAgent系统...")
        
        try:
            # 加载和验证配置
            await self._load_and_validate_config()
            
            # 初始化AgenticX核心组件
            await self._initialize_agenticx_components()
            
            # 初始化AgenticX-GUIAgent组件
            await self._initialize_agenticx_guiagent_components()
            
            # 启动核心组件
            await self._start_components()

            # 初始化智能体
            await self._initialize_agents()
            
            # 启动智能体
            await self._start_agents()
            
            # 初始化协调器
            await self._initialize_coordinator()
            
            logger.info("AgenticX-GUIAgent系统初始化完成！")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise

    async def _load_and_validate_config(self) -> None:
        """
        加载和验证配置文件
        """
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
            # 创建默认配置
            default_config = {
                'agenticx': {
                    'event_bus': {'enabled': True},
                    'components': {'auto_initialize': True},
                    'tools': {'timeout_default': 30.0},
                    'platform': {'name': 'AgenticX-GUIAgent', 'version': '2.0.0'}
                },
                'llm': {
                    'provider': 'openai',
                    'model': 'gpt-4o-mini',
                    'temperature': 0.3
                },
                'agents': [
                    {'id': 'manager_agent', 'name': 'Manager智能体'},
                    {'id': 'executor_agent', 'name': 'Executor智能体'},
                    {'id': 'actionreflector_agent', 'name': 'ActionReflector智能体'},
                    {'id': 'notetaker_agent', 'name': 'Notetaker智能体'}
                ],
                'mobile': {'platform': 'android'}
            }
            config_data = default_config
        else:
            # 加载配置文件
            config_data = load_config(config_file)
        
        # 验证AgenticX配置
        validate_agenticx_config(config_data)
        
        # 创建配置对象
        self.config = AgenticXGUIAgentConfig.from_dict(config_data)
        logger.info("配置加载和验证完成")
    
    async def _initialize_agenticx_components(self) -> None:
        """
        初始化AgenticX核心组件
        """
        # 初始化事件总线
        self.event_bus = EventBus()
        logger.info("AgenticX EventBus初始化完成")
        
        # Platform已设置为Android
        logger.info(f"Platform设置为: {self.platform}")
        
        # 初始化LLM提供者
        llm_config = self.config.llm
        
        # 检查是否使用百炼模型
        if llm_config.model in ['qwen-vl-max', 'qwen-vl-plus', 'qwen-max', 'qwen-plus']:
            # 使用百炼提供者
            import os
            api_key = os.getenv('BAILIAN_API_KEY') or llm_config.api_key
            self.llm_provider = BailianProvider(
                api_key=api_key,
                model=llm_config.model,
                temperature=getattr(llm_config, 'temperature', 0.3)
            )
            logger.info(f"百炼LLM提供者初始化完成，模型: {llm_config.model}")
        else:
            # 使用OpenAI提供者
            self.llm_provider = OpenAIProvider(
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                timeout=getattr(llm_config, 'timeout', 30.0),
                max_retries=getattr(llm_config, 'max_retries', 3)
            )
            logger.info(f"OpenAI LLM提供者初始化完成，模型: {llm_config.model}")
    
    async def _initialize_agenticx_guiagent_components(self) -> None:
        """
        初始化AgenticX-GUIAgent组件
        """
        # 初始化信息池
        self.info_pool = InfoPool(event_bus=self.event_bus)
        logger.info("InfoPool初始化完成")
        
        # 初始化学习引擎（如果可用）
        try:
            self.learning_engine = LearningEngine(
                info_pool=self.info_pool,
                config=self.config.learning
            )
            await self.learning_engine.initialize()
            logger.info("学习引擎初始化完成")
        except Exception as e:
            logger.warning(f"学习引擎初始化失败，将跳过: {e}")
        
        # 初始化工具管理器（如果可用）
        try:
            self.tool_manager = GUIToolManager(
                event_bus=self.event_bus,
                enable_monitoring=True,
                enable_caching=True
            )
            await self.tool_manager.initialize()
            logger.info("GUI工具管理器初始化完成")
        except Exception as e:
            logger.warning(f"工具管理器初始化失败，将跳过: {e}")
        
        # 初始化评估框架（如果可用）
        try:
            self.evaluation_framework = EvaluationFramework(
                base_dir=".",
                event_bus=self.event_bus
            )
            await self.evaluation_framework.initialize()
            logger.info("评估框架初始化完成")
        except Exception as e:
            logger.warning(f"评估框架初始化失败，将跳过: {e}")

    async def _start_components(self) -> None:
        """
        启动核心组件
        """
        logger.info("开始启动核心组件...")
        await self.info_pool.start()
        logger.info("InfoPool已启动")

    async def _initialize_agents(self) -> None:
        """
        初始化四个核心智能体
        """
        logger.info("初始化四个核心智能体...")
        
        try:
            # 创建智能体配置
            manager_config = self.config.get_agent_config('manager')
            executor_config = self.config.get_agent_config('executor')
            reflector_config = self.config.get_agent_config('reflector')
            notetaker_config = self.config.get_agent_config('notetaker')

            # 实例化所有智能体
            self.manager_agent = ManagerAgent(
                llm_provider=self.llm_provider,
                agent_config=manager_config,
                info_pool=self.info_pool,
                learning_engine=self.learning_engine
            )
            
            self.executor_agent = ExecutorAgent(
                llm_provider=self.llm_provider,
                agent_config=executor_config,
                info_pool=self.info_pool,
                tool_manager=self.tool_manager
            )
            
            self.reflector_agent = ActionReflectorAgent(
                llm_provider=self.llm_provider,
                agent_config=reflector_config,
                info_pool=self.info_pool,
                learning_engine=self.learning_engine
            )
            
            self.notetaker_agent = NotetakerAgent(
                llm_provider=self.llm_provider,
                agent_config=notetaker_config,
                info_pool=self.info_pool
            )
            logger.info("所有智能体实例化完成")

        except Exception as e:
            logger.error(f"智能体实例化失败: {e}")
            logger.info("尝试使用简化模式实例化智能体...")
            # 使用简化的智能体初始化
            manager_config = AgentConfig(id="manager", name="Manager智能体")
            executor_config = AgentConfig(id="executor", name="Executor智能体")
            reflector_config = AgentConfig(id="reflector", name="ActionReflector智能体")
            notetaker_config = AgentConfig(id="notetaker", name="Notetaker智能体")
            
            self.manager_agent = ManagerAgent(agent_config=manager_config, info_pool=self.info_pool)
            self.executor_agent = ExecutorAgent(agent_config=executor_config, info_pool=self.info_pool)
            self.reflector_agent = ActionReflectorAgent(agent_config=reflector_config, info_pool=self.info_pool)
            self.notetaker_agent = NotetakerAgent(agent_config=notetaker_config, info_pool=self.info_pool)
            logger.info("使用简化模式实例化智能体完成")

    async def _start_agents(self):
        """
        启动所有智能体
        """
        logger.info("开始启动所有智能体...")
        try:
            # 启动所有智能体
            await asyncio.gather(
                self.manager_agent.start(),
                self.executor_agent.start(),
                self.reflector_agent.start(),
                self.notetaker_agent.start()
            )
            logger.info("所有智能体已启动")
        except Exception as e:
            logger.error(f"启动智能体时出错: {e}")
            raise
    
    async def _initialize_coordinator(self) -> None:
        """
        初始化智能体协调器
        """
        try:
            # 使用AgentCoordinator
            self.agent_coordinator = AgentCoordinator(
                agents={
                    'manager': self.manager_agent,
                    'executor': self.executor_agent,
                    'reflector': self.reflector_agent,
                    'notetaker': self.notetaker_agent
                },
                info_pool=self.info_pool
            )
        except Exception as e:
            logger.warning(f"协调器初始化失败，使用简化版本: {e}")
            # 使用简化的协调器
            self.agent_coordinator = AgentCoordinator(
                agents={
                    'manager': self.manager_agent,
                    'executor': self.executor_agent,
                    'reflector': self.reflector_agent,
                    'notetaker': self.notetaker_agent
                },
                info_pool=self.info_pool
            )
        
        logger.info("智能体协调器初始化完成")
    
    async def execute_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        执行移动GUI任务
        
        Args:
            task_description: 任务描述
            **kwargs: 额外参数
            
        Returns:
            任务执行结果
        """
        if not self.agent_coordinator:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        logger.info(f"开始执行任务: {task_description}")
        
        try:
            # 通过协调器执行任务
            result = await self.agent_coordinator.execute_task(
                task_description=task_description,
                **kwargs
            )
            
            # 处理不同的结果格式
            if hasattr(result, 'success'):
                success = result.success
                logger.info(f"任务执行完成: {success}")
            else:
                success = result.get('status') == 'success' if isinstance(result, dict) else True
                logger.info(f"任务执行完成: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'task': task_description
            }
    
    async def learn_from_experience(self, experience_data: dict) -> None:
        """
        从经验中学习
        
        Args:
            experience_data: 经验数据
        """
        if self.learning_engine:
            try:
                await self.learning_engine.learn_from_experience(experience_data)
                logger.info("经验学习完成")
            except Exception as e:
                logger.error(f"经验学习失败: {e}")
    
    async def evaluate_performance(self, evaluation_config: dict = None) -> Dict[str, Any]:
        """
        评估系统性能
        
        Args:
            evaluation_config: 评估配置
            
        Returns:
            评估结果
        """
        if self.evaluation_framework:
            try:
                return await self.evaluation_framework.evaluate(
                    agents=[
                        self.manager_agent,
                        self.executor_agent,
                        self.reflector_agent,
                        self.notetaker_agent
                    ],
                    config=evaluation_config
                )
            except Exception as e:
                logger.error(f"性能评估失败: {e}")
                return {'status': 'failed', 'error': str(e)}
        else:
            logger.warning("评估框架未初始化")
            return {'status': 'skipped', 'reason': 'evaluation framework not available'}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        status = {
            'platform_status': f'active ({self.platform.value})' if self.platform else 'inactive',
            'event_bus_status': 'active' if self.event_bus else 'inactive',
            'agents_status': {
                'manager': 'active' if self.manager_agent else 'inactive',
                'executor': 'active' if self.executor_agent else 'inactive',
                'reflector': 'active' if self.reflector_agent else 'inactive',
                'notetaker': 'active' if self.notetaker_agent else 'inactive'
            },
            'components_status': {
                'info_pool': 'active' if self.info_pool else 'inactive',
                'coordinator': 'active' if self.agent_coordinator else 'inactive',
                'learning_engine': 'active' if self.learning_engine else 'inactive',
                'tool_manager': 'active' if self.tool_manager else 'inactive',
                'evaluation_framework': 'active' if self.evaluation_framework else 'inactive'
            }
        }
        
        return status
    
    async def shutdown(self) -> None:
        """
        关闭系统
        """
        logger.info("开始关闭AgenticX-GUIAgent系统...")
        
        try:
            # 停止所有智能体
            if all([self.manager_agent, self.executor_agent, 
                   self.reflector_agent, self.notetaker_agent]):
                await asyncio.gather(
                    self.manager_agent.stop() if hasattr(self.manager_agent, 'stop') else asyncio.sleep(0),
                    self.executor_agent.stop() if hasattr(self.executor_agent, 'stop') else asyncio.sleep(0),
                    self.reflector_agent.stop() if hasattr(self.reflector_agent, 'stop') else asyncio.sleep(0),
                    self.notetaker_agent.stop() if hasattr(self.notetaker_agent, 'stop') else asyncio.sleep(0),
                    return_exceptions=True
                )
            
            # 停止协调器
            if self.agent_coordinator and hasattr(self.agent_coordinator, 'shutdown'):
                await self.agent_coordinator.shutdown()
            
            # 清理组件
            if self.tool_manager and hasattr(self.tool_manager, 'cleanup'):
                await self.tool_manager.cleanup()
            
            if self.info_pool and hasattr(self.info_pool, 'cleanup'):
                await self.info_pool.cleanup()
            
            # Platform是枚举，无需停止操作
            logger.info(f"Platform {self.platform} 无需停止操作")
            
            logger.info("AgenticX-GUIAgent系统已关闭")
            
        except Exception as e:
            logger.error(f"关闭系统时出错: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    
    Returns:
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="AgenticX-GUIAgent - 基于AgenticX框架的移动GUI智能体系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                          # 交互模式
  python main.py --task "打开微信应用"      # 执行单个任务
  python main.py --evaluate              # 运行性能评估
  python main.py --config custom.yaml    # 使用自定义配置
  python main.py --status                # 显示系统状态
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        help="要执行的任务描述"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true", 
        help="运行性能评估"
    )
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="显示系统状态"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="强制进入交互模式"
    )
    
    return parser


async def run_interactive_mode(app: AgenticXGUIAgentApp) -> None:
    """
    运行交互模式
    
    Args:
        app: AgenticX-GUIAgent应用实例
    """
    # 示例任务
    task_examples = [
        "打开微信应用",
        "在设置中找到通知选项并打开",
        "发送一条消息给张三",
        "在应用商店搜索并下载抖音",
        "截取当前屏幕截图",
        "向下滑动页面"
    ]
    
    print("\n" + "=" * 60)
    print("🤖 AgenticX-GUIAgent 移动GUI智能体系统")
    print("基于AgenticX框架 v2.0.0")
    print("=" * 60)
    print("\n💡 支持的示例任务:")
    for i, task in enumerate(task_examples, 1):
        print(f"  {i}. {task}")
    
    print("\n📝 请输入任务描述 (输入 'help' 查看帮助, 'quit' 退出):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if user_input.lower() in ['help', 'h']:
                print("\n📖 帮助信息:")
                print("  - 输入任务描述来执行GUI操作")
                print("  - 'status' - 查看系统状态")
                print("  - 'eval' - 运行性能评估")
                print("  - 'quit' - 退出系统")
                continue
            
            if user_input.lower() == 'status':
                status = await app.get_system_status()
                print("\n📊 系统状态:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == 'eval':
                print("\n🔍 开始性能评估...")
                eval_result = await app.evaluate_performance()
                print(f"📈 评估结果: {eval_result}")
                continue
            
            if not user_input:
                continue
            
            # 执行任务
            # print(f"\n🚀 执行任务: {user_input}")
            result = await app.execute_task(user_input)
            # print(f"✅ 执行结果: {result}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


async def main() -> int:
    """
    主函数
    
    Returns:
        退出代码
    """
    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # loguru日志已在AgenticXGUIAgentApp初始化时配置
    
    # 创建应用实例
    app = AgenticXGUIAgentApp(config_path=args.config)
    
    try:
        # 初始化应用
        await app.initialize()
        
        # 根据参数执行不同操作
        if args.task:
            # 执行指定任务
            print(f"\n🚀 执行任务: {args.task}")
            result = await app.execute_task(args.task)
            print(f"✅ 任务执行结果: {result}")
        
        elif args.evaluate:
            # 运行性能评估
            print("\n🔍 开始性能评估...")
            evaluation_result = await app.evaluate_performance()
            print(f"📈 评估结果: {evaluation_result}")
        
        elif args.status:
            # 显示系统状态
            status = await app.get_system_status()
            print("\n📊 系统状态:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        
        else:
            # 交互模式
            await run_interactive_mode(app)
    
    except Exception as e:
        print(f"\n❌ 应用运行错误: {e}")
        logger.exception("详细错误信息:")
        return 1
    
    finally:
        # 关闭应用
        print("\n🔄 正在关闭系统...")
        await app.shutdown()
        print("✅ 系统已安全关闭")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        sys.exit(1)