#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent 系统测试脚本
用于验证系统各组件是否正常工作

Usage:
    python test_system.py
    python test_system.py --component agents
    python test_system.py --device-test
"""

import asyncio
import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
    
    def print_header(self, title: str):
        """打印测试标题"""
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    
    def print_test(self, test_name: str, status: str, message: str = ""):
        """打印测试结果"""
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_symbol} {test_name:<40} [{status}] {message}")
        
        if status == "PASS":
            self.passed_tests += 1
        elif status == "FAIL":
            self.failed_tests += 1
    
    def test_imports(self) -> bool:
        """测试基础导入"""
        self.print_header("基础导入测试")
        
        # 测试AgenticX导入
        try:
            import agenticx
            self.print_test("AgenticX框架", "PASS", f"版本: {getattr(agenticx, '__version__', 'unknown')}")
        except ImportError as e:
            self.print_test("AgenticX框架", "FAIL", str(e))
            return False
        
        # 测试核心依赖
        dependencies = [
            ("aiohttp", "异步HTTP客户端"),
            ("aiofiles", "异步文件操作"),
            ("pydantic", "数据验证"),
            ("yaml", "YAML配置解析"),
            ("numpy", "数值计算"),
            ("opencv-python", "图像处理"),
            ("PIL", "图像库"),
        ]
        
        for module_name, description in dependencies:
            try:
                if module_name == "opencv-python":
                    import cv2
                    self.print_test(f"{description}", "PASS", f"OpenCV {cv2.__version__}")
                elif module_name == "PIL":
                    from PIL import Image
                    self.print_test(f"{description}", "PASS", "Pillow")
                else:
                    module = __import__(module_name.replace("-", "_"))
                    version = getattr(module, '__version__', 'unknown')
                    self.print_test(f"{description}", "PASS", f"版本: {version}")
            except ImportError as e:
                self.print_test(f"{description}", "FAIL", str(e))
        
        return True
    
    def test_config(self) -> bool:
        """测试配置文件"""
        self.print_header("配置文件测试")
        
        # 检查配置文件
        config_files = [
            ("config.yaml", "主配置文件"),
            (".env.example", "环境变量模板"),
            ("requirements.txt", "依赖文件"),
        ]
        
        for file_name, description in config_files:
            file_path = project_root / file_name
            if file_path.exists():
                self.print_test(f"{description}", "PASS", f"文件存在: {file_path}")
            else:
                self.print_test(f"{description}", "FAIL", f"文件不存在: {file_path}")
        
        # 检查.env文件
        env_file = project_root / ".env"
        if env_file.exists():
            self.print_test("环境变量配置", "PASS", "已配置.env文件")
        else:
            self.print_test("环境变量配置", "WARN", "未找到.env文件，请复制.env.example并配置")
        
        # 测试配置加载
        try:
            from utils import load_config
            config = load_config("config.yaml")
            self.print_test("配置文件解析", "PASS", f"加载了{len(config)}个配置项")
        except Exception as e:
            self.print_test("配置文件解析", "FAIL", str(e))
        
        return True
    
    def test_agenticx_components(self) -> bool:
        """测试AgenticX组件"""
        self.print_header("AgenticX组件测试")
        
        # 测试核心组件
        components = [
            ("agenticx.core.platform", "Platform", "平台服务"),
            ("agenticx.core.event_bus", "EventBus", "事件总线"),
            ("agenticx.core.agent", "Agent", "智能体基类"),
            ("agenticx.tools.base", "BaseTool", "工具基类"),
            ("agenticx.memory.component", "Memory", "内存组件"),
        ]
        
        for module_name, class_name, description in components:
            try:
                module = __import__(module_name, fromlist=[class_name])
                component_class = getattr(module, class_name)
                self.print_test(f"{description}", "PASS", f"{module_name}.{class_name}")
            except (ImportError, AttributeError) as e:
                self.print_test(f"{description}", "FAIL", str(e))
        
        return True
    
    def test_agenticx_guiagent_components(self) -> bool:
        """测试AgenticX-GUIAgent组件"""
        self.print_header("AgenticX-GUIAgent组件测试")
        
        # 测试目录结构
        directories = [
            ("agents", "智能体模块"),
            ("core", "核心模块"),
            ("tools", "工具模块"),
            ("learning", "学习模块"),
            ("evaluation", "评估模块"),
            ("workflows", "工作流模块"),
        ]
        
        for dir_name, description in directories:
            dir_path = project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                files_count = len(list(dir_path.glob("*.py")))
                self.print_test(f"{description}", "PASS", f"目录存在，包含{files_count}个Python文件")
            else:
                self.print_test(f"{description}", "FAIL", f"目录不存在: {dir_path}")
        
        # 测试主要模块导入
        try:
            # 测试智能体导入
            from agents import ManagerAgent, ExecutorAgent, ActionReflectorAgent, NotetakerAgent
            self.print_test("四智能体导入", "PASS", "Manager, Executor, ActionReflector, Notetaker")
        except ImportError as e:
            self.print_test("四智能体导入", "FAIL", str(e))
        
        try:
            # 测试工具导入
            from tools.gui_tools import GUITool
            self.print_test("GUI工具导入", "PASS", "GUITool基类")
        except ImportError as e:
            self.print_test("GUI工具导入", "FAIL", str(e))
        
        return True
    
    def test_device_connection(self) -> bool:
        """测试设备连接"""
        self.print_header("设备连接测试")
        
        # 检查ADB
        import subprocess
        try:
            result = subprocess.run(["adb", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                self.print_test("ADB工具", "PASS", version_line)
            else:
                self.print_test("ADB工具", "FAIL", "ADB命令执行失败")
                return False
        except FileNotFoundError:
            self.print_test("ADB工具", "FAIL", "ADB未安装或不在PATH中")
            return False
        
        # 检查设备连接
        try:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                devices = [line for line in lines if line.strip() and 'device' in line]
                
                if devices:
                    self.print_test("设备连接", "PASS", f"检测到{len(devices)}个设备")
                    for device in devices:
                        print(f"    📱 {device}")
                else:
                    self.print_test("设备连接", "WARN", "未检测到已连接的设备")
            else:
                self.print_test("设备连接", "FAIL", "无法获取设备列表")
        except Exception as e:
            self.print_test("设备连接", "FAIL", str(e))
        
        return True
    
    async def test_system_integration(self) -> bool:
        """测试系统集成"""
        self.print_header("系统集成测试")
        
        try:
            # 测试主应用类导入
            from main import AgenticXGUIAgentApp
            self.print_test("主应用类", "PASS", "AgenticXGUIAgentApp导入成功")
            
            # 测试应用初始化（不实际启动）
            app = AgenticXGUIAgentApp()
            self.print_test("应用初始化", "PASS", "应用对象创建成功")
            
            # 测试配置加载
            await app._load_and_validate_config()
            self.print_test("配置加载", "PASS", "配置验证通过")
            
        except Exception as e:
            self.print_test("系统集成", "FAIL", str(e))
            return False
        
        return True
    
    def test_environment_variables(self) -> bool:
        """测试环境变量"""
        self.print_header("环境变量测试")
        
        import os
        from dotenv import load_dotenv
        
        # 加载.env文件
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            self.print_test(".env文件加载", "PASS", "环境变量已加载")
        else:
            self.print_test(".env文件加载", "WARN", "未找到.env文件")
        
        # 检查关键环境变量
        required_vars = [
            ("LLM_PROVIDER", "LLM提供商"),
        ]
        
        optional_vars = [
            ("OPENAI_API_KEY", "OpenAI API密钥"),
            ("DEEPSEEK_API_KEY", "DeepSeek API密钥"),
            ("KIMI_API_KEY", "Kimi API密钥"),
        ]
        
        for var_name, description in required_vars:
            value = os.getenv(var_name)
            if value:
                self.print_test(f"{description}", "PASS", f"{var_name}={value}")
            else:
                self.print_test(f"{description}", "FAIL", f"{var_name}未设置")
        
        # 检查至少有一个API密钥
        api_keys = [os.getenv(var) for var, _ in optional_vars]
        if any(api_keys):
            self.print_test("API密钥配置", "PASS", "至少配置了一个LLM API密钥")
        else:
            self.print_test("API密钥配置", "WARN", "未配置任何LLM API密钥")
        
        return True
    
    def print_summary(self):
        """打印测试总结"""
        self.print_header("测试总结")
        
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 测试统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {self.passed_tests} ✅")
        print(f"   失败: {self.failed_tests} ❌")
        print(f"   成功率: {success_rate:.1f}%")
        
        if self.failed_tests == 0:
            print(f"\n🎉 所有测试通过！系统准备就绪。")
            print(f"\n🚀 可以开始使用AgenticX-GUIAgent:")
            print(f"   python main.py --interactive")
        else:
            print(f"\n⚠️  有{self.failed_tests}个测试失败，请检查上述错误信息。")
            print(f"\n📚 参考文档: SETUP_GUIDE.md")
    
    async def run_all_tests(self, test_device: bool = False):
        """运行所有测试"""
        print("🧪 AgenticX-GUIAgent 系统测试")
        print(f"📁 项目路径: {project_root}")
        
        # 基础测试
        self.test_imports()
        self.test_config()
        self.test_environment_variables()
        
        # 组件测试
        self.test_agenticx_components()
        self.test_agenticx_guiagent_components()
        
        # 设备测试（可选）
        if test_device:
            self.test_device_connection()
        
        # 集成测试
        await self.test_system_integration()
        
        # 打印总结
        self.print_summary()
        
        return self.failed_tests == 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AgenticX-GUIAgent系统测试")
    parser.add_argument("--device-test", action="store_true", help="包含设备连接测试")
    parser.add_argument("--component", choices=["imports", "config", "agents", "tools"], help="只测试特定组件")
    
    args = parser.parse_args()
    
    tester = SystemTester()
    
    try:
        if args.component:
            # 运行特定组件测试
            if args.component == "imports":
                tester.test_imports()
            elif args.component == "config":
                tester.test_config()
            elif args.component == "agents":
                tester.test_agenticx_guiagent_components()
            elif args.component == "tools":
                tester.test_device_connection()
        else:
            # 运行所有测试
            success = asyncio.run(tester.run_all_tests(test_device=args.device_test))
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()