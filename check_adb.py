#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADB连接检查脚本

用于检查ADB设备连接状态和基本功能
"""

import subprocess
import sys
from pathlib import Path

def check_adb_installed():
    """检查ADB是否已安装"""
    try:
        result = subprocess.run(["adb", "version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ADB已安装")
            print(f"版本信息: {result.stdout.strip().split()[0]}")
            return True
        else:
            print("❌ ADB未正确安装")
            return False
    except FileNotFoundError:
        print("❌ ADB未安装或不在PATH中")
        print("请安装Android SDK Platform Tools或确保adb在系统PATH中")
        return False
    except Exception as e:
        print(f"❌ 检查ADB安装状态时出错: {e}")
        return False

def check_adb_devices():
    """检查连接的ADB设备"""
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"\n📱 ADB设备列表:")
            print(result.stdout.strip())
            
            # 分析设备状态
            devices = []
            for line in lines[1:]:  # 跳过标题行
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        device_id = parts[0]
                        status = parts[1]
                        devices.append((device_id, status))
            
            if not devices:
                print("⚠️ 没有检测到连接的设备")
                print("请确保:")
                print("1. 设备已通过USB连接到电脑")
                print("2. 设备已开启USB调试模式")
                print("3. 已授权电脑进行USB调试")
                return False
            
            connected_devices = [d for d in devices if d[1] == 'device']
            if connected_devices:
                print(f"\n✅ 发现 {len(connected_devices)} 个已连接的设备:")
                for device_id, status in connected_devices:
                    print(f"  - {device_id} ({status})")
                return True
            else:
                print(f"\n⚠️ 发现 {len(devices)} 个设备，但状态异常:")
                for device_id, status in devices:
                    print(f"  - {device_id} ({status})")
                print("\n请检查设备状态，确保已授权USB调试")
                return False
        else:
            print(f"❌ 执行adb devices失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 检查ADB设备时出错: {e}")
        return False

def test_adb_operations():
    """测试基本ADB操作"""
    print("\n🧪 测试基本ADB操作...")
    
    # 测试获取设备信息
    try:
        result = subprocess.run(["adb", "shell", "getprop", "ro.product.model"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            model = result.stdout.strip()
            print(f"✅ 设备型号: {model}")
        else:
            print(f"⚠️ 获取设备型号失败: {result.stderr}")
    except Exception as e:
        print(f"⚠️ 获取设备信息时出错: {e}")
    
    # 测试屏幕尺寸
    try:
        result = subprocess.run(["adb", "shell", "wm", "size"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            size_info = result.stdout.strip()
            print(f"✅ 屏幕尺寸: {size_info}")
        else:
            print(f"⚠️ 获取屏幕尺寸失败: {result.stderr}")
    except Exception as e:
        print(f"⚠️ 获取屏幕尺寸时出错: {e}")
    
    # 测试截图功能
    try:
        print("\n📸 测试截图功能...")
        # 在设备上截图
        result = subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/test_screenshot.png"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ 设备截图成功")
            
            # 拉取截图到本地
            local_path = "./test_screenshot.png"
            result = subprocess.run(["adb", "pull", "/sdcard/test_screenshot.png", local_path], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ 截图已保存到: {local_path}")
                
                # 清理设备上的临时文件
                subprocess.run(["adb", "shell", "rm", "/sdcard/test_screenshot.png"], 
                             capture_output=True, timeout=5)
            else:
                print(f"⚠️ 拉取截图失败: {result.stderr}")
        else:
            print(f"⚠️ 设备截图失败: {result.stderr}")
    except Exception as e:
        print(f"⚠️ 测试截图功能时出错: {e}")
    
    # 测试点击功能（安全位置）
    try:
        print("\n👆 测试点击功能（点击屏幕中央）...")
        # 先获取屏幕尺寸
        result = subprocess.run(["adb", "shell", "wm", "size"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            size_line = result.stdout.strip()
            if "Physical size:" in size_line:
                size_part = size_line.split("Physical size:")[1].strip()
                width, height = map(int, size_part.split('x'))
                center_x, center_y = width // 2, height // 2
                
                # 点击屏幕中央
                result = subprocess.run(["adb", "shell", "input", "tap", str(center_x), str(center_y)], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"✅ 点击测试成功: ({center_x}, {center_y})")
                else:
                    print(f"⚠️ 点击测试失败: {result.stderr}")
            else:
                print("⚠️ 无法解析屏幕尺寸")
        else:
            print(f"⚠️ 获取屏幕尺寸失败，跳过点击测试")
    except Exception as e:
        print(f"⚠️ 测试点击功能时出错: {e}")

def main():
    """主函数"""
    print("🔍 ADB连接检查工具")
    print("=" * 50)
    
    # 检查ADB安装
    if not check_adb_installed():
        print("\n❌ 请先安装ADB工具")
        sys.exit(1)
    
    # 检查设备连接
    if not check_adb_devices():
        print("\n❌ 没有可用的ADB设备")
        print("\n📋 设备连接步骤:")
        print("1. 在Android设备上开启开发者选项")
        print("2. 开启USB调试模式")
        print("3. 使用USB线连接设备到电脑")
        print("4. 在设备上授权USB调试")
        print("5. 重新运行此脚本")
        sys.exit(1)
    
    # 测试基本操作
    test_adb_operations()
    
    print("\n🎉 ADB连接检查完成！")
    print("\n💡 提示:")
    print("- 如果所有测试都通过，说明ADB工具可以正常使用")
    print("- 现在可以运行多模态ExecutorAgent进行真实设备操作")
    print("- 如果遇到权限问题，请确保设备已授权USB调试")

if __name__ == "__main__":
    main()