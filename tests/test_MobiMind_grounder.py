#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobiMind Grounder 测试脚本
支持定位UI元素并在图片上绘制bbox和coordinates
"""

import os
import base64
import json
import re
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import warnings

# 配置matplotlib中文字体和忽略字体警告
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# =====================================
# 🔧 配置区
# =====================================
IMAGE_DIR = "./images"          # 图片目录（和脚本同级）
MODEL_NAME = "IPADS-SAI/MobiMind-Grounder-3B"
BASE_URL = "http://localhost:6008/v1"
API_KEY = "sk-***"         # vLLM 不校验，非空即可
OUTPUT_DIR = "./output"     # 输出目录

# 目标图片和元素配置
TARGET_IMAGE = "4.jpg"      # 目标图片文件名
TARGET_ELEMENT = "微信应用图标，位于屏幕最底部dock栏中间偏右位置，图标是纯绿色背景（#00C853）上有两个白色圆形对话气泡，这是中国最流行的聊天应用WeChat，绝对不是Gallery、相机或其他彩色图标"   # 要定位的元素描述
USER_INTENT = "点击微信"  # 用户意图

# =====================================
# =====================================
def image_to_base64(image_path: str, max_size: int = 1024) -> Tuple[str, float]:
    """将图片转换为base64编码，并压缩大图片，返回base64和缩放比例"""
    # 打开并可能压缩图片
    with Image.open(image_path) as img:
        # 获取原始尺寸
        original_width, original_height = img.size
        print(f"📏 原始图片尺寸: {original_width}x{original_height}")
        
        scale_factor = 1.0  # 默认缩放比例
        
        # 如果图片太大，进行压缩
        if max(original_width, original_height) > max_size:
            # 计算缩放比例
            scale_factor = max_size / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            print(f"🔄 压缩图片到: {new_width}x{new_height} (缩放比例: {scale_factor:.3f})")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            print(f"✅ 图片尺寸合适，无需压缩")
        
        # 转换为RGB（如果是RGBA）
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 保存到内存中的字节流
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        image_data = buffer.getvalue()
        
        print(f"📦 压缩后大小: {len(image_data)/1024:.1f}KB")
        
        # 转换为base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded}", scale_factor

def load_prompt_template(template_path: str) -> str:
    """加载提示词模板"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def format_prompt(template: str, reasoning: str, description: str) -> str:
    """格式化提示词"""
    return template.replace('{reasoning}', reasoning).replace('{description}', description)

def extract_json_from_response(response: str) -> Optional[Dict]:
    """从响应中提取JSON对象"""
    # 尝试直接解析
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # 尝试提取JSON块
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

def scale_coordinates_to_original(coords: List[int], scale_factor: float) -> List[int]:
    """将压缩图片的坐标缩放回原始图片坐标"""
    if scale_factor == 1.0:
        return coords  # 无需缩放
    
    # 缩放坐标（除以缩放因子）
    scaled_coords = [int(coord / scale_factor) for coord in coords]
    print(f"🔄 坐标缩放: {coords} -> {scaled_coords} (缩放因子: {1/scale_factor:.3f})")
    return scaled_coords

def draw_bbox_on_image(image_path: str, bbox: List[int], output_path: str):
    """在图片上绘制极简边界框，无标题和文字"""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)
    
    x_min, y_min, x_max, y_max = bbox
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=5, edgecolor='#ff0000', facecolor='none',
                             alpha=0.8)
    ax.add_patch(rect)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"🖼️  已保存极简BBox图片到: {output_path}")

def draw_coordinates_on_image(image_path: str, coordinates: List[int], output_path: str):
    """在图片上绘制极简坐标点，无标题和文字"""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)
    
    x, y = coordinates
    ax.plot(x, y, 'o', markersize=20, markerfacecolor='#ff0000', markeredgecolor='white', markeredgewidth=3)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"🖼️  已保存极简坐标点图片到: {output_path}")

def call_grounder_api(client: OpenAI, image_path: str, prompt: str, mode: str = "bbox") -> Tuple[Optional[Dict], float]:
    """调用Grounder API，返回结果和缩放因子"""
    # 将图片转为base64，获取缩放因子
    image_url, scale_factor = image_to_base64(image_path)
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]
    
    try:
        print(f"🤖 正在调用{mode.upper()}模式...")
        
        # 调用API（非流式返回，避免卡住）
        print(f"⏰ 设置10秒超时，开始请求...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.1,  # 降低温度以获得更稳定的结果
            stream=False,  # 改为非流式
            timeout=10,    # 缩短为10秒超时
        )
        
        # 获取非流式响应
        collected_content = response.choices[0].message.content
        print(f"📝 {mode.upper()}响应: {collected_content}")
        
        # 解析JSON响应
        result = extract_json_from_response(collected_content)
        if result:
            print(f"✅ {mode.upper()}解析成功: {result}")
            return result, scale_factor
        else:
            print(f"❌ {mode.upper()}JSON解析失败")
            print(f"原始响应: {collected_content}")
            return None, scale_factor
            
    except Exception as e:
        print(f"❌ {mode.upper()}调用失败: {e}")
        return None, scale_factor

# =====================================
# 🚀 主程序
# =====================================
def main():
    """主程序"""
    print("🎯 MobiMind Grounder 测试脚本")
    print("=" * 50)
    
    # 检查目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 创建输出目录: {OUTPUT_DIR}")
    
    # 检查目标图片
    image_path = os.path.join(IMAGE_DIR, TARGET_IMAGE)
    if not os.path.exists(image_path):
        print(f"❌ 目标图片不存在: {image_path}")
        return
    
    print(f"🖼️  目标图片: {image_path}")
    print(f"🎯 目标元素: {TARGET_ELEMENT}")
    print(f"💭 用户意图: {USER_INTENT}")
    print()
    
    # 初始化OpenAI客户端（添加默认超时）
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=10.0  # 设置默认10秒超时
    )
    
    # 加载提示词模板
    bbox_template_path = "grounder_bbox.md"
    coordinates_template_path = "grounder_coordinates.md"
    
    if not os.path.exists(bbox_template_path) or not os.path.exists(coordinates_template_path):
        print(f"❌ 提示词模板文件不存在")
        return
    
    bbox_template = load_prompt_template(bbox_template_path)
    coordinates_template = load_prompt_template(coordinates_template_path)
    
    # 格式化提示词
    bbox_prompt = format_prompt(bbox_template, USER_INTENT, TARGET_ELEMENT)
    coordinates_prompt = format_prompt(coordinates_template, USER_INTENT, TARGET_ELEMENT)
    
    # 1. 测试BBox模式
    print("🔍 开始BBox定位...")
    bbox_result, bbox_scale_factor = call_grounder_api(client, image_path, bbox_prompt, "bbox")
    
    if bbox_result and "bbox" in bbox_result:
        bbox = bbox_result["bbox"]
        print(f"📍 BBox结果 (压缩图片坐标): {bbox}")
        
        # 将坐标缩放回原始图片
        scaled_bbox = scale_coordinates_to_original(bbox, bbox_scale_factor)
        print(f"📍 BBox结果 (原始图片坐标): {scaled_bbox}")
        
        # 绘制BBox（使用缩放后的坐标）
        bbox_output_path = os.path.join(OUTPUT_DIR, f"{TARGET_IMAGE}_bbox.png")
        draw_bbox_on_image(image_path, scaled_bbox, bbox_output_path)
    else:
        print("❌ BBox定位失败")
    
    print()
    
    # 2. 测试Coordinates模式
    print("🔍 开始Coordinates定位...")
    coordinates_result, coordinates_scale_factor = call_grounder_api(client, image_path, coordinates_prompt, "coordinates")
    
    if coordinates_result and "coordinates" in coordinates_result:
        coordinates = coordinates_result["coordinates"]
        print(f"📍 Coordinates结果 (压缩图片坐标): {coordinates}")
        
        # 将坐标缩放回原始图片
        scaled_coordinates = scale_coordinates_to_original(coordinates, coordinates_scale_factor)
        print(f"📍 Coordinates结果 (原始图片坐标): {scaled_coordinates}")
        
        # 绘制Coordinates（使用缩放后的坐标）
        coordinates_output_path = os.path.join(OUTPUT_DIR, f"{TARGET_IMAGE}_coordinates.png")
        draw_coordinates_on_image(image_path, scaled_coordinates, coordinates_output_path)
    else:
        print("❌ Coordinates定位失败")
    
    print()
    print("🎉 测试完成！")
    print(f"📁 结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
