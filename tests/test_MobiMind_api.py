"""
测试 MobiMind API
"""

import os
import base64
import glob
from openai import OpenAI

# =====================================
# 🔧 配置区
# =====================================
IMAGE_DIR = "./images"          # 图片目录（和脚本同级）
MODEL_NAME = "IPADS-SAI/MobiMind-Grounder-3B"
BASE_URL = "http://localhost:6008/v1"
API_KEY = "sk-***"         # vLLM 不校验，非空即可
QUESTION = "请详细描述这张图片中的内容。"

# =====================================
# 💡 辅助函数：读取图片并转为 base64
# =====================================
def image_to_base64(image_path):
    """读取本地图片，转为 data URL 格式 base64 字符串"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 推测 MIME 类型（简单支持 jpg/png）
    ext = os.path.splitext(image_path)[-1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"  # 默认 fallback

    return f"data:{mime_type};base64,{encoded}"

# =====================================
# 🚀 主程序
# =====================================
def main():
    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    # 获取所有图片路径
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print(f"❌ 在 '{IMAGE_DIR}' 目录下未找到任何图片！")
        return

    print(f"📁 找到 {len(image_paths)} 张图片，开始处理...\n")

    for idx, image_path in enumerate(image_paths, 1):
        print(f"--- 图片 {idx}: {os.path.basename(image_path)} ---")

        # 1. 将图片转为 base64
        image_url = image_to_base64(image_path)

        # 2. 构建多模态消息（符合 OpenAI API 格式）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
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
            # 3. 调用模型（启用流式返回）
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=512,  # 可选：限制最大输出长度
                temperature=0.7, # 可选：控制随机性
                stream=True,     # 启用流式返回
            )

            # 4. 流式输出结果
            print(f"🤖 回答:")
            collected_chunks = []
            collected_content = ""
            
            # 遍历流式响应
            for chunk in response:
                # 提取当前块的内容
                if hasattr(chunk.choices[0].delta, 'content'):
                    content_delta = chunk.choices[0].delta.content
                    if content_delta:
                        print(content_delta, end="", flush=True)  # 实时打印，不换行
                        collected_chunks.append(content_delta)
                        collected_content += content_delta
            
            # 完成后打印换行
            print("\n")

        except Exception as e:
            print(f"❌ 调用失败: {e}\n")

        print("="*60 + "\n")

if __name__ == "__main__":
    main()
