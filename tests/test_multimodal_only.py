#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试多模态embedding的脚本
验证修复后的配置是否能正确路由到多模态embedding
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from knowledge.embedding_config import EmbeddingRequest, ContentType
from knowledge.config_loader import load_embedding_config
from knowledge.embedding_factory import EmbeddingFactory
from dotenv import load_dotenv

# 加载环境变量
from dotenv import load_dotenv
# 明确指定.env文件路径
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

@pytest.mark.asyncio
async def test_multimodal_embedding():
    """测试多模态embedding"""
    print("\n🔍 开始多模态embedding专项测试")
    print("="*60)
    
    try:
        # 1. 加载配置
        print("\n📋 1. 加载配置...")
        config = load_embedding_config()
        print(f"配置加载成功: {config.get('provider')}")
        print(f"多模态启用: {config.get('multimodal', {}).get('enabled')}")
        print(f"文本模型: {config.get('multimodal', {}).get('text_model')}")
        print(f"多模态模型: {config.get('multimodal', {}).get('multimodal_model')}")
        
        # 2. 创建混合管理器
        print("\n🏗️ 2. 创建混合embedding管理器...")
        hybrid_manager = EmbeddingFactory.create_hybrid_from_config(config)
        print(f"管理器创建成功: {type(hybrid_manager).__name__}")
        
        # 3. 测试多模态内容
        print("\n🎯 3. 测试多模态内容...")
        
        # 测试用例1: 文本+图片 - 强制使用多模态模型
        multimodal_content_1 = [
            {'text': 'GUI界面截图分析'},
            {'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}  # 使用百炼官方示例图片
        ]
        
        print(f"\n测试内容1: {multimodal_content_1}")
        request1 = EmbeddingRequest(
            content=multimodal_content_1,
            content_type=ContentType.MULTIMODAL  # 强制使用多模态
        )
        
        result1 = await hybrid_manager.embed(request1)
        print(f"\n✅ 测试1结果:")
        print(f"  - Embedding类型: {result1.embedding_type}")
        print(f"  - 向量数量: {len(result1.embeddings)}")
        print(f"  - 向量维度: {len(result1.embeddings[0]) if result1.embeddings else 0}")
        print(f"  - 缓存命中: {result1.cache_hit}")
        print(f"  - 处理时间: {result1.processing_time:.3f}s")
        print(f"  - 成本估算: {result1.cost_estimate}")
        
        # 测试用例2: 多轮对话场景
        print("\n测试用例2: 多轮对话场景 - 拆分处理多图片请求")
        
        # 第一轮：文本 + 图片1
        round1_content = [
            {'text': '移动应用界面设计'},
            {'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}
        ]
        print(f"\n第一轮对话: {round1_content}")
        request2_1 = EmbeddingRequest(
            content=round1_content,
            content_type=ContentType.MULTIMODAL  # 强制使用多模态
        )
        result2_1 = await hybrid_manager.embed(request2_1)
        
        # 第二轮：文本 + 图片2
        round2_content = [
            {'text': '包含按钮、导航栏等UI元素'},
            {'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}
        ]
        print(f"\n第二轮对话: {round2_content}")
        request2_2 = EmbeddingRequest(
            content=round2_content,
            content_type=ContentType.MULTIMODAL  # 强制使用多模态
        )
        result2_2 = await hybrid_manager.embed(request2_2)
        
        # 合并结果
        result2 = result2_1
        result2.embeddings.extend(result2_2.embeddings)
        result2.processing_time += result2_2.processing_time
        result2.cost_estimate += result2_2.cost_estimate
        print(f"\n✅ 测试2结果:")
        print(f"  - Embedding类型: {result2.embedding_type}")
        print(f"  - 向量数量: {len(result2.embeddings)}")
        print(f"  - 向量维度: {len(result2.embeddings[0]) if result2.embeddings else 0}")
        print(f"  - 缓存命中: {result2.cache_hit}")
        print(f"  - 处理时间: {result2.processing_time:.3f}s")
        print(f"  - 成本估算: {result2.cost_estimate}")
        
        # 测试用例3: 多轮对话中的纯文本 - 强制使用多模态模型
        # 第一轮对话
        round3_1_content = [
            {'text': '这是多轮对话中的第一轮纯文本内容'}
        ]
        # 第二轮对话
        round3_2_content = [
            {'text': '这是多轮对话中的第二轮纯文本内容，为了保持向量空间一致性，也使用多模态模型'}
        ]
        
        print(f"\n测试用例3: 多轮对话中的纯文本")
        # 第一轮对话请求
        print(f"\n第一轮对话: {round3_1_content}")
        request3_1 = EmbeddingRequest(
            content=round3_1_content,
            content_type=ContentType.MULTIMODAL  # 强制使用多模态
        )
        result3_1 = await hybrid_manager.embed(request3_1)
        
        # 第二轮对话请求
        print(f"\n第二轮对话: {round3_2_content}")
        request3_2 = EmbeddingRequest(
            content=round3_2_content,
            content_type=ContentType.MULTIMODAL  # 强制使用多模态
        )
        result3_2 = await hybrid_manager.embed(request3_2)
        
        # 合并结果
        result3 = result3_1
        result3.embeddings.extend(result3_2.embeddings)
        result3.processing_time += result3_2.processing_time
        result3.cost_estimate += result3_2.cost_estimate
        
        print(f"\n✅ 测试3结果:")
        print(f"  - Embedding类型: {result3.embedding_type}")
        print(f"  - 向量数量: {len(result3.embeddings)}")
        print(f"  - 向量维度: {len(result3.embeddings[0]) if result3.embeddings else 0}")
        print(f"  - 缓存命中: {result3.cache_hit}")
        print(f"  - 处理时间: {result3.processing_time:.3f}s")
        print(f"  - 成本估算: {result3.cost_estimate}")
        
        # 4. 总结
        print("\n📊 4. 测试总结")
        print("="*60)
        
        # 计算实际API调用次数（包括拆分的请求）
        all_results = [result1, result2_1, result2_2, result3_1, result3_2]
        multimodal_count = sum(1 for r in all_results 
                              if r.embedding_type.value == 'multimodal')
        text_count = sum(1 for r in all_results 
                        if r.embedding_type.value == 'text')
        
        print(f"多模态embedding调用次数: {multimodal_count}")
        print(f"文本embedding调用次数: {text_count}")
        
        # 验证多轮对话场景
        all_multimodal = all(r.embedding_type.value == 'multimodal' 
                            for r in all_results)
        
        if all_multimodal:
            print("\n🎉 成功！多轮对话场景下所有内容均使用多模态embedding，保证了向量空间一致性")
        else:
            print("\n❌ 问题！部分内容未使用多模态embedding，可能影响检索效果")
            
        # 5. 关闭管理器
        await hybrid_manager.close()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multimodal_embedding())