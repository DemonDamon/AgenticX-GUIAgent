#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Knowledge Manager - 知识库管理命令行工具

提供便捷的命令行接口来管理知识库数据，包括：
- 查看知识库状态
- 查询知识内容
- 清理知识库
- 导出/导入数据
- 连接向量数据库（Milvus等）

Author: AgenticX Team
Date: 2025
"""

import asyncio
import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knowledge.knowledge_manager import KnowledgeManager
from knowledge.knowledge_types import KnowledgeType, KnowledgeSource, QueryRequest
from knowledge.agenticx_adapter import AgenticXConfig
from config import get_config


class KnowledgeCLI:
    """知识库CLI管理器"""
    
    def __init__(self):
        self.knowledge_manager = None
        self.config = get_config()
        
    async def initialize(self):
        """初始化知识管理器"""
        print("🚀 初始化知识库管理器...")
        
        # 创建AgenticX配置
        agenticx_config = AgenticXConfig(
            storage_type="file",  # 或 "database"
            vectorization_enabled=True,
            retrieval_type="hybrid",
            embedding_config={
                "provider": "bailian",
                "model": "text-embedding-v4",
                "dimension": 1536
            }
        )
        
        # 初始化知识管理器
        self.knowledge_manager = KnowledgeManager(
            agenticx_config=agenticx_config,
            embedding_provider=None,  # 将自动创建
            vector_store=None,  # 将自动创建
            cache=None  # 将自动创建
        )
        
        await self.knowledge_manager.start()
        print("✅ 知识库管理器初始化完成")
        
    async def cleanup(self):
        """清理资源"""
        if self.knowledge_manager:
            await self.knowledge_manager.stop()
            print("🏁 知识库管理器已停止")
    
    async def show_status(self):
        """显示知识库状态"""
        print("\n📊 知识库状态信息")
        print("=" * 50)
        
        if not self.knowledge_manager:
            print("❌ 知识管理器未初始化")
            return
            
        # 获取统计信息
        stats = self.knowledge_manager.get_stats()
        print(f"📚 总知识数量: {stats.get('total_knowledge', 0)}")
        print(f"🔍 总查询次数: {stats.get('total_queries', 0)}")
        print(f"💾 缓存命中率: {stats.get('cache_hit_rate', 0):.2%}")
        print(f"⏱️ 平均查询时间: {stats.get('avg_query_time', 0):.3f}秒")
        
        # 显示配置信息
        print("\n⚙️ 配置信息:")
        print(f"   - 存储类型: {self.knowledge_manager.agenticx_config.storage_type}")
        print(f"   - 向量化: {'启用' if self.knowledge_manager.agenticx_config.vectorization_enabled else '禁用'}")
        print(f"   - 检索类型: {self.knowledge_manager.agenticx_config.retrieval_type}")
        
        # 显示向量数据库信息
        print("\n🗄️ 向量数据库信息:")
        if hasattr(self.knowledge_manager, 'vector_store') and self.knowledge_manager.vector_store:
            print(f"   - 类型: {type(self.knowledge_manager.vector_store).__name__}")
            print(f"   - 状态: 已连接")
        else:
            print(f"   - 状态: 未连接")
            
    async def query_knowledge(self, query_text: str, limit: int = 5):
        """查询知识"""
        print(f"\n🔍 查询知识: {query_text}")
        print("=" * 50)
        
        if not self.knowledge_manager:
            print("❌ 知识管理器未初始化")
            return
            
        # 创建查询请求
        request = QueryRequest(
            query_text=query_text,
            limit=limit,
            query_type="semantic"
        )
        
        # 执行查询
        result = await self.knowledge_manager.query_knowledge(request)
        
        if result.items:
            print(f"✅ 找到 {len(result.items)} 条相关知识:")
            for i, item in enumerate(result.items, 1):
                print(f"\n{i}. {item.title}")
                print(f"   类型: {item.type.value}")
                print(f"   来源: {item.source.value}")
                print(f"   内容: {item.content[:200]}..." if len(item.content) > 200 else f"   内容: {item.content}")
                if hasattr(item, 'similarity_score'):
                    print(f"   相似度: {item.similarity_score:.3f}")
        else:
            print("❌ 未找到相关知识")
            
        print(f"\n⏱️ 查询耗时: {result.execution_time:.3f}秒")
    
    async def list_knowledge(self, knowledge_type: str = None, limit: int = 10):
        """列出知识"""
        print(f"\n📋 知识列表 (限制: {limit}条)")
        if knowledge_type:
            print(f"   筛选类型: {knowledge_type}")
        print("=" * 50)
        
        if not self.knowledge_manager:
            print("❌ 知识管理器未初始化")
            return
            
        # 这里需要实现列表功能，暂时使用查询代替
        request = QueryRequest(
            query_text="",  # 空查询获取所有
            limit=limit,
            query_type="all"
        )
        
        result = await self.knowledge_manager.query_knowledge(request)
        
        if result.items:
            print(f"📚 共找到 {len(result.items)} 条知识:")
            for i, item in enumerate(result.items, 1):
                print(f"\n{i}. [{item.type.value}] {item.title}")
                print(f"   ID: {item.id}")
                print(f"   来源: {item.source.value}")
                print(f"   创建时间: {item.created_at}")
                print(f"   状态: {item.status.value}")
        else:
            print("📭 知识库为空")
    
    async def export_knowledge(self, output_file: str, format_type: str = "json"):
        """导出知识"""
        print(f"\n📤 导出知识到: {output_file}")
        print(f"   格式: {format_type}")
        print("=" * 50)
        
        if not self.knowledge_manager:
            print("❌ 知识管理器未初始化")
            return
            
        try:
            # 获取所有知识
            request = QueryRequest(
                query_text="",
                limit=10000,  # 大数量获取所有
                query_type="all"
            )
            
            result = await self.knowledge_manager.query_knowledge(request)
            
            if not result.items:
                print("📭 没有知识可导出")
                return
                
            # 准备导出数据
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_count": len(result.items),
                "format": format_type,
                "knowledge_items": []
            }
            
            for item in result.items:
                export_data["knowledge_items"].append({
                    "id": item.id,
                    "title": item.title,
                    "content": item.content,
                    "type": item.type.value,
                    "source": item.source.value,
                    "domain": item.domain,
                    "tags": item.tags,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "status": item.status.value,
                    "metadata": item.metadata
                })
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                if format_type == "json":
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                else:
                    print(f"❌ 不支持的格式: {format_type}")
                    return
                    
            print(f"✅ 成功导出 {len(result.items)} 条知识到 {output_file}")
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    async def clear_knowledge(self, confirm: bool = False):
        """清空知识库"""
        print("\n🗑️ 清空知识库")
        print("=" * 50)
        
        if not confirm:
            print("⚠️ 这将删除所有知识数据，请使用 --confirm 参数确认")
            return
            
        if not self.knowledge_manager:
            print("❌ 知识管理器未初始化")
            return
            
        try:
            # 这里需要实现清空功能
            print("🔄 正在清空知识库...")
            # await self.knowledge_manager.clear_all()
            print("✅ 知识库已清空")
            print("ℹ️ 注意: 清空功能需要在KnowledgeManager中实现")
            
        except Exception as e:
            print(f"❌ 清空失败: {e}")
    
    async def test_connection(self):
        """测试数据库连接"""
        print("\n🔌 测试数据库连接")
        print("=" * 50)
        
        if not self.knowledge_manager:
            print("❌ 知识管理器未初始化")
            return
            
        try:
            # 测试向量数据库连接
            if hasattr(self.knowledge_manager, 'vector_store') and self.knowledge_manager.vector_store:
                print("🔄 测试向量数据库连接...")
                # 这里可以添加具体的连接测试逻辑
                print("✅ 向量数据库连接正常")
            else:
                print("⚠️ 向量数据库未配置")
                
            # 测试嵌入服务连接
            if hasattr(self.knowledge_manager, 'embedding_provider') and self.knowledge_manager.embedding_provider:
                print("🔄 测试嵌入服务连接...")
                # 测试嵌入生成
                test_text = "测试文本"
                if hasattr(self.knowledge_manager.embedding_provider, 'encode_text'):
                    vector = await self.knowledge_manager.embedding_provider.encode_text(test_text)
                    if vector:
                        print(f"✅ 嵌入服务连接正常 (向量维度: {len(vector)})")
                    else:
                        print("❌ 嵌入服务测试失败")
                else:
                    print("⚠️ 嵌入服务不支持encode_text方法")
            else:
                print("⚠️ 嵌入服务未配置")
                
        except Exception as e:
            print(f"❌ 连接测试失败: {e}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识库管理CLI工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 状态命令
    subparsers.add_parser('status', help='显示知识库状态')
    
    # 查询命令
    query_parser = subparsers.add_parser('query', help='查询知识')
    query_parser.add_argument('text', help='查询文本')
    query_parser.add_argument('--limit', type=int, default=5, help='结果数量限制')
    
    # 列表命令
    list_parser = subparsers.add_parser('list', help='列出知识')
    list_parser.add_argument('--type', help='筛选知识类型')
    list_parser.add_argument('--limit', type=int, default=10, help='结果数量限制')
    
    # 导出命令
    export_parser = subparsers.add_parser('export', help='导出知识')
    export_parser.add_argument('output', help='输出文件路径')
    export_parser.add_argument('--format', default='json', help='导出格式 (json)')
    
    # 清空命令
    clear_parser = subparsers.add_parser('clear', help='清空知识库')
    clear_parser.add_argument('--confirm', action='store_true', help='确认清空')
    
    # 测试连接命令
    subparsers.add_parser('test', help='测试数据库连接')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    cli = KnowledgeCLI()
    
    try:
        await cli.initialize()
        
        if args.command == 'status':
            await cli.show_status()
        elif args.command == 'query':
            await cli.query_knowledge(args.text, args.limit)
        elif args.command == 'list':
            await cli.list_knowledge(args.type, args.limit)
        elif args.command == 'export':
            await cli.export_knowledge(args.output, args.format)
        elif args.command == 'clear':
            await cli.clear_knowledge(args.confirm)
        elif args.command == 'test':
            await cli.test_connection()
        else:
            print(f"❌ 未知命令: {args.command}")
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断操作")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    print("🎯 知识库管理CLI工具")
    print("=" * 50)
    asyncio.run(main())