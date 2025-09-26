#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Embedding Performance Test
混合embedding性能测试和调优工具

Author: AgenticX Team
Date: 2025
"""

import asyncio
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.embedding_config import (
    EmbeddingStrategy, EmbeddingRequest, ContentType, EmbeddingType, EmbeddingConfig
)
from knowledge.embedding_factory import EmbeddingFactory
from knowledge.config_loader import load_embedding_config, validate_config
from utils import setup_logger, get_iso_timestamp

# 加载环境变量
from dotenv import load_dotenv
# 明确指定.env文件路径
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)


class EmbeddingPerformanceTester:
    """Embedding性能测试器"""
    
    def __init__(self):
        self.logger = logger
        self.test_results = []
        self.hybrid_manager = None
    
    async def setup(self) -> bool:
        """设置测试环境"""
        try:
            # 验证配置
            if not validate_config():
                logger.error("Embedding配置验证失败")
                return False
            
            # 加载配置
            embedding_config = load_embedding_config()
            
            # 创建混合管理器
            self.hybrid_manager = EmbeddingFactory.create_hybrid_from_config(embedding_config)
            
            logger.info("测试环境设置成功")
            return True
            
        except Exception as e:
            logger.error(f"测试环境设置失败: {e}")
            return False
    
    async def test_text_embedding_performance(self) -> Dict[str, Any]:
        """测试文本embedding性能"""
        logger.info("开始文本embedding性能测试")
        
        test_cases = [
            "简单的文本内容",
            "这是一个包含更多信息的较长文本，用于测试embedding的处理能力和性能表现。",
            "GUI操作：点击登录按钮，输入用户名和密码，然后点击确认按钮完成登录流程。",
            "移动应用界面分析：当前屏幕显示主页面，包含导航栏、搜索框、推荐内容列表等UI元素。"
        ]
        
        results = []
        
        for i, text in enumerate(test_cases):
            start_time = time.time()
            
            try:
                request = EmbeddingRequest(
                    content=text,
                    content_type=ContentType.PURE_TEXT,
                    priority="normal"
                )
                
                result = await self.hybrid_manager.embed(request)
                
                processing_time = time.time() - start_time
                
                test_result = {
                    'test_case': f'text_{i+1}',
                    'content_length': len(text),
                    'embedding_type': result.embedding_type.value,
                    'processing_time': processing_time,
                    'cache_hit': result.cache_hit,
                    'cost_estimate': result.cost_estimate,
                    'embedding_dimension': len(result.embeddings[0]) if result.embeddings else 0,
                    'success': True
                }
                
                results.append(test_result)
                logger.info(f"文本测试 {i+1}: {processing_time:.3f}s, 缓存命中: {result.cache_hit}")
                
            except Exception as e:
                logger.error(f"文本测试 {i+1} 失败: {e}")
                results.append({
                    'test_case': f'text_{i+1}',
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'test_type': 'text_embedding',
            'total_cases': len(test_cases),
            'results': results,
            'avg_processing_time': sum(r.get('processing_time', 0) for r in results) / len(results),
            'cache_hit_rate': sum(1 for r in results if r.get('cache_hit', False)) / len(results)
        }
    
    async def test_multimodal_embedding_performance(self) -> Dict[str, Any]:
        """测试多模态embedding性能"""
        logger.info("开始多模态embedding性能测试")
        
        test_cases = [
            # 纯文本（应该被智能路由到文本embedding）
            [{'text': '纯文本内容测试'}],
            
            # 文本+图片
            [
                {'text': '登录页面截图'},
                {'image': 'https://i.imgur.com/CzXTtJV.jpg'}  # 真实可访问的测试图片
            ],
            
            # 多个文本和图片
            [
                {'text': '移动应用主界面'},
                {'image': 'https://farm4.staticflickr.com/3075/3168662394_7d7103de7d_z_d.jpg'},  # 真实可访问的测试图片
                {'text': '包含导航栏、搜索框等UI元素'},
                {'image': 'https://farm9.staticflickr.com/8295/8007075227_dc958c1fe6_z_d.jpg'}  # 真实可访问的测试图片
            ],
            
            # 复杂多模态内容
            [
                {'text': 'GUI自动化测试场景'},
                {'image': 'https://farm2.staticflickr.com/1449/24800673529_64272a66ec_z_d.jpg'},  # 真实可访问的测试图片
                {'text': '操作步骤：1. 打开应用 2. 点击按钮 3. 验证结果'},
                {'image': 'https://farm4.staticflickr.com/3827/11349066413_99c32dee4a_z_d.jpg'}  # 真实可访问的测试图片
            ]
        ]
        
        results = []
        
        for i, content in enumerate(test_cases):
            start_time = time.time()
            
            try:
                request = EmbeddingRequest(
                    content=content,
                    content_type=ContentType.AUTO,  # 自动检测
                    priority="normal"
                )
                
                result = await self.hybrid_manager.embed(request)
                
                processing_time = time.time() - start_time
                
                test_result = {
                    'test_case': f'multimodal_{i+1}',
                    'content_items': len(content),
                    'embedding_type': result.embedding_type.value,
                    'processing_time': processing_time,
                    'cache_hit': result.cache_hit,
                    'cost_estimate': result.cost_estimate,
                    'embedding_dimension': len(result.embeddings[0]) if result.embeddings else 0,
                    'success': True
                }
                
                results.append(test_result)
                logger.info(f"多模态测试 {i+1}: {processing_time:.3f}s, 类型: {result.embedding_type.value}")
                
            except Exception as e:
                logger.error(f"多模态测试 {i+1} 失败: {e}")
                results.append({
                    'test_case': f'multimodal_{i+1}',
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'test_type': 'multimodal_embedding',
            'total_cases': len(test_cases),
            'results': results,
            'avg_processing_time': sum(r.get('processing_time', 0) for r in results) / len(results),
            'text_routing_rate': sum(1 for r in results if r.get('embedding_type') == 'text') / len(results),
            'multimodal_routing_rate': sum(1 for r in results if r.get('embedding_type') == 'multimodal') / len(results)
        }
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """测试缓存性能"""
        logger.info("开始缓存性能测试")
        
        # 测试内容
        test_content = "GUI操作测试：点击按钮，验证响应"
        
        # 第一次请求（应该缓存miss）
        start_time = time.time()
        request = EmbeddingRequest(
            content=test_content,
            content_type=ContentType.PURE_TEXT
        )
        
        first_result = await self.hybrid_manager.embed(request)
        first_time = time.time() - start_time
        
        # 第二次请求（应该缓存hit）
        start_time = time.time()
        second_result = await self.hybrid_manager.embed(request)
        second_time = time.time() - start_time
        
        # 批量测试缓存性能
        batch_times = []
        for i in range(10):
            start_time = time.time()
            await self.hybrid_manager.embed(request)
            batch_times.append(time.time() - start_time)
        
        return {
            'test_type': 'cache_performance',
            'first_request': {
                'time': first_time,
                'cache_hit': first_result.cache_hit
            },
            'second_request': {
                'time': second_time,
                'cache_hit': second_result.cache_hit
            },
            'speedup_ratio': first_time / second_time if second_time > 0 else 0,
            'batch_avg_time': sum(batch_times) / len(batch_times),
            'cache_stats': self.hybrid_manager.get_stats()['cache_stats']
        }
    
    async def test_cost_optimization(self) -> Dict[str, Any]:
        """测试成本优化"""
        logger.info("开始成本优化测试")
        
        # 测试不同策略的成本
        strategies = [
            EmbeddingStrategy(
                cost_threshold_multimodal=0.1,  # 低阈值，偏向文本
                prefer_multimodal_for_gui=False
            ),
            EmbeddingStrategy(
                cost_threshold_multimodal=1.0,  # 高阈值，偏向多模态
                prefer_multimodal_for_gui=True
            )
        ]
        
        test_content = [
            {'text': 'GUI操作描述'},
            {'image': 'https://i.imgur.com/OnwEDW3.jpg'}  # 真实可访问的测试图片
        ]
        
        results = []
        
        for i, strategy in enumerate(strategies):
            # 更新策略
            self.hybrid_manager.update_strategy(strategy)
            
            start_time = time.time()
            request = EmbeddingRequest(
                content=test_content,
                content_type=ContentType.AUTO
            )
            
            result = await self.hybrid_manager.embed(request)
            processing_time = time.time() - start_time
            
            results.append({
                'strategy': f'strategy_{i+1}',
                'embedding_type': result.embedding_type.value,
                'processing_time': processing_time,
                'cost_estimate': result.cost_estimate,
                'threshold': strategy.cost_threshold_multimodal
            })
        
        return {
            'test_type': 'cost_optimization',
            'results': results,
            'cost_difference': abs(results[0]['cost_estimate'] - results[1]['cost_estimate']) if len(results) == 2 else 0
        }
    
    async def test_batch_processing(self) -> Dict[str, Any]:
        """测试批量处理性能"""
        logger.info("开始批量处理性能测试")
        
        # 生成测试数据
        batch_sizes = [1, 5, 10, 20, 50]
        results = []
        
        for batch_size in batch_sizes:
            # 生成批量文本
            texts = [f"测试文本 {i+1}: GUI操作描述" for i in range(batch_size)]
            
            start_time = time.time()
            
            # 并发处理
            tasks = []
            for text in texts:
                request = EmbeddingRequest(
                    content=text,
                    content_type=ContentType.PURE_TEXT
                )
                tasks.append(self.hybrid_manager.embed(request))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            processing_time = time.time() - start_time
            
            # 统计成功率
            success_count = sum(1 for r in batch_results if not isinstance(r, Exception))
            
            results.append({
                'batch_size': batch_size,
                'processing_time': processing_time,
                'avg_time_per_item': processing_time / batch_size,
                'success_rate': success_count / batch_size,
                'throughput': batch_size / processing_time if processing_time > 0 else 0
            })
            
            logger.info(f"批量测试 {batch_size}: {processing_time:.3f}s, 吞吐量: {results[-1]['throughput']:.2f} items/s")
        
        return {
            'test_type': 'batch_processing',
            'results': results,
            'optimal_batch_size': max(results, key=lambda x: x['throughput'])['batch_size']
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合性能测试"""
        logger.info("开始综合性能测试")
        
        if not await self.setup():
            return {'error': '测试环境设置失败'}
        
        test_results = {
            'test_timestamp': get_iso_timestamp(),
            'test_environment': {
                'python_version': sys.version,
                'embedding_config': load_embedding_config()
            }
        }
        
        try:
            # 运行各项测试
            test_results['text_embedding'] = await self.test_text_embedding_performance()
            test_results['multimodal_embedding'] = await self.test_multimodal_embedding_performance()
            test_results['cache_performance'] = await self.test_cache_performance()
            test_results['cost_optimization'] = await self.test_cost_optimization()
            test_results['batch_processing'] = await self.test_batch_processing()
            
            # 获取最终统计
            test_results['final_stats'] = self.hybrid_manager.get_stats()
            
            # 生成性能报告
            test_results['performance_summary'] = self._generate_performance_summary(test_results)
            
        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            test_results['error'] = str(e)
        
        finally:
            # 清理资源
            if self.hybrid_manager:
                await self.hybrid_manager.close()
        
        return test_results
    
    def _generate_performance_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能摘要"""
        summary = {
            'overall_performance': 'good',
            'recommendations': [],
            'key_metrics': {}
        }
        
        try:
            # 分析文本embedding性能
            text_results = test_results.get('text_embedding', {})
            if text_results:
                avg_time = text_results.get('avg_processing_time', 0)
                summary['key_metrics']['text_avg_time'] = avg_time
                
                if avg_time > 1.0:
                    summary['recommendations'].append('文本embedding处理时间较长，建议优化批量大小或启用缓存')
            
            # 分析缓存性能
            cache_results = test_results.get('cache_performance', {})
            if cache_results:
                speedup = cache_results.get('speedup_ratio', 0)
                summary['key_metrics']['cache_speedup'] = speedup
                
                if speedup < 5:
                    summary['recommendations'].append('缓存加速效果不明显，建议检查缓存配置')
            
            # 分析批量处理性能
            batch_results = test_results.get('batch_processing', {})
            if batch_results:
                optimal_batch = batch_results.get('optimal_batch_size', 0)
                summary['key_metrics']['optimal_batch_size'] = optimal_batch
                
                if optimal_batch < 10:
                    summary['recommendations'].append('建议增加批量处理大小以提高吞吐量')
            
            # 分析成本优化
            cost_results = test_results.get('cost_optimization', {})
            if cost_results:
                cost_diff = cost_results.get('cost_difference', 0)
                summary['key_metrics']['cost_optimization_potential'] = cost_diff
                
                if cost_diff > 0.1:
                    summary['recommendations'].append('不同策略间成本差异较大，建议根据使用场景调整策略')
            
            # 综合评估
            if len(summary['recommendations']) == 0:
                summary['overall_performance'] = 'excellent'
            elif len(summary['recommendations']) <= 2:
                summary['overall_performance'] = 'good'
            else:
                summary['overall_performance'] = 'needs_improvement'
        
        except Exception as e:
            summary['error'] = f'性能摘要生成失败: {e}'
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """保存测试结果"""
        if filename is None:
            timestamp = get_iso_timestamp().replace(':', '-').replace('.', '-')
            filename = f'embedding_performance_test_{timestamp}.json'
        
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"测试结果已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
            return ""


async def main():
    """主函数"""
    print("🚀 AgenticX-GUIAgent 混合Embedding性能测试")
    print("=" * 50)
    
    tester = EmbeddingPerformanceTester()
    
    # 运行综合测试
    results = await tester.run_comprehensive_test()
    
    # 保存结果
    filepath = tester.save_results(results)
    
    # 打印摘要
    if 'performance_summary' in results:
        summary = results['performance_summary']
        print(f"\n📊 性能测试摘要:")
        print(f"总体性能: {summary.get('overall_performance', 'unknown')}")
        
        if 'key_metrics' in summary:
            print(f"\n🔑 关键指标:")
            for metric, value in summary['key_metrics'].items():
                print(f"  - {metric}: {value}")
        
        if 'recommendations' in summary and summary['recommendations']:
            print(f"\n💡 优化建议:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
    
    if 'error' in results:
        print(f"\n❌ 测试失败: {results['error']}")
    else:
        print(f"\n✅ 测试完成，结果已保存到: {filepath}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(main())