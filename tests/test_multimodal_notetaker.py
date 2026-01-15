#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多模态NotetakerAgent的知识管理功能

验证基于AgenticX框架的真正多模态LLM驱动的NotetakerAgent实现
"""

import asyncio
from loguru import logger
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

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
from agents.notetaker_agent import NotetakerAgent
from config import AgentConfig
from utils import setup_logger

# 设置日志
logger = setup_logger("test_multimodal_notetaker", level="INFO")


async def _validate_knowledge_granularity(results: List[Dict[str, Any]], query_case: Dict[str, Any]) -> Dict[str, Any]:
    """验证知识检索结果的粒度特征"""
    granularity = query_case.get('granularity', 'unknown')
    expected_features = query_case.get('expected_features', [])
    expected_granularities = query_case.get('expected_granularities', [])
    expected_domains = query_case.get('expected_domains', [])
    
    validation_result = {
        'valid': True,
        'message': '',
        'details': {}
    }
    
    if not results:
        validation_result['valid'] = False
        validation_result['message'] = '未找到任何知识项'
        return validation_result
    
    # 验证特定粒度的特征
    if expected_features:
        feature_found_count = 0
        total_items = len(results)
        
        for result in results:
            content = result.get('content', {})
            if isinstance(content, dict):
                for feature in expected_features:
                    if feature in content:
                        feature_found_count += 1
                        break
        
        feature_coverage = feature_found_count / total_items if total_items > 0 else 0
        validation_result['details']['feature_coverage'] = feature_coverage
        
        if feature_coverage >= 0.5:  # 至少50%的结果包含预期特征
            validation_result['message'] += f'{granularity}粒度特征覆盖率: {feature_coverage:.1%}'
        else:
            validation_result['valid'] = False
            validation_result['message'] += f'{granularity}粒度特征覆盖率过低: {feature_coverage:.1%}'
    
    # 验证跨粒度检索
    if expected_granularities:
        found_granularities = set()
        for result in results:
            tags = result.get('tags', [])
            for granularity_tag in expected_granularities:
                if granularity_tag in tags:
                    found_granularities.add(granularity_tag)
        
        coverage = len(found_granularities) / len(expected_granularities)
        validation_result['details']['granularity_coverage'] = coverage
        
        if coverage >= 0.6:  # 至少覆盖60%的预期粒度
            validation_result['message'] += f'跨粒度覆盖率: {coverage:.1%}'
        else:
            validation_result['valid'] = False
            validation_result['message'] += f'跨粒度覆盖率不足: {coverage:.1%}'
    
    # 验证领域分布
    if expected_domains:
        found_domains = set()
        for result in results:
            domain = result.get('domain', '')
            if domain in expected_domains:
                found_domains.add(domain)
        
        domain_coverage = len(found_domains) / len(expected_domains)
        validation_result['details']['domain_coverage'] = domain_coverage
        
        if domain_coverage >= 0.5:  # 至少覆盖50%的预期领域
            validation_result['message'] += f', 领域覆盖率: {domain_coverage:.1%}'
        else:
            validation_result['message'] += f', 领域覆盖率较低: {domain_coverage:.1%}'
    
    return validation_result


def _analyze_granularity_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """分析知识检索结果的粒度分布"""
    granularity_stats = {
        'complete_task': 0,
        'subtask': 0,
        'atomic_operation': 0,
        'other': 0
    }
    
    for result in results:
        tags = result.get('tags', [])
        knowledge_type = result.get('type', '')
        
        # 根据标签和类型判断粒度
        if 'complete_task' in tags or knowledge_type == 'task_workflow':
            granularity_stats['complete_task'] += 1
        elif 'subtask' in tags or knowledge_type == 'best_practice':
            granularity_stats['subtask'] += 1
        elif 'atomic' in tags or knowledge_type == 'action_pattern':
            granularity_stats['atomic_operation'] += 1
        else:
            granularity_stats['other'] += 1
    
    # 只返回非零的统计
    return {k: v for k, v in granularity_stats.items() if v > 0}

async def test_multimodal_knowledge_management():
    """测试多模态知识管理功能"""
    
    logger.info("🚀 开始测试多模态NotetakerAgent")
    
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
    
    # 创建NotetakerAgent配置
    agent_config = AgentConfig(
        id="test_notetaker",
        name="TestNotetakerAgent",
        role="notetaker",
        goal="测试多模态LLM驱动的知识管理",
        backstory="我是一个测试用的知识记录智能体，能够使用多模态LLM智能提取和管理知识。"
    )
    
    # 初始化NotetakerAgent
    notetaker_agent = NotetakerAgent(
        llm_provider=llm_provider,
        agent_id="test_notetaker",
        event_bus=event_bus,
        agent_config=agent_config
    )
    
    logger.info("NotetakerAgent初始化完成")
    
    # 测试用例 - 多粒度知识捕获、查询、组织
    test_cases = [
        {
            "name": "测试1: 完整任务级别知识 - 电商购物流程",
            "granularity": "complete_task",
            "task_context": {
                "task_type": "capture",
                "knowledge_data": {
                    "type": "task_workflow",
                    "title": "淘宝购物完整流程最佳实践",
                    "description": "用户在淘宝购买商品的完整任务流程",
                    "content": {
                        "task_description": "帮我在淘宝上买一件衣服",
                        "workflow_steps": [
                            "打开淘宝应用",
                            "登录账户",
                            "搜索商品",
                            "筛选和比较",
                            "选择商品",
                            "加入购物车",
                            "确认订单",
                            "选择支付方式",
                            "完成支付"
                        ],
                        "success_rate": 0.92,
                        "average_duration": 180,
                        "complexity_level": "high",
                        "user_satisfaction": 0.88,
                        "common_challenges": [
                            "商品选择困难",
                            "支付流程复杂",
                            "网络延迟影响"
                        ],
                        "optimization_tips": [
                            "预先准备购物清单",
                            "使用收藏夹管理商品",
                            "选择合适的支付方式"
                        ]
                    },
                    "domain": "e_commerce",
                    "tags": ["shopping", "complete_task", "e_commerce", "workflow"],
                    "source": "ManagerAgent"
                }
            }
        },
        {
            "name": "测试2: 子任务级别知识 - 点击操作最佳实践",
            "granularity": "subtask",
            "task_context": {
                "task_type": "capture",
                "knowledge_data": {
                    "type": "best_practice",
                    "title": "移动设备点击操作最佳实践",
                    "description": "Manager拆解后的点击子任务执行技巧",
                    "content": {
                        "subtask_type": "click_operation",
                        "parent_task": "商品选择和购买",
                        "success_rate": 0.95,
                        "efficiency_score": 0.88,
                        "best_practices": [
                            "确保目标元素可见",
                            "等待页面加载完成",
                            "使用精确的坐标定位",
                            "验证点击后的状态变化"
                        ],
                        "common_errors": [
                            "元素未找到",
                            "点击位置偏移",
                            "页面未完全加载"
                        ],
                        "applicable_scenarios": [
                            "按钮点击",
                            "链接点击",
                            "图标点击",
                            "列表项选择"
                        ],
                        "conditions": "适用于所有点击操作",
                        "confidence": 0.9
                    },
                    "domain": "mobile_automation",
                    "tags": ["click", "subtask", "best_practice", "mobile"],
                    "source": "ExecutorAgent"
                }
            }
        },
        {
            "name": "测试3: 原子操作级别知识 - tap坐标优化",
            "granularity": "atomic_operation",
            "task_context": {
                "task_type": "capture",
                "knowledge_data": {
                    "type": "action_pattern",
                    "title": "tap(x,y)坐标精度优化模式",
                    "description": "原子级别的tap操作参数优化知识",
                    "content": {
                        "operation_type": "tap",
                        "parameters": {
                            "coordinate_precision": 0.1,
                            "tap_duration": 100,
                            "pressure_level": "medium"
                        },
                        "optimization_strategies": [
                            "中心点偏移补偿",
                            "元素边界检测",
                            "多点采样验证",
                            "动态坐标调整"
                        ],
                        "performance_metrics": {
                            "accuracy_rate": 0.98,
                            "response_time": 50,
                            "retry_rate": 0.02
                        },
                        "device_compatibility": [
                            "Android 9+",
                            "iOS 13+",
                            "不同屏幕分辨率"
                        ],
                        "error_handling": {
                            "coordinate_out_of_bounds": "边界裁剪",
                            "element_moved": "重新定位",
                            "tap_failed": "重试机制"
                        }
                    },
                    "domain": "atomic_operations",
                    "tags": ["tap", "atomic", "coordinates", "optimization"],
                    "source": "low_level_executor"
                }
            }
        },
        {
            "name": "测试2: 捕获错误解决方案",
            "task_context": {
                "task_type": "capture",
                "knowledge_data": {
                    "type": "error_solution",
                    "content": {
                        "error_type": "element_not_found",
                        "error_frequency": 0.15,
                        "solutions": [
                            "增加等待时间",
                            "使用备用定位策略",
                            "检查页面加载状态"
                        ],
                        "prevention": [
                            "预先验证元素存在",
                            "使用动态等待机制"
                        ],
                        "success_rate_after_fix": 0.92
                    },
                    "source": "ActionReflectorAgent"
                }
            }
        },
        {
            "name": "测试3: 捕获最佳实践",
            "task_context": {
                "task_type": "capture",
                "knowledge_data": {
                    "type": "best_practice",
                    "content": {
                        "area": "mobile_automation",
                        "practices": [
                            "操作前先截图保存状态",
                            "使用多模态LLM进行智能分析",
                            "建立操作前后对比机制",
                            "实施多模型降级策略"
                        ],
                        "benefits": [
                            "提高操作成功率",
                            "增强系统可靠性",
                            "改善用户体验"
                        ],
                        "applicable_scenarios": "所有移动设备自动化场景"
                    },
                    "source": "ManagerAgent"
                }
            }
        },
        {
            "name": "测试4: 捕获性能洞察",
            "task_context": {
                "task_type": "capture",
                "knowledge_data": {
                    "type": "performance_insight",
                    "content": {
                        "metric": "execution_time",
                        "average_time": 2.3,
                        "optimization_suggestions": [
                            "并行执行非依赖操作",
                            "缓存常用元素定位",
                            "优化截图处理流程"
                        ],
                        "performance_trends": {
                            "improving": True,
                            "trend_description": "执行时间逐步优化"
                        }
                    },
                    "source": "system_monitor"
                }
            }
        }
    ]
    
    # 执行知识捕获测试
    logger.info("\n📝 开始知识捕获测试")
    captured_knowledge_ids = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"知识捕获 {i}: {test_case['name']}")
        logger.info(f"{'='*50}")
        
        try:
            # 执行知识捕获
            result = await notetaker_agent._execute_task_impl(test_case['task_context'])
            
            if result.get('success', False):
                knowledge_id = result.get('knowledge_id')
                knowledge_type = result.get('knowledge_type')
                captured_knowledge_ids.append(knowledge_id)
                
                logger.info(f"✅ 【知识捕获成功】")
                logger.info(f"📋 知识ID: {knowledge_id}")
                logger.info(f"🏷️ 知识类型: {knowledge_type}")
                logger.info(f"💾 存储路径: {result.get('file_path', 'N/A')}")
                
                # 显示结构化知识信息
                structured_knowledge = result.get('structured_knowledge', {})
                if structured_knowledge:
                    logger.info(f"📝 标题: {structured_knowledge.get('title', 'N/A')}")
                    logger.info(f"🏷️ 标签: {', '.join(structured_knowledge.get('tags', []))}")
                    logger.info(f"⭐ 重要性: {structured_knowledge.get('importance', 0):.2f}")
                    logger.info(f"🔗 可靠性: {structured_knowledge.get('metadata', {}).get('reliability', 0):.2f}")
            else:
                logger.error(f"❌ 【知识捕获失败】")
                logger.error(f"💥 错误信息: {result.get('error', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ 知识捕获异常: {e}")
        
        await asyncio.sleep(0.5)
    
    # 测试多粒度知识检索功能
    logger.info("\n🔍 开始多粒度知识检索测试")
    
    query_test_cases = [
        {
            "name": "检索完整任务级别知识 - 电商购物流程",
            "granularity": "complete_task",
            "task_context": {
                "task_type": "query",
                "query": "淘宝购物 电商",
                "knowledge_type": "task_workflow",
                "tags": ["shopping", "complete_task", "e_commerce"],
                "limit": 5
            },
            "expected_features": ["workflow_steps", "complexity_level", "user_satisfaction"]
        },
        {
            "name": "检索子任务级别知识 - 点击操作技巧",
            "granularity": "subtask",
            "task_context": {
                "task_type": "query",
                "query": "点击操作 click",
                "knowledge_type": "best_practice",
                "tags": ["click", "subtask", "mobile"],
                "limit": 5
            },
            "expected_features": ["subtask_type", "parent_task", "applicable_scenarios"]
        },
        {
            "name": "检索原子操作级别知识 - tap坐标优化",
            "granularity": "atomic_operation",
            "task_context": {
                "task_type": "query",
                "query": "tap 坐标 优化",
                "knowledge_type": "action_pattern",
                "tags": ["tap", "atomic", "coordinates"],
                "limit": 5
            },
            "expected_features": ["operation_type", "parameters", "performance_metrics"]
        },
        {
            "name": "跨粒度知识检索 - 点击相关的所有知识",
            "granularity": "cross_granularity",
            "task_context": {
                "task_type": "query",
                "query": "点击 click tap",
                "limit": 10
            },
            "expected_granularities": ["complete_task", "subtask", "atomic_operation"]
        },
        {
            "name": "领域特定知识检索 - 移动自动化最佳实践",
            "granularity": "domain_specific",
            "task_context": {
                "task_type": "query",
                "query": "mobile_automation",
                "tags": ["best_practice", "mobile"],
                "limit": 5
            },
            "expected_domains": ["mobile_automation", "e_commerce", "atomic_operations"]
        },
        {
            "name": "错误解决方案检索",
            "granularity": "error_handling",
            "task_context": {
                "task_type": "query",
                "query": "error 错误",
                "knowledge_type": "error_solution",
                "limit": 3
            },
            "expected_features": ["error_type", "solutions", "prevention"]
        }
    ]
    
    for i, query_case in enumerate(query_test_cases, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"多粒度知识检索 {i}: {query_case['name']}")
        logger.info(f"🎯 测试粒度: {query_case.get('granularity', 'N/A')}")
        logger.info(f"{'='*50}")
        
        try:
            result = await notetaker_agent._execute_task_impl(query_case['task_context'])
            
            if result.get('success', False):
                results = result.get('results', [])
                total_count = result.get('total_count', 0)
                returned_count = result.get('returned_count', 0)
                
                logger.info(f"🔍 【检索成功】")
                logger.info(f"📊 总计找到: {total_count}条, 返回: {returned_count}条")
                logger.info(f"🔎 查询条件: {result.get('query', 'N/A')}")
                
                # 验证多粒度知识特征
                granularity_validation = await _validate_knowledge_granularity(
                    results, query_case
                )
                
                if granularity_validation['valid']:
                    logger.info(f"✅ 【粒度验证通过】: {granularity_validation['message']}")
                else:
                    logger.warning(f"⚠️ 【粒度验证警告】: {granularity_validation['message']}")
                
                # 显示检索结果详情
                for j, knowledge_item in enumerate(results[:3], 1):  # 只显示前3条
                    logger.info(f"\n  📋 结果 {j}:")
                    logger.info(f"     🏷️ 类型: [{knowledge_item.get('type', 'unknown')}]")
                    logger.info(f"     📝 标题: {knowledge_item.get('title', 'N/A')}")
                    logger.info(f"     🌐 领域: {knowledge_item.get('domain', 'N/A')}")
                    logger.info(f"     🏷️ 标签: {', '.join(knowledge_item.get('tags', []))}")
                    logger.info(f"     ⭐ 重要性: {knowledge_item.get('importance', 0):.2f}")
                    logger.info(f"     📅 创建时间: {knowledge_item.get('created_at', 'N/A')}")
                    logger.info(f"     🔢 访问次数: {knowledge_item.get('access_count', 0)}")
                    
                    # 显示内容特征（用于验证粒度）
                    content = knowledge_item.get('content', {})
                    if isinstance(content, dict):
                        key_features = []
                        expected_features = query_case.get('expected_features', [])
                        for feature in expected_features:
                            if feature in content:
                                key_features.append(f"{feature}✓")
                            else:
                                key_features.append(f"{feature}✗")
                        if key_features:
                            logger.info(f"     🔍 特征验证: {', '.join(key_features)}")
                
                # 统计不同粒度的知识分布
                granularity_stats = _analyze_granularity_distribution(results)
                if granularity_stats:
                    logger.info(f"\n📈 粒度分布统计:")
                    for granularity, count in granularity_stats.items():
                        logger.info(f"     {granularity}: {count}条")
                        
            else:
                logger.error(f"❌ 【检索失败】")
                logger.error(f"💥 错误信息: {result.get('error', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ 知识检索异常: {e}")
        
        await asyncio.sleep(0.5)
    
    # 测试知识组织功能
    logger.info("\n🗂️ 开始知识组织测试")
    
    organization_test_cases = [
        {
            "name": "知识分类整理",
            "task_context": {
                "task_type": "organize",
                "organization_type": "categorize"
            }
        },
        {
            "name": "关联知识链接",
            "task_context": {
                "task_type": "organize",
                "organization_type": "link"
            }
        },
        {
            "name": "生成知识摘要",
            "task_context": {
                "task_type": "summary"
            }
        }
    ]
    
    for i, org_case in enumerate(organization_test_cases, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"知识组织 {i}: {org_case['name']}")
        logger.info(f"{'='*50}")
        
        try:
            result = await notetaker_agent._execute_task_impl(org_case['task_context'])
            
            if result.get('success', False):
                logger.info(f"🗂️ 【组织成功】")
                
                if 'categories' in result:
                    categories = result['categories']
                    logger.info(f"📂 分类结果: {len(categories)}个类别")
                    for category, items in categories.items():
                        logger.info(f"  📁 {category}: {len(items)}项")
                
                if 'links_created' in result:
                    logger.info(f"🔗 创建链接: {result['links_created']}个")
                
                if 'summary' in result:
                    summary = result['summary']
                    logger.info(f"📊 知识库摘要:")
                    logger.info(f"  📝 总计项目: {summary.get('total_items', 0)}")
                    
                    by_type = summary.get('by_type', {})
                    if by_type:
                        logger.info(f"  📂 按类型分布: {by_type}")
                    
                    by_importance = summary.get('by_importance', {})
                    if by_importance:
                        logger.info(f"  ⭐ 按重要性分布: {by_importance}")
                    
                    recent_activity = summary.get('recent_activity', {})
                    if recent_activity:
                        logger.info(f"  🔄 最近活动: 捕获{recent_activity.get('recent_captures', 0)}次, 查询{recent_activity.get('recent_queries', 0)}次")
            else:
                logger.error(f"❌ 【组织失败】")
                logger.error(f"💥 错误信息: {result.get('error', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ 知识组织异常: {e}")
        
        await asyncio.sleep(0.5)
    
    logger.info("\n🎉 所有知识管理测试完成")
    
    # 显示知识管理统计
    knowledge_stats = notetaker_agent.get_knowledge_stats()
    recent_captures = notetaker_agent.get_recent_captures()
    query_history = notetaker_agent.get_query_history()
    
    logger.info(f"\n📊 知识管理统计:")
    logger.info(f"📝 最近捕获: {len(recent_captures)}次")
    logger.info(f"🔍 查询历史: {len(query_history)}次")
    
    if knowledge_stats:
        logger.info(f"📈 知识库统计: {knowledge_stats}")
    
    return notetaker_agent

async def test_knowledge_retrieval_capabilities(notetaker_agent: NotetakerAgent):
    """专门测试NotetakerAgent的知识检索能力"""
    logger.info("\n🎯 开始测试NotetakerAgent知识检索能力")
    
    retrieval_tests = [
        {
            "name": "基础搜索能力测试",
            "test_type": "basic_search",
            "queries": [
                {"query": "click", "expected_min": 1},
                {"query": "tap", "expected_min": 1},
                {"query": "购物", "expected_min": 1},
                {"query": "error", "expected_min": 1}
            ]
        },
        {
            "name": "类型特定检索测试",
            "test_type": "type_specific",
            "queries": [
                {"area": "mobile_automation", "method": "get_best_practices"},
                {"area": "click_operations", "method": "get_best_practices"},
                {"error_type": "element_not_found", "method": "get_error_solutions"},
                {"error_type": None, "method": "get_error_solutions"}
            ]
        },
        {
            "name": "多粒度检索验证",
            "test_type": "multi_granularity",
            "queries": [
                {"query": "淘宝购物", "expected_granularity": "complete_task"},
                {"query": "点击操作", "expected_granularity": "subtask"},
                {"query": "tap坐标", "expected_granularity": "atomic_operation"}
            ]
        },
        {
            "name": "语义理解检索测试",
            "test_type": "semantic_search",
            "queries": [
                {"query": "如何点击按钮", "expected_concepts": ["click", "button", "tap"]},
                {"query": "购买商品流程", "expected_concepts": ["shopping", "workflow", "e_commerce"]},
                {"query": "操作失败怎么办", "expected_concepts": ["error", "solution", "recovery"]}
            ]
        }
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_group in retrieval_tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 {test_group['name']}")
        logger.info(f"{'='*60}")
        
        for i, query_test in enumerate(test_group['queries'], 1):
            total_tests += 1
            test_passed = False
            
            try:
                if test_group['test_type'] == 'basic_search':
                    # 基础搜索测试
                    query = query_test['query']
                    expected_min = query_test['expected_min']
                    
                    results = await notetaker_agent.search_knowledge(query, limit=5)
                    
                    if len(results) >= expected_min:
                        logger.info(f"✅ 基础搜索 {i}: '{query}' -> 找到{len(results)}条结果 (≥{expected_min})")
                        test_passed = True
                    else:
                        logger.warning(f"⚠️ 基础搜索 {i}: '{query}' -> 仅找到{len(results)}条结果 (<{expected_min})")
                
                elif test_group['test_type'] == 'type_specific':
                    # 类型特定检索测试
                    method = query_test['method']
                    
                    if method == 'get_best_practices':
                        area = query_test.get('area')
                        results = await notetaker_agent.get_best_practices(area)
                        logger.info(f"✅ 最佳实践检索 {i}: 领域'{area}' -> 找到{len(results)}条")
                        test_passed = True
                    
                    elif method == 'get_error_solutions':
                        error_type = query_test.get('error_type')
                        results = await notetaker_agent.get_error_solutions(error_type)
                        logger.info(f"✅ 错误解决方案检索 {i}: 类型'{error_type}' -> 找到{len(results)}条")
                        test_passed = True
                
                elif test_group['test_type'] == 'multi_granularity':
                    # 多粒度检索验证
                    query = query_test['query']
                    expected_granularity = query_test['expected_granularity']
                    
                    results = await notetaker_agent.search_knowledge(query, limit=5)
                    
                    # 检查结果中是否包含预期粒度的知识
                    granularity_found = False
                    for result in results:
                        tags = result.get('tags', [])
                        knowledge_type = result.get('type', '')
                        
                        if expected_granularity == 'complete_task' and ('complete_task' in tags or knowledge_type == 'task_workflow'):
                            granularity_found = True
                            break
                        elif expected_granularity == 'subtask' and ('subtask' in tags or knowledge_type == 'best_practice'):
                            granularity_found = True
                            break
                        elif expected_granularity == 'atomic_operation' and ('atomic' in tags or knowledge_type == 'action_pattern'):
                            granularity_found = True
                            break
                    
                    if granularity_found:
                        logger.info(f"✅ 多粒度检索 {i}: '{query}' -> 找到{expected_granularity}级别知识")
                        test_passed = True
                    else:
                        logger.warning(f"⚠️ 多粒度检索 {i}: '{query}' -> 未找到{expected_granularity}级别知识")
                
                elif test_group['test_type'] == 'semantic_search':
                    # 语义理解检索测试
                    query = query_test['query']
                    expected_concepts = query_test['expected_concepts']
                    
                    results = await notetaker_agent.search_knowledge(query, limit=5)
                    
                    # 检查结果中是否包含预期概念
                    concepts_found = 0
                    for result in results:
                        content_str = str(result.get('content', '')).lower()
                        title_str = str(result.get('title', '')).lower()
                        tags_str = ' '.join(result.get('tags', [])).lower()
                        
                        full_text = f"{content_str} {title_str} {tags_str}"
                        
                        for concept in expected_concepts:
                            if concept.lower() in full_text:
                                concepts_found += 1
                                break
                    
                    concept_coverage = concepts_found / len(results) if results else 0
                    
                    if concept_coverage >= 0.3:  # 至少30%的结果包含预期概念
                        logger.info(f"✅ 语义检索 {i}: '{query}' -> 概念覆盖率{concept_coverage:.1%}")
                        test_passed = True
                    else:
                        logger.warning(f"⚠️ 语义检索 {i}: '{query}' -> 概念覆盖率仅{concept_coverage:.1%}")
                
                if test_passed:
                    passed_tests += 1
                    
            except Exception as e:
                logger.error(f"❌ 检索测试 {i} 异常: {e}")
            
            await asyncio.sleep(0.2)
    
    # 输出测试总结
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    logger.info(f"\n📊 知识检索能力测试总结:")
    logger.info(f"   总测试数: {total_tests}")
    logger.info(f"   通过测试: {passed_tests}")
    logger.info(f"   成功率: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        logger.info(f"🎉 NotetakerAgent知识检索能力测试 - 优秀！")
    elif success_rate >= 0.6:
        logger.info(f"👍 NotetakerAgent知识检索能力测试 - 良好")
    else:
        logger.warning(f"⚠️ NotetakerAgent知识检索能力需要改进")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate
    }


async def test_convenience_methods():
    """测试便捷方法"""
    logger.info("\n🛠️ 开始测试便捷方法")
    
    # 初始化NotetakerAgent
    api_key = os.getenv('BAILIAN_API_KEY')
    if api_key:
        llm_provider = BailianProvider(
            api_key=api_key,
            model="qwen-vl-max",
            temperature=0.3
        )
        
        notetaker_agent = NotetakerAgent(
            llm_provider=llm_provider,
            agent_id="convenience_test_notetaker"
        )
        
        # 测试搜索知识
        logger.info("🔍 测试搜索知识功能")
        try:
            search_results = await notetaker_agent.search_knowledge("click", limit=3)
            logger.info(f"搜索结果: 找到{len(search_results)}条相关知识")
            for i, item in enumerate(search_results, 1):
                logger.info(f"  {i}. {item.get('title', 'N/A')}")
        except Exception as e:
            logger.error(f"搜索知识失败: {e}")
        
        # 测试获取最佳实践
        logger.info("\n💡 测试获取最佳实践")
        try:
            best_practices = await notetaker_agent.get_best_practices("mobile_automation")
            logger.info(f"最佳实践: 找到{len(best_practices)}条")
            for i, practice in enumerate(best_practices, 1):
                logger.info(f"  {i}. {practice.get('title', 'N/A')}")
        except Exception as e:
            logger.error(f"获取最佳实践失败: {e}")
        
        # 测试获取错误解决方案
        logger.info("\n🔧 测试获取错误解决方案")
        try:
            error_solutions = await notetaker_agent.get_error_solutions("element_not_found")
            logger.info(f"错误解决方案: 找到{len(error_solutions)}条")
            for i, solution in enumerate(error_solutions, 1):
                logger.info(f"  {i}. {solution.get('title', 'N/A')}")
        except Exception as e:
            logger.error(f"获取错误解决方案失败: {e}")
        
        # 运行知识检索能力测试
        await test_knowledge_retrieval_capabilities(notetaker_agent)
        
    else:
        logger.warning("未设置BAILIAN_API_KEY，跳过便捷方法测试")

def main():
    """主函数"""
    try:
        # 运行基础测试
        notetaker_agent = asyncio.run(test_multimodal_knowledge_management())
        
        # 运行便捷方法测试
        asyncio.run(test_convenience_methods())
        
        logger.info("\n🎊 所有知识管理测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        logger.error("\n💡 请检查:")
        logger.error("   1. 是否在.env文件中设置了 BAILIAN_API_KEY")
        logger.error("   2. API Key是否有效且有足够额度")
        logger.error("   3. 网络连接是否可以访问百炼API服务")
        logger.error("   4. 是否已安装所需依赖")
        logger.error("   5. knowledge_base目录是否可写")

if __name__ == "__main__":
    main()