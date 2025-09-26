#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ActionReflectorAgent - 多模态动作反思器智能体

基于AgenticX框架和Mobile Agent v3设计精髓，实现真正的多模态LLM驱动的动作反思分析。
负责：
1. 多模态分析执行前后的屏幕状态变化
2. 基于视觉理解判断操作成功性
3. 生成智能化的改进建议和学习洞察
4. 支持多模型降级策略确保可靠性
"""

import asyncio
import copy
import json
from rich import print
from rich.json import JSON
from loguru import logger
import base64
import os
from typing import Dict, Any, List, Optional, Tuple
import json
from collections import defaultdict

# 使用AgenticX核心组件
from agenticx.core.agent import Agent, AgentResult
from agenticx.core.tool import BaseTool
from agenticx.core.event import Event, TaskStartEvent, TaskEndEvent, ReplanningRequiredEvent, ActionCorrectionEvent
from agenticx.core.event_bus import EventBus
from agenticx.llms.base import BaseLLMProvider
from agenticx.memory.component import MemoryComponent

from core.base_agent import BaseAgenticXGUIAgentAgent
from config import AgentConfig
from utils import get_iso_timestamp


class MultimodalActionAnalysisTool(BaseTool):
    """多模态动作分析工具 - 基于Mobile Agent v3的ActionReflector设计精髓"""
    
    name: str = "multimodal_action_analysis"
    description: str = "使用多模态LLM分析执行前后的屏幕状态变化，判断操作成功性"
    
    # 添加自定义属性
    llm_provider: Optional[BaseLLMProvider] = None
    model_fallback_chain: List[Dict[str, str]] = []
    
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm_provider = llm_provider
        
        # 定义模型降级策略
        self.model_fallback_chain = [
            {"provider": "bailian", "model": "qwen-vl-max"},
            {"provider": "bailian", "model": "qwen-vl-plus"},
            {"provider": "kimi", "model": "moonshot-v1-8k"}
        ]
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法（简化版本，建议使用异步版本）"""
        # 从kwargs中提取action_data参数
        action_data = kwargs.get('action_data', {})
        
        if not self.llm_provider:
            logger.error("未配置LLM提供者，无法执行多模态分析")
            return {"success": False, "error": "未配置LLM提供者"}
        
        try:
            # 尝试同步调用LLM（如果支持）
            return self._sync_multimodal_analysis(action_data)
        except Exception as e:
            logger.error(f"同步多模态分析失败: {e}")
            return {
                "success": False,
                "error": f"同步分析失败: {str(e)}",
                "analysis_time": get_iso_timestamp(),
                "note": "建议使用异步版本aexecute以获得多模型降级支持"
            }
    
    def _extract_coordinate_feedback_from_analysis(self, coordinate_analysis: str, 
                                                 comparison_analysis: str, 
                                                 improvement_suggestions: str) -> Optional[Dict[str, Any]]:
        """从分析文本中提取坐标反馈信息 - MultimodalActionAnalysisTool版本"""
        try:
            # 合并所有相关文本
            combined_text = f"{coordinate_analysis} {comparison_analysis} {improvement_suggestions}"
            
            if not combined_text.strip():
                return None
            
            # 尝试提取精确的像素调整
            import re
            
            adjustment_x = 0
            adjustment_y = 0
            reasons = []
            
            # 水平方向调整
            horizontal_patterns = [
                r'向左调整(\d+)像素',
                r'向右调整(\d+)像素',
                r'左移(\d+)像素',
                r'右移(\d+)像素'
            ]
            
            for pattern in horizontal_patterns:
                matches = re.findall(pattern, combined_text)
                for match in matches:
                    pixels = int(match)
                    if '左' in pattern:
                        adjustment_x -= pixels
                        reasons.append(f"向左调整{pixels}像素")
                    else:
                        adjustment_x += pixels
                        reasons.append(f"向右调整{pixels}像素")
            
            # 垂直方向调整
            vertical_patterns = [
                r'向上调整(\d+)像素',
                r'向下调整(\d+)像素',
                r'上移(\d+)像素',
                r'下移(\d+)像素'
            ]
            
            for pattern in vertical_patterns:
                matches = re.findall(pattern, combined_text)
                for match in matches:
                    pixels = int(match)
                    if '上' in pattern:
                        adjustment_y -= pixels
                        reasons.append(f"向上调整{pixels}像素")
                    else:
                        adjustment_y += pixels
                        reasons.append(f"向下调整{pixels}像素")
            
            if adjustment_x != 0 or adjustment_y != 0:
                return {
                    "original_coordinates": [0, 0],  # 占位符
                    "suggested_adjustment": [adjustment_x, adjustment_y],
                    "reason": "基于多模态分析的坐标调整: " + ", ".join(set(reasons)),
                    "confidence": 0.85,
                    "analysis_method": "multimodal_tool_extraction"
                }
            
            return None
            
        except Exception as e:
              logger.error(f"MultimodalActionAnalysisTool坐标反馈提取失败: {e}")
              return None
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """异步执行多模态动作分析 - 支持多模型降级策略
        
        Args:
            action_data: 动作数据，包含:
                - before_screenshot: 操作前截图路径
                - after_screenshot: 操作后截图路径
                - action: 执行的动作信息
                - expectation: 期望的结果
            **kwargs: 额外参数
        
        Returns:
            分析结果，包含成功判断、错误分析、改进建议等
        """
        # 从kwargs中提取action_data参数
        action_data = kwargs.get('action_data', {})
        
        if not self.llm_provider:
            logger.error("未配置LLM提供者，无法执行多模态分析")
            return {"success": False, "error": "未配置LLM提供者"}
        
        # 尝试多模型降级策略
        for i, model_config in enumerate(self.model_fallback_chain):
            model_name = f"{model_config['provider']}/{model_config['model']}"
            try:
                logger.info(f"🤖 尝试使用 {model_name} 进行动作反思分析...")
                
                # 创建对应的LLM提供者
                provider = await self._create_provider(model_config)
                if not provider:
                    continue
                
                # 执行多模态分析
                result = await self._multimodal_reflection_analysis(
                    provider, action_data, model_config
                )
                
                logger.info(f"✅ {model_name} 动作反思分析成功")
                return result
                
            except Exception as e:
                logger.warning(f"❌ {model_name} 分析失败: {e}")
                if i == len(self.model_fallback_chain) - 1:
                    # 所有模型都失败了
                    logger.error("🚨 所有LLM模型都失败，动作反思分析无法完成")
                    return {
                        "success": False,
                        "error": f"所有模型都失败: {str(e)}",
                        "attempted_models": [f"{m['provider']}/{m['model']}" for m in self.model_fallback_chain],
                        "analysis_time": get_iso_timestamp()
                    }
                else:
                    next_model = self.model_fallback_chain[i+1]
                    next_model_name = f"{next_model['provider']}/{next_model['model']}"
                    logger.info(f"🔄 降级到下一个模型: {next_model_name}")
                    continue
        
        return {"success": False, "error": "未知错误"}
    
    async def _create_provider(self, model_config: Dict[str, str]):
        """根据配置创建LLM提供者"""
        try:
            import os
            
            if model_config["provider"] == "bailian":
                from agenticx.llms.bailian_provider import BailianProvider
                api_key = os.getenv('BAILIAN_API_KEY')
                if not api_key:
                    model_name = f"{model_config['provider']}/{model_config['model']}"
                    logger.warning(f"未设置BAILIAN_API_KEY，跳过{model_name}")
                    return None
                
                return BailianProvider(
                    api_key=api_key,
                    model=model_config["model"],
                    temperature=0.3,
                    timeout=60.0
                )
            
            elif model_config["provider"] == "kimi":
                from agenticx.llms.kimi_provider import KimiProvider
                api_key = os.getenv('MOONSHOT_API_KEY') or os.getenv('KIMI_API_KEY')
                if not api_key:
                    model_name = f"{model_config['provider']}/{model_config['model']}"
                    logger.warning(f"未设置MOONSHOT_API_KEY或KIMI_API_KEY，跳过{model_name}")
                    return None
                
                return KimiProvider(
                    api_key=api_key,
                    model=model_config["model"],
                    temperature=0.3,
                    timeout=60.0
                )
            
            else:
                logger.warning(f"不支持的提供者: {model_config['provider']}")
                return None
                
        except Exception as e:
            model_name = f"{model_config['provider']}/{model_config['model']}"
            logger.error(f"创建{model_name}提供者失败: {e}")
            return None
    
    async def _multimodal_reflection_analysis(
        self, 
        provider, 
        action_data: Dict[str, Any], 
        model_config: Dict[str, str]
    ) -> Dict[str, Any]:
        """使用指定提供者执行多模态反思分析 - 参考Mobile Agent v3的ActionReflector设计"""
        
        # 构建多模态反思提示词
        prompt = self._build_reflection_prompt(action_data)
        
        # 添加操作前后的截图进行对比分析
        before_screenshot = action_data.get("before_screenshot")
        after_screenshot = action_data.get("after_screenshot")
        
        # 准备图像内容
        if before_screenshot and after_screenshot:
            try:
                # 读取操作前截图
                with open(before_screenshot, "rb") as f:
                    before_image_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                # 读取操作后截图
                with open(after_screenshot, "rb") as f:
                    after_image_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                # 构建多模态消息
                messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_image_base64}"}},
                        {"type": "text", "text": "\n\n### 操作后截图 ###"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_image_base64}"}}
                    ]
                }]
            except Exception as e:
                logger.warning(f"读取截图文件失败: {e}，将使用纯文本模式")
                # 如果读取失败，回退到纯文本模式
                messages = [{"role": "user", "content": prompt}]
        else:
            # 使用纯文本模式
            messages = [{"role": "user", "content": prompt}]
        
        # 调用LLM进行分析
        # 为了日志记录，截断base64字符串
        log_messages = copy.deepcopy(messages)
        for message in log_messages:
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "image_url" and item.get("image_url", {}).get("url", "").startswith("data:image"):
                        item["image_url"]["url"] = item["image_url"]["url"][:50] + "..."
        logger.info(f"发送给reflector的指令: {log_messages}")
        response = await provider.ainvoke(messages)
        result = self._parse_reflection_response(response.content, action_data)
        
        # 添加模型信息
        result["model_used"] = f"{model_config['provider']}/{model_config['model']}"
        result["provider"] = model_config["provider"]
        
        return result
    
    def _build_reflection_prompt(self, action_data: Dict[str, Any]) -> str:
        """构建反思分析提示词 - 增强版本，支持坐标精度分析"""
        
        action_info = action_data.get("action", {})
        expectation = action_data.get("expectation", "")
        
        prompt = "你是一个专业的移动设备操作分析专家，能够通过对比操作前后的截图来判断操作是否成功，并提供精确的坐标优化建议。\n\n"
        
        prompt += "### 任务信息 ###\n"
        prompt += f"执行的操作：{action_info}\n"
        prompt += f"期望结果：{expectation}\n\n"
        
        # 添加坐标信息
        if "coordinates" in action_info:
            coords = action_info["coordinates"]
            prompt += f"执行坐标：({coords.get('x', 0)}, {coords.get('y', 0)})\n\n"
        
        prompt += "### 分析要求 ###\n"
        prompt += "请仔细对比操作前后的两张截图，特别注意：\n"
        prompt += "1. 操作是否达到了预期效果\n"
        prompt += "2. 如果截图中有紫色点标注，分析点击位置是否精确\n"
        prompt += "3. 点击位置与目标元素中心的偏移情况\n"
        prompt += "4. 提供具体的像素级坐标调整建议\n\n"
        
        prompt += "### 判断标准 ###\n"
        prompt += "A: 成功或部分成功 - 操作结果符合预期\n"
        prompt += "B: 失败 - 操作导致错误页面或意外结果\n"
        prompt += "C: 失败 - 操作没有产生任何变化\n\n"
        
        prompt += "### 特别注意 ###\n"
        prompt += "对于滑动操作：如果操作前后内容完全相同，则认为是C类失败（可能已滑动到底部）\n"
        prompt += "对于点击操作：\n"
        prompt += "- 检查是否打开了新页面、弹出了菜单或产生了预期的界面变化\n"
        prompt += "- 如果有紫色点标注，分析点击位置是否在目标元素的有效区域内\n"
        prompt += "- 评估点击位置与目标元素中心的偏移距离\n"
        prompt += "对于输入操作：检查文本是否正确输入到目标位置\n\n"
        
        prompt += "请按以下格式提供分析结果：\n"
        prompt += "### 对比分析 ###\n"
        prompt += "详细描述操作前后截图的差异和变化\n\n"
        
        prompt += "### 成功判断 ###\n"
        prompt += "选择A、B或C，并说明判断理由\n\n"
        
        prompt += "### 坐标精度分析 ###\n"
        prompt += "如果有紫色点标注，分析：\n"
        prompt += "- 点击位置是否准确命中目标元素\n"
        prompt += "- 与目标元素中心的偏移方向和距离（像素）\n"
        prompt += "- 具体的坐标调整建议（如：向右调整10像素，向下调整5像素）\n\n"
        
        prompt += "### 错误分析 ###\n"
        prompt += "如果操作失败，分析可能的原因和错误类型\n\n"
        
        prompt += "### 改进建议 ###\n"
        prompt += "提供具体的改进建议和优化方案，包括：\n"
        prompt += "- 精确的坐标调整数值\n"
        prompt += "- 操作时机优化\n"
        prompt += "- 其他执行策略建议\n"
        
        return prompt
    
    def _parse_reflection_response(self, response_content: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析反思分析响应 - 增强版本，支持坐标精度分析"""
        try:
            import re
            
            # 提取对比分析
            comparison_match = re.search(r'### 对比分析 ###\s*(.+?)\s*### 成功判断 ###', response_content, re.DOTALL)
            comparison_analysis = comparison_match.group(1).strip() if comparison_match else ""
            
            # 提取成功判断
            judgment_match = re.search(r'### 成功判断 ###\s*(.+?)\s*### (坐标精度分析|错误分析) ###', response_content, re.DOTALL)
            success_judgment = judgment_match.group(1).strip() if judgment_match else ""
            
            # 提取坐标精度分析（新增）
            coordinate_match = re.search(r'### 坐标精度分析 ###\s*(.+?)\s*### 错误分析 ###', response_content, re.DOTALL)
            coordinate_analysis = coordinate_match.group(1).strip() if coordinate_match else ""
            
            # 提取错误分析
            error_match = re.search(r'### 错误分析 ###\s*(.+?)\s*### 改进建议 ###', response_content, re.DOTALL)
            error_analysis = error_match.group(1).strip() if error_match else ""
            
            # 提取改进建议
            improvement_match = re.search(r'### 改进建议 ###\s*(.+?)$', response_content, re.DOTALL)
            improvement_suggestions = improvement_match.group(1).strip() if improvement_match else ""
            
            # 判断操作结果
            outcome = "A"  # 默认成功
            if "B" in success_judgment.upper():
                outcome = "B"
            elif "C" in success_judgment.upper():
                outcome = "C"
            
            success = outcome == "A"
            
            # 提取坐标调整信息
            coordinate_feedback = self._extract_coordinate_feedback_from_analysis(
                coordinate_analysis, comparison_analysis, improvement_suggestions
            )
            
            result = {
                "success": True,  # 分析成功
                "operation_success": success,  # 操作是否成功
                "outcome": outcome,
                "comparison_analysis": comparison_analysis,
                "success_judgment": success_judgment,
                "coordinate_analysis": coordinate_analysis,  # 新增坐标分析
                "error_analysis": error_analysis,
                "improvement_suggestions": improvement_suggestions,
                "coordinate_feedback": coordinate_feedback,  # 新增坐标反馈
                "full_response": response_content,
                "analysis_time": get_iso_timestamp(),
                "method": "enhanced_multimodal_llm_reflection"
            }
            
            # 记录坐标分析结果
            if coordinate_feedback:
                logger.info(f"🎯 提取到坐标反馈: {coordinate_feedback}")
            
            return result
            
        except Exception as e:
            logger.error(f"解析反思响应失败: {e}")
            return {
                "success": False,
                "error": f"解析响应失败: {str(e)}",
                "full_response": response_content,
                "analysis_time": get_iso_timestamp()
            }
    
    def _sync_multimodal_analysis(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步多模态分析（简化版本）"""
        # 简化的同步实现，提供基本的分析功能
        try:
            # 基本的成功性判断
            action_info = action_data.get("action", {})
            task_type = action_data.get("task_type", "unknown")
            
            # 简单的启发式分析
            success_score = 0.7  # 默认成功分数
            quality_score = 0.6  # 默认质量分数
            
            return {
                "success": True,
                "task_type": task_type,
                "success_score": success_score,
                "quality_score": quality_score,
                "analysis_method": "sync_heuristic",
                "analysis_time": get_iso_timestamp(),
                "note": "同步简化版本，建议使用异步版本aexecute以获得完整的多模态LLM支持"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"同步分析失败: {str(e)}",
                "analysis_time": get_iso_timestamp()
            }
    
    def _calculate_efficiency_score(self, action_data: Dict[str, Any]) -> float:
        """计算效率分数"""
        result = action_data.get("result", {})
        duration = result.get("duration", 1.0)
        retry_count = action_data.get("retry_count", 0)
        
        # 基础分数
        base_score = 1.0
        
        # 根据执行时间调整
        if duration > 5.0:
            base_score *= 0.7
        elif duration > 2.0:
            base_score *= 0.9
        
        # 根据重试次数调整
        base_score *= (1.0 - retry_count * 0.2)
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_accuracy_score(self, action_data: Dict[str, Any]) -> float:
        """计算准确性分数"""
        success = action_data.get("success", False)
        result = action_data.get("result", {})
        
        if not success:
            return 0.0
        
        # 根据结果中的置信度调整
        confidence = result.get("confidence", 0.9)
        return confidence
    
    def _analyze_errors(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析错误"""
        error_info = action_data.get("error")
        if not error_info:
            return {"has_error": False}
        
        return {
            "has_error": True,
            "error_type": self._classify_error(error_info),
            "error_message": str(error_info),
            "potential_causes": self._identify_error_causes(error_info),
            "recovery_suggestions": self._suggest_error_recovery(error_info)
        }
    
    def _classify_error(self, error_info: str) -> str:
        """分类错误类型"""
        error_str = str(error_info).lower()
        
        if "timeout" in error_str:
            return "timeout_error"
        elif "element" in error_str and "not found" in error_str:
            return "element_not_found"
        elif "permission" in error_str:
            return "permission_error"
        elif "network" in error_str or "connection" in error_str:
            return "network_error"
        else:
            return "unknown_error"
    
    def _identify_error_causes(self, error_info: str) -> List[str]:
        """识别错误原因"""
        error_type = self._classify_error(error_info)
        
        causes_map = {
            "timeout_error": ["网络延迟", "设备响应慢", "操作复杂度高"],
            "element_not_found": ["UI界面变化", "元素定位不准确", "页面加载未完成"],
            "permission_error": ["权限不足", "安全策略限制", "应用状态异常"],
            "network_error": ["网络连接问题", "服务器不可用", "代理设置问题"],
            "unknown_error": ["未知系统问题", "代码逻辑错误", "环境配置问题"]
        }
        
        return causes_map.get(error_type, ["未知原因"])
    
    def _suggest_error_recovery(self, error_info: str) -> List[str]:
        """建议错误恢复方案"""
        error_type = self._classify_error(error_info)
        
        recovery_map = {
            "timeout_error": ["增加超时时间", "分解复杂操作", "检查网络状态"],
            "element_not_found": ["更新元素定位策略", "等待页面加载", "使用备用定位方法"],
            "permission_error": ["检查应用权限", "重启应用", "使用管理员权限"],
            "network_error": ["检查网络连接", "重试操作", "使用离线模式"],
            "unknown_error": ["重启系统", "检查日志", "联系技术支持"]
        }
        
        return recovery_map.get(error_type, ["重试操作"])
    
    def _generate_suggestions(self, action_data: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 基于效率分数的建议
        efficiency = self._calculate_efficiency_score(action_data)
        if efficiency < 0.7:
            suggestions.append("考虑优化操作流程以提高效率")
        
        # 基于准确性分数的建议
        accuracy = self._calculate_accuracy_score(action_data)
        if accuracy < 0.8:
            suggestions.append("提高元素定位的准确性")
        
        # 基于重试次数的建议
        retry_count = action_data.get("retry_count", 0)
        if retry_count > 1:
            suggestions.append("分析重试原因，改进初次执行成功率")
        
        # 基于任务类型的建议
        task_type = action_data.get("task_type", "")
        if task_type == "click_action":
            suggestions.append("确保点击目标元素可见且可交互")
        elif task_type == "input_text":
            suggestions.append("验证输入框状态和文本格式")
        elif task_type == "swipe_action":
            suggestions.append("调整滑动速度和距离")
        
        return suggestions if suggestions else ["当前操作表现良好"]


class PerformanceAnalysisTool(BaseTool):
    """性能分析工具"""
    
    name: str = "performance_analysis"
    description: str = "分析整体性能表现"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法"""
        # 从kwargs中提取action_history参数
        action_history = kwargs.get('action_history', [])
        
        # 直接返回模拟结果，避免异步调用问题
        return {
            "performance_id": f"perf_{get_iso_timestamp()}",
            "total_actions": len(action_history),
            "success_rate": 0.9,
            "avg_execution_time": 2.5,
            "bottlenecks": [],
            "improvements": ["优化执行速度"],
            "success": True
        }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """分析性能
        
        Args:
            action_history: 动作历史
            **kwargs: 额外参数
        
        Returns:
            性能分析结果
        """
        # 从kwargs中提取action_history参数
        action_history = kwargs.get('action_history', [])
        
        await asyncio.sleep(1.0)  # 模拟分析时间
        
        if not action_history:
            return {
                "total_actions": 0,
                "success_rate": 0.0,
                "average_efficiency": 0.0,
                "performance_trends": [],
                "analysis_time": get_iso_timestamp()
            }
        
        # 统计基本指标
        total_actions = len(action_history)
        successful_actions = sum(1 for action in action_history if action.get("success", False))
        success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
        
        # 计算平均效率
        efficiency_scores = []
        for action in action_history:
            result = action.get("result", {})
            duration = result.get("duration", 1.0)
            retry_count = action.get("retry_count", 0)
            
            efficiency = 1.0 - (duration / 10.0) - (retry_count * 0.2)
            efficiency_scores.append(max(0.0, min(1.0, efficiency)))
        
        average_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
        
        # 分析性能趋势
        performance_trends = self._analyze_trends(action_history)
        
        # 识别问题模式
        problem_patterns = self._identify_problem_patterns(action_history)
        
        # 生成优化建议
        optimization_suggestions = self._generate_optimization_suggestions(
            success_rate, average_efficiency, problem_patterns
        )
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": success_rate,
            "average_efficiency": average_efficiency,
            "performance_trends": performance_trends,
            "problem_patterns": problem_patterns,
            "optimization_suggestions": optimization_suggestions,
            "analysis_time": get_iso_timestamp()
        }
    
    def _analyze_trends(self, action_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析性能趋势"""
        if len(action_history) < 5:
            return []
        
        # 按时间窗口分析
        window_size = max(5, len(action_history) // 4)
        trends = []
        
        for i in range(0, len(action_history) - window_size + 1, window_size):
            window = action_history[i:i + window_size]
            window_success_rate = sum(1 for action in window if action.get("success", False)) / len(window)
            
            trends.append({
                "window_start": i,
                "window_end": i + window_size - 1,
                "success_rate": window_success_rate,
                "action_count": len(window)
            })
        
        return trends
    
    def _identify_problem_patterns(self, action_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """识别问题模式"""
        patterns = {
            "frequent_failures": [],
            "slow_operations": [],
            "retry_patterns": [],
            "error_clusters": []
        }
        
        # 统计失败频繁的操作类型
        failure_counts = defaultdict(int)
        total_counts = defaultdict(int)
        
        for action in action_history:
            task_type = action.get("task_type", "unknown")
            total_counts[task_type] += 1
            if not action.get("success", False):
                failure_counts[task_type] += 1
        
        for task_type, failure_count in failure_counts.items():
            total_count = total_counts[task_type]
            failure_rate = failure_count / total_count
            if failure_rate > 0.3:  # 失败率超过30%
                patterns["frequent_failures"].append({
                    "task_type": task_type,
                    "failure_rate": failure_rate,
                    "failure_count": failure_count,
                    "total_count": total_count
                })
        
        # 识别慢操作
        for action in action_history:
            result = action.get("result", {})
            duration = result.get("duration", 0)
            if duration > 3.0:  # 超过3秒的操作
                patterns["slow_operations"].append({
                    "task_type": action.get("task_type", "unknown"),
                    "duration": duration,
                    "timestamp": action.get("timestamp")
                })
        
        # 识别重试模式
        for action in action_history:
            retry_count = action.get("retry_count", 0)
            if retry_count > 0:
                patterns["retry_patterns"].append({
                    "task_type": action.get("task_type", "unknown"),
                    "retry_count": retry_count,
                    "final_success": action.get("success", False)
                })
        
        return patterns
    
    def _generate_optimization_suggestions(self, success_rate: float, efficiency: float, patterns: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于成功率的建议
        if success_rate < 0.8:
            suggestions.append("整体成功率偏低，需要改进操作策略")
        
        # 基于效率的建议
        if efficiency < 0.7:
            suggestions.append("操作效率有待提升，考虑优化执行流程")
        
        # 基于问题模式的建议
        if patterns["frequent_failures"]:
            suggestions.append("某些操作类型失败率较高，需要重点优化")
        
        if patterns["slow_operations"]:
            suggestions.append("存在执行缓慢的操作，建议优化或分解")
        
        if patterns["retry_patterns"]:
            suggestions.append("重试次数较多，建议改进初次执行成功率")
        
        return suggestions if suggestions else ["整体表现良好，继续保持"]


class LearningInsightTool(BaseTool):
    """学习洞察工具"""
    
    name: str = "learning_insight"
    description: str = "生成学习洞察和知识"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """同步执行方法"""
        # 从kwargs中提取analysis_results参数
        analysis_results = kwargs.get('analysis_results', [])
        
        # 直接返回模拟结果，避免异步调用问题
        return {
            "insight_id": f"insight_{get_iso_timestamp()}",
            "patterns_identified": ["用户偏好模式"],
            "learning_points": ["提高准确性"],
            "strategy_updates": [],
            "confidence_score": 0.8,
            "success": True
        }
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """生成学习洞察
        
        Args:
            analysis_results: 分析结果列表
            **kwargs: 额外参数
        
        Returns:
            学习洞察
        """
        # 从kwargs中提取analysis_results参数
        analysis_results = kwargs.get('analysis_results', [])
        
        await asyncio.sleep(0.8)  # 模拟分析时间
        
        insights = {
            "key_learnings": self._extract_key_learnings(analysis_results),
            "best_practices": self._identify_best_practices(analysis_results),
            "common_pitfalls": self._identify_common_pitfalls(analysis_results),
            "improvement_opportunities": self._identify_improvements(analysis_results),
            "knowledge_patterns": self._extract_knowledge_patterns(analysis_results),
            "insight_time": get_iso_timestamp()
        }
        
        return insights
    
    def _extract_key_learnings(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """提取关键学习点"""
        learnings = []
        
        # 分析成功模式
        successful_patterns = []
        for result in analysis_results:
            if result.get("success", False) and result.get("efficiency_score", 0) > 0.8:
                successful_patterns.append(result.get("task_type", "unknown"))
        
        if successful_patterns:
            most_successful = max(set(successful_patterns), key=successful_patterns.count)
            learnings.append(f"{most_successful}操作表现最佳，可作为标准模式")
        
        # 分析失败模式
        failed_patterns = []
        for result in analysis_results:
            if not result.get("success", False):
                failed_patterns.append(result.get("task_type", "unknown"))
        
        if failed_patterns:
            most_failed = max(set(failed_patterns), key=failed_patterns.count)
            learnings.append(f"{most_failed}操作失败率较高，需要重点关注")
        
        return learnings
    
    def _identify_best_practices(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """识别最佳实践"""
        practices = []
        
        # 基于高效率操作的实践
        high_efficiency_actions = [r for r in analysis_results if r.get("efficiency_score", 0) > 0.9]
        if high_efficiency_actions:
            practices.append("保持简洁的操作流程，避免不必要的步骤")
        
        # 基于高准确性操作的实践
        high_accuracy_actions = [r for r in analysis_results if r.get("accuracy_score", 0) > 0.95]
        if high_accuracy_actions:
            practices.append("使用精确的元素定位策略")
        
        # 基于零重试操作的实践
        zero_retry_actions = [r for r in analysis_results if r.get("retry_count", 0) == 0]
        if len(zero_retry_actions) > len(analysis_results) * 0.8:
            practices.append("充分的预检查可以减少重试次数")
        
        return practices
    
    def _identify_common_pitfalls(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """识别常见陷阱"""
        pitfalls = []
        
        # 分析错误模式
        error_types = []
        for result in analysis_results:
            error_analysis = result.get("error_analysis", {})
            if error_analysis.get("has_error"):
                error_types.append(error_analysis.get("error_type", "unknown"))
        
        if error_types:
            most_common_error = max(set(error_types), key=error_types.count)
            pitfalls.append(f"最常见的错误类型是{most_common_error}")
        
        # 分析低效率模式
        low_efficiency_count = sum(1 for r in analysis_results if r.get("efficiency_score", 1) < 0.5)
        if low_efficiency_count > len(analysis_results) * 0.3:
            pitfalls.append("操作效率普遍偏低，可能存在系统性问题")
        
        return pitfalls
    
    def _identify_improvements(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """识别改进机会"""
        improvements = []
        
        # 收集所有改进建议
        all_suggestions = []
        for result in analysis_results:
            suggestions = result.get("improvement_suggestions", [])
            all_suggestions.extend(suggestions)
        
        # 统计最频繁的建议
        if all_suggestions:
            suggestion_counts = defaultdict(int)
            for suggestion in all_suggestions:
                suggestion_counts[suggestion] += 1
            
            # 取前3个最频繁的建议
            top_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            improvements = [suggestion for suggestion, count in top_suggestions]
        
        return improvements
    
    def _extract_knowledge_patterns(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取知识模式"""
        patterns = {
            "success_factors": [],
            "failure_factors": [],
            "efficiency_factors": [],
            "timing_patterns": []
        }
        
        # 分析成功因素
        successful_results = [r for r in analysis_results if r.get("success", False)]
        if successful_results:
            avg_efficiency = sum(r.get("efficiency_score", 0) for r in successful_results) / len(successful_results)
            patterns["success_factors"].append(f"成功操作的平均效率: {avg_efficiency:.2f}")
        
        # 分析失败因素
        failed_results = [r for r in analysis_results if not r.get("success", False)]
        if failed_results:
            common_error_types = [r.get("error_analysis", {}).get("error_type") for r in failed_results]
            if common_error_types:
                most_common = max(set(common_error_types), key=common_error_types.count)
                patterns["failure_factors"].append(f"最常见失败原因: {most_common}")
        
        return patterns


class ActionReflectorAgent(BaseAgenticXGUIAgentAgent):
    """多模态动作反思器智能体 - 基于AgenticX框架和Mobile Agent v3设计精髓
    
    负责：
    1. 多模态分析执行前后的屏幕状态变化
    2. 基于视觉理解判断操作成功性
    3. 生成智能化的改进建议和学习洞察
    4. 支持多模型降级策略确保可靠性
    5. 与其他智能体协作进行持续学习
    """
    
    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        agent_id: str = "action_reflector",
        platform = None,
        info_pool = None,
        learning_engine = None,
        agent_config: Optional[AgentConfig] = None,
        memory: Optional[MemoryComponent] = None
    ):
        # 存储额外参数
        self.agent_id = agent_id
        self.platform = platform
        self.info_pool = info_pool
        self.learning_engine = learning_engine
        
        # 创建默认配置（如果未提供）
        if agent_config is None:
            agent_config = AgentConfig(
                id=agent_id,
                name="ActionReflectorAgent",
                role="action_reflector",
                goal="基于多模态LLM分析操作前后的屏幕变化，判断操作成功性并提供改进建议",
                backstory="我是一个多模态动作反思智能体，能够通过视觉理解分析操作效果，参考Mobile Agent v3的ActionReflector设计，基于AgenticX框架实现。",
                tools=[]
            )
        
        # 初始化多模态工具
        tools = [
            MultimodalActionAnalysisTool(llm_provider=llm_provider),
            # 保留一些传统分析工具作为备用
            PerformanceAnalysisTool(),
            LearningInsightTool()
        ]
        
        super().__init__(agent_config, llm_provider, memory, tools, info_pool=info_pool)
        
        # 反思分析状态
        self.reflection_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.learning_insights: Dict[str, Any] = {}
        self.screenshot_pairs: List[Tuple[str, str]] = []  # 存储操作前后截图对
    
    async def _execute_task_impl(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行反思分析 - ActionReflector核心功能
        
        输入：Executor的执行结果
        功能：执行质量评估、问题识别、改进建议
        输出：反思分析和改进建议
        
        Args:
            task_context: 任务上下文，包含:
                - task_id: 任务ID
                - execution_result: Executor的执行结果
                - prompt: 反思提示词
                - before_screenshot: 操作前截图路径
                - after_screenshot: 操作后截图路径
                - action_info: 执行的动作信息
                - expectation: 期望结果
        
        Returns:
            反思分析结果
        """
        task_id = task_context.get("task_id")
        execution_result = task_context.get("execution_result", {})
                
        # 检查是否有execution_result（来自collaboration.py的调用）
        if execution_result:
            return await self._analyze_execution_result(task_context)
        
        # 否则按原有逻辑处理（直接的多模态分析调用）
        analysis_type = task_context.get("analysis_type", "multimodal_reflection")
        logger.info(f"🔍 执行多模态反思分析: {analysis_type}")
        
        try:
            if analysis_type == "multimodal_reflection":
                result = await self._multimodal_action_reflection(task_context)
            elif analysis_type == "performance_analysis":
                result = await self._analyze_performance(task_context)
            elif analysis_type == "learning_insight":
                result = await self._generate_learning_insights(task_context)
            elif analysis_type == "comprehensive_analysis":
                result = await self._comprehensive_multimodal_analysis(task_context)
            else:
                # 默认使用多模态反思分析
                result = await self._multimodal_action_reflection(task_context)
            
            # 记录反思历史
            reflection_record = {
                "analysis_type": analysis_type,
                "task_context": task_context,
                "result": result,
                "timestamp": get_iso_timestamp(),
                "model_used": result.get("model_used", "unknown")
            }
            self.reflection_history.append(reflection_record)
            
            # 保持历史记录在合理范围内
            if len(self.reflection_history) > 100:
                self.reflection_history = self.reflection_history[-100:]
            
            # 发布反思结果事件
            reflection_event = Event(
                type="multimodal_reflection_result",
                data={
                    "agent_id": self.config.id,
                    "reflection_record": reflection_record
                },
                agent_id=self.config.id
            )
            await self.info_pool.publish_async(reflection_event)
            
            # 如果操作失败，发送具体的改进建议给ExecutorAgent
            if not result.get("operation_success", True):
                await self._send_improvement_feedback_to_executor(result, task_context)
            
            logger.info(f"✅ 多模态反思分析完成: {analysis_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 多模态反思分析失败: {analysis_type}, 错误: {e}")
            raise
    
    async def _multimodal_action_reflection(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """多模态动作反思分析 - 核心方法
        
        Args:
            task_context: 任务上下文
        
        Returns:
            反思分析结果
        """
        # 准备分析数据
        action_info = task_context.get("action_info", {})
        action_data = {
            "before_screenshot": task_context.get("before_screenshot"),
            "after_screenshot": task_context.get("after_screenshot"),
            "action": action_info,
            "expectation": task_context.get("expectation", ""),
            "task_type": action_info.get("action", task_context.get("task_type", "unknown"))
        }
        
        # 验证必要的输入
        if not action_data["before_screenshot"] or not action_data["after_screenshot"]:
            logger.warning("缺少操作前后截图，无法进行多模态分析")
            return {
                "success": False,
                "error": "缺少必要的截图文件",
                "analysis_time": get_iso_timestamp()
            }
        
        # 检查截图文件是否存在
        if not os.path.exists(action_data["before_screenshot"]) or not os.path.exists(action_data["after_screenshot"]):
            logger.error("截图文件不存在")
            return {
                "success": False,
                "error": "截图文件不存在",
                "analysis_time": get_iso_timestamp()
            }
        
        # 使用多模态分析工具
        analysis_tool = self.get_tool("multimodal_action_analysis")
        if analysis_tool:
            result = analysis_tool.execute(action_data=action_data)
        else:
            result = {"success": False, "error": "未找到分析工具"}
        
        logger.info(f"单个动作分析完成: {action_data.get('task_type', 'unknown')}")
        return result
    
    async def _analyze_execution_result(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """分析Executor执行结果 - ActionReflector核心功能
        
        输入：Executor的执行结果
        功能：执行质量评估、问题识别、改进建议
        输出：反思分析和改进建议
        """
        task_id = task_context.get("task_id")
        execution_result = task_context.get("execution_result", {})
        execution_details = execution_result.get("execution_details", {})
        before_screenshot = execution_details.get("before_screenshot")
        after_screenshot = execution_details.get("after_screenshot")
        action_info = execution_details.get("llm_action_plan", task_context.get("action_info", {}))
        expectation = task_context.get("expectation", "操作成功执行")
        
        logger.info(f"reflector分析任务ID: {task_id}")
        logger.info("上一步执行结果:"); print(execution_result)
        logger.info(f"动作执行「前」状态: {before_screenshot}")
        logger.info(f"动作执行「后」状态: {after_screenshot}")
        logger.info("动作执行细节:"); print(action_info)
        logger.info(f"预期结果: {expectation}")
        
        try:
            # 1. 执行质量评估
            logger.info(f"🔍 步骤1: 执行质量评估")
            quality_assessment = await self._assess_execution_quality(execution_result, action_info)
            
            # 2. 问题识别
            logger.info(f"🔍 步骤2: 问题识别")
            problem_identification = await self._identify_problems(execution_result, action_info)
            
            # 3. 多模态分析（如果有截图）
            multimodal_analysis = None
            if before_screenshot and after_screenshot:
                logger.info(f"🔍 步骤3: 多模态分析")
                multimodal_analysis = await self._perform_multimodal_analysis(
                    before_screenshot, after_screenshot, action_info, expectation
                )
            else:
                logger.warning(f"⚠️ 跳过多模态分析: 缺少截图文件")
            
            # 4. 改进建议生成
            logger.info(f"🔍 步骤4: 改进建议生成")
            improvement_suggestions = await self._generate_improvement_suggestions(
                quality_assessment, problem_identification, multimodal_analysis
            )
            
            # 5. 综合反思分析结果
            logger.info(f"🔍 步骤5: 综合评分计算")
            overall_score = self._calculate_overall_score(quality_assessment, problem_identification)
            
            # 检查是否存在逻辑问题：执行成功但多模态分析显示失败
            execution_success = execution_result.get("success", False)
            multimodal_success = multimodal_analysis.get("operation_success", True) if multimodal_analysis else True
            
            logger.info(f"成功性对比检查:")
            logger.info(f"  - 执行器报告成功: {execution_success}")
            logger.info(f"  - 多模态分析成功: {multimodal_success}")
            
            if execution_success and not multimodal_success:
                logger.warning(f"⚠️ 发现逻辑矛盾: 执行器报告成功但多模态分析显示失败")
                # 应该降低总体评分
                overall_score *= 0.3  # 大幅降低评分
                logger.info(f"矛盾惩罚后评分: {overall_score:.3f}")
            
            reflection_result = {
                "success": True,
                "task_id": task_id,
                "quality_assessment": quality_assessment,
                "problem_identification": problem_identification,
                "multimodal_analysis": multimodal_analysis,
                "improvement_suggestions": improvement_suggestions,
                "overall_score": overall_score,
                "reflection_time": get_iso_timestamp(),
                "analysis_method": "comprehensive_reflection",
                "execution_success": execution_success,
                "multimodal_success": multimodal_success,
                "has_logic_contradiction": execution_success and not multimodal_success
            }
            
            logger.info(f"✅ 执行结果分析完成，总体评分: {reflection_result['overall_score']:.3f}")
            logger.info(f"最终反思结果摘要:")
            logger.info(f"  - 质量评估: {quality_assessment.get('assessment')}")
            logger.info(f"  - 问题数量: {problem_identification.get('total_problems')}")
            logger.info(f"  - 多模态分析: {'成功' if multimodal_analysis and multimodal_analysis.get('success') else '失败/跳过'}")
            logger.info(f"  - 改进建议数: {len(improvement_suggestions)}")

            # 决定并发布下一步事件
            await self._decide_and_publish_next_step(reflection_result, task_context)
            
            return reflection_result
            
        except Exception as e:
            logger.error(f"❌ 执行结果分析失败: {e}")
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "reflection_time": get_iso_timestamp()
            }

    async def _decide_and_publish_next_step(self, reflection_result: Dict[str, Any], task_context: Dict[str, Any]):
        """
        根据反思分析决定下一步，并发布相应事件。
        """
        execution_success = reflection_result.get("execution_success", False)
        multimodal_success = reflection_result.get("multimodal_success", True)
        improvement_suggestions = reflection_result.get("improvement_suggestions", [])
        task_id = task_context.get("task_id")

        if not execution_success or not multimodal_success:
            logger.info("🤔 操作失败，决策下一步...")

            # 优先处理有明确修正建议的情况
            # 在真实场景中，这里的逻辑会更复杂，例如判断建议类型
            if improvement_suggestions and improvement_suggestions[0].get("type") == "correction":
                corrected_action = improvement_suggestions[0].get("action")
                reason = improvement_suggestions[0].get("reason")
                
                logger.info(f"✅ 发现可行的修正建议，发布 ActionCorrectionEvent")
                
                correction_event = ActionCorrectionEvent(
                    task_id=task_id,
                    original_action=task_context.get("action_info"),
                    corrected_action=corrected_action,
                    reason=reason,
                    reflection_result=reflection_result
                )
                await self.info_pool.publish_async(correction_event)

            else:
                # 如果没有明确的修正建议，或者问题比较严重，则请求重规划
                logger.warning("❌ 未找到简单的修正方法，发布 ReplanningRequiredEvent")
                
                replanning_event = ReplanningRequiredEvent(
                    task_id=task_id,
                    reason="操作失败，且没有直接的修正建议。",
                    failure_details=reflection_result
                )
                await self.info_pool.publish_async(replanning_event)
        else:
            logger.info("✅ 操作成功，无需进一步操作。")
    
    async def _assess_execution_quality(self, execution_result: Dict[str, Any], action_info: Dict[str, Any]) -> Dict[str, Any]:
        """执行质量评估"""
        # logger.info(f"🔍 开始执行质量评估...")
        # logger.info(f"原始execution_result: {execution_result}")
        # logger.info(f"action_info: {action_info}")
        
        # 从execution_result中提取实际的执行信息
        execution_details = execution_result.get('execution_details', {})
        # logger.info(f"execution_details: {execution_details}")
        
        # 处理AgentResult对象
        if hasattr(execution_details, 'success'):
            success = execution_details.success
            execution_time = getattr(execution_details, 'execution_time', 0)
            error_info = execution_details.error
            # logger.info(f"从AgentResult对象提取: success={success}, execution_time={execution_time}, error={error_info}")
        else:
            # 从字典中提取
            success = execution_result.get("success", execution_details.get("success", False))
            execution_time = execution_result.get("execution_time", execution_details.get("execution_time", 0))
            error_info = execution_result.get("error", execution_details.get("error"))
            # logger.info(f"从字典提取: success={success}, execution_time={execution_time}, error={error_info}")
        
        # 基础质量指标
        efficiency_score = self._calculate_efficiency_score_from_result(execution_result)
        reliability_score = self._calculate_reliability_score(execution_result)
        error_severity = self._assess_error_severity(error_info) if error_info else 0.0
        
        quality_metrics = {
            "success_rate": 1.0 if success else 0.0,
            "efficiency_score": efficiency_score,
            "reliability_score": reliability_score,
            "error_severity": error_severity
        }
        
        # logger.info(f"质量指标详情:")
        # logger.info(f"  - success_rate: {quality_metrics['success_rate']} (基于success={success})")
        # logger.info(f"  - efficiency_score: {quality_metrics['efficiency_score']}")
        # logger.info(f"  - reliability_score: {quality_metrics['reliability_score']}")
        # logger.info(f"  - error_severity: {quality_metrics['error_severity']}")
        
        # 综合质量评分
        overall_quality = (
            quality_metrics["success_rate"] * 0.4 +
            quality_metrics["efficiency_score"] * 0.3 +
            quality_metrics["reliability_score"] * 0.2 +
            (1.0 - quality_metrics["error_severity"]) * 0.1
        )
        
        assessment = "优秀" if overall_quality > 0.8 else "良好" if overall_quality > 0.6 else "需改进"
        
        # logger.info(f"综合质量评分计算:")
        # logger.info(f"  - 公式: {quality_metrics['success_rate']}*0.4 + {quality_metrics['efficiency_score']}*0.3 + {quality_metrics['reliability_score']}*0.2 + {1.0-quality_metrics['error_severity']}*0.1")
        # logger.info(f"  - 结果: {overall_quality:.3f} ({assessment})")
        
        result = {
            "overall_quality": overall_quality,
            "metrics": quality_metrics,
            "assessment": assessment
        }
        
        logger.info(f"✅ 执行质量评估完成: {result}")
        return result
    
    async def _identify_problems(self, execution_result: Dict[str, Any], action_info: Dict[str, Any]) -> Dict[str, Any]:
        """问题识别"""
        # logger.info(f"🔍 开始问题识别...")
        
        problems = []
        problem_categories = {
            "execution_errors": [],
            "performance_issues": [],
            "logic_problems": [],
            "environment_issues": []
        }
        
        # 从execution_result中提取实际的执行信息
        execution_details = execution_result.get('execution_details', {})
        
        # 处理AgentResult对象
        if hasattr(execution_details, 'success'):
            success = execution_details.success
            execution_time = getattr(execution_details, 'execution_time', 0)
            error_info = execution_details.error
            retry_count = getattr(execution_details, 'retry_count', 0)
            # logger.info(f"从AgentResult提取问题信息: success={success}, time={execution_time}, error={error_info}, retry={retry_count}")
        else:
            # 从字典中提取
            success = execution_result.get("success", execution_details.get("success", False))
            execution_time = execution_result.get("execution_time", execution_details.get("execution_time", 0))
            error_info = execution_result.get("error", execution_details.get("error"))
            retry_count = execution_result.get("retry_count", execution_details.get("retry_count", 0))
            # logger.info(f"从字典提取问题信息: success={success}, time={execution_time}, error={error_info}, retry={retry_count}")
        
        # 检查执行错误
        if not success:
            error_info = error_info or "未知错误"
            problem_categories["execution_errors"].append({
                "type": "execution_failure",
                "description": f"执行失败: {error_info}",
                "severity": "high"
            })
            logger.warning(f"❌ 发现执行错误: {error_info}")
        else:
            logger.info(f"✅ 执行成功，无执行错误")
        
        # 检查性能问题
        if execution_time is not None and execution_time > 10.0:  # 超过10秒认为是性能问题
            problem_categories["performance_issues"].append({
                "type": "slow_execution",
                "description": f"执行时间过长: {execution_time:.2f}秒",
                "severity": "medium"
            })
            logger.warning(f"⏱️ 发现性能问题: 执行时间{execution_time:.2f}秒")
        
        # 检查重试问题
        if retry_count is not None and retry_count > 0:
            problem_categories["logic_problems"].append({
                "type": "multiple_retries",
                "description": f"需要重试{retry_count}次才成功",
                "severity": "medium"
            })
            logger.warning(f"🔄 发现重试问题: 重试{retry_count}次")
        
        # 统计问题总数
        total_problems = sum(len(problems) for problems in problem_categories.values())
        has_critical = any(
            problem.get("severity") == "high" 
            for category in problem_categories.values() 
            for problem in category
        )
        
        result = {
            "total_problems": total_problems,
            "problem_categories": problem_categories,
            "has_critical_issues": has_critical
        }
        
        # logger.info(f"问题识别结果:")
        # logger.info(f"  - 总问题数: {total_problems}")
        # logger.info(f"  - 严重问题: {has_critical}")
        # logger.info(f"  - 问题分类: {problem_categories}")
        
        return result
    
    async def _perform_multimodal_analysis(self, before_screenshot: str, after_screenshot: str, 
                                         action_info: Dict[str, Any], expectation: str) -> Dict[str, Any]:
        """执行多模态分析"""
        logger.info(f"""🔍 开始多模态分析...
分析参数:
  - before_screenshot: {before_screenshot}
  - after_screenshot: {after_screenshot}
  - action_info: {action_info}
  - expectation: {expectation}""")
        
        try:
            # 使用多模态分析工具
            analysis_tool = self.get_tool("multimodal_action_analysis")
            if analysis_tool:
                action_data = {
                    "before_screenshot": before_screenshot,
                    "after_screenshot": after_screenshot,
                    "action": action_info,
                    "expectation": expectation,
                    "task_type": action_info.get("action", "unknown")
                }
                logger.info(f"调用多模态分析工具，action_data: {action_data}")
                result = await analysis_tool.aexecute(action_data=action_data)
                
                logger.info(f"""多模态分析结果:
  - success: {result.get('success')}
  - operation_success: {result.get('operation_success')}
  - outcome: {result.get('outcome')}
  - comparison_analysis: {result.get('comparison_analysis', '')[:100]}...
  - improvement_suggestions: {result.get('improvement_suggestions', '')[:100]}...""")
                
                return result
            else:
                logger.error(f"❌ 多模态分析工具不可用")
                return {"success": False, "error": "多模态分析工具不可用"}
        except Exception as e:
            logger.error(f"❌ 多模态分析失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_improvement_suggestions(self, quality_assessment: Dict[str, Any], 
                                              problem_identification: Dict[str, Any],
                                              multimodal_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 基于质量评估的建议
        overall_quality = quality_assessment.get("overall_quality", 0.0)
        if overall_quality < 0.6:
            suggestions.append("整体执行质量偏低，建议优化执行策略")
        
        metrics = quality_assessment.get("metrics", {})
        if metrics.get("efficiency_score", 1.0) < 0.7:
            suggestions.append("执行效率有待提升，考虑优化算法或减少不必要的步骤")
        
        # 基于问题识别的建议
        problem_categories = problem_identification.get("problem_categories", {})
        
        if problem_categories.get("execution_errors"):
            suggestions.append("存在执行错误，建议增强错误处理和异常恢复机制")
        
        if problem_categories.get("performance_issues"):
            suggestions.append("存在性能问题，建议优化执行流程或增加超时处理")
        
        if problem_categories.get("logic_problems"):
            suggestions.append("存在逻辑问题，建议检查任务分解和执行顺序")
        
        # 基于多模态分析的建议
        if multimodal_analysis and multimodal_analysis.get("success"):
            if not multimodal_analysis.get("operation_success", True):
                suggestions.append("多模态分析显示操作未达到预期效果，建议检查操作精度")
            
            improvement_suggestions = multimodal_analysis.get("improvement_suggestions", "")
            if improvement_suggestions:
                suggestions.append(f"多模态分析建议: {improvement_suggestions}")
        
        # 如果没有具体建议，提供通用建议
        if not suggestions:
            suggestions.append("执行表现良好，建议继续保持当前策略")
        
        return suggestions
    
    async def _send_improvement_feedback_to_executor(self, reflection_result: Dict[str, Any], task_context: Dict[str, Any]) -> None:
        """向ExecutorAgent发送具体的改进反馈"""
        try:
            # 分析反思结果，生成具体的改进建议
            improvement_suggestions = reflection_result.get("improvement_suggestions", "")
            multimodal_analysis = reflection_result.get("multimodal_analysis", {})
            
            # 提取坐标相关的改进建议
            coordinate_feedback = self._extract_coordinate_feedback(multimodal_analysis, task_context)
            if coordinate_feedback:
                coordinate_event = Event(
                    type="execution_improvement_suggestion",
                    data={
                        "type": "coordinate_adjustment",
                        "content": "基于多模态分析的坐标调整建议",
                        "coordinates": coordinate_feedback["original_coordinates"],
                        "adjustment": coordinate_feedback["suggested_adjustment"],
                        "reason": coordinate_feedback["reason"],
                        "confidence": coordinate_feedback.get("confidence", 0.7)
                    },
                    agent_id=self.config.id
                )
                await self.info_pool.publish_async(coordinate_event)
                logger.info(f"📤 发送坐标调整建议: {coordinate_feedback['suggested_adjustment']}")
            
            # 提取执行策略相关的改进建议
            strategy_feedback = self._extract_strategy_feedback(reflection_result, task_context)
            if strategy_feedback:
                strategy_event = Event(
                    type="execution_improvement_suggestion",
                    data={
                        "type": "execution_strategy",
                        "content": "基于反思分析的执行策略优化",
                        "task_type": strategy_feedback["task_type"],
                        "strategy": strategy_feedback["strategy"],
                        "reason": strategy_feedback["reason"]
                    },
                    agent_id=self.config.id
                )
                await self.info_pool.publish_async(strategy_event)
                logger.info(f"📤 发送策略优化建议: {strategy_feedback['task_type']}")
            
            # 发送通用改进建议
            if improvement_suggestions:
                general_event = Event(
                    type="execution_improvement_suggestion",
                    data={
                        "type": "general",
                        "content": improvement_suggestions,
                        "reflection_result": reflection_result,
                        "task_context": task_context
                    },
                    agent_id=self.config.id
                )
                await self.info_pool.publish_async(general_event)
                logger.info(f"📤 发送通用改进建议: {improvement_suggestions[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ 发送改进反馈失败: {e}")
    
    def _extract_coordinate_feedback(self, multimodal_analysis: Dict[str, Any], task_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从多模态分析中提取坐标反馈 - 增强版本，支持紫色点标注分析"""
        try:
            # 检查是否有坐标相关的分析
            analysis_text = str(multimodal_analysis)
            action_info = task_context.get("action_info", {})
            
            if "coordinates" not in action_info:
                return None
            
            original_coords = [action_info["coordinates"]["x"], action_info["coordinates"]["y"]]
            
            # 增强的坐标分析 - 基于紫色点标注和视觉分析
            coordinate_feedback = self._analyze_coordinate_precision(
                multimodal_analysis, original_coords, task_context
            )
            
            if coordinate_feedback:
                return coordinate_feedback
            
            # 回退到文本分析
            if any(keyword in analysis_text for keyword in ["坐标", "位置", "偏移", "偏上", "偏下", "偏左", "偏右", "紫色点", "标注"]):
                # 基于分析内容推断调整方向
                if "偏上" in analysis_text or "太高" in analysis_text or "上方" in analysis_text:
                    adjustment = [0, 15]  # 向下调整
                    reason = "点击位置偏上，建议向下调整"
                elif "偏下" in analysis_text or "太低" in analysis_text or "下方" in analysis_text:
                    adjustment = [0, -15]  # 向上调整
                    reason = "点击位置偏下，建议向上调整"
                elif "偏左" in analysis_text or "左侧" in analysis_text:
                    adjustment = [15, 0]  # 向右调整
                    reason = "点击位置偏左，建议向右调整"
                elif "偏右" in analysis_text or "右侧" in analysis_text:
                    adjustment = [-15, 0]  # 向左调整
                    reason = "点击位置偏右，建议向左调整"
                elif "不准确" in analysis_text or "偏差" in analysis_text:
                    adjustment = [0, 0]  # 需要更精确的分析
                    reason = "坐标存在偏差，建议重新校准"
                else:
                    return None
                
                return {
                    "original_coordinates": original_coords,
                    "suggested_adjustment": adjustment,
                    "reason": reason,
                    "confidence": 0.8
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"提取坐标反馈失败: {e}")
            return None
    
    def _analyze_coordinate_precision(self, multimodal_analysis: Dict[str, Any], 
                                    original_coords: List[int], task_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分析坐标精度 - 基于紫色点标注和多模态分析"""
        try:
            # 从多模态分析结果中提取坐标精度信息
            analysis_text = multimodal_analysis.get("comparison_analysis", "")
            improvement_suggestions = multimodal_analysis.get("improvement_suggestions", "")
            full_response = multimodal_analysis.get("full_response", "")
            
            # 合并所有分析文本
            combined_text = f"{analysis_text} {improvement_suggestions} {full_response}"
            
            logger.info(f"🔍 分析坐标精度，原始坐标: {original_coords}")
            logger.info(f"📝 分析文本: {combined_text[:200]}...")
            
            # 检查是否有紫色点标注相关的描述
            if any(keyword in combined_text for keyword in ["紫色点", "标注", "点击位置", "偏移", "中心"]):
                return self._extract_precise_coordinate_adjustment(combined_text, original_coords)
            
            # 检查是否有具体的像素调整建议
            pixel_adjustment = self._extract_pixel_adjustment_from_text(combined_text)
            if pixel_adjustment:
                return {
                    "original_coordinates": original_coords,
                    "suggested_adjustment": pixel_adjustment["adjustment"],
                    "reason": pixel_adjustment["reason"],
                    "confidence": pixel_adjustment["confidence"],
                    "analysis_method": "text_pixel_extraction"
                }
            
            # 检查是否有方向性调整建议
            directional_adjustment = self._extract_directional_adjustment(combined_text)
            if directional_adjustment:
                return {
                    "original_coordinates": original_coords,
                    "suggested_adjustment": directional_adjustment["adjustment"],
                    "reason": directional_adjustment["reason"],
                    "confidence": directional_adjustment["confidence"],
                    "analysis_method": "directional_analysis"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 坐标精度分析失败: {e}")
            return None
    
    def _extract_precise_coordinate_adjustment(self, analysis_text: str, original_coords: List[int]) -> Optional[Dict[str, Any]]:
        """从分析文本中提取精确的坐标调整"""
        import re
        
        # 查找具体的像素调整建议 - 优化版本，避免重复匹配
        adjustment_x = 0
        adjustment_y = 0
        reasons = []
        processed_matches = set()  # 避免重复处理
        
        # 水平方向调整
        horizontal_patterns = [
            r'向左调整(\d+)像素',
            r'向右调整(\d+)像素',
            r'左移(\d+)像素',
            r'右移(\d+)像素',
            r'偏左(\d+)像素',
            r'偏右(\d+)像素'
        ]
        
        for i, pattern in enumerate(horizontal_patterns):
            matches = re.findall(pattern, analysis_text)
            for match in matches:
                pixels = int(match)
                match_key = f"h_{i}_{pixels}"
                if match_key not in processed_matches:
                    processed_matches.add(match_key)
                    
                    if '左' in pattern:
                        adjustment_x -= pixels
                        reasons.append(f"向左调整{pixels}像素")
                    else:
                        adjustment_x += pixels
                        reasons.append(f"向右调整{pixels}像素")
        
        # 垂直方向调整
        vertical_patterns = [
            r'向上调整(\d+)像素',
            r'向下调整(\d+)像素',
            r'上移(\d+)像素',
            r'下移(\d+)像素',
            r'偏上(\d+)像素',
            r'偏下(\d+)像素'
        ]
        
        for i, pattern in enumerate(vertical_patterns):
            matches = re.findall(pattern, analysis_text)
            for match in matches:
                pixels = int(match)
                match_key = f"v_{i}_{pixels}"
                if match_key not in processed_matches:
                    processed_matches.add(match_key)
                    
                    if '上' in pattern:
                        adjustment_y -= pixels
                        reasons.append(f"向上调整{pixels}像素")
                    else:
                        adjustment_y += pixels
                        reasons.append(f"向下调整{pixels}像素")
        
        if adjustment_x != 0 or adjustment_y != 0:
            return {
                "original_coordinates": original_coords,
                "suggested_adjustment": [adjustment_x, adjustment_y],
                "reason": "基于多模态分析的精确调整: " + ", ".join(set(reasons)),  # 去重
                "confidence": 0.9,
                "analysis_method": "precise_pixel_extraction"
            }
        
        return None
    
    def _extract_pixel_adjustment_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取像素级调整建议"""
        import re
        
        # 更复杂的像素调整模式
        patterns = [
            r'调整坐标.*?([+-]?\d+).*?([+-]?\d+)',
            r'偏移.*?([+-]?\d+).*?([+-]?\d+)',
            r'移动.*?([+-]?\d+).*?([+-]?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    x_adj = int(match.group(1))
                    y_adj = int(match.group(2))
                    
                    return {
                        "adjustment": [x_adj, y_adj],
                        "reason": f"基于文本分析的坐标调整: X{x_adj:+d}, Y{y_adj:+d}",
                        "confidence": 0.7
                    }
                except ValueError:
                    continue
        
        return None
    
    def _extract_directional_adjustment(self, text: str) -> Optional[Dict[str, Any]]:
        """提取方向性调整建议"""
        # 默认调整像素数
        default_pixels = 15
        
        adjustment_x = 0
        adjustment_y = 0
        reasons = []
        
        # 检查方向性描述
        if any(word in text for word in ['偏左', '太左', '左侧']):
            adjustment_x = default_pixels
            reasons.append("向右调整")
        elif any(word in text for word in ['偏右', '太右', '右侧']):
            adjustment_x = -default_pixels
            reasons.append("向左调整")
        
        if any(word in text for word in ['偏上', '太高', '上方']):
            adjustment_y = default_pixels
            reasons.append("向下调整")
        elif any(word in text for word in ['偏下', '太低', '下方']):
            adjustment_y = -default_pixels
            reasons.append("向上调整")
        
        if adjustment_x != 0 or adjustment_y != 0:
            return {
                "adjustment": [adjustment_x, adjustment_y],
                "reason": "基于方向性分析的调整: " + ", ".join(reasons),
                "confidence": 0.6
            }
        
        return None
    
    def _extract_coordinate_feedback_from_analysis(self, coordinate_analysis: str, 
                                                 comparison_analysis: str, 
                                                 improvement_suggestions: str) -> Optional[Dict[str, Any]]:
        """从分析文本中提取坐标反馈信息"""
        try:
            # 合并所有相关文本
            combined_text = f"{coordinate_analysis} {comparison_analysis} {improvement_suggestions}"
            
            if not combined_text.strip():
                return None
            
            logger.info(f"🔍 从分析文本中提取坐标反馈")
            logger.info(f"📝 分析文本: {combined_text[:150]}...")
            
            # 尝试提取精确的像素调整
            precise_adjustment = self._extract_precise_coordinate_adjustment(combined_text, [0, 0])
            if precise_adjustment:
                logger.info(f"✅ 提取到精确调整: {precise_adjustment['suggested_adjustment']}")
                return precise_adjustment
            
            # 尝试提取像素级调整
            pixel_adjustment = self._extract_pixel_adjustment_from_text(combined_text)
            if pixel_adjustment:
                logger.info(f"✅ 提取到像素调整: {pixel_adjustment['adjustment']}")
                return {
                    "original_coordinates": [0, 0],  # 占位符
                    "suggested_adjustment": pixel_adjustment["adjustment"],
                    "reason": pixel_adjustment["reason"],
                    "confidence": pixel_adjustment["confidence"],
                    "analysis_method": "coordinate_analysis_extraction"
                }
            
            # 尝试提取方向性调整
            directional_adjustment = self._extract_directional_adjustment(combined_text)
            if directional_adjustment:
                logger.info(f"✅ 提取到方向调整: {directional_adjustment['adjustment']}")
                return {
                    "original_coordinates": [0, 0],  # 占位符
                    "suggested_adjustment": directional_adjustment["adjustment"],
                    "reason": directional_adjustment["reason"],
                    "confidence": directional_adjustment["confidence"],
                    "analysis_method": "coordinate_analysis_directional"
                }
            
            # 检查是否有紫色点相关的描述
            if any(keyword in combined_text for keyword in ["紫色点", "标注", "点击位置", "目标中心"]):
                # 基于紫色点描述生成调整建议
                purple_dot_feedback = self._analyze_purple_dot_feedback(combined_text)
                if purple_dot_feedback:
                    logger.info(f"✅ 基于紫色点分析: {purple_dot_feedback['adjustment']}")
                    return {
                        "original_coordinates": [0, 0],  # 占位符
                        "suggested_adjustment": purple_dot_feedback["adjustment"],
                        "reason": purple_dot_feedback["reason"],
                        "confidence": purple_dot_feedback["confidence"],
                        "analysis_method": "purple_dot_analysis"
                    }
            
            logger.info(f"ℹ️ 未能从分析文本中提取坐标反馈")
            return None
            
        except Exception as e:
            logger.error(f"❌ 提取坐标反馈失败: {e}")
            return None
    
    def _analyze_purple_dot_feedback(self, text: str) -> Optional[Dict[str, Any]]:
        """分析紫色点相关的反馈"""
        import re
        
        # 查找紫色点相关的描述
        purple_patterns = [
            r'紫色点.*?([偏离|距离|远离]).*?([上下左右]).*?(\d+)',
            r'点击位置.*?([偏离|距离]).*?目标.*?([上下左右]).*?(\d+)',
            r'标注.*?([偏离|距离]).*?中心.*?([上下左右]).*?(\d+)'
        ]
        
        for pattern in purple_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    direction = match.group(2)
                    distance = int(match.group(3))
                    
                    # 根据方向计算调整
                    adjustment_x = 0
                    adjustment_y = 0
                    
                    if direction == '左':
                        adjustment_x = distance
                    elif direction == '右':
                        adjustment_x = -distance
                    elif direction == '上':
                        adjustment_y = distance
                    elif direction == '下':
                        adjustment_y = -distance
                    
                    return {
                        "adjustment": [adjustment_x, adjustment_y],
                        "reason": f"基于紫色点标注分析，点击位置偏{direction}{distance}像素",
                        "confidence": 0.85
                    }
                except (ValueError, IndexError):
                    continue
        
        # 如果没有找到具体数值，使用默认调整
        if "紫色点" in text or "标注" in text:
            if any(word in text for word in ['偏左', '左侧']):
                return {"adjustment": [20, 0], "reason": "紫色点显示偏左", "confidence": 0.6}
            elif any(word in text for word in ['偏右', '右侧']):
                return {"adjustment": [-20, 0], "reason": "紫色点显示偏右", "confidence": 0.6}
            elif any(word in text for word in ['偏上', '上方']):
                return {"adjustment": [0, 20], "reason": "紫色点显示偏上", "confidence": 0.6}
            elif any(word in text for word in ['偏下', '下方']):
                return {"adjustment": [0, -20], "reason": "紫色点显示偏下", "confidence": 0.6}
        
        return None
    
    def _extract_strategy_feedback(self, reflection_result: Dict[str, Any], task_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从反思结果中提取策略反馈"""
        try:
            action_info = task_context.get("action_info", {})
            task_type = action_info.get("task_type", "unknown")
            
            # 基于反思结果生成策略建议
            quality_assessment = reflection_result.get("quality_assessment", {})
            problem_identification = reflection_result.get("problem_identification", {})
            
            strategy = {}
            reason_parts = []
            
            # 基于效率问题调整策略
            efficiency_score = quality_assessment.get("metrics", {}).get("efficiency_score", 1.0)
            if efficiency_score < 0.5:
                strategy["timeout"] = 15.0  # 增加超时时间
                strategy["retry_delay"] = 2.0  # 增加重试延迟
                reason_parts.append("效率偏低")
            
            # 基于错误类型调整策略
            problem_categories = problem_identification.get("problem_categories", {})
            if problem_categories.get("performance_issues"):
                strategy["pre_wait"] = 1.0  # 操作前等待
                reason_parts.append("性能问题")
            
            if problem_categories.get("execution_errors"):
                strategy["verification_required"] = True  # 需要验证
                reason_parts.append("执行错误")
            
            if strategy:
                return {
                    "task_type": task_type,
                    "strategy": strategy,
                    "reason": f"基于{', '.join(reason_parts)}的策略优化"
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"提取策略反馈失败: {e}")
            return None
    
    def _calculate_efficiency_score_from_result(self, execution_result: Dict[str, Any]) -> float:
        """从执行结果计算效率分数"""
        # 从execution_result中提取实际的执行信息
        execution_details = execution_result.get('execution_details', {})
        
        # 处理AgentResult对象
        if hasattr(execution_details, 'execution_time'):
            execution_time = getattr(execution_details, 'execution_time', 1.0)
            retry_count = getattr(execution_details, 'retry_count', 0)
        else:
            # 从字典中提取
            execution_time = execution_result.get("execution_time", execution_details.get("execution_time", 1.0))
            retry_count = execution_result.get("retry_count", execution_details.get("retry_count", 0))
        
        # 基础效率分数
        base_score = 1.0
        
        # 根据执行时间调整
        if execution_time is not None:
            if execution_time > 10.0:
                base_score *= 0.5
            elif execution_time > 5.0:
                base_score *= 0.7
            elif execution_time > 2.0:
                base_score *= 0.9
        
        # 根据重试次数调整
        if retry_count is not None:
            base_score *= (1.0 - retry_count * 0.2)
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_reliability_score(self, execution_result: Dict[str, Any]) -> float:
        """计算可靠性分数"""
        # 从execution_result中提取实际的执行信息
        execution_details = execution_result.get('execution_details', {})
        
        # 处理AgentResult对象
        if hasattr(execution_details, 'success'):
            success = execution_details.success
            error_info = execution_details.error
        else:
            # 从字典中提取
            success = execution_result.get("success", execution_details.get("success", False))
            error_info = execution_result.get("error", execution_details.get("error"))
        
        if success:
            return 1.0
        elif error_info:
            # 根据错误类型调整可靠性分数
            error_str = str(error_info).lower()
            if "timeout" in error_str:
                return 0.3
            elif "permission" in error_str:
                return 0.2
            elif "network" in error_str:
                return 0.4
            else:
                return 0.1
        else:
            return 0.0
    
    def _assess_error_severity(self, error_info: str) -> float:
        """评估错误严重程度 (0.0-1.0, 1.0为最严重)"""
        error_str = str(error_info).lower()
        
        if any(keyword in error_str for keyword in ["crash", "fatal", "critical"]):
            return 1.0
        elif any(keyword in error_str for keyword in ["error", "failed", "exception"]):
            return 0.7
        elif any(keyword in error_str for keyword in ["warning", "timeout"]):
            return 0.4
        else:
            return 0.2
    
    def _calculate_overall_score(self, quality_assessment: Dict[str, Any], 
                               problem_identification: Dict[str, Any]) -> float:
        """计算总体评分"""
        logger.info(f"🔍 开始计算总体评分...")
        
        quality_score = quality_assessment.get("overall_quality", 0.0)
        problem_count = problem_identification.get("total_problems", 0)
        has_critical = problem_identification.get("has_critical_issues", False)
        
        logger.info(f"评分计算输入:")
        logger.info(f"  - 质量分数: {quality_score}")
        logger.info(f"  - 问题数量: {problem_count}")
        logger.info(f"  - 严重问题: {has_critical}")
        
        # 基础分数来自质量评估
        overall_score = quality_score
        logger.info(f"基础分数: {overall_score}")
        
        # 根据问题数量调整
        problem_penalty = problem_count * 0.1
        overall_score *= (1.0 - problem_penalty)
        logger.info(f"问题调整后: {overall_score} (问题惩罚: {problem_penalty})")
        
        # 如果有严重问题，大幅降低分数
        if has_critical:
            overall_score *= 0.5
            logger.warning(f"严重问题惩罚后: {overall_score}")
        
        final_score = max(0.0, min(1.0, overall_score))
        
        logger.info(f"最终评分: {final_score:.3f}")
        logger.info(f"评分等级: {'优秀' if final_score > 0.8 else '良好' if final_score > 0.6 else '需改进'}")
        
        return final_score
    
    async def _comprehensive_multimodal_analysis(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """综合多模态分析 - 结合多种分析方法"""
        # 执行多模态反思分析
        multimodal_result = await self._multimodal_action_reflection(task_context)
        
        # 获取参数
        action_history = task_context.get("action_history", [])
        analysis_results = task_context.get("analysis_results", [])
        
        # 执行性能分析
        performance_result = {"success": False, "error": "未执行性能分析"}
        performance_tool = self.get_tool("performance_analysis")
        if performance_tool:
            result = performance_tool.execute(action_history=action_history)
        else:
            result = {"success": False, "error": "未找到性能分析工具"}
        
        # 更新性能指标
        self.performance_metrics = result
        
        # 生成学习洞察
        insight_result = {"success": False, "error": "未执行学习洞察"}
        insight_tool = self.get_tool("learning_insight")
        if insight_tool:
            result = insight_tool.execute(analysis_results=analysis_results)
        else:
            result = {"success": False, "error": "未找到学习洞察工具"}
        
        # 更新学习洞察
        self.learning_insights = result
        
        logger.info(f"学习洞察生成完成: {len(analysis_results)}个分析结果")
        return result
    
    def _generate_comprehensive_summary(
        self, 
        multimodal_result: Dict[str, Any], 
        performance_result: Dict[str, Any], 
        insight_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成综合分析摘要"""
        return {
            "operation_success": multimodal_result.get("operation_success", False),
            "outcome_category": multimodal_result.get("outcome", "unknown"),
            "model_used": multimodal_result.get("model_used", "unknown"),
            "analysis_method": "multimodal_llm_reflection",
            "performance_score": performance_result.get("success_rate", 0.0),
            "insights_generated": len(insight_result.get("key_learnings", [])),
            "recommendations_count": len(multimodal_result.get("improvement_suggestions", "")),
            "analysis_quality": "high" if multimodal_result.get("success") else "limited"
        }
    
    async def _analyze_single_action(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """分析单个动作"""
        action_data = task_context.get("action_data", {})
        
        analysis_tool = self.get_tool("action_analysis")
        if analysis_tool:
            result = analysis_tool.execute(action_data=action_data)
        else:
            result = {"success": False, "error": "未找到分析工具"}
        
        logger.info(f"单个动作分析完成: {action_data.get('task_type', 'unknown')}")
        return result
    
    async def _analyze_performance(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """分析整体性能"""
        action_history = task_context.get("action_history", [])
        
        performance_tool = self.get_tool("performance_analysis")
        if performance_tool:
            result = performance_tool.execute(action_history=action_history)
        else:
            result = {"success": False, "error": "未找到性能分析工具"}
        
        # 更新性能指标
        self.performance_metrics = result
        
        logger.info(f"性能分析完成: {len(action_history)}个动作")
        return result
    
    async def _generate_learning_insights(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """生成学习洞察"""
        analysis_results = task_context.get("analysis_results", self.reflection_history)
        
        insight_tool = self.get_tool("learning_insight")
        if insight_tool:
            result = insight_tool.execute(analysis_results=analysis_results)
        else:
            result = {"success": False, "error": "未找到学习洞察工具"}
        
        # 更新学习洞察
        self.learning_insights = result
        
        logger.info(f"学习洞察生成完成: {len(analysis_results)}个分析结果")
        return result
    
    async def _comprehensive_analysis(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """综合分析"""
        action_history = task_context.get("action_history", [])
        
        # 执行性能分析
        performance_result = await self._analyze_performance({"action_history": action_history})
        
        # 分析每个动作
        action_analyses = []
        for action in action_history[-10:]:  # 只分析最近10个动作
            try:
                analysis = await self._analyze_single_action({"action_data": action})
                action_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"动作分析失败: {e}")
        
        # 生成学习洞察
        insight_result = await self._generate_learning_insights({"analysis_results": action_analyses})
        
        comprehensive_result = {
            "performance_analysis": performance_result,
            "action_analyses": action_analyses,
            "learning_insights": insight_result,
            "summary": self._generate_analysis_summary(performance_result, insight_result),
            "analysis_time": get_iso_timestamp()
        }
        
        logger.info("综合分析完成")
        return comprehensive_result
    
    def _generate_analysis_summary(self, performance_result: Dict[str, Any], insight_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析摘要"""
        return {
            "total_actions_analyzed": performance_result.get("total_actions", 0),
            "overall_success_rate": performance_result.get("success_rate", 0.0),
            "overall_efficiency": performance_result.get("average_efficiency", 0.0),
            "key_insights_count": len(insight_result.get("key_learnings", [])),
            "improvement_suggestions_count": len(insight_result.get("improvement_opportunities", [])),
            "analysis_quality": "good" if performance_result.get("success_rate", 0) > 0.8 else "needs_improvement"
        }
    
    def _handle_action_result(self, info_entry) -> None:
        """处理动作结果信息 - 触发多模态反思分析"""
        try:
            logger.info(f"🔍 [ACTION_RESULT_HANDLER] Received info_entry: {info_entry}")
            if hasattr(info_entry, 'data'):
                logger.info(f"🔍 [ACTION_RESULT_HANDLER] info_entry.data: {info_entry.data}")
            else:
                logger.warning(f"🔍 [ACTION_RESULT_HANDLER] info_entry has no 'data' attribute.")
            
            action_record = info_entry.data.get("action_record", {})
            
            logger.info(f"📨 收到动作结果: {action_record.get('task_type', 'unknown')}")
            
            # 如果有操作前后截图，触发多模态反思分析
            if self._should_trigger_reflection(action_record):
                asyncio.create_task(self._trigger_reflection_analysis(action_record))
            
        except Exception as e:
            logger.error(f"❌ 处理动作结果失败: {e}", exc_info=True)
    
    def _handle_screenshot_taken(self, info_entry) -> None:
        """处理截图事件"""
        try:
            screenshot_data = info_entry.data
            screenshot_path = screenshot_data.get("screenshot_path")
            screenshot_type = screenshot_data.get("type", "unknown")  # before/after
            
            logger.info(f"📸 收到截图事件: {screenshot_type} - {screenshot_path}")
            
            # 存储截图信息用于后续分析
            if screenshot_type in ["before", "after"]:
                self._store_screenshot_info(screenshot_path, screenshot_type)
            
        except Exception as e:
            logger.error(f"❌ 处理截图事件失败: {e}")
    
    def _handle_operation_completed(self, info_entry) -> None:
        """处理操作完成事件 - 自动触发反思分析"""
        try:
            operation_data = info_entry.data
            
            logger.info(f"✅ 收到操作完成事件: {operation_data.get('operation_type', 'unknown')}")
            
            # 自动触发反思分析
            asyncio.create_task(self._auto_reflection_analysis(operation_data))
            
        except Exception as e:
            logger.error(f"❌ 处理操作完成事件失败: {e}")
    
    def _should_trigger_reflection(self, action_record: Dict[str, Any]) -> bool:
        """判断是否应该触发反思分析"""
        # 检查是否有必要的信息
        task_type = action_record.get("task_type")
        
        # 对于这些操作类型，需要进行反思分析
        reflection_worthy_actions = [
            "click_action", "input_text", "swipe_action", 
            "long_press_action", "open_app_action"
        ]
        
        return task_type in reflection_worthy_actions
    
    def _store_screenshot_info(self, screenshot_path: str, screenshot_type: str) -> None:
        """存储截图信息"""
        # 简化的截图存储逻辑
        # 实际应用中可能需要更复杂的配对逻辑
        if not hasattr(self, '_temp_screenshots'):
            self._temp_screenshots = {}
        
        self._temp_screenshots[screenshot_type] = screenshot_path
        
        # 如果有操作前后截图对，存储到历史中
        if 'before' in self._temp_screenshots and 'after' in self._temp_screenshots:
            self.screenshot_pairs.append((
                self._temp_screenshots['before'],
                self._temp_screenshots['after']
            ))
            # 保持合理的历史记录数量
            if len(self.screenshot_pairs) > 50:
                self.screenshot_pairs = self.screenshot_pairs[-50:]
            
            # 清空临时存储
            self._temp_screenshots = {}
    
    async def _trigger_reflection_analysis(self, action_record: Dict[str, Any]) -> None:
        """触发反思分析"""
        try:
            # 构建分析上下文
            task_context = {
                "analysis_type": "multimodal_reflection",
                "action_info": action_record,
                "expectation": action_record.get("expectation", "操作成功完成"),
            }
            
            # 从 action_record 中提取截图信息
            action_result = action_record.get("result", {})
            before_screenshot = action_result.get("before_screenshot")
            after_screenshot = action_result.get("after_screenshot")

            # 如果在 action_record 中找到截图，则使用它们
            if before_screenshot and after_screenshot:
                logger.info(f"从 action_record 中找到截图: before='{before_screenshot}', after='{after_screenshot}'")
                task_context.update({
                    "before_screenshot": before_screenshot,
                    "after_screenshot": after_screenshot
                })
            # 回退到使用 screenshot_pairs
            elif self.screenshot_pairs:
                logger.warning("在 action_record 中未找到截图，回退到使用 screenshot_pairs")
                before_screenshot, after_screenshot = self.screenshot_pairs[-1]
                task_context.update({
                    "before_screenshot": before_screenshot,
                    "after_screenshot": after_screenshot
                })
            
            # 执行反思分析
            result = await self._execute_task_impl(task_context)
            
            logger.info(f"🔍 自动反思分析完成: {result.get('operation_success', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ 触发反思分析失败: {e}")
    
    async def _auto_reflection_analysis(self, operation_data: Dict[str, Any]) -> None:
        """自动反思分析"""
        try:
            # 构建分析上下文
            task_context = {
                "analysis_type": "multimodal_reflection",
                "action_info": operation_data,
                "expectation": operation_data.get("expected_outcome", "操作按预期完成"),
                "before_screenshot": operation_data.get("before_screenshot"),
                "after_screenshot": operation_data.get("after_screenshot")
            }
            
            # 执行反思分析
            result = await self._execute_task_impl(task_context)
            
            logger.info(f"🤖 自动反思分析完成: 操作{'成功' if result.get('operation_success') else '失败'}")
            
        except Exception as e:
            logger.error(f"❌ 自动反思分析失败: {e}")
    
    def get_reflection_history(self) -> List[Dict[str, Any]]:
        """获取反思分析历史"""
        return self.reflection_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        return self.learning_insights.copy()
    
    def get_screenshot_pairs(self) -> List[Tuple[str, str]]:
        """获取截图对历史"""
        return self.screenshot_pairs.copy()
    
    def clear_reflection_history(self) -> None:
        """清空反思历史"""
        self.reflection_history.clear()
        self.performance_metrics.clear()
        self.learning_insights.clear()
        self.screenshot_pairs.clear()
        if hasattr(self, '_temp_screenshots'):
            self._temp_screenshots.clear()
        logger.info("🧹 反思分析历史已清空")
    
    async def manual_reflection_analysis(
        self, 
        before_screenshot: str, 
        after_screenshot: str, 
        action_info: Dict[str, Any], 
        expectation: str = ""
    ) -> Dict[str, Any]:
        """手动触发反思分析 - 便捷方法
        
        Args:
            before_screenshot: 操作前截图路径
            after_screenshot: 操作后截图路径
            action_info: 动作信息
            expectation: 期望结果
        
        Returns:
            反思分析结果
        """
        task_context = {
            "analysis_type": "multimodal_reflection",
            "before_screenshot": before_screenshot,
            "after_screenshot": after_screenshot,
            "action_info": action_info,
            "expectation": expectation
        }
        
        return await self._execute_task_impl(task_context)
    
    def get_recent_reflections(self, count: int = 5) -> List[Dict[str, Any]]:
        """获取最近的反思分析结果"""
        return self.reflection_history[-count:] if self.reflection_history else []
    
    def get_success_rate(self) -> float:
        """获取操作成功率"""
        if not self.reflection_history:
            return 0.0
        
        successful_operations = sum(
            1 for reflection in self.reflection_history 
            if reflection.get("result", {}).get("operation_success", False)
        )
        
        return successful_operations / len(self.reflection_history)
    
    def get_model_usage_stats(self) -> Dict[str, int]:
        """获取模型使用统计"""
        model_stats = defaultdict(int)
        
        for reflection in self.reflection_history:
            model_used = reflection.get("model_used", "unknown")
            model_stats[model_used] += 1
        
        return dict(model_stats)