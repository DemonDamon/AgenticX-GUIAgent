#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体协作工作流模块

实现四智能体协作的工作流编排，基于AgenticX框架构建。
协调Manager、Executor、ActionReflector、Notetaker的协作执行。
"""

import asyncio
import json
from rich import print
from rich.json import JSON
from loguru import logger
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from agenticx.core.workflow import Workflow
from agenticx.core.component import Component
from agenticx.core.task import Task

from core.info_pool import InfoPool, InfoType, InfoPriority


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    RECORDING = "recording"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentCoordinator(Component):
    """
    智能体协调器
    
    基于AgenticX框架的四智能体协作协调器，
    实现Manager、Executor、ActionReflector、Notetaker的协作流程。
    """
    
    def __init__(self, agents: Dict[str, Any], info_pool: InfoPool):
        """
        初始化协调器
        
        Args:
            agents: 智能体字典 {agent_name: agent_instance}
            info_pool: 信息池实例
        """
        super().__init__()
        self.agents = agents
        self.info_pool = info_pool
        self.current_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_counter = 0
        
    async def execute_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        执行移动GUI任务
        
        Args:
            task_description: 任务描述
            **kwargs: 额外参数
            
        Returns:
            任务执行结果
        """
        # 生成任务ID
        self._task_counter += 1
        task_id = f"task_{self._task_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化任务状态
        task_info = {
            "id": task_id,
            "description": task_description,
            "status": TaskStatus.PENDING,
            "start_time": datetime.now(),
            "end_time": None,
            "result": None,
            "error": None,
            "steps": [],
            "kwargs": kwargs
        }
        
        self.current_tasks[task_id] = task_info
        
        try:
            # 更新InfoPool中的当前任务
            await self.info_pool.update_task(task_description, source_agent="coordinator")
            
            # 执行四阶段协作流程
            result = await self._execute_collaboration_workflow(task_id, task_description, **kwargs)
            
            # 更新任务状态
            task_info["status"] = TaskStatus.COMPLETED
            task_info["end_time"] = datetime.now()
            task_info["result"] = result
            
            return {
                "task_id": task_id,
                "status": "success",
                "result": result,
                "execution_time": (task_info["end_time"] - task_info["start_time"]).total_seconds()
            }
            
        except Exception as e:
            # 处理执行错误
            task_info["status"] = TaskStatus.FAILED
            task_info["end_time"] = datetime.now()
            task_info["error"] = str(e)
            
            await self.info_pool.add_error(f"任务执行失败: {str(e)}", source_agent="coordinator")
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "execution_time": (task_info["end_time"] - task_info["start_time"]).total_seconds()
            }
    
    async def _execute_collaboration_workflow(self, task_id: str, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        执行四智能体协作工作流
        
        Args:
            task_id: 任务ID
            task_description: 任务描述
            **kwargs: 额外参数
            
        Returns:
            执行结果
        """
        task_info = self.current_tasks[task_id]
        
        # 阶段1: Manager - 任务规划
        task_info["status"] = TaskStatus.PLANNING
        planning_result = await self._manager_planning_phase(task_id, task_description, **kwargs)
        task_info["steps"].append({"phase": "planning", "result": planning_result})
        
        # 阶段2: Executor - 操作执行
        task_info["status"] = TaskStatus.EXECUTING
        execution_result = await self._executor_execution_phase(task_id, planning_result, **kwargs)
        task_info["steps"].append({"phase": "execution", "result": execution_result})
        
        # 阶段3: ActionReflector - 结果反思
        task_info["status"] = TaskStatus.REFLECTING
        reflection_result = await self._reflector_reflection_phase(task_id, execution_result, **kwargs)
        task_info["steps"].append({"phase": "reflection", "result": reflection_result})
        
        # 阶段4: Notetaker - 知识记录
        task_info["status"] = TaskStatus.RECORDING
        recording_result = await self._notetaker_recording_phase(task_id, {
            "planning": planning_result,
            "execution": execution_result,
            "reflection": reflection_result
        }, **kwargs)
        task_info["steps"].append({"phase": "recording", "result": recording_result})
        
        # 汇总结果
        final_result = {
            "task_id": task_id,
            "task_description": task_description,
            "planning_result": planning_result,
            "execution_result": execution_result,
            "reflection_result": reflection_result,
            "recording_result": recording_result,
            "overall_success": execution_result.get("success", False),
            "workflow_completed": True
        }
        
        return final_result
    
    async def _manager_planning_phase(self, task_id: str, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Manager智能体规划阶段
        
        Args:
            task_id: 任务ID
            task_description: 任务描述
            **kwargs: 额外参数
            
        Returns:
            规划结果
        """
        manager = self.agents.get("manager")
        if not manager:
            raise ValueError("Manager智能体未找到")

        try:
            # 1. Manager Agent takes a screenshot
            logger.info(f"Manager agent starts taking screenshot (TASK {task_id})")
            screenshot_path = await manager.take_screenshot()
            logger.info(f"Manager agent finished taking screenshot (TASK {task_id}), path: {screenshot_path}")
            
            # 获取当前共享状态
            shared_state = await self.info_pool.get_shared_state()
            logger.info(f"Manager agent fetched shared state (TASK {task_id}): "); 
            print(JSON(json.dumps(shared_state, ensure_ascii=False)))

            # 调用Manager智能体
            logger.info(f"Manager agent invokes for planning (TASK {task_id})")
            agent_result = await manager.run({
                "task_id": task_id,
                "description": task_description,
                "screenshot_path": screenshot_path
            })
            
            # 从AgentResult中提取实际结果
            if hasattr(agent_result, 'output'):
                planning_result = agent_result.output
            elif hasattr(agent_result, 'success') and agent_result.success:
                planning_result = getattr(agent_result, 'output', agent_result)
            else:
                planning_result = agent_result
            
            logger.info(f"Manager agent returned planning result (TASK {task_id}): ")
            print(planning_result)
            
            # 构建结构化结果
            structured_result = {
                "agent": "manager",
                "phase": "planning",
                "task_id": task_id,
                "plan": planning_result,
                "screenshot_path": screenshot_path,
                "original_task": task_description,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # 更新InfoPool
            await self.info_pool.update_execution_plan(structured_result, source_agent="manager")
            
            return structured_result
            
        except Exception as e:
            logger.error(f"Manager agent planning phase failed (TASK {task_id}): {e}", exc_info=True)
            error_result = {
                "agent": "manager",
                "phase": "planning",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "screenshot_path": screenshot_path if 'screenshot_path' in locals() else None
            }
            
            await self.info_pool.add_error(f"Manager规划失败: {str(e)}", source_agent="manager")
            return error_result
    
    async def _executor_execution_phase(self, task_id: str, planning_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Executor智能体执行阶段
        
        Args:
            task_id: 任务ID
            planning_result: 规划结果
            **kwargs: 额外参数
            
        Returns:
            执行结果
        """
        executor = self.agents.get("executor")
        if not executor:
            raise ValueError("Executor智能体未找到")
        
        try:
            # 从规划结果中提取截图路径
            screenshot_path = planning_result.get("screenshot_path")
            if not screenshot_path:
                raise ValueError("截图路径未在规划结果中找到")

            # 构建执行提示
            execution_prompt = f"""
            作为Executor智能体，请根据以下计划执行移动GUI操作：
            
            执行计划: {planning_result.get('plan', '')}
            任务ID: {task_id}
            当前屏幕截图路径: {screenshot_path}
            
            请使用提供的截图作为参考，执行具体的GUI操作并报告结果。
            """
            
            # 调用Executor智能体
            agent_result = await executor.run({
                "task_id": task_id,
                "planning_result": planning_result,
                "prompt": execution_prompt,
                "description": planning_result.get('original_task', ''),
                "screenshot_path": screenshot_path,
                "task_type": "multimodal_analysis",
                "use_multimodal_analysis": True
            })
            
            # 从AgentResult中提取实际结果
            if hasattr(agent_result, 'output'):
                execution_result = agent_result.output
            elif hasattr(agent_result, 'success') and agent_result.success:
                execution_result = getattr(agent_result, 'output', agent_result)
            else:
                execution_result = agent_result
            
            # 构建结构化结果
            structured_result = {
                "agent": "executor",
                "phase": "execution",
                "task_id": task_id,
                "execution_details": execution_result,
                "actions_performed": [],  # 实际应用中会包含具体的GUI操作
                "timestamp": datetime.now().isoformat(),
                "success": True  # 实际应用中会根据执行结果判断
            }
            
            # 更新InfoPool
            await self.info_pool.add_action_result(structured_result, source_agent="executor")
            
            return structured_result
            
        except Exception as e:
            error_result = {
                "agent": "executor",
                "phase": "execution",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            await self.info_pool.add_error(f"Executor执行失败: {str(e)}", source_agent="executor")
            return error_result
    
    async def _reflector_reflection_phase(self, task_id: str, execution_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        ActionReflector智能体反思阶段
        
        Args:
            task_id: 任务ID
            execution_result: 执行结果
            **kwargs: 额外参数
            
        Returns:
            反思结果
        """
        reflector = self.agents.get("reflector")
        if not reflector:
            raise ValueError("ActionReflector智能体未找到")
        
        try:
            # 构建反思提示
            reflection_prompt = f"""
            作为ActionReflector智能体，请分析以下执行结果：
            
            执行结果: {execution_result}
            任务ID: {task_id}
            
            请提供：
            1. 执行质量评估
            2. 问题识别
            3. 改进建议
            4. 学习洞察
            """
            
            # 从执行结果中提取截图信息
            execution_details = execution_result.get('execution_details', {})
            
            # 处理AgentResult对象
            if hasattr(execution_details, 'output'):
                execution_output = execution_details.output
            elif hasattr(execution_details, '__dict__'):
                # 如果是AgentResult对象，直接使用其属性
                execution_output = execution_details.__dict__
            else:
                execution_output = execution_details
            
            # 确保execution_output是字典类型
            if not isinstance(execution_output, dict):
                execution_output = {}
            
            # 调用ActionReflector智能体
            reflection_result = await reflector.run({
                "task_id": task_id,
                "execution_result": execution_result,
                "prompt": reflection_prompt,
                "before_screenshot": execution_output.get('screenshot_path'),  # 操作前截图
                "after_screenshot": execution_output.get('marked_screenshot_path'),  # 操作后截图
                "action_info": execution_output.get('llm_action_plan', {}),
                "expectation": "操作成功执行"
            })
            
            # 构建结构化结果
            structured_result = {
                "agent": "reflector",
                "phase": "reflection",
                "task_id": task_id,
                "reflection_analysis": reflection_result,
                "quality_score": 0.8,  # 实际应用中会计算具体分数
                "improvements": [],
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # 更新InfoPool
            await self.info_pool.add_reflection(structured_result, source_agent="reflector")
            
            return structured_result
            
        except Exception as e:
            error_result = {
                "agent": "reflector",
                "phase": "reflection",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            await self.info_pool.add_error(f"ActionReflector反思失败: {str(e)}", source_agent="reflector")
            return error_result
    
    async def _notetaker_recording_phase(self, task_id: str, all_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Notetaker智能体记录阶段
        
        Args:
            task_id: 任务ID
            all_results: 所有阶段的结果
            **kwargs: 额外参数
            
        Returns:
            记录结果
        """
        notetaker = self.agents.get("notetaker")
        if not notetaker:
            raise ValueError("Notetaker智能体未找到")
        
        try:
            # 构建记录提示
            recording_prompt = f"""
            作为Notetaker智能体，请记录以下完整的任务执行过程：
            
            任务ID: {task_id}
            所有阶段结果: {all_results}
            
            请生成：
            1. 执行摘要
            2. 关键知识点
            3. 经验总结
            4. 知识库更新
            """
            
            # 调用Notetaker智能体
            recording_result = await notetaker.run({
                "task_id": task_id,
                "all_results": all_results,
                "prompt": recording_prompt
            })
            
            # 构建结构化结果
            structured_result = {
                "agent": "notetaker",
                "phase": "recording",
                "task_id": task_id,
                "knowledge_record": recording_result,
                "knowledge_items": [],  # 实际应用中会包含具体的知识项
                "patterns_identified": [],
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # 更新InfoPool
            await self.info_pool.add_knowledge(structured_result, source_agent="notetaker")
            
            return structured_result
            
        except Exception as e:
            error_result = {
                "agent": "notetaker",
                "phase": "recording",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            await self.info_pool.add_error(f"Notetaker记录失败: {str(e)}", source_agent="notetaker")
            return error_result
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        return self.current_tasks.get(task_id)
    
    async def list_active_tasks(self) -> List[Dict[str, Any]]:
        """
        列出所有活跃任务
        
        Returns:
            活跃任务列表
        """
        active_tasks = []
        for task_id, task_info in self.current_tasks.items():
            if task_info["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                active_tasks.append(task_info)
        
        return active_tasks
    
    async def shutdown(self):
        """
        关闭协调器
        """
        # 等待所有活跃任务完成或超时
        active_tasks = await self.list_active_tasks()
        if active_tasks:
            print(f"等待 {len(active_tasks)} 个活跃任务完成...")
            # 实际应用中可以实现更复杂的关闭逻辑
        
        # 清理资源
        self.current_tasks.clear()
        print("AgentCoordinator已关闭")


# 导出类和枚举
__all__ = ["AgentCoordinator", "TaskStatus"]