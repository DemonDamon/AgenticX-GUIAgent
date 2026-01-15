"""Core agents for AgenticX-GUIAgent system.

基于AgenticX框架的核心智能体实现。
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json

# 使用AgenticX核心组件
from agenticx.core.agent import Agent, AgentResult
from agenticx.core.event import Event, TaskStartEvent, TaskEndEvent
from agenticx.core.tool import BaseTool
from agenticx.llms.base import BaseLLMProvider
from agenticx.memory.component import MemoryComponent

from .base_agent import BaseAgenticXGUIAgentAgent, AgentState
from config import AgentConfig


class ManagerAgent(BaseAgenticXGUIAgentAgent):
    """Manager Agent - 负责任务分解、分配和整体协调。
    
    作为系统的核心协调者，负责：
    - 接收和分析用户任务
    - 将复杂任务分解为子任务
    - 分配任务给其他智能体
    - 监控任务执行进度
    - 协调智能体间的协作
    """
    
    def __init__(self, agent_id: str = "manager", **kwargs):
        super().__init__(agent_id, **kwargs)
        self.agent_type = "ManagerAgent"
        self.managed_agents: Set[str] = set()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
    async def _initialize_impl(self) -> None:
        """Initialize manager-specific components."""
        await super()._initialize_impl()
        logger.info(f"Manager Agent {self.agent_id} initialized")
        
        # Register with info pool
        if self.info_pool:
            await self.info_pool.add_info(
                InfoType.SYSTEM,
                f"Manager Agent {self.agent_id} online",
                source=self.agent_id,
                priority=InfoPriority.HIGH
            )
    
    async def process_user_task(self, task_description: str, task_context: Optional[Dict[str, Any]] = None) -> str:
        """Process user task and coordinate execution.
        
        Args:
            task_description: Description of the task to execute
            task_context: Additional context for the task
            
        Returns:
            Task ID for tracking
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_tasks)}"
        
        # Analyze and decompose task
        subtasks = await self._decompose_task(task_description, task_context)
        
        # Create task record
        self.active_tasks[task_id] = {
            "description": task_description,
            "context": task_context or {},
            "subtasks": subtasks,
            "status": "planning",
            "created_at": datetime.now(),
            "assigned_agents": [],
            "progress": 0.0
        }
        
        # Share task information
        if self.info_pool:
            await self.info_pool.add_info(
                InfoType.TASK,
                f"New task created: {task_description}",
                source=self.agent_id,
                priority=InfoPriority.HIGH,
                metadata={"task_id": task_id, "subtasks_count": len(subtasks)}
            )
        
        # Start task execution
        asyncio.create_task(self._execute_task(task_id))
        
        return task_id
    
    async def _decompose_task(self, task_description: str, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Decompose complex task into subtasks.
        
        Args:
            task_description: Main task description
            context: Task context
            
        Returns:
            List of subtasks
        """
        # Simple task decomposition logic
        # In a real implementation, this would use more sophisticated AI reasoning
        
        subtasks = []
        
        # Basic decomposition based on keywords
        if "screenshot" in task_description.lower() or "capture" in task_description.lower():
            subtasks.append({
                "type": "screen_capture",
                "description": "Capture current screen",
                "assigned_agent": "executor",
                "priority": "high"
            })
        
        if "click" in task_description.lower() or "tap" in task_description.lower():
            subtasks.append({
                "type": "ui_interaction",
                "description": "Perform UI interaction",
                "assigned_agent": "executor",
                "priority": "high"
            })
        
        if "analyze" in task_description.lower() or "check" in task_description.lower():
            subtasks.append({
                "type": "analysis",
                "description": "Analyze results",
                "assigned_agent": "action_reflector",
                "priority": "medium"
            })
        
        # Always add documentation task
        subtasks.append({
            "type": "documentation",
            "description": "Document task execution",
            "assigned_agent": "notetaker",
            "priority": "low"
        })
        
        return subtasks
    
    async def _execute_task(self, task_id: str) -> None:
        """Execute task by coordinating subtasks.
        
        Args:
            task_id: ID of task to execute
        """
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task["status"] = "executing"
        
        try:
            # Execute subtasks
            for i, subtask in enumerate(task["subtasks"]):
                agent_id = subtask.get("assigned_agent")
                if agent_id and self.coordinator:
                    # Send subtask to assigned agent
                    await self.coordinator.send_message(
                        from_agent=self.agent_id,
                        to_agent=agent_id,
                        message_type=MessageType.TASK,
                        content={
                            "task_id": task_id,
                            "subtask_index": i,
                            "subtask": subtask
                        }
                    )
                
                # Update progress
                task["progress"] = (i + 1) / len(task["subtasks"]) * 100
            
            task["status"] = "completed"
            task["completed_at"] = datetime.now()
            
            # Share completion info
            if self.info_pool:
                await self.info_pool.add_info(
                    InfoType.TASK,
                    f"Task completed: {task['description']}",
                    source=self.agent_id,
                    priority=InfoPriority.MEDIUM,
                    metadata={"task_id": task_id, "duration": str(task["completed_at"] - task["created_at"])}
                )
        
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            logger.error(f"Task {task_id} failed: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task.
        
        Args:
            task_id: ID of task to check
            
        Returns:
            Task status information or None if not found
        """
        return self.active_tasks.get(task_id)
    
    async def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks.
        
        Returns:
            List of active task information
        """
        return [
            {"task_id": tid, **task}
            for tid, task in self.active_tasks.items()
            if task["status"] in ["planning", "executing"]
        ]


class ExecutorAgent(BaseAgenticXGUIAgentAgent):
    """Executor Agent - 负责执行具体的GUI操作和任务。
    
    专门负责：
    - 执行GUI操作（点击、滑动、输入等）
    - 屏幕截图和元素识别
    - 与移动设备交互
    - 执行自动化脚本
    """
    
    def __init__(self, agent_id: str = "executor", **kwargs):
        super().__init__(agent_id, **kwargs)
        self.agent_type = "ExecutorAgent"
        self.execution_history: List[Dict[str, Any]] = []
        self.current_screen_state: Optional[Dict[str, Any]] = None
        
    async def _initialize_impl(self) -> None:
        """Initialize executor-specific components."""
        await super()._initialize_impl()
        logger.info(f"Executor Agent {self.agent_id} initialized")
        
        # Initialize GUI tools (placeholder)
        # In real implementation, this would initialize actual GUI automation tools
        self.gui_tools = {
            "screen_capture": True,
            "element_finder": True,
            "touch_operations": True,
            "keyboard_input": True
        }
    
    async def execute_action(self, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific GUI action.
        
        Args:
            action_type: Type of action to execute
            parameters: Action parameters
            
        Returns:
            Execution result
        """
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.execution_history)}"
        
        execution_record = {
            "execution_id": execution_id,
            "action_type": action_type,
            "parameters": parameters,
            "timestamp": datetime.now(),
            "status": "executing"
        }
        
        self.execution_history.append(execution_record)
        
        try:
            # Execute based on action type
            if action_type == "screen_capture":
                result = await self._capture_screen(parameters)
            elif action_type == "click":
                result = await self._click_element(parameters)
            elif action_type == "input_text":
                result = await self._input_text(parameters)
            elif action_type == "swipe":
                result = await self._swipe_gesture(parameters)
            else:
                result = {"success": False, "error": f"Unknown action type: {action_type}"}
            
            execution_record["status"] = "completed" if result.get("success") else "failed"
            execution_record["result"] = result
            
            # Share execution info
            if self.info_pool:
                await self.info_pool.add_info(
                    InfoType.ACTION,
                    f"Executed {action_type}: {result.get('success', False)}",
                    source=self.agent_id,
                    priority=InfoPriority.MEDIUM,
                    metadata={"execution_id": execution_id, "action_type": action_type}
                )
            
            return result
        
        except Exception as e:
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            logger.error(f"Action execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _capture_screen(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Capture screen screenshot."""
        # Placeholder implementation
        # In real implementation, this would use actual screen capture tools
        await asyncio.sleep(0.1)  # Simulate capture time
        
        self.current_screen_state = {
            "timestamp": datetime.now(),
            "resolution": parameters.get("resolution", "1080x1920"),
            "format": parameters.get("format", "png"),
            "file_path": f"/tmp/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        }
        
        return {
            "success": True,
            "screen_state": self.current_screen_state,
            "message": "Screen captured successfully"
        }
    
    async def _click_element(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Click on screen element."""
        # Placeholder implementation
        await asyncio.sleep(0.05)  # Simulate click time
        
        x = parameters.get("x", 0)
        y = parameters.get("y", 0)
        element_id = parameters.get("element_id")
        
        return {
            "success": True,
            "coordinates": {"x": x, "y": y},
            "element_id": element_id,
            "message": f"Clicked at ({x}, {y})"
        }
    
    async def _input_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Input text into element."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate input time
        
        text = parameters.get("text", "")
        element_id = parameters.get("element_id")
        
        return {
            "success": True,
            "text": text,
            "element_id": element_id,
            "message": f"Input text: {text[:50]}..."
        }
    
    async def _swipe_gesture(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform swipe gesture."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate swipe time
        
        start_x = parameters.get("start_x", 0)
        start_y = parameters.get("start_y", 0)
        end_x = parameters.get("end_x", 0)
        end_y = parameters.get("end_y", 0)
        
        return {
            "success": True,
            "start_coordinates": {"x": start_x, "y": start_y},
            "end_coordinates": {"x": end_x, "y": end_y},
            "message": f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y})"
        }


class ActionReflectorAgent(BaseAgenticXGUIAgentAgent):
    """Action Reflector Agent - 负责分析和反思执行结果。
    
    专门负责：
    - 分析执行结果的正确性
    - 识别执行中的问题和错误
    - 提供改进建议
    - 学习和优化策略
    """
    
    def __init__(self, agent_id: str = "action_reflector", **kwargs):
        super().__init__(agent_id, **kwargs)
        self.agent_type = "ActionReflectorAgent"
        self.analysis_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}
        
    async def _initialize_impl(self) -> None:
        """Initialize reflector-specific components."""
        await super()._initialize_impl()
        logger.info(f"Action Reflector Agent {self.agent_id} initialized")
        
        # Initialize analysis tools
        self.analysis_tools = {
            "result_validator": True,
            "pattern_recognizer": True,
            "error_analyzer": True,
            "improvement_suggester": True
        }
    
    async def analyze_execution(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution results and provide feedback.
        
        Args:
            execution_data: Data about the execution to analyze
            
        Returns:
            Analysis results and recommendations
        """
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.analysis_history)}"
        
        analysis_record = {
            "analysis_id": analysis_id,
            "execution_data": execution_data,
            "timestamp": datetime.now(),
            "status": "analyzing"
        }
        
        self.analysis_history.append(analysis_record)
        
        try:
            # Perform analysis
            success_rate = await self._calculate_success_rate(execution_data)
            error_patterns = await self._identify_error_patterns(execution_data)
            improvements = await self._suggest_improvements(execution_data)
            
            analysis_result = {
                "success_rate": success_rate,
                "error_patterns": error_patterns,
                "improvements": improvements,
                "confidence": self._calculate_confidence(execution_data),
                "recommendations": await self._generate_recommendations(execution_data)
            }
            
            analysis_record["status"] = "completed"
            analysis_record["result"] = analysis_result
            
            # Share analysis info
            if self.info_pool:
                await self.info_pool.add_info(
                    InfoType.ANALYSIS,
                    f"Analysis completed: {success_rate:.1%} success rate",
                    source=self.agent_id,
                    priority=InfoPriority.MEDIUM,
                    metadata={"analysis_id": analysis_id, "success_rate": success_rate}
                )
            
            return analysis_result
        
        except Exception as e:
            analysis_record["status"] = "error"
            analysis_record["error"] = str(e)
            logger.error(f"Analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _calculate_success_rate(self, execution_data: Dict[str, Any]) -> float:
        """Calculate success rate of executions."""
        # Placeholder implementation
        executions = execution_data.get("executions", [])
        if not executions:
            return 0.0
        
        successful = sum(1 for exec in executions if exec.get("status") == "completed")
        return successful / len(executions)
    
    async def _identify_error_patterns(self, execution_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify common error patterns."""
        # Placeholder implementation
        patterns = []
        executions = execution_data.get("executions", [])
        
        # Simple pattern detection
        error_types = {}
        for exec in executions:
            if exec.get("status") == "failed":
                error_type = exec.get("error", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            if count > 1:
                patterns.append({
                    "pattern_type": "recurring_error",
                    "error_type": error_type,
                    "frequency": count,
                    "severity": "high" if count > 3 else "medium"
                })
        
        return patterns
    
    async def _suggest_improvements(self, execution_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest improvements based on analysis."""
        # Placeholder implementation
        improvements = []
        
        success_rate = await self._calculate_success_rate(execution_data)
        
        if success_rate < 0.8:
            improvements.append({
                "type": "reliability",
                "description": "Add retry mechanism for failed operations",
                "priority": "high",
                "estimated_impact": "20% improvement in success rate"
            })
        
        if success_rate < 0.9:
            improvements.append({
                "type": "validation",
                "description": "Improve pre-execution validation",
                "priority": "medium",
                "estimated_impact": "10% improvement in success rate"
            })
        
        return improvements
    
    def _calculate_confidence(self, execution_data: Dict[str, Any]) -> float:
        """Calculate confidence in analysis results."""
        # Simple confidence calculation based on data volume
        executions = execution_data.get("executions", [])
        if len(executions) < 5:
            return 0.5
        elif len(executions) < 20:
            return 0.7
        else:
            return 0.9
    
    async def _generate_recommendations(self, execution_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        success_rate = await self._calculate_success_rate(execution_data)
        
        if success_rate < 0.5:
            recommendations.append("Consider reviewing task decomposition strategy")
            recommendations.append("Implement more robust error handling")
        elif success_rate < 0.8:
            recommendations.append("Add validation steps before execution")
            recommendations.append("Implement retry logic for transient failures")
        else:
            recommendations.append("Performance is good, consider optimizing for speed")
            recommendations.append("Document successful patterns for reuse")
        
        return recommendations


class NotetakerAgent(BaseAgenticXGUIAgentAgent):
    """Notetaker Agent - 负责记录和文档化系统活动。
    
    专门负责：
    - 记录系统活动和事件
    - 生成执行报告
    - 维护操作日志
    - 创建知识文档
    """
    
    def __init__(self, agent_id: str = "notetaker", **kwargs):
        super().__init__(agent_id, **kwargs)
        self.agent_type = "NotetakerAgent"
        self.notes: List[Dict[str, Any]] = []
        self.reports: Dict[str, Dict[str, Any]] = {}
        
    async def _initialize_impl(self) -> None:
        """Initialize notetaker-specific components."""
        await super()._initialize_impl()
        logger.info(f"Notetaker Agent {self.agent_id} initialized")
        
        # Initialize documentation tools
        self.documentation_tools = {
            "note_recorder": True,
            "report_generator": True,
            "log_analyzer": True,
            "knowledge_extractor": True
        }
    
    async def record_event(self, event_type: str, event_data: Dict[str, Any], importance: str = "medium") -> str:
        """Record a system event.
        
        Args:
            event_type: Type of event to record
            event_data: Event data
            importance: Importance level (low, medium, high, critical)
            
        Returns:
            Note ID
        """
        note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.notes)}"
        
        note = {
            "note_id": note_id,
            "event_type": event_type,
            "event_data": event_data,
            "importance": importance,
            "timestamp": datetime.now(),
            "source": event_data.get("source", "unknown")
        }
        
        self.notes.append(note)
        
        # Share note info if important
        if importance in ["high", "critical"] and self.info_pool:
            await self.info_pool.add_info(
                InfoType.LOG,
                f"Important event recorded: {event_type}",
                source=self.agent_id,
                priority=InfoPriority.HIGH if importance == "critical" else InfoPriority.MEDIUM,
                metadata={"note_id": note_id, "event_type": event_type}
            )
        
        return note_id
    
    async def generate_report(self, report_type: str, time_range: Optional[Dict[str, datetime]] = None, 
                            filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a report based on recorded events.
        
        Args:
            report_type: Type of report to generate
            time_range: Time range for the report
            filters: Additional filters
            
        Returns:
            Generated report
        """
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_type}"
        
        # Filter notes based on criteria
        filtered_notes = self._filter_notes(time_range, filters)
        
        # Generate report based on type
        if report_type == "activity_summary":
            report_content = await self._generate_activity_summary(filtered_notes)
        elif report_type == "error_analysis":
            report_content = await self._generate_error_analysis(filtered_notes)
        elif report_type == "performance_report":
            report_content = await self._generate_performance_report(filtered_notes)
        else:
            report_content = await self._generate_general_report(filtered_notes)
        
        report = {
            "report_id": report_id,
            "report_type": report_type,
            "generated_at": datetime.now(),
            "time_range": time_range,
            "filters": filters,
            "content": report_content,
            "notes_count": len(filtered_notes)
        }
        
        self.reports[report_id] = report
        
        # Share report info
        if self.info_pool:
            await self.info_pool.add_info(
                InfoType.REPORT,
                f"Report generated: {report_type}",
                source=self.agent_id,
                priority=InfoPriority.MEDIUM,
                metadata={"report_id": report_id, "notes_count": len(filtered_notes)}
            )
        
        return report
    
    def _filter_notes(self, time_range: Optional[Dict[str, datetime]], 
                     filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter notes based on criteria."""
        filtered = self.notes.copy()
        
        # Time range filter
        if time_range:
            start_time = time_range.get("start")
            end_time = time_range.get("end")
            
            if start_time:
                filtered = [note for note in filtered if note["timestamp"] >= start_time]
            if end_time:
                filtered = [note for note in filtered if note["timestamp"] <= end_time]
        
        # Additional filters
        if filters:
            for key, value in filters.items():
                if key == "event_type":
                    filtered = [note for note in filtered if note["event_type"] == value]
                elif key == "importance":
                    filtered = [note for note in filtered if note["importance"] == value]
                elif key == "source":
                    filtered = [note for note in filtered if note["source"] == value]
        
        return filtered
    
    async def _generate_activity_summary(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate activity summary report."""
        event_types = {}
        importance_levels = {}
        sources = {}
        
        for note in notes:
            # Count event types
            event_type = note["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Count importance levels
            importance = note["importance"]
            importance_levels[importance] = importance_levels.get(importance, 0) + 1
            
            # Count sources
            source = note["source"]
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_events": len(notes),
            "event_types": event_types,
            "importance_distribution": importance_levels,
            "source_distribution": sources,
            "time_span": {
                "start": min(note["timestamp"] for note in notes) if notes else None,
                "end": max(note["timestamp"] for note in notes) if notes else None
            }
        }
    
    async def _generate_error_analysis(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate error analysis report."""
        error_notes = [note for note in notes if "error" in note["event_type"].lower()]
        
        error_patterns = {}
        error_sources = {}
        
        for note in error_notes:
            # Analyze error patterns
            event_data = note.get("event_data", {})
            error_type = event_data.get("error_type", "unknown")
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
            
            # Count error sources
            source = note["source"]
            error_sources[source] = error_sources.get(source, 0) + 1
        
        return {
            "total_errors": len(error_notes),
            "error_rate": len(error_notes) / len(notes) if notes else 0,
            "error_patterns": error_patterns,
            "error_sources": error_sources,
            "recommendations": self._generate_error_recommendations(error_patterns)
        }
    
    async def _generate_performance_report(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance report."""
        performance_notes = [note for note in notes if "performance" in note["event_type"].lower()]
        
        metrics = {
            "execution_times": [],
            "success_rates": [],
            "resource_usage": []
        }
        
        for note in performance_notes:
            event_data = note.get("event_data", {})
            
            if "execution_time" in event_data:
                metrics["execution_times"].append(event_data["execution_time"])
            if "success_rate" in event_data:
                metrics["success_rates"].append(event_data["success_rate"])
            if "resource_usage" in event_data:
                metrics["resource_usage"].append(event_data["resource_usage"])
        
        return {
            "performance_events": len(performance_notes),
            "average_execution_time": sum(metrics["execution_times"]) / len(metrics["execution_times"]) if metrics["execution_times"] else 0,
            "average_success_rate": sum(metrics["success_rates"]) / len(metrics["success_rates"]) if metrics["success_rates"] else 0,
            "metrics": metrics
        }
    
    async def _generate_general_report(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate general report."""
        return {
            "summary": f"General report covering {len(notes)} events",
            "event_distribution": await self._generate_activity_summary(notes),
            "recent_events": notes[-10:] if len(notes) > 10 else notes
        }
    
    def _generate_error_recommendations(self, error_patterns: Dict[str, int]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        if error_patterns:
            most_common_error = max(error_patterns.items(), key=lambda x: x[1])
            recommendations.append(f"Focus on resolving '{most_common_error[0]}' errors (occurred {most_common_error[1]} times)")
            
            if len(error_patterns) > 3:
                recommendations.append("Consider implementing more robust error handling")
            
            total_errors = sum(error_patterns.values())
            if total_errors > 10:
                recommendations.append("High error frequency detected, review system stability")
        
        return recommendations
    
    async def get_notes(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get notes with optional filtering.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of filtered notes
        """
        return self._filter_notes(None, filters)
    
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific report.
        
        Args:
            report_id: ID of report to retrieve
            
        Returns:
            Report data or None if not found
        """
        return self.reports.get(report_id)