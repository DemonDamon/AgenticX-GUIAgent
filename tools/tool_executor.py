#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent Tool Executor
工具执行器：负责工具的执行调度、队列管理和结果处理

Author: AgenticX Team
Date: 2025
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .gui_tools import (
    GUITool, ToolParameters, ToolResult, ToolError,
    ToolType, ToolStatus, ExecutionMode
)
from .tool_adapters import ToolAdapter, AdaptedGUITool
from utils import get_iso_timestamp, setup_logger


class ExecutionPriority(Enum):
    """执行优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ExecutionStrategy(Enum):
    """执行策略"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"     # 并行执行
    PIPELINE = "pipeline"     # 流水线执行
    ADAPTIVE = "adaptive"     # 自适应执行


@dataclass
class ExecutionTask:
    """执行任务"""
    task_id: str
    tool: GUITool
    parameters: ToolParameters
    context: Optional[Dict[str, Any]] = None
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    created_at: str = field(default_factory=get_iso_timestamp)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: ToolStatus = ToolStatus.IDLE
    result: Optional[ToolResult] = None
    error: Optional[str] = None


@dataclass
class ExecutionBatch:
    """执行批次"""
    batch_id: str
    tasks: List[ExecutionTask]
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_concurrent: int = 5
    timeout: Optional[int] = None
    created_at: str = field(default_factory=get_iso_timestamp)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: ToolStatus = ToolStatus.IDLE
    results: List[ToolResult] = field(default_factory=list)
    progress: float = 0.0


class ExecutionQueue:
    """执行队列"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {
            ExecutionPriority.URGENT: deque(),
            ExecutionPriority.HIGH: deque(),
            ExecutionPriority.NORMAL: deque(),
            ExecutionPriority.LOW: deque()
        }
        self.task_map: Dict[str, ExecutionTask] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.waiting_tasks: Dict[str, ExecutionTask] = {}
        self._lock = asyncio.Lock()
    
    async def enqueue(self, task: ExecutionTask) -> bool:
        """入队任务"""
        async with self._lock:
            if len(self.task_map) >= self.max_size:
                return False
            
            # 检查依赖关系
            if task.dependencies:
                # 如果有未完成的依赖，放入等待队列
                unresolved_deps = [
                    dep for dep in task.dependencies
                    if dep not in self.task_map or 
                    self.task_map[dep].status != ToolStatus.COMPLETED
                ]
                
                if unresolved_deps:
                    self.waiting_tasks[task.task_id] = task
                    for dep in task.dependencies:
                        self.dependency_graph[dep].add(task.task_id)
                    return True
            
            # 添加到相应优先级队列
            self.queues[task.priority].append(task)
            self.task_map[task.task_id] = task
            return True
    
    async def dequeue(self) -> Optional[ExecutionTask]:
        """出队任务"""
        async with self._lock:
            # 按优先级顺序检查队列
            for priority in [ExecutionPriority.URGENT, ExecutionPriority.HIGH,
                           ExecutionPriority.NORMAL, ExecutionPriority.LOW]:
                if self.queues[priority]:
                    task = self.queues[priority].popleft()
                    task.status = ToolStatus.RUNNING
                    task.started_at = get_iso_timestamp()
                    return task
            
            return None
    
    async def complete_task(self, task_id: str, result: ToolResult) -> None:
        """完成任务"""
        async with self._lock:
            if task_id in self.task_map:
                task = self.task_map[task_id]
                task.status = result.status
                task.result = result
                task.completed_at = get_iso_timestamp()
                
                # 检查是否有等待此任务的依赖任务
                if task_id in self.dependency_graph:
                    dependent_tasks = self.dependency_graph[task_id].copy()
                    del self.dependency_graph[task_id]
                    
                    for dependent_id in dependent_tasks:
                        if dependent_id in self.waiting_tasks:
                            dependent_task = self.waiting_tasks[dependent_id]
                            
                            # 检查所有依赖是否都已完成
                            all_deps_completed = all(
                                dep in self.task_map and 
                                self.task_map[dep].status == ToolStatus.COMPLETED
                                for dep in dependent_task.dependencies
                            )
                            
                            if all_deps_completed:
                                # 移动到执行队列
                                del self.waiting_tasks[dependent_id]
                                self.queues[dependent_task.priority].append(dependent_task)
                                self.task_map[dependent_id] = dependent_task
    
    async def get_task(self, task_id: str) -> Optional[ExecutionTask]:
        """获取任务"""
        async with self._lock:
            return self.task_map.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            if task_id in self.task_map:
                task = self.task_map[task_id]
                if task.status in [ToolStatus.PENDING, ToolStatus.RUNNING]:
                    task.status = ToolStatus.CANCELLED
                    task.completed_at = get_iso_timestamp()
                    return True
            
            if task_id in self.waiting_tasks:
                del self.waiting_tasks[task_id]
                return True
            
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        async with self._lock:
            return {
                'total_tasks': len(self.task_map),
                'waiting_tasks': len(self.waiting_tasks),
                'queue_sizes': {
                    priority.value: len(queue)
                    for priority, queue in self.queues.items()
                },
                'status_counts': {
                    status.value: sum(
                        1 for task in self.task_map.values()
                        if task.status == status
                    )
                    for status in ToolStatus
                }
            }


class ToolExecutor:
    """工具执行器"""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        max_queue_size: int = 1000,
        default_timeout: int = 30
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.logger = logger
        
        # 执行队列
        self.queue = ExecutionQueue(max_queue_size)
        
        # 执行状态
        self.is_running = False
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, ExecutionTask] = {}
        self.failed_tasks: Dict[str, ExecutionTask] = {}
        
        # 批次管理
        self.batches: Dict[str, ExecutionBatch] = {}
        
        # 统计信息
        self.stats = {
            'total_executed': 0,
            'total_succeeded': 0,
            'total_failed': 0,
            'total_cancelled': 0,
            'average_execution_time': 0.0,
            'execution_times': deque(maxlen=1000)
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # 执行器任务
        self.executor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """启动执行器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.executor_task = asyncio.create_task(self._execution_loop())
        logger.info("Tool executor started")
        
        await self._emit_event('executor_started', {'timestamp': get_iso_timestamp()})
    
    async def stop(self) -> None:
        """停止执行器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 取消所有活动任务
        for task in self.active_tasks.values():
            task.cancel()
        
        # 等待执行器任务完成
        if self.executor_task:
            try:
                await self.executor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Tool executor stopped")
        await self._emit_event('executor_stopped', {'timestamp': get_iso_timestamp()})
    
    async def submit_task(
        self,
        tool: GUITool,
        parameters: ToolParameters,
        context: Optional[Dict[str, Any]] = None,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        timeout: Optional[int] = None,
        dependencies: Optional[List[str]] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """提交单个任务"""
        task_id = str(uuid.uuid4())
        
        task = ExecutionTask(
            task_id=task_id,
            tool=tool,
            parameters=parameters,
            context=context or {},
            priority=priority,
            timeout=timeout or self.default_timeout,
            dependencies=dependencies or [],
            callback=callback
        )
        
        success = await self.queue.enqueue(task)
        
        if not success:
            raise ToolError(
                "Failed to enqueue task: queue is full",
                "QUEUE_FULL",
                task_id
            )
        
        logger.info(f"Task submitted: {task_id}")
        await self._emit_event('task_submitted', {
            'task_id': task_id,
            'tool_name': tool.name,
            'priority': priority.value
        })
        
        return task_id
    
    async def submit_batch(
        self,
        tasks: List[Tuple[GUITool, ToolParameters, Optional[Dict[str, Any]]]],
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        max_concurrent: int = 5,
        timeout: Optional[int] = None,
        priority: ExecutionPriority = ExecutionPriority.NORMAL
    ) -> str:
        """提交批次任务"""
        batch_id = str(uuid.uuid4())
        
        execution_tasks = []
        for i, (tool, parameters, context) in enumerate(tasks):
            task_id = f"{batch_id}_{i}"
            
            # 为顺序执行设置依赖关系
            dependencies = []
            if strategy == ExecutionStrategy.SEQUENTIAL and i > 0:
                dependencies = [f"{batch_id}_{i-1}"]
            
            task = ExecutionTask(
                task_id=task_id,
                tool=tool,
                parameters=parameters,
                context=context or {},
                priority=priority,
                timeout=timeout or self.default_timeout,
                dependencies=dependencies
            )
            
            execution_tasks.append(task)
        
        batch = ExecutionBatch(
            batch_id=batch_id,
            tasks=execution_tasks,
            strategy=strategy,
            max_concurrent=max_concurrent,
            timeout=timeout
        )
        
        self.batches[batch_id] = batch
        
        # 提交所有任务到队列
        for task in execution_tasks:
            await self.queue.enqueue(task)
        
        logger.info(f"Batch submitted: {batch_id} with {len(tasks)} tasks")
        await self._emit_event('batch_submitted', {
            'batch_id': batch_id,
            'task_count': len(tasks),
            'strategy': strategy.value
        })
        
        return batch_id
    
    async def get_task_result(self, task_id: str) -> Optional[ToolResult]:
        """获取任务结果"""
        task = await self.queue.get_task(task_id)
        
        if task:
            return task.result
        
        # 检查已完成的任务
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id].result
        
        return None
    
    async def get_batch_results(self, batch_id: str) -> Optional[List[ToolResult]]:
        """获取批次结果"""
        if batch_id not in self.batches:
            return None
        
        batch = self.batches[batch_id]
        results = []
        
        for task in batch.tasks:
            result = await self.get_task_result(task.task_id)
            if result:
                results.append(result)
        
        return results
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 取消队列中的任务
        success = await self.queue.cancel_task(task_id)
        
        # 取消活动任务
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            del self.active_tasks[task_id]
            success = True
        
        if success:
            logger.info(f"Task cancelled: {task_id}")
            await self._emit_event('task_cancelled', {'task_id': task_id})
        
        return success
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """取消批次"""
        if batch_id not in self.batches:
            return False
        
        batch = self.batches[batch_id]
        cancelled_count = 0
        
        for task in batch.tasks:
            if await self.cancel_task(task.task_id):
                cancelled_count += 1
        
        batch.status = ToolStatus.CANCELLED
        batch.completed_at = get_iso_timestamp()
        
        logger.info(f"Batch cancelled: {batch_id}, {cancelled_count} tasks cancelled")
        await self._emit_event('batch_cancelled', {
            'batch_id': batch_id,
            'cancelled_count': cancelled_count
        })
        
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        queue_status = await self.queue.get_queue_status()
        
        return {
            'is_running': self.is_running,
            'active_tasks': len(self.active_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'queue_status': queue_status,
            'batch_count': len(self.batches),
            'statistics': self.stats.copy()
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = await self.queue.get_task(task_id)
        
        if not task:
            # 检查已完成的任务
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
            elif task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
            else:
                return None
        
        return {
            'task_id': task.task_id,
            'tool_name': task.tool.name,
            'status': task.status.value,
            'priority': task.priority.value,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'retry_count': task.retry_count,
            'max_retries': task.max_retries,
            'dependencies': task.dependencies,
            'has_result': task.result is not None,
            'error': task.error
        }
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """获取批次状态"""
        if batch_id not in self.batches:
            return None
        
        batch = self.batches[batch_id]
        
        # 计算进度
        completed_tasks = sum(
            1 for task in batch.tasks
            if task.status in [ToolStatus.COMPLETED, ToolStatus.FAILED, ToolStatus.CANCELLED]
        )
        
        progress = completed_tasks / len(batch.tasks) if batch.tasks else 0.0
        batch.progress = progress
        
        return {
            'batch_id': batch.batch_id,
            'status': batch.status.value,
            'strategy': batch.strategy.value,
            'task_count': len(batch.tasks),
            'progress': progress,
            'created_at': batch.created_at,
            'started_at': batch.started_at,
            'completed_at': batch.completed_at,
            'max_concurrent': batch.max_concurrent
        }
    
    def register_event_callback(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """注册事件回调"""
        self.event_callbacks[event_type].append(callback)
    
    def unregister_event_callback(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """注销事件回调"""
        if event_type in self.event_callbacks:
            try:
                self.event_callbacks[event_type].remove(callback)
            except ValueError:
                pass
    
    async def _execution_loop(self) -> None:
        """执行循环"""
        while self.is_running:
            try:
                # 清理已完成的活动任务
                await self._cleanup_completed_tasks()
                
                # 检查是否可以启动新任务
                if len(self.active_tasks) < self.max_concurrent_tasks:
                    task = await self.queue.dequeue()
                    
                    if task:
                        # 启动任务执行
                        execution_task = asyncio.create_task(
                            self._execute_task(task)
                        )
                        self.active_tasks[task.task_id] = execution_task
                
                # 短暂休眠避免忙等待
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task: ExecutionTask) -> None:
        """执行单个任务"""
        start_time = time.time()
        
        try:
            logger.info(f"Executing task: {task.task_id}")
            
            await self._emit_event('task_started', {
                'task_id': task.task_id,
                'tool_name': task.tool.name
            })
            
            # 执行工具
            if task.timeout:
                result = await asyncio.wait_for(
                    task.tool.execute(task.parameters, task.context),
                    timeout=task.timeout
                )
            else:
                result = await task.tool.execute(task.parameters, task.context)
            
            # 更新任务状态
            await self.queue.complete_task(task.task_id, result)
            
            # 更新统计信息
            execution_time = time.time() - start_time
            await self._update_stats(result.success, execution_time)
            
            # 存储已完成的任务
            if result.success:
                self.completed_tasks[task.task_id] = task
            else:
                self.failed_tasks[task.task_id] = task
            
            # 调用回调
            if task.callback:
                try:
                    await task.callback(result)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            logger.info(f"Task completed: {task.task_id}, success: {result.success}")
            
            await self._emit_event('task_completed', {
                'task_id': task.task_id,
                'tool_name': task.tool.name,
                'success': result.success,
                'execution_time': execution_time
            })
            
        except asyncio.TimeoutError:
            error_msg = f"Task timeout: {task.task_id}"
            logger.error(error_msg)
            
            # 创建超时结果
            result = ToolResult(
                tool_id=task.tool.tool_id,
                tool_type=task.tool.tool_type.value,
                status=ToolStatus.FAILED,
                success=False,
                start_time=task.started_at or get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                error_message="Task execution timeout",
                error_code="EXECUTION_TIMEOUT"
            )
            
            await self.queue.complete_task(task.task_id, result)
            self.failed_tasks[task.task_id] = task
            
            await self._update_stats(False, time.time() - start_time)
            
            await self._emit_event('task_timeout', {
                'task_id': task.task_id,
                'tool_name': task.tool.name
            })
            
        except asyncio.CancelledError:
            logger.info(f"Task cancelled: {task.task_id}")
            
            # 创建取消结果
            result = ToolResult(
                tool_id=task.tool.tool_id,
                tool_type=task.tool.tool_type.value,
                status=ToolStatus.CANCELLED,
                success=False,
                start_time=task.started_at or get_iso_timestamp(),
                end_time=get_iso_timestamp(),
                error_message="Task execution cancelled",
                error_code="EXECUTION_CANCELLED"
            )
            
            await self.queue.complete_task(task.task_id, result)
            
            await self._update_stats(False, time.time() - start_time)
            
            raise  # 重新抛出取消异常
            
        except Exception as e:
            error_msg = f"Task execution error: {task.task_id}, {str(e)}"
            logger.error(error_msg)
            
            # 检查是否需要重试
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = ToolStatus.PENDING
                task.started_at = None
                
                # 重新入队
                await self.queue.enqueue(task)
                
                logger.info(f"Task retry: {task.task_id}, attempt {task.retry_count}")
                
                await self._emit_event('task_retry', {
                    'task_id': task.task_id,
                    'tool_name': task.tool.name,
                    'retry_count': task.retry_count
                })
                
            else:
                # 创建失败结果
                result = ToolResult(
                    tool_id=task.tool.tool_id,
                    tool_type=task.tool.tool_type.value,
                    status=ToolStatus.FAILED,
                    success=False,
                    start_time=task.started_at or get_iso_timestamp(),
                    end_time=get_iso_timestamp(),
                    error_message=str(e),
                    error_code="EXECUTION_ERROR"
                )
                
                await self.queue.complete_task(task.task_id, result)
                self.failed_tasks[task.task_id] = task
                
                await self._update_stats(False, time.time() - start_time)
                
                await self._emit_event('task_failed', {
                    'task_id': task.task_id,
                    'tool_name': task.tool.name,
                    'error': str(e)
                })
    
    async def _cleanup_completed_tasks(self) -> None:
        """清理已完成的活动任务"""
        completed_task_ids = []
        
        for task_id, task in self.active_tasks.items():
            if task.done():
                completed_task_ids.append(task_id)
        
        for task_id in completed_task_ids:
            del self.active_tasks[task_id]
    
    async def _update_stats(self, success: bool, execution_time: float) -> None:
        """更新统计信息"""
        self.stats['total_executed'] += 1
        
        if success:
            self.stats['total_succeeded'] += 1
        else:
            self.stats['total_failed'] += 1
        
        self.stats['execution_times'].append(execution_time)
        
        # 计算平均执行时间
        if self.stats['execution_times']:
            self.stats['average_execution_time'] = sum(
                self.stats['execution_times']
            ) / len(self.stats['execution_times'])
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """发送事件"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")


class BatchExecutor:
    """批次执行器"""
    
    def __init__(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor
        self.logger = logger
    
    async def execute_sequential(
        self,
        tasks: List[Tuple[GUITool, ToolParameters, Optional[Dict[str, Any]]]],
        timeout: Optional[int] = None
    ) -> List[ToolResult]:
        """顺序执行任务"""
        results = []
        
        for tool, parameters, context in tasks:
            try:
                task_id = await self.tool_executor.submit_task(
                    tool, parameters, context, timeout=timeout
                )
                
                # 等待任务完成
                while True:
                    result = await self.tool_executor.get_task_result(task_id)
                    if result:
                        results.append(result)
                        break
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in sequential execution: {e}")
                # 创建错误结果
                error_result = ToolResult(
                    tool_id=tool.tool_id,
                    tool_type=tool.tool_type.value,
                    status=ToolStatus.FAILED,
                    success=False,
                    start_time=get_iso_timestamp(),
                    end_time=get_iso_timestamp(),
                    error_message=str(e),
                    error_code="BATCH_EXECUTION_ERROR"
                )
                results.append(error_result)
        
        return results
    
    async def execute_parallel(
        self,
        tasks: List[Tuple[GUITool, ToolParameters, Optional[Dict[str, Any]]]],
        max_concurrent: int = 5,
        timeout: Optional[int] = None
    ) -> List[ToolResult]:
        """并行执行任务"""
        task_ids = []
        
        # 提交所有任务
        for tool, parameters, context in tasks:
            try:
                task_id = await self.tool_executor.submit_task(
                    tool, parameters, context, timeout=timeout
                )
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Error submitting parallel task: {e}")
                task_ids.append(None)
        
        # 等待所有任务完成
        results = []
        for task_id in task_ids:
            if task_id:
                while True:
                    result = await self.tool_executor.get_task_result(task_id)
                    if result:
                        results.append(result)
                        break
                    await asyncio.sleep(0.1)
            else:
                # 创建错误结果
                error_result = ToolResult(
                    tool_id="unknown",
                    tool_type="unknown",
                    status=ToolStatus.FAILED,
                    success=False,
                    start_time=get_iso_timestamp(),
                    end_time=get_iso_timestamp(),
                    error_message="Failed to submit task",
                    error_code="TASK_SUBMISSION_ERROR"
                )
                results.append(error_result)
        
        return results