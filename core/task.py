"""任务管理模块

基于AgenticX Task组件提供任务定义、状态管理和执行跟踪功能。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

# 使用AgenticX的任务组件
from agenticx.core.task import Task as AgenticXTask
from agenticx.core.component import Component


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TaskType(Enum):
    """任务类型枚举"""
    EXPLORATION = "exploration"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    LEARNING = "learning"
    COORDINATION = "coordination"
    EVALUATION = "evaluation"


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Task:
    """任务定义"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    task_type: TaskType = TaskType.EXECUTION
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    
    # 任务参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 依赖关系
    dependencies: List[str] = field(default_factory=list)
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 执行信息
    assigned_agent: Optional[str] = None
    result: Optional[TaskResult] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self, agent_id: str) -> None:
        """开始执行任务"""
        self.status = TaskStatus.RUNNING
        self.assigned_agent = agent_id
        self.started_at = datetime.now()
    
    def complete(self, result: TaskResult) -> None:
        """完成任务"""
        self.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        self.result = result
        self.completed_at = datetime.now()
    
    def cancel(self) -> None:
        """取消任务"""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
    
    def pause(self) -> None:
        """暂停任务"""
        self.status = TaskStatus.PAUSED
    
    def resume(self) -> None:
        """恢复任务"""
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
    
    @property
    def duration(self) -> Optional[float]:
        """获取任务执行时长（秒）"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """检查任务是否已完成"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    @property
    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        return self.status == TaskStatus.RUNNING


class TaskManager(Component):
    """任务管理器 - 基于AgenticX Component"""
    
    def __init__(self):
        super().__init__(name="task_manager")
        self._tasks: Dict[str, Task] = {}
        self._task_queue: List[str] = []
    
    def create_task(
        self,
        name: str,
        description: str = "",
        task_type: TaskType = TaskType.EXECUTION,
        priority: TaskPriority = TaskPriority.NORMAL,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> Task:
        """创建新任务"""
        task = Task(
            name=name,
            description=description,
            task_type=task_type,
            priority=priority,
            parameters=parameters or {},
            dependencies=dependencies or []
        )
        
        self._tasks[task.id] = task
        self._add_to_queue(task.id)
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """根据状态获取任务列表"""
        return [task for task in self._tasks.values() if task.status == status]
    
    def get_tasks_by_agent(self, agent_id: str) -> List[Task]:
        """获取分配给特定智能体的任务"""
        return [task for task in self._tasks.values() if task.assigned_agent == agent_id]
    
    def get_next_task(self, agent_id: Optional[str] = None) -> Optional[Task]:
        """获取下一个待执行任务"""
        # 按优先级排序
        pending_tasks = self.get_tasks_by_status(TaskStatus.PENDING)
        if not pending_tasks:
            return None
        
        # 检查依赖关系
        available_tasks = []
        for task in pending_tasks:
            if self._are_dependencies_satisfied(task):
                available_tasks.append(task)
        
        if not available_tasks:
            return None
        
        # 按优先级排序
        available_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return available_tasks[0]
    
    def start_task(self, task_id: str, agent_id: str) -> bool:
        """开始执行任务"""
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        if not self._are_dependencies_satisfied(task):
            return False
        
        task.start(agent_id)
        return True
    
    def complete_task(self, task_id: str, result: TaskResult) -> bool:
        """完成任务"""
        task = self.get_task(task_id)
        if not task or not task.is_running:
            return False
        
        task.complete(result)
        self._remove_from_queue(task_id)
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.get_task(task_id)
        if not task or task.is_completed:
            return False
        
        task.cancel()
        self._remove_from_queue(task_id)
        return True
    
    def get_task_statistics(self) -> Dict[str, int]:
        """获取任务统计信息"""
        stats = {status.value: 0 for status in TaskStatus}
        for task in self._tasks.values():
            stats[task.status.value] += 1
        return stats
    
    def _add_to_queue(self, task_id: str) -> None:
        """添加任务到队列"""
        if task_id not in self._task_queue:
            self._task_queue.append(task_id)
    
    def _remove_from_queue(self, task_id: str) -> None:
        """从队列中移除任务"""
        if task_id in self._task_queue:
            self._task_queue.remove(task_id)
    
    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """检查任务依赖是否满足"""
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True