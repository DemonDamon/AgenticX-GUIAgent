import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent

# 加载.env文件
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"已加载环境变量文件: {env_path}")
except ImportError:
    print("未安装python-dotenv，跳过.env文件加载")

from agenticx.core.event import ActionCorrectionEvent, ReplanningRequiredEvent, TaskStartEvent, TaskEndEvent, Event
from agenticx.core.event_bus import EventBus
from core.info_pool import InfoPool
from agenticx.llms.base import BaseLLMProvider
from agents.manager_agent import ManagerAgent
from agents.action_reflector_agent import ActionReflectorAgent
from agents.executor_agent import ExecutorAgent

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.fixture
def info_pool(event_bus):
    return InfoPool(event_bus=event_bus)

@pytest.fixture
def llm_provider():
    llm = MagicMock(spec=BaseLLMProvider)
    llm.acreate_completion = AsyncMock()
    return llm

@pytest.fixture
def manager_agent(event_bus, info_pool, llm_provider):
    agent = ManagerAgent(
        agent_id="manager_agent",
        info_pool=info_pool,
        llm_provider=llm_provider,
    )
    agent._plan_task = AsyncMock()
    agent._coordinate_agents = AsyncMock()
    return agent

@pytest.fixture
def action_reflector_agent(event_bus, info_pool, llm_provider):
    return ActionReflectorAgent(
        agent_id="action_reflector_agent",
        info_pool=info_pool,
        llm_provider=llm_provider,
    )

@pytest.fixture
def executor_agent(event_bus, info_pool):
    agent = ExecutorAgent(
        agent_id="executor_agent",
        info_pool=info_pool,
    )
    agent.execute_action = AsyncMock()
    return agent

@pytest.mark.asyncio
async def test_correction_and_replanning_flow(
    event_bus,
    info_pool,
    manager_agent,
    action_reflector_agent,
    executor_agent,
    llm_provider,
):
    # 1. Start the system
    await info_pool.start()
    await manager_agent.start()
    await action_reflector_agent.start()
    await executor_agent.start()

    task_id = "test_task_123"
    user_instruction = "发微信给张三"

    # Mock the task decomposition, planning and coordination methods
    sub_tasks = [{"agent": "executor_agent", "instruction": "打开微信"}]
    plan = {"steps": sub_tasks, "plan_id": "test_plan_123"}
    
    # Mock all the methods that involve LLM calls
    manager_agent._decompose_task = AsyncMock(return_value={"subtasks": sub_tasks})
    manager_agent._plan_task = AsyncMock(return_value=plan)
    manager_agent._coordinate_agents = AsyncMock(return_value={"assigned_tasks": {"executor_agent": []}})


    # 2. ManagerAgent starts the task
    # We run this in the background, as it contains the main monitoring loop
    task = asyncio.create_task(manager_agent.execute_task(user_instruction, task_id=task_id))
    await asyncio.sleep(0.1)  # Allow events to propagate

    # 3. ExecutorAgent fails the first time
    failed_action = {"tool_name": "click", "parameters": {"selector": "#submit"}}
    
    # Simulate the executor failing and publishing the result via InfoPool
    await info_pool.publish("action_result", {
        "status": "failure",
        "error": "Element not found",
        "action": failed_action,
        "task_id": task_id,
    }, "executor_agent")


    # 4. ActionReflectorAgent analyzes and decides on a tactical correction
    corrected_action = {"tool_name": "click", "parameters": {"selector": "button[type='submit']"}}
    llm_provider.acreate_completion.return_value = {
        "type": "tactical",
        "corrected_action": corrected_action,
    }

    # Let the reflector process the failure and publish a correction
    await asyncio.sleep(0.2)

    # 5. ExecutorAgent fails again, reflector decides to replan
    # Simulate the executor failing with the corrected action via InfoPool
    await info_pool.publish("action_result", {
        "status": "failure",
        "error": "Still cannot find the element",
        "action": corrected_action,
        "task_id": task_id,
    }, "executor_agent")

    llm_provider.acreate_completion.return_value = {
        "type": "strategic",
        "reason": "The UI seems to have changed fundamentally.",
    }

    # Let the reflector process the second failure and request replanning
    await asyncio.sleep(0.2)
    
    # Simulate ActionReflectorAgent publishing a replanning_required event via InfoPool
    await info_pool.publish("task_status", {
        "type": "replanning_required",
        "reason": "The UI seems to have changed fundamentally.",
        "task_id": task_id,
    }, "action_reflector_agent")
    
    await asyncio.sleep(0.1)  # Allow time for event processing

    # 6. ManagerAgent should receive a replanning event and trigger replanning
    # The execute_task should complete with a "replanning_required" status
    result = await asyncio.wait_for(task, timeout=10)
    
    # 验证结果
    print(f"DEBUG: result = {result}")
    print(f"DEBUG: result.output = {result.output}")
    print(f"DEBUG: type(result.output) = {type(result.output)}")
    assert result is not None, "Task result should not be None"
    assert result.output is not None, "Task result output should not be None"
    assert result.output["status"] == "replanning_required"
    
    # Stop agents
    await manager_agent.stop()
    await action_reflector_agent.stop()
    await executor_agent.stop()
    await info_pool.stop()