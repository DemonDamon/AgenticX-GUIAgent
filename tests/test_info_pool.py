import asyncio
import pytest
from typing import List, Dict, Any
from agenticx.core.event import Event
from agenticx.core.event_bus import EventBus
from core.info_pool import InfoPool, InfoType, InfoPriority, InfoEntry

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.fixture
def info_pool(event_bus):
    pool = InfoPool(event_bus=event_bus)
    asyncio.run(pool.start())
    yield pool
    asyncio.run(pool.stop())

@pytest.mark.asyncio
async def test_publish_and_subscribe(info_pool: InfoPool):
    received_events: List[Event] = []

    async def callback(event: Event):
        received_events.append(event)

    info_pool.subscribe(callback, [InfoType.TASK_STATUS])

    await info_pool.publish(
        info_type=InfoType.TASK_STATUS,
        data={"status": "running"},
        source_agent="test_agent"
    )

    await asyncio.sleep(0.1)  # Allow time for event propagation

    assert len(received_events) == 1
    event = received_events[0]
    assert event.type == InfoType.TASK_STATUS.value
    entry = InfoEntry.from_event(event)
    assert entry.data["status"] == "running"

@pytest.mark.asyncio
async def test_reply_and_conversation(info_pool: InfoPool):
    entry_id = await info_pool.publish(
        info_type=InfoType.TASK_PLAN,
        data={"plan": "step 1"},
        source_agent="manager"
    )

    await info_pool.reply_to(
        original_entry_id=entry_id,
        source_agent="executor",
        data={"status": "acknowledged"}
    )

    await asyncio.sleep(0.1)

    # The entry_id from the first publish IS the conversation_id
    conversation = info_pool.get_conversation(entry_id)
    print(f"Conversation entries: {conversation}")

    assert len(conversation) == 2
    assert conversation[0].id == entry_id
    assert conversation[1].reply_to_id == entry_id
    assert conversation[1].source_agent == "executor"

@pytest.mark.asyncio
async def test_get_stats(info_pool: InfoPool):
    await info_pool.publish(
        info_type=InfoType.AGENT_STATE,
        data={"state": "idle"},
        source_agent="agent1",
        conversation_id="conv1"
    )
    await info_pool.publish(
        info_type=InfoType.AGENT_STATE,
        data={"state": "busy"},
        source_agent="agent2",
        conversation_id="conv1"
    )

    stats = info_pool.get_stats()
    assert stats["total_entries"] == 2
    assert stats["entries_by_type"]["agent_state"] == 2
    assert stats["active_conversations"] == 1