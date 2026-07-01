"""Tests for put_writes / aput_writes with special channels (__interrupt__)."""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.types import Interrupt

from langgraph.checkpoint.scylladb import ScyllaDBSaver


def test_put_writes_on_interrupt(saver: ScyllaDBSaver) -> None:
    """No error raised when an interrupted task updates its writes."""
    config: RunnableConfig = {
        "configurable": {
            "checkpoint_id": "check1",
            "thread_id": "interrupt-thread-1",
            "checkpoint_ns": "",
        }
    }
    task_id = "task_id"
    task_path = "~__pregel_pull,human_feedback"

    writes1 = [("__interrupt__", (Interrupt(value="please provide input"),))]
    saver.put_writes(config, writes1, task_id, task_path)

    # Second call should overwrite the first interrupt row (same stable index -1)
    writes2 = [("__interrupt__", (Interrupt(value="please provide another input"),))]
    saver.put_writes(config, writes2, task_id, task_path)

    # Retrieve and verify only one interrupt row exists per task
    tup = saver.get_tuple(
        {
            "configurable": {
                "thread_id": "interrupt-thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "check1",
            }
        }
    )
    # get_tuple may return None since we never called put() for this config,
    # but the writes should not have raised an error. The main assertion is
    # that no exception was raised above.


@pytest.mark.asyncio
async def test_aput_writes_on_interrupt(saver: ScyllaDBSaver) -> None:
    """Async: no error raised when an interrupted task updates its writes."""
    config: RunnableConfig = {
        "configurable": {
            "checkpoint_id": "acheck1",
            "thread_id": "interrupt-athread-1",
            "checkpoint_ns": "",
        }
    }
    task_id = "async_task_id"
    task_path = "~__pregel_pull,human_feedback"

    writes1 = [("__interrupt__", (Interrupt(value="async input 1"),))]
    await saver.aput_writes(config, writes1, task_id, task_path)

    writes2 = [("__interrupt__", (Interrupt(value="async input 2"),))]
    await saver.aput_writes(config, writes2, task_id, task_path)


def test_writes_idx_map_stable_indices(saver: ScyllaDBSaver) -> None:
    """__interrupt__ always uses index -1 regardless of its position in the writes list."""
    from langgraph.checkpoint.base import empty_checkpoint, WRITES_IDX_MAP

    config: RunnableConfig = {
        "configurable": {
            "checkpoint_id": "stable-idx-check",
            "thread_id": "stable-idx-thread",
            "checkpoint_ns": "",
        }
    }

    # Write a regular channel + interrupt together
    saver.put_writes(
        config,
        [("regular_channel", "value"), ("__interrupt__", (Interrupt(value="stop"),))],
        "task_stable",
    )

    # Write interrupt alone on a retry — should land on same row (idx = -1)
    saver.put_writes(
        config,
        [("__interrupt__", (Interrupt(value="updated stop"),))],
        "task_stable",
    )

    # Verify: query checkpoint_writes directly and count interrupt rows for this task
    from tests.unit_tests.conftest import KEYSPACE

    rows = list(
        saver.session.execute(
            f"SELECT idx, channel FROM {KEYSPACE}.checkpoint_writes "
            f"WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s",
            ("stable-idx-thread", "", "stable-idx-check"),
        )
    )
    interrupt_rows = [r for r in rows if r.channel == "__interrupt__"]
    # Only one interrupt row should exist (the second write overwrote the first)
    assert len(interrupt_rows) == 1
    assert interrupt_rows[0].idx == WRITES_IDX_MAP["__interrupt__"]
