"""Interrupt-related integration tests for ScyllaDBSaver.

Verify that put_writes / aput_writes handle __interrupt__ channels correctly:
- writing twice with the same task_id is idempotent (upsert, not append)
- the stored write is retrievable via get_tuple / aget_tuple

Mirrors langgraph-checkpoint-mongodb unit_tests/test_interrupt.py.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Interrupt

from langgraph.checkpoint.scylladb import ScyllaDBSaver


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------


def test_put_writes_on_interrupt(saver: ScyllaDBSaver) -> None:
    """Writing an __interrupt__ write twice with the same task_id is idempotent."""
    from langgraph.checkpoint.base import empty_checkpoint, CheckpointMetadata

    chkpnt = empty_checkpoint()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "interrupt-thread-sync-1",
            "checkpoint_ns": "",
            "checkpoint_id": chkpnt["id"],
        }
    }

    saver.put(
        config,
        chkpnt,
        CheckpointMetadata(source="input", step=0, writes={}),
        {},
    )

    task_id = "task-interrupt-sync"
    task_path = "~__pregel_pull,human_feedback"

    interrupt_val = Interrupt(value="please provide input")
    writes1 = [("__interrupt__", (interrupt_val,))]
    saver.put_writes(config, writes1, task_id, task_path)

    # Second write with the same task_id: CQL upsert must replace, not duplicate
    interrupt_val2 = Interrupt(value="please provide another input")
    writes2 = [("__interrupt__", (interrupt_val2,))]
    saver.put_writes(config, writes2, task_id, task_path)

    tup = saver.get_tuple(config)
    assert tup is not None

    interrupt_writes = [w for w in tup.pending_writes if w[1] == "__interrupt__"]
    # Upsert semantics: only one row per (task_id, idx)
    assert len(interrupt_writes) == 1
    # The second write wins
    stored_interrupts = interrupt_writes[0][2]
    assert stored_interrupts[0].value == "please provide another input"


def test_put_writes_multiple_channels(saver: ScyllaDBSaver) -> None:
    """put_writes stores each channel as a separate row, all retrievable."""
    from langgraph.checkpoint.base import empty_checkpoint, CheckpointMetadata

    chkpnt = empty_checkpoint()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "interrupt-thread-sync-2",
            "checkpoint_ns": "",
            "checkpoint_id": chkpnt["id"],
        }
    }
    saver.put(config, chkpnt, CheckpointMetadata(source="input", step=0, writes={}), {})

    saver.put_writes(
        config,
        [("channel_a", "value_a"), ("channel_b", 42), ("channel_c", {"nested": True})],
        "task-multi",
    )

    tup = saver.get_tuple(config)
    assert tup is not None
    channels = {w[1]: w[2] for w in tup.pending_writes}
    assert channels["channel_a"] == "value_a"
    assert channels["channel_b"] == 42
    assert channels["channel_c"] == {"nested": True}


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aput_writes_on_interrupt(saver: ScyllaDBSaver) -> None:
    """Async: writing an __interrupt__ write twice with the same task_id is idempotent."""
    from langgraph.checkpoint.base import empty_checkpoint, CheckpointMetadata

    chkpnt = empty_checkpoint()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "interrupt-thread-async-1",
            "checkpoint_ns": "",
            "checkpoint_id": chkpnt["id"],
        }
    }

    await saver.aput(
        config,
        chkpnt,
        CheckpointMetadata(source="input", step=0, writes={}),
        {},
    )

    task_id = "task-interrupt-async"
    task_path = "~__pregel_pull,human_feedback"

    interrupt_val = Interrupt(value="please provide input")
    await saver.aput_writes(config, [("__interrupt__", (interrupt_val,))], task_id, task_path)

    interrupt_val2 = Interrupt(value="please provide another input")
    await saver.aput_writes(config, [("__interrupt__", (interrupt_val2,))], task_id, task_path)

    tup = await saver.aget_tuple(config)
    assert tup is not None

    interrupt_writes = [w for w in tup.pending_writes if w[1] == "__interrupt__"]
    assert len(interrupt_writes) == 1
    stored_interrupts = interrupt_writes[0][2]
    assert stored_interrupts[0].value == "please provide another input"


@pytest.mark.asyncio
async def test_aput_writes_multiple_channels(saver: ScyllaDBSaver) -> None:
    """aput_writes stores each channel as a separate row, all retrievable."""
    from langgraph.checkpoint.base import empty_checkpoint, CheckpointMetadata

    chkpnt = empty_checkpoint()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "interrupt-thread-async-2",
            "checkpoint_ns": "",
            "checkpoint_id": chkpnt["id"],
        }
    }
    await saver.aput(config, chkpnt, CheckpointMetadata(source="input", step=0, writes={}), {})

    await saver.aput_writes(
        config,
        [("ch_x", "hello"), ("ch_y", [1, 2, 3])],
        "task-amulti",
    )

    tup = await saver.aget_tuple(config)
    assert tup is not None
    channels = {w[1]: w[2] for w in tup.pending_writes}
    assert channels["ch_x"] == "hello"
    assert channels["ch_y"] == [1, 2, 3]
