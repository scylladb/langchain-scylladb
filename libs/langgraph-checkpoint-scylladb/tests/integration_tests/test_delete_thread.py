"""delete_thread integration tests for ScyllaDBSaver.

Verify that delete_thread / adelete_thread remove all checkpoints and writes
for the target thread while leaving other threads' data intact.

Mirrors langgraph-checkpoint-mongodb unit_tests/test_delete_thread.py.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint

from langgraph.checkpoint.scylladb import ScyllaDBSaver


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def test_delete_thread(saver: ScyllaDBSaver) -> None:
    """delete_thread removes all checkpoints and writes for the target thread
    while leaving a second thread's data intact."""
    chkpnt_1 = empty_checkpoint()
    thread_1_id = "del-thread-sync-1"
    config_1 = RunnableConfig(
        configurable=dict(
            thread_id=thread_1_id,
            checkpoint_ns="",
            checkpoint_id=chkpnt_1["id"],
        )
    )
    metadata_1 = CheckpointMetadata(source="input", step=1, writes={"foo": "bar"})

    chkpnt_2 = empty_checkpoint()
    thread_2_id = "del-thread-sync-2"
    config_2 = RunnableConfig(
        configurable=dict(
            thread_id=thread_2_id,
            checkpoint_ns="",
            checkpoint_id=chkpnt_2["id"],
        )
    )
    metadata_2 = CheckpointMetadata(source="output", step=1, writes={"baz": "qux"})

    # Save checkpoints and associated writes for both threads
    saver.put(config_1, chkpnt_1, metadata_1, {})
    saver.put(config_2, chkpnt_2, metadata_2, {})
    saver.put_writes(config_1, [("channel1", "value1")], "task1")
    saver.put_writes(config_2, [("channel2", "value2")], "task2")

    # Both threads must be present before deletion
    assert saver.get_tuple(config_1) is not None
    assert saver.get_tuple(config_2) is not None

    # Delete only thread 1
    saver.delete_thread(thread_1_id)

    # Thread 1 should be gone
    assert saver.get_tuple(config_1) is None
    remaining_1 = list(
        saver.list(RunnableConfig(configurable={"thread_id": thread_1_id}))
    )
    assert remaining_1 == []

    # Thread 2 must still be intact
    assert saver.get_tuple(config_2) is not None
    remaining_2 = list(
        saver.list(RunnableConfig(configurable={"thread_id": thread_2_id}))
    )
    assert len(remaining_2) == 1


def test_delete_thread_with_multiple_checkpoints(saver: ScyllaDBSaver) -> None:
    """delete_thread removes all checkpoints (not just the latest) for a thread."""
    from langgraph.checkpoint.base import create_checkpoint

    thread_id = "del-thread-sync-multi"
    ns = ""

    chk1 = empty_checkpoint()
    cfg1 = RunnableConfig(
        configurable=dict(thread_id=thread_id, checkpoint_ns=ns, checkpoint_id=chk1["id"])
    )
    saver.put(cfg1, chk1, CheckpointMetadata(source="input", step=0, writes={}), {})

    chk2 = create_checkpoint(chk1, {}, 1)
    cfg2 = RunnableConfig(
        configurable=dict(thread_id=thread_id, checkpoint_ns=ns, checkpoint_id=chk1["id"])
    )
    saver.put(cfg2, chk2, CheckpointMetadata(source="loop", step=1, writes={}), {})

    # Two checkpoints stored
    thread_cfg = RunnableConfig(configurable={"thread_id": thread_id, "checkpoint_ns": ns})
    assert len(list(saver.list(thread_cfg))) == 2

    saver.delete_thread(thread_id)

    assert list(saver.list(thread_cfg)) == []


def test_delete_nonexistent_thread_is_noop(saver: ScyllaDBSaver) -> None:
    """delete_thread on a thread that never existed does not raise."""
    saver.delete_thread("ghost-thread-that-never-existed")


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adelete_thread(saver: ScyllaDBSaver) -> None:
    """adelete_thread removes all checkpoints and writes for the target thread
    while leaving a second thread's data intact."""
    chkpnt_1 = empty_checkpoint()
    thread_1_id = "del-thread-async-1"
    config_1 = RunnableConfig(
        configurable=dict(
            thread_id=thread_1_id,
            checkpoint_ns="",
            checkpoint_id=chkpnt_1["id"],
        )
    )
    metadata_1 = CheckpointMetadata(source="input", step=1, writes={"foo": "bar"})

    chkpnt_2 = empty_checkpoint()
    thread_2_id = "del-thread-async-2"
    config_2 = RunnableConfig(
        configurable=dict(
            thread_id=thread_2_id,
            checkpoint_ns="",
            checkpoint_id=chkpnt_2["id"],
        )
    )
    metadata_2 = CheckpointMetadata(source="output", step=1, writes={"baz": "qux"})

    await saver.aput(config_1, chkpnt_1, metadata_1, {})
    await saver.aput(config_2, chkpnt_2, metadata_2, {})
    await saver.aput_writes(config_1, [("channel1", "value1")], "task1")
    await saver.aput_writes(config_2, [("channel2", "value2")], "task2")

    # Both threads present
    assert await saver.aget_tuple(config_1) is not None
    assert await saver.aget_tuple(config_2) is not None

    await saver.adelete_thread(thread_1_id)

    # Thread 1 gone
    assert await saver.aget_tuple(config_1) is None
    remaining_1 = [
        c async for c in saver.alist(RunnableConfig(configurable={"thread_id": thread_1_id}))
    ]
    assert remaining_1 == []

    # Thread 2 still intact
    assert await saver.aget_tuple(config_2) is not None
    remaining_2 = [
        c async for c in saver.alist(RunnableConfig(configurable={"thread_id": thread_2_id}))
    ]
    assert len(remaining_2) == 1


@pytest.mark.asyncio
async def test_adelete_nonexistent_thread_is_noop(saver: ScyllaDBSaver) -> None:
    """adelete_thread on a thread that never existed does not raise."""
    await saver.adelete_thread("async-ghost-thread-never-existed")
