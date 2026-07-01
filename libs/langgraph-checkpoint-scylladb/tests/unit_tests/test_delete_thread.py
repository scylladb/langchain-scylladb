"""Tests for delete_thread / adelete_thread."""

import pytest
from cassandra.cluster import Session
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint

from langgraph.checkpoint.scylladb import ScyllaDBSaver
from .conftest import KEYSPACE


def _count_checkpoints(session: Session, thread_id: str) -> int:
    rows = session.execute(
        f"SELECT COUNT(*) FROM {KEYSPACE}.checkpoints WHERE thread_id = %s",
        (thread_id,),
    )
    return rows.one().count


def _count_writes(session: Session, thread_id: str) -> int:
    # checkpoint_writes has a composite partition key; we need ALLOW FILTERING
    # for a thread-only scan — acceptable in tests.
    rows = session.execute(
        f"SELECT COUNT(*) FROM {KEYSPACE}.checkpoint_writes "
        f"WHERE thread_id = %s ALLOW FILTERING",
        (thread_id,),
    )
    return rows.one().count


def test_delete_thread(saver: ScyllaDBSaver, session: Session) -> None:
    thread_1 = "del-thread-1"
    thread_2 = "del-thread-2"

    chkpnt_1 = empty_checkpoint()
    chkpnt_2 = empty_checkpoint()
    cfg_1 = RunnableConfig(
        configurable=dict(thread_id=thread_1, checkpoint_ns="", checkpoint_id=chkpnt_1["id"])
    )
    cfg_2 = RunnableConfig(
        configurable=dict(thread_id=thread_2, checkpoint_ns="", checkpoint_id=chkpnt_2["id"])
    )
    meta: CheckpointMetadata = {"source": "input", "step": 1, "writes": {}}

    saver.put(cfg_1, chkpnt_1, meta, {})
    saver.put(cfg_2, chkpnt_2, meta, {})
    saver.put_writes(cfg_1, [("channel1", "value1")], "task1")
    saver.put_writes(cfg_2, [("channel2", "value2")], "task2")

    assert saver.get_tuple(cfg_1) is not None
    assert saver.get_tuple(cfg_2) is not None
    assert _count_checkpoints(session, thread_1) > 0
    assert _count_writes(session, thread_1) > 0

    saver.delete_thread(thread_1)

    assert saver.get_tuple(cfg_1) is None
    assert _count_checkpoints(session, thread_1) == 0
    assert _count_writes(session, thread_1) == 0

    # thread_2 unaffected
    assert saver.get_tuple(cfg_2) is not None
    assert _count_checkpoints(session, thread_2) > 0


@pytest.mark.asyncio
async def test_adelete_thread(saver: ScyllaDBSaver, session: Session) -> None:
    thread_1 = "adel-thread-1"
    thread_2 = "adel-thread-2"

    chkpnt_1 = empty_checkpoint()
    chkpnt_2 = empty_checkpoint()
    cfg_1 = RunnableConfig(
        configurable=dict(thread_id=thread_1, checkpoint_ns="", checkpoint_id=chkpnt_1["id"])
    )
    cfg_2 = RunnableConfig(
        configurable=dict(thread_id=thread_2, checkpoint_ns="", checkpoint_id=chkpnt_2["id"])
    )
    meta: CheckpointMetadata = {"source": "input", "step": 1, "writes": {}}

    await saver.aput(cfg_1, chkpnt_1, meta, {})
    await saver.aput(cfg_2, chkpnt_2, meta, {})
    await saver.aput_writes(cfg_1, [("channel1", "value1")], "task1")
    await saver.aput_writes(cfg_2, [("channel2", "value2")], "task2")

    assert await saver.aget_tuple(cfg_1) is not None
    assert await saver.aget_tuple(cfg_2) is not None

    await saver.adelete_thread(thread_1)

    assert await saver.aget_tuple(cfg_1) is None
    assert _count_checkpoints(session, thread_1) == 0
    assert _count_writes(session, thread_1) == 0

    # thread_2 unaffected
    assert await saver.aget_tuple(cfg_2) is not None
    assert _count_checkpoints(session, thread_2) > 0
