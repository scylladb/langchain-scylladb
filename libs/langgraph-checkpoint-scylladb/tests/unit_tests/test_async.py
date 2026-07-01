"""Async tests for ScyllaDBSaver: aput / aget_tuple / alist / aput_writes."""

from typing import Any

import pytest

from langgraph.checkpoint.scylladb import ScyllaDBSaver


@pytest.mark.asyncio
async def test_asearch(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """alist() with various metadata filters returns correct checkpoints."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    # filter by single key
    results = [c async for c in saver.alist(None, filter={"source": "input"})]
    assert len(results) == 1
    assert results[0].metadata["source"] == "input"

    # filter by multiple keys
    results = [c async for c in saver.alist(None, filter={"step": 1, "writes": {"foo": "bar"}})]
    assert len(results) == 1
    assert results[0].metadata == input_data["metadata_2"]

    # no filter → all
    results = [c async for c in saver.alist(None, filter={})]
    assert len(results) == 3

    # no match
    results = [c async for c in saver.alist(None, filter={"source": "update", "step": 1})]
    assert len(results) == 0


@pytest.mark.asyncio
async def test_alist_by_thread_id(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() scoped to a thread_id returns only that thread's checkpoints."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = [
        c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
    ]
    assert len(results) == 2
    assert {r.config["configurable"]["checkpoint_ns"] for r in results} == {"", "inner"}


@pytest.mark.asyncio
async def test_aget_tuple_latest(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """aget_tuple without checkpoint_id returns the latest checkpoint."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})

    tup = await saver.aget_tuple({"configurable": {"thread_id": "thread-1"}})
    assert tup is not None
    assert tup.config["configurable"]["thread_id"] == "thread-1"
    assert tup.metadata["source"] == "input"


@pytest.mark.asyncio
async def test_aget_tuple_by_id(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """aget_tuple with an explicit checkpoint_id retrieves that specific checkpoint."""
    saved = await saver.aput(
        input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {}
    )
    checkpoint_id = saved["configurable"]["checkpoint_id"]

    tup = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }
    )
    assert tup is not None
    assert tup.config["configurable"]["checkpoint_id"] == checkpoint_id


@pytest.mark.asyncio
async def test_aget_tuple_missing(saver: ScyllaDBSaver) -> None:
    """aget_tuple returns None when no matching checkpoint exists."""
    tup = await saver.aget_tuple({"configurable": {"thread_id": "ghost-thread"}})
    assert tup is None


@pytest.mark.asyncio
async def test_aput_writes_and_pending_writes(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """aput_writes stores channel values visible in pending_writes on retrieval."""
    saved = await saver.aput(
        input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {}
    )
    write_config = {**saved, "configurable": {**saved["configurable"]}}
    await saver.aput_writes(write_config, [("chan_a", "value_a")], "async-task-1")

    tup = await saver.aget_tuple(saved)
    assert tup is not None
    channels = {w[1] for w in tup.pending_writes}
    assert "chan_a" in channels


@pytest.mark.asyncio
async def test_alist_limit(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """alist(limit=N) yields at most N checkpoints."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = [c async for c in saver.alist(None, limit=2)]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_alist_no_config(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """alist(config=None) returns all checkpoints via ALLOW FILTERING."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})

    results = [c async for c in saver.alist(None)]
    assert len(results) >= 2
