"""Asynchronous integration tests for ScyllaDBSaver low-level API.

These tests exercise aput / aget_tuple / alist / aput_writes directly, without
going through a compiled LangGraph graph.  They mirror test_sync.py but use
the async methods.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.scylladb import ScyllaDBSaver


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def input_data() -> dict[str, Any]:
    """Common checkpoint data shared across tests in this module."""
    inputs: dict[str, Any] = {}

    inputs["config_1"] = RunnableConfig(
        configurable=dict(thread_id="async-thread-1", checkpoint_ns="")
    )
    inputs["config_2"] = RunnableConfig(
        configurable=dict(
            thread_id="async-thread-2", checkpoint_id="2", checkpoint_ns=""
        )
    )
    inputs["config_3"] = RunnableConfig(
        configurable=dict(
            thread_id="async-thread-2", checkpoint_id="2-inner", checkpoint_ns="inner"
        )
    )

    inputs["chkpnt_1"] = empty_checkpoint()
    inputs["chkpnt_2"] = create_checkpoint(inputs["chkpnt_1"], {}, 1)
    inputs["chkpnt_3"] = empty_checkpoint()

    inputs["metadata_1"] = CheckpointMetadata(source="input", step=2, writes={}, score=1)
    inputs["metadata_2"] = CheckpointMetadata(
        source="loop", step=1, writes={"foo": "bar"}, score=None
    )
    inputs["metadata_3"] = CheckpointMetadata()

    return inputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aput_and_aget_tuple(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """aput() persists a checkpoint that aget_tuple() can retrieve by id."""
    saved_config = await saver.aput(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )

    result = await saver.aget_tuple(saved_config)
    assert result is not None
    assert result.checkpoint["id"] == input_data["chkpnt_1"]["id"]
    assert result.metadata["source"] == "input"
    assert result.metadata["step"] == 2


@pytest.mark.asyncio
async def test_aget_latest_when_no_checkpoint_id(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """aget_tuple() with only thread_id returns the most-recently written checkpoint."""
    await saver.aput(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    second = create_checkpoint(input_data["chkpnt_1"], {}, 1)
    cfg2 = RunnableConfig(
        configurable=dict(
            thread_id="async-thread-1",
            checkpoint_ns="",
            checkpoint_id=input_data["chkpnt_1"]["id"],
        )
    )
    await saver.aput(cfg2, second, input_data["metadata_2"], {})

    latest_config = RunnableConfig(
        configurable=dict(thread_id="async-thread-1", checkpoint_ns="")
    )
    tup = await saver.aget_tuple(latest_config)
    assert tup is not None
    assert tup.checkpoint["id"] == second["id"]


@pytest.mark.asyncio
async def test_aput_writes_appear_as_pending(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """aput_writes() associates pending writes with a checkpoint."""
    saved = await saver.aput(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    await saver.aput_writes(saved, [("my_channel", "hello"), ("other", 42)], "task-1")

    tup = await saver.aget_tuple(saved)
    assert tup is not None
    write_channels = {w[1] for w in tup.pending_writes}
    assert "my_channel" in write_channels
    assert "other" in write_channels


@pytest.mark.asyncio
async def test_alist_filter_by_source(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() with filter={'source': 'input'} returns only matching checkpoints."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = [c async for c in saver.alist(None, filter={"source": "input"})]
    assert len(results) == 1
    assert results[0].metadata["source"] == "input"


@pytest.mark.asyncio
async def test_alist_filter_by_step(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() with filter={'step': 1} returns only matching checkpoints."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = [c async for c in saver.alist(None, filter={"step": 1})]
    assert len(results) == 1
    assert results[0].metadata["step"] == 1


@pytest.mark.asyncio
async def test_alist_empty_filter_returns_all(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() with an empty filter returns all stored checkpoints."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = [c async for c in saver.alist(None, filter={})]
    assert len(results) == 3


@pytest.mark.asyncio
async def test_alist_no_match_returns_empty(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() with a filter that matches nothing returns an empty list."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})

    results = [c async for c in saver.alist(None, filter={"source": "update", "step": 1})]
    assert len(results) == 0


@pytest.mark.asyncio
async def test_alist_filter_by_writes_key(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() can filter on a nested metadata key (writes dict)."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})

    results = [
        c async for c in saver.alist(None, filter={"step": 1, "writes": {"foo": "bar"}})
    ]
    assert len(results) == 1
    assert results[0].metadata["writes"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_alist_by_thread_crosses_namespaces(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() with only thread_id in config returns checkpoints across all namespaces."""
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    thread_config = RunnableConfig(configurable={"thread_id": "async-thread-2"})
    results = [c async for c in saver.alist(thread_config)]
    assert len(results) == 2
    namespaces = {r.config["configurable"]["checkpoint_ns"] for r in results}
    assert namespaces == {"", "inner"}


@pytest.mark.asyncio
async def test_alist_limit(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """alist() with limit respects the upper bound on returned checkpoints."""
    await saver.aput(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    await saver.aput(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    await saver.aput(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = [c async for c in saver.alist(None, limit=2)]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_aget_tuple_returns_none_for_unknown_thread(saver: ScyllaDBSaver) -> None:
    """aget_tuple() returns None when no checkpoint exists for the thread."""
    config = RunnableConfig(
        configurable=dict(thread_id="nonexistent-async-thread", checkpoint_ns="")
    )
    assert await saver.aget_tuple(config) is None


@pytest.mark.asyncio
async def test_aput_writes_idempotent(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """Calling aput_writes twice with the same task_id overwrites, not duplicates."""
    saved = await saver.aput(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    await saver.aput_writes(saved, [("ch", "v1")], "task-idem")
    await saver.aput_writes(saved, [("ch", "v2")], "task-idem")

    tup = await saver.aget_tuple(saved)
    assert tup is not None
    ch_writes = [w for w in tup.pending_writes if w[1] == "ch"]
    # CQL INSERT has upsert semantics: only the latest value survives
    assert len(ch_writes) == 1
    assert ch_writes[0][2] == "v2"
