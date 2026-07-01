"""Synchronous integration tests for ScyllaDBSaver low-level API.

These tests exercise put / get_tuple / list / put_writes directly, without
going through a compiled LangGraph graph.  They mirror the MongoDB
langgraph-checkpoint-mongodb unit_tests/test_sync.py but are adapted to
ScyllaDB/CQL semantics (no MQL injection concern, CQL WHERE clauses, etc.).
"""

from __future__ import annotations

from typing import Any

import pytest
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
        configurable=dict(thread_id="sync-thread-1", checkpoint_ns="")
    )
    inputs["config_2"] = RunnableConfig(
        configurable=dict(
            thread_id="sync-thread-2", checkpoint_id="2", checkpoint_ns=""
        )
    )
    inputs["config_3"] = RunnableConfig(
        configurable=dict(
            thread_id="sync-thread-2", checkpoint_id="2-inner", checkpoint_ns="inner"
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


def test_put_and_get_tuple(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """put() persists a checkpoint that get_tuple() can retrieve by id."""
    saved_config = saver.put(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )

    result = saver.get_tuple(saved_config)
    assert result is not None
    assert result.checkpoint["id"] == input_data["chkpnt_1"]["id"]
    assert result.metadata["source"] == "input"
    assert result.metadata["step"] == 2


def test_get_latest_when_no_checkpoint_id(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """get_tuple() with only thread_id returns the most-recently written checkpoint."""
    saver.put(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    second = create_checkpoint(input_data["chkpnt_1"], {}, 1)
    cfg2 = RunnableConfig(
        configurable=dict(
            thread_id="sync-thread-1",
            checkpoint_ns="",
            checkpoint_id=input_data["chkpnt_1"]["id"],
        )
    )
    saved = saver.put(cfg2, second, input_data["metadata_2"], {})

    # Ask for latest without a specific id
    latest_config = RunnableConfig(
        configurable=dict(thread_id="sync-thread-1", checkpoint_ns="")
    )
    tup = saver.get_tuple(latest_config)
    assert tup is not None
    assert tup.checkpoint["id"] == second["id"]


def test_put_writes_appear_as_pending(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """put_writes() associates pending writes with a checkpoint."""
    saved = saver.put(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    saver.put_writes(saved, [("my_channel", "hello"), ("other", 42)], "task-1")

    tup = saver.get_tuple(saved)
    assert tup is not None
    write_channels = {w[1] for w in tup.pending_writes}
    assert "my_channel" in write_channels
    assert "other" in write_channels


def test_list_filter_by_source(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """list() with filter={'source': 'input'} returns only matching checkpoints."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = list(saver.list(None, filter={"source": "input"}))
    assert len(results) == 1
    assert results[0].metadata["source"] == "input"


def test_list_filter_by_step(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """list() with filter={'step': 1} returns only matching checkpoints."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = list(saver.list(None, filter={"step": 1}))
    assert len(results) == 1
    assert results[0].metadata["step"] == 1


def test_list_empty_filter_returns_all(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """list() with an empty filter returns all stored checkpoints."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = list(saver.list(None, filter={}))
    assert len(results) == 3


def test_list_no_match_returns_empty(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """list() with a filter that matches nothing returns an empty list."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})

    results = list(saver.list(None, filter={"source": "update", "step": 1}))
    assert len(results) == 0


def test_list_filter_by_writes_key(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """list() can filter on a nested metadata key (writes.foo)."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})

    results = list(saver.list(None, filter={"step": 1, "writes": {"foo": "bar"}}))
    assert len(results) == 1
    assert results[0].metadata["writes"] == {"foo": "bar"}


def test_list_by_thread_crosses_namespaces(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """list() with only thread_id in config returns checkpoints across all namespaces."""
    # config_2 and config_3 both belong to 'sync-thread-2' but different namespaces
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    thread_config = RunnableConfig(configurable={"thread_id": "sync-thread-2"})
    results = list(saver.list(thread_config))
    assert len(results) == 2
    namespaces = {r.config["configurable"]["checkpoint_ns"] for r in results}
    assert namespaces == {"", "inner"}


def test_list_limit(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """list() with limit respects the upper bound on returned checkpoints."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = list(saver.list(None, limit=2))
    assert len(results) == 2


def test_list_before(saver: ScyllaDBSaver) -> None:
    """list() with before=<config> returns only checkpoints older than that id."""
    thread_id = "sync-before-test"
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
    saved2 = saver.put(cfg2, chk2, CheckpointMetadata(source="loop", step=1, writes={}), {})

    before_config = RunnableConfig(
        configurable=dict(
            thread_id=thread_id,
            checkpoint_ns=ns,
            checkpoint_id=saved2["configurable"]["checkpoint_id"],
        )
    )
    thread_cfg = RunnableConfig(
        configurable=dict(thread_id=thread_id, checkpoint_ns=ns)
    )
    results = list(saver.list(thread_cfg, before=before_config))
    # Only chk1 is before chk2
    assert len(results) == 1
    assert results[0].checkpoint["id"] == chk1["id"]


def test_get_tuple_returns_none_for_unknown_thread(saver: ScyllaDBSaver) -> None:
    """get_tuple() returns None when no checkpoint exists for the thread."""
    config = RunnableConfig(
        configurable=dict(thread_id="nonexistent-thread", checkpoint_ns="")
    )
    assert saver.get_tuple(config) is None


def test_put_writes_idempotent(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """Calling put_writes twice with the same task_id overwrites, not duplicates."""
    saved = saver.put(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    saver.put_writes(saved, [("ch", "v1")], "task-idem")
    saver.put_writes(saved, [("ch", "v2")], "task-idem")

    tup = saver.get_tuple(saved)
    assert tup is not None
    ch_writes = [w for w in tup.pending_writes if w[1] == "ch"]
    # CQL INSERT has upsert semantics: only the latest value survives
    assert len(ch_writes) == 1
    assert ch_writes[0][2] == "v2"
