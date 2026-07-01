"""Sync tests for ScyllaDBSaver: put / get_tuple / list / filter / pagination."""

from typing import Any

import pytest

from langgraph.checkpoint.scylladb import ScyllaDBSaver


def test_search(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """list() with various metadata filters returns correct checkpoints."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    # filter by single key
    results = list(saver.list(None, filter={"source": "input"}))
    assert len(results) == 1
    assert results[0].metadata["source"] == "input"

    # filter by multiple keys
    results = list(saver.list(None, filter={"step": 1, "writes": {"foo": "bar"}}))
    assert len(results) == 1
    assert results[0].metadata == input_data["metadata_2"]

    # no filter → all checkpoints
    results = list(saver.list(None, filter={}))
    assert len(results) == 3

    # filter with no match
    results = list(saver.list(None, filter={"source": "update", "step": 1}))
    assert len(results) == 0


def test_list_no_config_emits_warning(
    saver: ScyllaDBSaver, input_data: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """list(config=None) should log a warning about ALLOW FILTERING."""
    import logging

    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.scylladb.saver"):
        results = list(saver.list(None))
    assert len(results) >= 1
    assert any("ALLOW FILTERING" in r.message for r in caplog.records)


def test_list_by_thread_id(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """list() scoped to a thread_id returns only that thread's checkpoints."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
    assert len(results) == 2
    assert {r.config["configurable"]["checkpoint_ns"] for r in results} == {"", "inner"}


def test_get_tuple_latest(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """get_tuple without checkpoint_id returns the latest checkpoint."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})

    tup = saver.get_tuple({"configurable": {"thread_id": "thread-1"}})
    assert tup is not None
    assert tup.config["configurable"]["thread_id"] == "thread-1"
    assert tup.metadata["source"] == "input"


def test_get_tuple_by_id(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """get_tuple with an explicit checkpoint_id retrieves that specific checkpoint."""
    saved = saver.put(
        input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {}
    )
    checkpoint_id = saved["configurable"]["checkpoint_id"]

    tup = saver.get_tuple(
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


def test_get_tuple_missing(saver: ScyllaDBSaver) -> None:
    """get_tuple returns None when no matching checkpoint exists."""
    tup = saver.get_tuple({"configurable": {"thread_id": "nonexistent-thread"}})
    assert tup is None


def test_list_before_pagination(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """list(before=...) returns only checkpoints older than the given one."""
    from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
    from langchain_core.runnables import RunnableConfig

    base = empty_checkpoint()
    ck1 = base
    ck2 = create_checkpoint(ck1, {}, 1)
    ck3 = create_checkpoint(ck2, {}, 2)

    cfg = RunnableConfig(configurable=dict(thread_id="pagination-thread", checkpoint_ns=""))
    saved1 = saver.put(cfg, ck1, {}, {})
    saved2 = saver.put(
        {**cfg, "configurable": {**cfg["configurable"], "checkpoint_id": ck1["id"]}},
        ck2, {}, {},
    )
    saver.put(
        {**cfg, "configurable": {**cfg["configurable"], "checkpoint_id": ck2["id"]}},
        ck3, {}, {},
    )

    before_cfg = {"configurable": {"thread_id": "pagination-thread", "checkpoint_ns": "", "checkpoint_id": ck3["id"]}}
    results = list(saver.list(cfg, before=before_cfg))
    returned_ids = {r.config["configurable"]["checkpoint_id"] for r in results}
    assert ck3["id"] not in returned_ids


def test_list_limit(saver: ScyllaDBSaver, input_data: dict[str, Any]) -> None:
    """list(limit=N) returns at most N checkpoints."""
    saver.put(input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {})
    saver.put(input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {})
    saver.put(input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {})

    results = list(saver.list(None, limit=2))
    assert len(results) == 2


def test_put_returns_updated_config(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """put() returns a config that includes the stored checkpoint_id."""
    saved = saver.put(
        input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {}
    )
    assert saved["configurable"]["checkpoint_id"] == input_data["chkpnt_1"]["id"]


def test_put_writes_and_pending_writes(
    saver: ScyllaDBSaver, input_data: dict[str, Any]
) -> None:
    """put_writes stores channel values that appear in pending_writes on retrieval."""
    saved = saver.put(
        input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {}
    )
    write_config = {
        **saved,
        "configurable": {**saved["configurable"]},
    }
    saver.put_writes(write_config, [("my_channel", "hello")], "task-1")

    tup = saver.get_tuple(saved)
    assert tup is not None
    task_ids = {w[0] for w in tup.pending_writes}
    channels = {w[1] for w in tup.pending_writes}
    assert "task-1" in task_ids
    assert "my_channel" in channels
