"""Integration tests for ScyllaDB index utilities."""
from __future__ import annotations

import pytest
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy

from langchain_scylladb.index import (
    create_secondary_index,
    create_vector_index,
    drop_vector_index,
    list_indexes,
    wait_for_index,
)

_TABLE = "index_test"


@pytest.fixture()
def session(scylladb_service):
    info = scylladb_service
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(info["local_dc"])
    )
    cluster = Cluster(
        [info["host"]],
        port=info["port"],
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        protocol_version=4,
    )
    s = cluster.connect()
    ks = info["keyspace"]
    s.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ks}.{_TABLE} (
            id   TEXT PRIMARY KEY,
            tag  TEXT
        )
        """
    )
    yield s, ks
    s.execute(f"DROP TABLE IF EXISTS {ks}.{_TABLE}")
    cluster.shutdown()


def test_create_and_list_secondary_index(session) -> None:
    db_session, keyspace = session
    create_secondary_index(db_session, keyspace, _TABLE, "tag", index_name="tag_idx")

    indexes = list_indexes(db_session, keyspace, _TABLE)
    assert "tag_idx" in indexes


def test_wait_for_secondary_index(session) -> None:
    db_session, keyspace = session
    create_secondary_index(db_session, keyspace, _TABLE, "tag", index_name="wait_idx")
    # Should not raise — index was just created
    wait_for_index(db_session, keyspace, _TABLE, "wait_idx", timeout=30)



