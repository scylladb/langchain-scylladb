"""Integration tests for ScyllaDBCache."""
from __future__ import annotations

import pytest
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy
from langchain_core.outputs import Generation

from langchain_scylladb.cache import ScyllaDBCache


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
    yield s, info["keyspace"]
    s.execute(f"DROP TABLE IF EXISTS {info['keyspace']}.cache_test")
    cluster.shutdown()


def _make_cache(db_session, keyspace):
    return ScyllaDBCache(db_session, keyspace=keyspace, table_name="cache_test")


# ---------------------------------------------------------------------------
# Exact-match cache
# ---------------------------------------------------------------------------


def test_exact_cache_write_then_read(session) -> None:
    db_session, keyspace = session
    cache = _make_cache(db_session, keyspace)

    gens = [Generation(text="answer")]
    cache.update("prompt1", "llm1", gens)
    result = cache.lookup("prompt1", "llm1")

    assert result is not None
    assert result[0].text == "answer"


def test_exact_cache_different_llm_string_is_miss(session) -> None:
    db_session, keyspace = session
    cache = _make_cache(db_session, keyspace)

    cache.update("prompt2", "llm_a", [Generation(text="cached")])
    result = cache.lookup("prompt2", "llm_b")

    assert result is None


def test_exact_cache_clear_causes_miss(session) -> None:
    db_session, keyspace = session
    cache = _make_cache(db_session, keyspace)

    cache.update("prompt3", "llm1", [Generation(text="to clear")])
    cache.clear()
    result = cache.lookup("prompt3", "llm1")

    assert result is None



