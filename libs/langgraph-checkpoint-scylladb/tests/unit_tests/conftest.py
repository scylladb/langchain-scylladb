"""Shared fixtures for ScyllaDB checkpointer unit tests.

Requires a running Cassandra-compatible node. Configure via env vars:
  SCYLLADB_HOST      default: localhost
  SCYLLADB_PORT      default: 9042
  SCYLLADB_USERNAME  default: cassandra
  SCYLLADB_PASSWORD  default: cassandra
  SCYLLADB_KEYSPACE  default: langgraph_test
"""

import os
from collections.abc import Generator
from typing import Any

import pytest
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, Session
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.scylladb import ScyllaDBSaver

KEYSPACE = os.environ.get("SCYLLADB_KEYSPACE", "langgraph_test")


def _make_cluster_and_session() -> tuple[Cluster, Session]:
    """Create a Cluster and Session pointed at the test keyspace.

    Connection details are read from environment variables at call time so that
    the testcontainers fixture in tests/conftest.py can set them before the
    session-scoped ``session`` fixture runs.
    """
    host = os.environ.get("SCYLLADB_HOST", "localhost")
    port = int(os.environ.get("SCYLLADB_PORT", "9042"))
    username = os.environ.get("SCYLLADB_USERNAME", "cassandra")
    password = os.environ.get("SCYLLADB_PASSWORD", "cassandra")
    cluster = Cluster(
        contact_points=[host],
        port=port,
        auth_provider=PlainTextAuthProvider(username, password),
    )
    session = cluster.connect()
    session.execute(f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}")
    session.set_keyspace(KEYSPACE)
    return cluster, session


def _truncate_tables(session: Session) -> None:
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoints")
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoint_writes")


@pytest.fixture(scope="session")
def session() -> Generator[Session, None, None]:
    cluster, sess = _make_cluster_and_session()
    yield sess
    sess.execute(f"DROP KEYSPACE IF EXISTS {KEYSPACE}")
    sess.shutdown()
    cluster.shutdown()


@pytest.fixture
def saver(session: Session) -> Generator[ScyllaDBSaver, None, None]:
    s = ScyllaDBSaver(session, KEYSPACE)
    s.setup()
    yield s
    _truncate_tables(session)


@pytest.fixture(scope="session")
def input_data() -> dict[str, Any]:
    """Standard set of configs, checkpoints, and metadata used across tests."""
    inputs: dict[str, Any] = {}

    inputs["config_1"] = RunnableConfig(
        configurable=dict(thread_id="thread-1", checkpoint_ns="")
    )
    inputs["config_2"] = RunnableConfig(
        configurable=dict(thread_id="thread-2", checkpoint_id="2", checkpoint_ns="")
    )
    inputs["config_3"] = RunnableConfig(
        configurable=dict(
            thread_id="thread-2", checkpoint_id="2-inner", checkpoint_ns="inner"
        )
    )

    inputs["chkpnt_1"] = empty_checkpoint()
    inputs["chkpnt_2"] = create_checkpoint(inputs["chkpnt_1"], {}, 1)
    inputs["chkpnt_3"] = empty_checkpoint()

    inputs["metadata_1"] = CheckpointMetadata(
        source="input", step=2, writes={}, score=1
    )
    inputs["metadata_2"] = CheckpointMetadata(
        source="loop", step=1, writes={"foo": "bar"}, score=None
    )
    inputs["metadata_3"] = CheckpointMetadata()

    return inputs
