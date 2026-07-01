"""Shared fixtures for ScyllaDB checkpointer integration tests.

Connection details are read from env vars set by the top-level
tests/conftest.py testcontainers fixture:
  SCYLLADB_HOST      default: localhost
  SCYLLADB_PORT      default: 9042
  SCYLLADB_USERNAME  default: cassandra
  SCYLLADB_PASSWORD  default: cassandra
  SCYLLADB_KEYSPACE  default: langgraph_integration_test
"""

import os
from collections.abc import Generator

import pytest
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, Session

from langgraph.checkpoint.scylladb import ScyllaDBSaver

KEYSPACE = os.environ.get("SCYLLADB_KEYSPACE", "langgraph_integration_test")


def _make_session() -> tuple[Cluster, Session]:
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


@pytest.fixture(scope="session")
def session() -> Generator[Session, None, None]:
    cluster, sess = _make_session()
    yield sess
    sess.execute(f"DROP KEYSPACE IF EXISTS {KEYSPACE}")
    sess.shutdown()
    cluster.shutdown()


@pytest.fixture
def saver(session: Session) -> Generator[ScyllaDBSaver, None, None]:
    s = ScyllaDBSaver(session, KEYSPACE)
    s.setup()
    yield s
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoints")
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoint_writes")
