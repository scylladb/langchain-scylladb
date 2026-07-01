"""Top-level conftest: spins up a ScyllaDB container for the whole test session.

If SCYLLADB_HOST is already set in the environment the container is skipped and
the existing connection details are used instead.
"""

import os
import time

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

_SCYLLA_IMAGE = "scylladb/scylla:2026.2"
_SCYLLA_CMD = "--smp 1 --memory 1G --overprovisioned 1 --broadcast-rpc-address 127.0.0.1"


@pytest.fixture(scope="session", autouse=True)
def scylladb_container():
    """Session-scoped fixture that starts a ScyllaDB container.

    Sets SCYLLADB_HOST and SCYLLADB_PORT env vars so all sub-conftest files
    and fixtures pick up the container's coordinates.
    """
    if os.environ.get("SCYLLADB_HOST"):
        # An external ScyllaDB is already configured — nothing to do.
        yield
        return

    container = DockerContainer(_SCYLLA_IMAGE)
    container.with_command(_SCYLLA_CMD)
    container.with_exposed_ports(9042)
    container.waiting_for(LogMessageWaitStrategy("Starting listening for CQL clients").with_startup_timeout(120))

    container.start()
    try:
        time.sleep(5)  # let the CQL port fully settle

        os.environ["SCYLLADB_HOST"] = container.get_container_host_ip()
        os.environ["SCYLLADB_PORT"] = str(container.get_exposed_port(9042))

        yield
    finally:
        container.stop()
        os.environ.pop("SCYLLADB_HOST", None)
        os.environ.pop("SCYLLADB_PORT", None)
