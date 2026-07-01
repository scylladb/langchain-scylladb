from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT, Session
from cassandra.policies import DCAwareRoundRobinPolicy
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    WRITES_IDX_MAP,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CQL DDL
# ---------------------------------------------------------------------------

_CREATE_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS {keyspace}.checkpoints (
    thread_id         text,
    checkpoint_ns     text,
    checkpoint_id     text,
    parent_checkpoint_id text,
    checkpoint_type   text,
    checkpoint        blob,
    metadata_type     text,
    metadata          blob,
    source            text,
    step              int,
    PRIMARY KEY ((thread_id), checkpoint_ns, checkpoint_id)
) WITH CLUSTERING ORDER BY (checkpoint_ns ASC, checkpoint_id DESC)
"""

_CREATE_WRITES = """
CREATE TABLE IF NOT EXISTS {keyspace}.checkpoint_writes (
    thread_id     text,
    checkpoint_ns text,
    checkpoint_id text,
    task_id       text,
    idx           int,
    channel       text,
    type          text,
    value         blob,
    task_path     text,
    PRIMARY KEY ((thread_id, checkpoint_ns, checkpoint_id), task_id, idx)
)
"""

# ---------------------------------------------------------------------------
# Asyncio bridge for cassandra ResponseFuture
# ---------------------------------------------------------------------------

async def _aexecute(session: Session, query: Any, params: tuple = ()) -> Any:
    """Bridge cassandra execute_async() to asyncio."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def on_success(rows: Any) -> None:
        loop.call_soon_threadsafe(fut.set_result, rows)

    def on_error(exc: BaseException) -> None:
        loop.call_soon_threadsafe(fut.set_exception, exc)

    response = session.execute_async(query, params)
    response.add_callbacks(on_success, on_error)
    return await fut


# ---------------------------------------------------------------------------
# Saver
# ---------------------------------------------------------------------------

class ScyllaDBSaver(BaseCheckpointSaver):
    """LangGraph checkpointer backed by ScyllaDB.

    Uses prepared statements throughout.

    Example — sync usage::

        cluster = Cluster(
            contact_points=["node-0.example.scylla.cloud"],
            auth_provider=PlainTextAuthProvider("scylla", "secret"),
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="AWS_US_EAST_1"),
        )
        session = cluster.connect()
        saver = ScyllaDBSaver(session, keyspace="langgraph")
        saver.setup()

    Example — async usage via from_conn_string::

        async with ScyllaDBSaver.from_conn_string(
            "scylladb://scylla:secret@host/keyspace?dc=AWS_US_EAST_1"
        ) as saver:
            ...
    """

    session: Session
    keyspace: str

    def __init__(
        self,
        session: Session,
        keyspace: str,
        *,
        serde: Any = None,
        ttl: Optional[int] = None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.session = session
        self.keyspace = keyspace
        self.ttl = ttl
        self._ps: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create tables and prepare statements. Call once before use."""
        ks = self.keyspace
        self.session.execute(_CREATE_CHECKPOINTS.format(keyspace=ks))
        self.session.execute(_CREATE_WRITES.format(keyspace=ks))
        self._prepare_statements()

    async def asetup(self) -> None:
        """Async version of setup."""
        ks = self.keyspace
        await _aexecute(self.session, _CREATE_CHECKPOINTS.format(keyspace=ks))
        await _aexecute(self.session, _CREATE_WRITES.format(keyspace=ks))
        # _prepare_statements() calls session.prepare() which is blocking I/O;
        # run it in a thread pool so we don't block the event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._prepare_statements)

    def _prepare_statements(self) -> None:
        ks = self.keyspace
        ttl_clause = f" USING TTL {self.ttl}" if self.ttl is not None else ""
        self._ps["put"] = self.session.prepare(
            f"INSERT INTO {ks}.checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            "checkpoint_type, checkpoint, metadata_type, metadata, source, step) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?){ttl_clause}"
        )
        self._ps["put_write"] = self.session.prepare(
            f"INSERT INTO {ks}.checkpoint_writes "
            "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
            "channel, type, value, task_path) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?){ttl_clause}"
        )
        self._ps["get_by_id"] = self.session.prepare(
            f"SELECT * FROM {ks}.checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
        )
        # Get latest: DESC order on checkpoint_id means first row = latest
        self._ps["get_latest"] = self.session.prepare(
            f"SELECT * FROM {ks}.checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? LIMIT 1"
        )
        self._ps["list_all"] = self.session.prepare(
            f"SELECT * FROM {ks}.checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ?"
        )
        self._ps["list_all_by_thread"] = self.session.prepare(
            f"SELECT * FROM {ks}.checkpoints "
            "WHERE thread_id = ?"
        )
        self._ps["list_before"] = self.session.prepare(
            f"SELECT * FROM {ks}.checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id < ?"
        )
        self._ps["get_writes"] = self.session.prepare(
            f"SELECT channel, type, value, task_id, idx, task_path "
            f"FROM {ks}.checkpoint_writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
        )
        self._ps["select_checkpoint_refs"] = self.session.prepare(
            f"SELECT checkpoint_ns, checkpoint_id FROM {ks}.checkpoints "
            "WHERE thread_id = ?"
        )
        self._ps["delete_checkpoints"] = self.session.prepare(
            f"DELETE FROM {ks}.checkpoints WHERE thread_id = ?"
        )
        self._ps["delete_writes_for_checkpoint"] = self.session.prepare(
            f"DELETE FROM {ks}.checkpoint_writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_tuple(
        self, row: Any, pending_writes: list[tuple[str, str, Any]]
    ) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed((row.checkpoint_type, bytes(row.checkpoint)))
        metadata: CheckpointMetadata = self.serde.loads_typed(
            (row.metadata_type, bytes(row.metadata))
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": row.thread_id,
                "checkpoint_ns": row.checkpoint_ns,
                "checkpoint_id": row.checkpoint_id,
            }
        }
        parent_config: Optional[RunnableConfig] = None
        if row.parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _load_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        rows = self.session.execute(
            self._ps["get_writes"], (thread_id, checkpoint_ns, checkpoint_id)
        )
        writes = []
        for w in rows:
            value = self.serde.loads_typed((w.type, bytes(w.value)))
            writes.append((w.task_id, w.channel, value))
        return writes

    async def _aload_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        rows = await _aexecute(
            self.session, self._ps["get_writes"],
            (thread_id, checkpoint_ns, checkpoint_id),
        )
        writes = []
        for w in rows:
            value = self.serde.loads_typed((w.type, bytes(w.value)))
            writes.append((w.task_id, w.channel, value))
        return writes

    def _matches_filter(
        self, row: Any, filter: Optional[dict[str, Any]]
    ) -> bool:
        if not filter:
            return True
        # source and step are stored as dedicated columns for efficient access.
        # Other metadata keys require post-query filtering via deserialized metadata.
        fast_keys = {"source", "step"}
        metadata: Optional[dict] = None
        for k, v in filter.items():
            if k in fast_keys:
                if getattr(row, k, None) != v:
                    return False
            else:
                # Deserialize lazily and cache so multi-key filters only pay once.
                if metadata is None:
                    metadata = self.serde.loads_typed(
                        (row.metadata_type, bytes(row.metadata))
                    )
                if metadata.get(k) != v:
                    return False
        return True

    # ------------------------------------------------------------------
    # Sync methods
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: str = c.get("checkpoint_ns", "")
        parent_id: Optional[str] = get_checkpoint_id(config)

        cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
        metadata = get_checkpoint_metadata(config, metadata)
        meta_type, meta_bytes = self.serde.dumps_typed(metadata)

        source: str = metadata.get("source", "")
        step: int = metadata.get("step", -1)

        self.session.execute(
            self._ps["put"],
            (
                thread_id, checkpoint_ns, checkpoint["id"],
                parent_id,
                cp_type, cp_bytes,
                meta_type, meta_bytes,
                source, step,
            ),
        )
        return {
            **config,
            "configurable": {
                **c,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            },
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: str = c.get("checkpoint_ns", "")
        checkpoint_id: str = c["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            val_type, val_bytes = self.serde.dumps_typed(value)
            # INSERT … uses upsert semantics in CQL (idempotent).
            # WRITES_IDX_MAP assigns stable negative indices to special channels
            # (e.g. __interrupt__ → -1) so retried tasks always overwrite the
            # same row rather than accumulating duplicate entries.
            self.session.execute(
                self._ps["put_write"],
                (
                    thread_id, checkpoint_ns, checkpoint_id,
                    task_id, WRITES_IDX_MAP.get(channel, idx),
                    channel, val_type, val_bytes, task_path,
                ),
            )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: str = c.get("checkpoint_ns", "")
        checkpoint_id: Optional[str] = c.get("checkpoint_id") or get_checkpoint_id(config)

        if checkpoint_id:
            rows = self.session.execute(
                self._ps["get_by_id"], (thread_id, checkpoint_ns, checkpoint_id)
            )
        else:
            rows = self.session.execute(
                self._ps["get_latest"], (thread_id, checkpoint_ns)
            )

        row = next(iter(rows), None)
        if row is None:
            return None

        pending_writes = self._load_writes(thread_id, checkpoint_ns, row.checkpoint_id)
        return self._row_to_tuple(row, pending_writes)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            # Full-table scan: requires ALLOW FILTERING.
            # Use only for admin/debug purposes; always pass a config with thread_id
            # in production.
            logger.warning(
                "list() called with config=None: performing a full-table scan "
                "(ALLOW FILTERING). This is expensive on large datasets. "
                "Pass a config with thread_id for efficient queries."
            )
            ks = self.keyspace
            rows = self.session.execute(
                f"SELECT * FROM {ks}.checkpoints ALLOW FILTERING"
            )
            yielded = 0
            for row in rows:
                if limit is not None and yielded >= limit:
                    break
                if not self._matches_filter(row, filter):
                    continue
                pending_writes = self._load_writes(
                    row.thread_id, row.checkpoint_ns, row.checkpoint_id
                )
                yield self._row_to_tuple(row, pending_writes)
                yielded += 1
            return

        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: Optional[str] = c.get("checkpoint_ns")

        before_id: Optional[str] = None
        if before is not None:
            before_id = get_checkpoint_id(before) or before.get("configurable", {}).get(
                "checkpoint_id"
            )

        if checkpoint_ns is None:
            # No namespace filter: return checkpoints across all namespaces for
            # this thread (e.g. list(config={"configurable": {"thread_id": ...}})).
            rows = self.session.execute(
                self._ps["list_all_by_thread"], (thread_id,)
            )
        elif before_id:
            rows = self.session.execute(
                self._ps["list_before"], (thread_id, checkpoint_ns, before_id)
            )
        else:
            rows = self.session.execute(
                self._ps["list_all"], (thread_id, checkpoint_ns)
            )

        yielded = 0
        for row in rows:
            if limit is not None and yielded >= limit:
                break
            if not self._matches_filter(row, filter):
                continue
            row_ns = row.checkpoint_ns if checkpoint_ns is None else checkpoint_ns
            pending_writes = self._load_writes(
                thread_id, row_ns, row.checkpoint_id
            )
            yield self._row_to_tuple(row, pending_writes)
            yielded += 1

    def delete_thread(self, thread_id: str) -> None:
        # Fetch (checkpoint_ns, checkpoint_id) pairs for this thread so we can
        # delete writes, which require the full composite partition key.
        cp_rows = list(
            self.session.execute(
                self._ps["select_checkpoint_refs"], (thread_id,)
            )
        )
        for r in cp_rows:
            self.session.execute(
                self._ps["delete_writes_for_checkpoint"],
                (thread_id, r.checkpoint_ns, r.checkpoint_id),
            )
        self.session.execute(self._ps["delete_checkpoints"], (thread_id,))

    def close(self) -> None:
        """Shut down the cluster connection (and all its sessions)."""
        self.session.cluster.shutdown()

    # ------------------------------------------------------------------
    # Async methods
    # ------------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: str = c.get("checkpoint_ns", "")
        parent_id: Optional[str] = get_checkpoint_id(config)

        cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
        metadata = get_checkpoint_metadata(config, metadata)
        meta_type, meta_bytes = self.serde.dumps_typed(metadata)

        source: str = metadata.get("source", "")
        step: int = metadata.get("step", -1)

        await _aexecute(
            self.session,
            self._ps["put"],
            (
                thread_id, checkpoint_ns, checkpoint["id"],
                parent_id,
                cp_type, cp_bytes,
                meta_type, meta_bytes,
                source, step,
            ),
        )
        return {
            **config,
            "configurable": {
                **c,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            },
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: str = c.get("checkpoint_ns", "")
        checkpoint_id: str = c["checkpoint_id"]

        coros = []
        for idx, (channel, value) in enumerate(writes):
            val_type, val_bytes = self.serde.dumps_typed(value)
            coros.append(
                _aexecute(
                    self.session,
                    self._ps["put_write"],
                    (
                        thread_id, checkpoint_ns, checkpoint_id,
                        task_id, WRITES_IDX_MAP.get(channel, idx),
                        channel, val_type, val_bytes, task_path,
                    ),
                )
            )
        await asyncio.gather(*coros)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: str = c.get("checkpoint_ns", "")
        checkpoint_id: Optional[str] = c.get("checkpoint_id") or get_checkpoint_id(config)

        if checkpoint_id:
            rows = await _aexecute(
                self.session, self._ps["get_by_id"],
                (thread_id, checkpoint_ns, checkpoint_id),
            )
        else:
            rows = await _aexecute(
                self.session, self._ps["get_latest"],
                (thread_id, checkpoint_ns),
            )

        row = next(iter(rows), None)
        if row is None:
            return None

        pending_writes = await self._aload_writes(
            thread_id, checkpoint_ns, row.checkpoint_id
        )
        return self._row_to_tuple(row, pending_writes)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            logger.warning(
                "alist() called with config=None: performing a full-table scan "
                "(ALLOW FILTERING). This is expensive on large datasets. "
                "Pass a config with thread_id for efficient queries."
            )
            ks = self.keyspace
            rows = await _aexecute(
                self.session,
                f"SELECT * FROM {ks}.checkpoints ALLOW FILTERING",
            )
            yielded = 0
            for row in rows:
                if limit is not None and yielded >= limit:
                    break
                if not self._matches_filter(row, filter):
                    continue
                pending_writes = await self._aload_writes(
                    row.thread_id, row.checkpoint_ns, row.checkpoint_id
                )
                yield self._row_to_tuple(row, pending_writes)
                yielded += 1
            return

        c = config["configurable"]
        thread_id: str = c["thread_id"]
        checkpoint_ns: Optional[str] = c.get("checkpoint_ns")

        before_id: Optional[str] = None
        if before is not None:
            before_id = get_checkpoint_id(before) or before.get("configurable", {}).get(
                "checkpoint_id"
            )

        if checkpoint_ns is None:
            rows = await _aexecute(
                self.session, self._ps["list_all_by_thread"], (thread_id,)
            )
        elif before_id:
            rows = await _aexecute(
                self.session, self._ps["list_before"],
                (thread_id, checkpoint_ns, before_id),
            )
        else:
            rows = await _aexecute(
                self.session, self._ps["list_all"],
                (thread_id, checkpoint_ns),
            )

        yielded = 0
        for row in rows:
            if limit is not None and yielded >= limit:
                break
            if not self._matches_filter(row, filter):
                continue
            row_ns = row.checkpoint_ns if checkpoint_ns is None else checkpoint_ns
            pending_writes = await self._aload_writes(
                thread_id, row_ns, row.checkpoint_id
            )
            yield self._row_to_tuple(row, pending_writes)
            yielded += 1

    async def adelete_thread(self, thread_id: str) -> None:
        cp_rows = list(
            await _aexecute(
                self.session, self._ps["select_checkpoint_refs"], (thread_id,)
            )
        )
        delete_write_coros = [
            _aexecute(
                self.session,
                self._ps["delete_writes_for_checkpoint"],
                (thread_id, r.checkpoint_ns, r.checkpoint_id),
            )
            for r in cp_rows
        ]
        if delete_write_coros:
            await asyncio.gather(*delete_write_coros)
        await _aexecute(
            self.session, self._ps["delete_checkpoints"], (thread_id,)
        )

    # ------------------------------------------------------------------
    # from_conn_string — idiomatic async entry point
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Any = None,
        ttl: Optional[int] = None,
    ) -> AsyncIterator["ScyllaDBSaver"]:
        """Async context manager.  conn_string format::

            scylladb://user:password@host:9042/keyspace?dc=DATACENTER_NAME

        The ``dc`` query parameter is required for ScyllaDB Cloud
        (DC-aware load balancing policy).
        """
        parsed = urlparse(conn_string)
        host = parsed.hostname or "localhost"
        port = parsed.port or 9042
        username = parsed.username
        password = parsed.password
        keyspace = parsed.path.lstrip("/") or "langgraph"
        qs = parse_qs(parsed.query)
        dc = qs.get("dc", [None])[0]

        profile = ExecutionProfile(
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=dc) if dc else None,
        )
        cluster = Cluster(
            contact_points=[host],
            port=port,
            auth_provider=(
                PlainTextAuthProvider(username=username, password=password)
                if username
                else None
            ),
            execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        )
        session = cluster.connect()
        # Ensure the keyspace exists.
        session.execute(f"CREATE KEYSPACE IF NOT EXISTS {keyspace}")
        session.set_keyspace(keyspace)
        saver = cls(session, keyspace, serde=serde, ttl=ttl)
        await saver.asetup()
        try:
            yield saver
        finally:
            cluster.shutdown()

    # ------------------------------------------------------------------
    # Convenience factory for sync usage
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        hosts: list[str],
        keyspace: str,
        username: str,
        password: str,
        dc: str,
        port: int = 9042,
        serde: Any = None,
        ttl: Optional[int] = None,
    ) -> "ScyllaDBSaver":
        """Create a saver with explicit config (sync).  Call .setup() afterwards."""
        profile = ExecutionProfile(
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=dc),
        )
        cluster = Cluster(
            contact_points=hosts,
            port=port,
            auth_provider=PlainTextAuthProvider(username=username, password=password),
            execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        )
        session = cluster.connect(keyspace)
        return cls(session, keyspace, serde=serde, ttl=ttl)



