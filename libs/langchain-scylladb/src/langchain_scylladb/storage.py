"""ScyllaDB-backed LangChain stores: ScyllaDBStore, ScyllaDBByteStore."""
from __future__ import annotations

import base64
import json
import logging
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.documents import Document
from langchain_core.stores import BaseStore, ByteStore

from cassandra.cluster import Session
from cassandra.query import BatchStatement, BatchType

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


class _ScyllaDBStoreMixin:
    """Internal mixin that provides CQL-backed mget/mset/mdelete/yield_keys.

    Concrete subclasses must set ``_session``, ``_keyspace``, and
    ``_table_name`` (done by calling ``_init_mixin``) and must implement
    ``_decode_value`` / ``_encode_value``.
    """

    def _init_mixin(
        self,
        session: Session,
        keyspace: str,
        table_name: str,
    ) -> None:
        self._session = session
        self._keyspace = keyspace
        self._table_name = table_name
        self._setup_schema()

    def _setup_schema(self) -> None:
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._table_name} (
                key    TEXT PRIMARY KEY,
                value  TEXT
            )
            """
        )

    def _decode_value(self, value: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _encode_value(self, value: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # BaseStore / ByteStore interface
    # ------------------------------------------------------------------

    def mget(self, keys: Sequence[str]) -> List[Optional[Any]]:
        if not keys:
            return []
        stmt = self._session.prepare(
            f"SELECT key, value FROM {self._keyspace}.{self._table_name} WHERE key = ?"
        )
        result_map: dict[str, Any] = {}
        for key in keys:
            row = self._session.execute(stmt, (key,)).one()
            if row is not None:
                result_map[key] = self._decode_value(row.value)
        return [result_map.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Any]]) -> None:
        if not key_value_pairs:
            return
        stmt = self._session.prepare(
            f"INSERT INTO {self._keyspace}.{self._table_name} (key, value) VALUES (?, ?)"
        )
        for i in range(0, len(key_value_pairs), _BATCH_SIZE):
            chunk = key_value_pairs[i : i + _BATCH_SIZE]
            batch = BatchStatement(batch_type=BatchType.UNLOGGED)
            for key, value in chunk:
                batch.add(stmt, (key, self._encode_value(value)))
            self._session.execute(batch)

    def mdelete(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        stmt = self._session.prepare(
            f"DELETE FROM {self._keyspace}.{self._table_name} WHERE key = ?"
        )
        for key in keys:
            self._session.execute(stmt, (key,))

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        rows = self._session.execute(
            f"SELECT key FROM {self._keyspace}.{self._table_name}"
        )
        for row in rows:
            if prefix is None or row.key.startswith(prefix):
                yield row.key


class ScyllaDBStore(_ScyllaDBStoreMixin, BaseStore):
    """ScyllaDB-backed store for arbitrary JSON-serialisable values.

    Suitable for use as the ``store`` argument to LangChain's
    ``CacheBackedEmbeddings`` and similar utilities.

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: ScyllaDB keyspace (must already exist).
        table_name: Table name within the keyspace.
    """

    def __init__(
        self,
        session: Session,
        *,
        keyspace: str = "langchain_scylladb",
        table_name: str = "store",
    ) -> None:
        self._init_mixin(session, keyspace, table_name)

    def _decode_value(self, value: str) -> Any:
        return json.loads(value)

    def _encode_value(self, value: Any) -> str:
        return json.dumps(value)


class ScyllaDBByteStore(_ScyllaDBStoreMixin, ByteStore):
    """ScyllaDB-backed byte store for binary blobs.

    Values are base64-encoded in the ``value`` TEXT column.

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: ScyllaDB keyspace (must already exist).
        table_name: Table name within the keyspace.
    """

    def __init__(
        self,
        session: Session,
        *,
        keyspace: str = "langchain_scylladb",
        table_name: str = "byte_store",
    ) -> None:
        self._init_mixin(session, keyspace, table_name)

    def _decode_value(self, value: str) -> bytes:
        return base64.b64decode(value)

    def _encode_value(self, value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")


class ScyllaDBDocStore(_ScyllaDBStoreMixin, BaseStore):
    """ScyllaDB-backed store for :class:`langchain_core.documents.Document` objects.

    Documents are serialised as JSON (``page_content`` + ``metadata``).

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: ScyllaDB keyspace (must already exist).
        table_name: Table name within the keyspace.
    """

    def __init__(
        self,
        session: Session,
        *,
        keyspace: str = "langchain_scylladb",
        table_name: str = "doc_store",
    ) -> None:
        self._init_mixin(session, keyspace, table_name)

    def _decode_value(self, value: str) -> Document:
        data = json.loads(value)
        return Document(page_content=data["page_content"], metadata=data.get("metadata", {}))

    def _encode_value(self, value: Document) -> str:
        return json.dumps({"page_content": value.page_content, "metadata": value.metadata})

