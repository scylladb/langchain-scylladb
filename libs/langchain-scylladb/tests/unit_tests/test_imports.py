from langchain_scylladb import (
    ScyllaDBByteStore,
    ScyllaDBCache,
    ScyllaDBChatMessageHistory,
    ScyllaDBSemanticCache,
    ScyllaDBStore,
    ScyllaDBVectorStore,
    __all__,
    create_secondary_index,
    create_vector_index,
    drop_vector_index,
    list_indexes,
    wait_for_index,
)

_EXPECTED_ALL = [
    "ScyllaDBVectorStore",
    "ScyllaDBChatMessageHistory",
    "ScyllaDBCache",
    "ScyllaDBSemanticCache",
    "ScyllaDBStore",
    "ScyllaDBByteStore",
    "ScyllaDBDocStore",
    "create_vector_index",
    "drop_vector_index",
    "create_secondary_index",
    "wait_for_index",
    "list_indexes",
]


def test_all_exports_present() -> None:
    assert sorted(_EXPECTED_ALL) == sorted(__all__)


def test_all_symbols_importable() -> None:
    assert ScyllaDBVectorStore is not None
    assert ScyllaDBChatMessageHistory is not None
    assert ScyllaDBCache is not None
    assert ScyllaDBSemanticCache is not None
    assert ScyllaDBStore is not None
    assert ScyllaDBByteStore is not None
