"""Tests for custom serializer injection via serde= parameter."""

from typing import Any

from cassandra.cluster import Session
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.checkpoint.scylladb import ScyllaDBSaver
from .conftest import KEYSPACE, _make_cluster_and_session


class TrackingSerializer(JsonPlusSerializer):
    """Wraps JsonPlusSerializer and tracks call counts."""

    def __init__(self) -> None:
        super().__init__()
        self.dumps_count = 0
        self.loads_count = 0

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        self.dumps_count += 1
        return super().dumps_typed(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        self.loads_count += 1
        return super().loads_typed(data)


def test_custom_serde(input_data: dict[str, Any], session: Session) -> None:
    """ScyllaDBSaver uses the injected serde for all serialization."""
    custom_serde = TrackingSerializer()
    saver = ScyllaDBSaver(session, KEYSPACE, serde=custom_serde)
    saver.setup()

    put_config = saver.put(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )

    assert custom_serde.dumps_count > 0, "dumps_typed should have been called during put()"

    tup = saver.get_tuple(put_config)

    assert custom_serde.loads_count > 0, "loads_typed should have been called during get_tuple()"
    assert tup is not None
    assert tup.checkpoint == input_data["chkpnt_1"]
    assert tup.metadata["source"] == input_data["metadata_1"]["source"]

    # Clean up the extra rows this test introduced
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoints")
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoint_writes")
