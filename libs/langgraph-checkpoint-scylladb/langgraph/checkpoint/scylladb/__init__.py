"""ScyllaDB checkpointer for LangGraph.

Uses scylla-driver with shard-aware routing.
"""

from langgraph.checkpoint.scylladb.saver import ScyllaDBSaver

__all__ = ["ScyllaDBSaver"]
