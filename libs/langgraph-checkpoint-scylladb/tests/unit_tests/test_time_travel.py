"""Time-travel test: compile a StateGraph, run it, then replay from a past checkpoint."""

from collections.abc import Generator
from operator import add
from typing import Annotated, Any, TypedDict

import pytest
from cassandra.cluster import Session
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import StateSnapshot
from typing_extensions import NotRequired

from langgraph.checkpoint.scylladb import ScyllaDBSaver
from .conftest import KEYSPACE, _make_cluster_and_session


class ExpenseState(TypedDict):
    amount: NotRequired[int]
    version: NotRequired[int]
    approved: NotRequired[bool]
    messages: Annotated[list[str], add]


def add_expense_node(state: ExpenseState) -> dict[str, Any]:
    return dict(amount=100, version=1, approved=False, messages=["Added expense"])


def validate_expense_node(state: ExpenseState) -> dict[str, Any]:
    if state.get("amount", 0) == 200:
        return dict(approved=True, messages=["approved"])
    return dict(approved=False, messages=["denied"])


@pytest.fixture
def time_travel_saver(session: Session) -> Generator[ScyllaDBSaver, None, None]:
    saver = ScyllaDBSaver(session, KEYSPACE)
    saver.setup()
    yield saver
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoints")
    session.execute(f"TRUNCATE {KEYSPACE}.checkpoint_writes")


def test_time_travel(time_travel_saver: ScyllaDBSaver) -> None:
    """Run a graph, find an intermediate checkpoint, update state, and resume."""
    initial_state: ExpenseState = dict(
        amount=0, version=0, approved=False, messages=["Initial state"]
    )
    config: RunnableConfig = dict(
        configurable=dict(thread_id="test-time-travel")
    )

    workflow = StateGraph(ExpenseState)
    workflow.add_node("add_expense", add_expense_node)
    workflow.add_node("validate_expense", validate_expense_node)
    workflow.add_edge(START, "add_expense")
    workflow.add_edge("add_expense", "validate_expense")
    workflow.add_edge("validate_expense", END)
    graph = workflow.compile(checkpointer=time_travel_saver)

    graph.invoke(input=initial_state, config=config, stream_mode="checkpoints")  # type: ignore[call-overload]

    final_state = graph.get_state(config=config)
    # With amount=100, validate_expense denies
    assert not final_state.values["approved"]

    # Find checkpoint right after add_expense (before validate_expense)
    checkpoints: list[StateSnapshot] = list(graph.get_state_history(config))
    target_checkpoint = None
    for ckpt in checkpoints:
        if (
            ckpt.metadata
            and ckpt.metadata.get("step") == 1
            and "validate_expense" in ckpt.next
        ):
            target_checkpoint = ckpt
            break

    assert target_checkpoint is not None, "Expected a checkpoint before validate_expense"

    # Update the expense amount so validation will approve it
    past_state = graph.get_state(target_checkpoint.config)
    updated_values = dict(**past_state.values)
    updated_values["amount"] += 100   # 100 → 200 triggers approval
    updated_values["version"] += 1
    updated_values["messages"] = updated_values["messages"] + ["Updated state"]

    updated_config = graph.update_state(
        config=target_checkpoint.config,
        values=updated_values,
        as_node="add_expense",
    )

    # Resume from the updated checkpoint
    final_step = None
    for step in graph.stream(None, updated_config):
        final_step = step

    assert isinstance(final_step, dict)
    assert final_step["validate_expense"]["approved"]

    final_state = graph.get_state(updated_config)
    assert final_state.values["amount"] == 200
