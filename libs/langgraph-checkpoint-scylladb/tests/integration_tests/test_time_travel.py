"""Time-travel integration tests for ScyllaDBSaver.

These tests compile a real LangGraph graph with ScyllaDBSaver as the
checkpointer and verify that state can be rewound to an earlier checkpoint,
patched, and re-executed from that point forward.

They mirror langgraph-checkpoint-mongodb's unit_tests/test_time_travel.py.
"""

from __future__ import annotations

from collections.abc import Generator
from operator import add
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import StateSnapshot
from typing_extensions import NotRequired

from langgraph.checkpoint.scylladb import ScyllaDBSaver


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------


class ExpenseState(TypedDict):
    amount: NotRequired[int]
    version: NotRequired[int]
    approved: NotRequired[bool]
    messages: Annotated[list[str], add]


def add_expense_node(state: ExpenseState) -> dict[str, Any]:
    """Adds an expense and records a message."""
    return dict(amount=100, version=1, approved=False, messages=["Added new expense"])


def validate_expense_node(state: ExpenseState) -> dict[str, Any]:
    """Approves the expense only when the amount is exactly 200."""
    if state.get("amount") == 200:
        return dict(approved=True, messages=["expense approved"])
    return dict(approved=False, messages=["expense denied"])


def _build_expense_graph(checkpointer: ScyllaDBSaver) -> Any:
    workflow = StateGraph(ExpenseState)
    workflow.add_node("add_expense", add_expense_node)
    workflow.add_node("validate_expense", validate_expense_node)
    workflow.add_edge(START, "add_expense")
    workflow.add_edge("add_expense", "validate_expense")
    workflow.add_edge("validate_expense", END)
    return workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_time_travel_update_and_rerun(saver: ScyllaDBSaver) -> None:
    """Update state at an intermediate checkpoint to trigger a different outcome.

    1. Run the graph — expense amount=100 → not approved.
    2. Find the checkpoint right before validate_expense.
    3. Patch amount to 200 via update_state, re-run.
    4. Verify the expense is now approved.
    """
    config: RunnableConfig = dict(configurable=dict(thread_id="tt-expense-1"))
    initial_state: ExpenseState = dict(
        amount=0, version=0, approved=False, messages=["Initial state"]
    )

    graph = _build_expense_graph(saver)
    graph.invoke(input=initial_state, config=config)

    # Verify the default outcome: amount=100 does not pass validation
    final_state = graph.get_state(config=config)
    assert not final_state.values["approved"]

    # Find the checkpoint right after add_expense (i.e. validate_expense is next)
    checkpoints: list[StateSnapshot] = list(graph.get_state_history(config))
    target_checkpoint = next(
        (
            cp
            for cp in checkpoints
            if cp.metadata and cp.metadata.get("step") == 1 and "validate_expense" in cp.next
        ),
        None,
    )
    assert target_checkpoint is not None, "Expected checkpoint with validate_expense as next node"

    # Patch the amount to 200 at that checkpoint
    updated_values = dict(**target_checkpoint.values)
    updated_values["amount"] = 200
    updated_values["version"] = updated_values.get("version", 0) + 1
    updated_values["messages"] = list(updated_values.get("messages", [])) + ["Updated state"]

    updated_config = graph.update_state(
        config=target_checkpoint.config,
        values=updated_values,
        as_node="add_expense",
    )

    # Continue execution from the patched checkpoint
    final_step = None
    for step in graph.stream(None, updated_config):
        final_step = step

    assert final_step is not None
    assert final_step["validate_expense"]["approved"]

    # Full state should reflect amount=200
    final = graph.get_state(updated_config)
    assert final.values["amount"] == 200


def test_time_travel_history_length(saver: ScyllaDBSaver) -> None:
    """After a full run the checkpoint history contains at least one entry per node."""
    config: RunnableConfig = dict(configurable=dict(thread_id="tt-history-1"))
    graph = _build_expense_graph(saver)
    graph.invoke(
        input=dict(amount=0, version=0, approved=False, messages=[]),
        config=config,
    )

    history: list[StateSnapshot] = list(graph.get_state_history(config))
    # Expect: START input + after add_expense + after validate_expense (+ possible empty)
    assert len(history) >= 2


def test_time_travel_rewound_state_matches_patch(saver: ScyllaDBSaver) -> None:
    """get_state at a historical checkpoint reflects the state at that point in time."""
    config: RunnableConfig = dict(configurable=dict(thread_id="tt-rewind-1"))
    graph = _build_expense_graph(saver)
    graph.invoke(
        input=dict(amount=0, version=0, approved=False, messages=["start"]),
        config=config,
    )

    history: list[StateSnapshot] = list(graph.get_state_history(config))
    pre_validate = next(
        (cp for cp in history if cp.metadata and "validate_expense" in cp.next),
        None,
    )
    assert pre_validate is not None

    # At this checkpoint amount should be 100 (set by add_expense)
    past_state = graph.get_state(pre_validate.config)
    assert past_state.values["amount"] == 100
