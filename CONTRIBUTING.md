# Contributing

## Repository structure

This is a monorepo. Each package lives in `libs/`:

- `libs/langchain-scylladb/` — LangChain VectorStore integration
- `libs/langgraph-checkpoint-scylladb/` — LangGraph checkpointer

## Development setup

Each package is developed independently. Navigate to the package you want to work on:

```bash
cd libs/langchain-scylladb
uv sync --all-groups
```

or

```bash
cd libs/langgraph-checkpoint-scylladb
uv sync --all-groups
```

## Running tests

```bash
uv run pytest tests/integration_tests/ -v
```

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes and add tests where appropriate
3. Ensure tests pass
4. Open a pull request
