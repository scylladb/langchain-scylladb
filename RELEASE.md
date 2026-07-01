# Release Process

This repository contains two independently versioned and published packages:

| Package | Location | PyPI |
|---------|----------|------|
| `langchain-scylladb` | `libs/langchain-scylladb/` | langchain-scylladb - not published yet |
| `langgraph-checkpoint-scylladb` | `libs/langgraph-checkpoint-scylladb/` | [pypi.org/project/langgraph-checkpoint-scylladb](https://pypi.org/project/langgraph-checkpoint-scylladb/) |

## CI

Workflow: `.github/workflows/ci.yml` — triggers on push to `main` and on pull requests. It runs tests for each package.

## Integration tests

Workflow: `.github/workflows/integration-tests.yml` — triggers on push to `main` and manually. It uses testcontainers.

## Releasing a package

Each package is released independently with a **prefixed tag**:

| Package | Tag format | Example |
|---------|-----------|---------|
| `langchain-scylladb` | `langchain-scylladb-MAJOR.MINOR.PATCH` | `langchain-scylladb-0.2.0` |
| `langgraph-checkpoint-scylladb` | `langgraph-checkpoint-scylladb-MAJOR.MINOR.PATCH` | `langgraph-checkpoint-scylladb-0.1.0` |

Steps (replace paths/versions as appropriate):

1. Bump `version` in the package's `pyproject.toml`.
2. Update `libs/<package>/CHANGELOG.md`.
3. Commit and push:
   ```
   git add libs/<package>/pyproject.toml libs/<package>/CHANGELOG.md
   git commit -m "chore: release <package> X.Y.Z"
   git push origin main
   ```
4. Tag and push:
   ```
   git tag <package>-X.Y.Z
   git push origin <package>-X.Y.Z
   ```

Pushing the tag triggers the publish workflow for that package only.

## Publish to PyPI

Workflow: `.github/workflows/publish.yml` — triggered by a prefixed tag or manually via workflow dispatch.

