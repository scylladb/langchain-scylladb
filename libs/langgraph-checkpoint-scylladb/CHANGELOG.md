# Changelog


## [0.1.0] - 2026-07-01

### Added

- `ScyllaDBSaver`: LangGraph `CheckpointSaver` backed by ScyllaDB, with full sync and async support.
- `ScyllaDBSaver.setup()` / `ascync` variant to create the required keyspace and tables.
- `ScyllaDBSaver.from_conn_string()` async context manager for connection-string-based setup.
- TTL support: pass `ttl` (seconds) to automatically expire checkpoints.
- Shard-aware routing via `scylla-driver`.

