# ADR-019: Local Persistence Substrate

## Context

Phase 10a makes provenance, artifacts, and checkpoints durable before
side-effecting specialists such as Terraform. The project still needs default
tests to run without cloud credentials, services, GPUs, or model downloads.

## Decision

Use local persistence as the first durable substrate:

- `SqliteProvenanceLedger` stores append-only provenance events in SQLite with
  WAL mode enabled.
- Ledger order is the autoincrement `position`, not timestamps or UUIDs.
- The in-memory `ProvenanceLedger` remains the default for lightweight tests and
  callers that do not opt into persistence.
- `ContentAddressedStore` remains file-backed and verifies SHA-256 on restore.
- `DiskCheckpointStore` stores each checkpoint as a directory containing
  `memory_values.npy`, `metadata.json`, and `digest.sha256`.
- The intended local runtime root is repo-local `.vecl/`, ignored by git.

SQLite is used before Postgres, GCS, or S3 because it is embedded, inspectable,
deterministic in tests, and suitable for the local desktop/workbench path.
Cloud/object-store backends can be added later behind the same interfaces.

## Consequences

- Phase 8 Terraform plan artifacts can survive process restart and be verified
  by hash before review.
- Local development and CI remain service-free.
- WAL supports concurrent readers and crash-safe completed transactions, but
  Phase 10a assumes serialized writes from one ledger authority.
- Future schema changes should be handled by explicit migrations in
  `SqliteProvenanceLedger._ensure_schema`.

## Rejected Alternatives

- Postgres first — adds a service dependency before the local runtime needs it.
- GCS/S3 first — couples core persistence to a cloud provider too early.
- Timestamp ordering — unsafe for chain hashes because timestamps can collide
  and are not the append authority.
