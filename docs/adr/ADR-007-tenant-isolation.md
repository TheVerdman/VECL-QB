# ADR-007: Tenant Isolation By Construction

## Context

Replay, tokens, ledger queries, rollback, and QB requests all carry tenant identity.

## Decision

Tenant id is part of every mutation path and must match before state can move.

## Consequences

Cross-tenant replay and rollback are rejected.

## Rejected Alternatives

- Global replay batches with later filtering.
- Tenant separation only at the UI or API layer.
