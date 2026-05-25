# ADR-006: Provenance Ledger Is Append-Only

## Context

Auditability requires historical events to remain visible.

## Decision

The ledger does not delete or rewrite events. Quarantine, rollback, approval, and rejection are represented as new events.

## Consequences

Chain hashes can detect event order or payload changes.

## Rejected Alternatives

- Delete bad evidence.
- Mutate old events to reflect current status.
