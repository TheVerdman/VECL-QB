# ADR-005: Deterministic TopK Tie-Breaking

## Context

Equal scores can otherwise produce platform-dependent slot selection.

## Decision

TopK sorts by score descending, then slot id ascending.

## Consequences

Tests, traces, and rollbacks are reproducible.

## Rejected Alternatives

- Preserve input set iteration order.
- Use nondeterministic GPU atomics to resolve ties.
