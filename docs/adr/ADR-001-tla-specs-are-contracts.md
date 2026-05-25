# ADR-001: TLA+ Specs Are Contracts, Not Implementation Generators

## Context

VECL-QB uses TLA+ to describe abstract state transitions and invariants.

## Decision

TLA+ specs are behavioral contracts. Production implementations refine them through tests, monitor checks, and trace comparison.

## Consequences

Engineers can reason about safety without coupling implementation language to the spec language.

## Rejected Alternatives

- Generate production Python or CUDA directly from TLA+.
- Treat model-checking artifacts as deployment artifacts.
