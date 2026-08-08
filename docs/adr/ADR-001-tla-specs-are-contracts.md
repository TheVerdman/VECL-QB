# ADR-001: TLA+ Is Planned Contract Work, Not Current Verification

Status: planned; no executable model is implemented in this repository.

## Context

VECL-QB has Markdown behavioral contracts and placeholder TLA+ modules. The
placeholder modules do not currently define variables, actions, invariants, or
TLC model configurations.

## Decision

Do not describe the current repository as TLA+-specified or TLC-verified. If
executable TLA+ models are added later, they will serve as behavioral contracts;
they will not generate Python or CUDA implementation code.

## Consequences

- Current guarantees come from executable Python tests, property tests, runtime
  monitor checks, and provenance-chain validation.
- `specs/tla/` remains an explicitly labeled future-work scaffold.
- Future model-checking claims require checked-in invariants, model
  configurations, commands, and reproducible CI evidence.

## Rejected Alternatives

- Generate production Python or CUDA directly from TLA+.
- Treat model-checking artifacts as deployment artifacts.
