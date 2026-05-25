# ADR-002: CPU Oracle Before CUDA

## Context

Sparse update correctness depends on score, eligibility, deterministic selection, update magnitude, and provenance records.

## Decision

The CPU oracle is the reference implementation before any CUDA kernels are trusted.

## Consequences

Accelerated paths must pass differential tests against the oracle.

## Rejected Alternatives

- Build CUDA first and infer semantics from kernels.
- Accept approximate accelerator behavior for governance-relevant state.
