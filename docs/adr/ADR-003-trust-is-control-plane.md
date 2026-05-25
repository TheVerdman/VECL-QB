# ADR-003: Trust Is Control-Plane, Not GPU Data-Plane

## Context

Trust decisions require governance, credentials, policy versions, and provenance.

## Decision

CUDA may process numeric masks and updates, but it must not compute trust or policy authority.

## Consequences

Host code owns trust, provenance, tenant isolation, and commit decisions.

## Rejected Alternatives

- Let kernels decide eligibility from raw governance data.
- Move source trust updates into accelerator code.
