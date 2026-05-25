# ADR-004: No Memory Update Without Learning Event Token

## Context

Continual learning needs an auditable authorization boundary.

## Decision

Sparse update execution requires a prepared learning event token or equivalent non-empty event/provenance identifiers at the oracle boundary.

## Consequences

Updates are tied to tenant, batch, source set, policy versions, and provenance root.

## Rejected Alternatives

- Allow direct memory mutation from replay code.
- Reconstruct authorization after mutation.
