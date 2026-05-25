# ADR-008: Sleep-Cycle Learning Is Promoted Through Release Gates

## Context

Sleep-cycle consolidation can change memory after offline replay.

## Decision

Promoted checkpoints must pass invariant tests, regressions, adversarial simulation bounds, rollback checks, verification calibration, and tenant isolation checks.

## Consequences

Candidate memory states remain provisional until the release gate approves them.

## Rejected Alternatives

- Promote every successful sleep cycle immediately.
- Use only aggregate quality metrics without safety gates.
