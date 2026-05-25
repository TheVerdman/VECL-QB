# Refinement Bridge

This document maps abstract TLA+ state to production state for engineers who know either formal specs or production systems.

## State Mapping

- TLA `memoryValues` maps to production `memory_values`.
- TLA `memoryScores` maps to sparse `scores`.
- TLA `selectedSlots` maps to `selected_slots`.
- TLA `provenanceLog` maps to an append-only ledger projection.
- TLA `activePolicies` maps to `policy_version` and `trust_policy_version`.
- TLA `sourceTrust` maps to `TrustAnchorRegistry` plus `BoundedTrustUpdater`.
- TLA sleep-cycle state maps to `SleepCycleReport`.

## Refinement Idea

For every concrete production learning event, there exists a sequence of abstract transitions:

```text
PrepareLearningEvent
-> ComputeScores
-> SelectTopSlots
-> ApplyMemoryUpdate
-> CommitLearningEvent
```

The production implementation may have more bookkeeping, hashes, timestamps, and audit fields, but it must preserve the abstract transition order and invariants.

## Invariant Coverage

- TLC checks abstract transition invariants and counterexamples.
- Unit tests check concrete edge cases and data validation.
- Property tests check broad input spaces and architectural failure modes.
- Runtime monitor checks the concrete update before commit.
- Release gate checks promoted checkpoints against invariants, regressions, adversarial simulations, rollback, verification calibration, and tenant isolation.
