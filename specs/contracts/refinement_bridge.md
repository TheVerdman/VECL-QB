# Planned Refinement Bridge

This document records a proposed mapping between a future executable TLA+ model
and production state. No such executable model or TLC configuration is currently
implemented in this repository.

## Intended State Mapping

- TLA `memoryValues` maps to production `memory_values`.
- TLA `memoryScores` maps to sparse `scores`.
- TLA `selectedSlots` maps to `selected_slots`.
- TLA `provenanceLog` maps to an append-only ledger projection.
- TLA `activePolicies` maps to `policy_version` and `trust_policy_version`.
- TLA `sourceTrust` maps to `TrustAnchorRegistry` plus `BoundedTrustUpdater`.
- TLA sleep-cycle state maps to `SleepCycleReport`.

## Intended Refinement Shape

For every concrete production learning event, there exists a sequence of abstract transitions:

```text
PrepareLearningEvent
-> ComputeScores
-> SelectTopSlots
-> ApplyMemoryUpdate
-> CommitLearningEvent
```

If executable models are added, production traces should preserve this abstract
transition order. Today, this ordering is checked by Python tests and runtime
validation rather than by TLC.

## Current Executable Coverage

- Unit tests check concrete edge cases and data validation.
- Property tests check broad input spaces and architectural failure modes.
- Runtime monitor checks the concrete update before commit.
- Provenance ledgers verify payload and chain integrity.
- The release gate requires complete, evidence-producing evaluator results before
  approval.

## Not Implemented

- TLA+ variables, initial states, next-state relations, or invariants.
- TLC model configuration and state-space exploration.
- CI model checking or trace-refinement comparison against a TLA+ model.
