# CUDA Plan (Not Implemented)

There is no CUDA implementation in this repository. The current Python
`sparse_update_cuda_like` function is an API scaffold that directly delegates to
the CPU oracle; it is not independent differential evidence.

CUDA is data-plane only. It may accelerate deterministic numeric operations, but it must not own trust, governance, provenance, policy, or tenant isolation authority.

## Planned Kernels

- Compute sparse scores: `score[i] = activation[i] * rarity[i] * authority[i]`.
- Compute eligibility flags from authority threshold, score threshold, and host-provided quarantine mask.
- Deterministic top-k selection helpers with tie-breaking by lower slot id.
- Masked authority-scaled update for selected slots.
- Delta buffer generation for host-side provenance records.

## CUDA Must Not Compute

- Source trust.
- Governance approval.
- Policy decisions.
- Provenance authority.
- Tenant isolation decisions.

## Host Verification Checklist

- Verify no NaN or Inf in CUDA outputs.
- Verify selected slots are a subset of eligible slots.
- Verify changed slots are a subset of selected slots.
- Verify changed slot count is bounded by `max_slots`.
- Verify every delta can be tied to a prepared learning event.
- Verify deterministic tie-breaking against the CPU oracle.
- Verify tenant and trust policy versions before commit.

Any future CUDA output must be independently and differentially tested against
the CPU oracle. Direct TLA+ to CUDA compilation would be a category error;
host-side checks decide whether a concrete run may commit.
