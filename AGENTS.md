# VECL-QB Agent Instructions

This repository is a Python-first prototype for provenance-first expert orchestration and bounded continual learning.

Coding rules:

- Prefer small pure functions. Keep mutation boundaries narrow and explicit.
- Every state mutation must be represented as an append-only provenance event.
- No memory update may run without a prepared learning event token or equivalent non-empty event/provenance identifiers at the oracle boundary.
- Tests must be deterministic. Sort sets before serializing or comparing, and use deterministic tie-breakers.
- All sparse update logic must be tested against the CPU oracle before any accelerator implementation is trusted.
- CUDA is data-plane only. CUDA must not implement trust, governance, provenance authority, policy decisions, or tenant isolation decisions.
- TLA+ specs are contracts for behavior. They are not code generators.
