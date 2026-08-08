# TLA+ status

This directory is a future-work scaffold. The three `.tla` files are deliberately
minimal placeholders: they contain no variables, actions, invariants, model
configuration, or TLC invocation.

Current safety evidence comes from the Python CPU oracle, property and integration
tests, runtime update verification, and provenance-chain integrity checks. See
[`../contracts/refinement_bridge.md`](../contracts/refinement_bridge.md) for the
intended—but not yet implemented—relationship to a future formal model.

Do not cite this directory as evidence of formal verification. A future change may
make that claim only after adding executable specifications, checked-in TLC
configuration, documented commands, and reproducible CI results.
