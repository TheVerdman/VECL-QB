# VECL-QB

VECL-QB is an early Python-first prototype for a provenance-first expert orchestration and bounded continual-learning architecture. The core idea is simple: state changes are not magic. They are prepared, authorized, checked against a CPU oracle, recorded as provenance events, and only then committed.

## What Is Implemented

- Sparse memory update types, deterministic TopK, and a CPU oracle.
- Learning event tokens and a sparse update monitor.
- Append-only provenance ledger with chain hashes.
- Trust anchors and bounded learned trust updates.
- A small provenance graph, QB specialist/router/verifier/synthesis flow, sleep replay and consolidation, rollback, traces, metrics, demos, and release gate scaffolding.
- Real Stockfish 18 specialist integration through UCI, with content-addressed session-log artifacts.
- Prompted Gemma 4 31B routing baseline for Stockfish, with fallback provenance and a bounded opt-in Vertex eval.
- DAG-based chain plans and a chain executor with per-step provenance, artifact threading, and abort-on-failure semantics.
- Optional CUDA-like API boundary that currently delegates to the CPU oracle.

## What Is Not Implemented

- Production storage, distributed execution, real model serving, real freight data integrations, and actual CUDA kernels.
- TLA+ specs are contracts, not code generators. Production code refines the contracts; it is not generated from them.
- This prototype demonstrates invariants and workflows. It is not a production security guarantee.

## Run

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
make all
brew install stockfish
.venv/bin/python examples/stockfish_qb/demo.py
.venv/bin/python examples/magellan_freight_qb/demo.py
.venv/bin/python examples/sleep_cycle_demo.py
```

## Sparse Update

Sparse updates compute `score[i] = activation[i] * rarity[i] * authority[i]`, apply eligibility thresholds, select deterministic TopK slots, then update only selected slots by `learning_rate * authority[i] * gradient[i]`. TopK ranks candidates; thresholds and authority-scaled updates suppress low-trust influence.

## Provenance

The ledger is append-only. Every event carries tenant, actor, timestamp, payload hash, parent IDs, and a chain hash. Quarantine and rollback are represented as new events rather than deletion.

## Trust

Trust roots are exogenous anchors. Learned trust may drift only within bounded limits and cannot self-promote from a source's own claims. Effective trust is anchored by root trust.

## QB Orchestration

QB routes a tenant request to specialists, collects typed claims, builds a claim graph, verifies support/conflict/root requirements, and synthesizes only when verification passes or marks the answer as needing review. For pre-built chain plans, QB executes specialist DAG steps in dependency order and records each step as provenance.

## Sleep Cycle

Sleep replay builds source-diverse, trust-capped batches, prepares a learning event token, computes sparse inputs, runs the monitor, and commits only after invariant verification.

## CUDA Boundary

CUDA is planned as data-plane acceleration only: scores, eligibility flags, deterministic selection helpers, masked authority-scaled updates, and delta buffers. Host code must verify all invariants before commit. Direct TLA+ to CUDA compilation is a category error; CUDA refines a contract through differential tests against the CPU oracle.

## Safety And Audit Principles

- No update without a learning event.
- No final answer without a claim graph.
- No trust without roots or bounded verification.
- No cross-tenant replay or rollback.
- No deletion from the provenance ledger.
