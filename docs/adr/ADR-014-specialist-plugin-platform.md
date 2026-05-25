# ADR-014: Declarative Specialist Plugin Platform

## Context

ADR-011 defines VECL-QB as an artifact-producing platform, not a hand-wired demo. Adding a specialist must be mechanical: declare the tool, trust anchor, task types, cost/latency hints, and class path; then the router and provenance system can use it.

## Decision

Specialists are implemented as Python classes but registered through declarative YAML/TOML config. The registry entry contains `id`, `class_path`, `task_types`, `trust_anchor_id`, `cost_hint`, `latency_hint`, and `version`.

Three base classes cover the initial execution shapes:

- `SubprocessSpecialist` for local CLI tools.
- `LibrarySpecialist` for importable Python callables.
- `ServiceSpecialist` for long-lived service connections.

Produced files are stored through a content-addressed artifact store. The ledger records `ARTIFACT_PRODUCED` events that carry the artifact record payload and parent event links.

## Consequences

- Adding a specialist does not require editing router code.
- Registry config becomes reviewable governance surface.
- Artifact outputs get stable hashes and paths before real tool-chain execution arrives in Phase 4.
- Python class loading remains simple and local; remote plugin loading is intentionally not supported.

## Rejected Alternatives

- Pure Python registration only — easy for tests, but poor as an operator-facing registry.
- Dynamic remote plugins — too much supply-chain risk for the current provenance model.
- One universal specialist base class — hides important differences between subprocess, library, and service execution.
