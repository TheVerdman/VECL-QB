# ADR-024: Release Evaluation Harness

## Context

VECL-QB has accumulated real validation scripts for routing, tool-call payloads,
ethics, Terraform, TimesFM, persistence, and chain execution. Before Phase 9a,
those scripts were trusted manually: each printed a summary and returned a
process exit code, but ReleaseGate still accepted hand-supplied booleans.

## Decision

Introduce a small `vecl.evaluation` harness with evaluator results, metric
thresholds, release reports, and a `vecl release evaluate|approve|reject` CLI.
Default evaluators are local summary evaluators so `make all` remains CPU-only
and dependency-light. Real eval scripts can be attached through a manifest using
`ScriptEvaluator`, which runs a command, parses a named `*_SUMMARY` line, and
applies thresholds.

Release reports map evaluator categories onto the existing ReleaseGate fields:
routing/tool-call/chain results feed regression, ethics feeds adversarial,
rollback feeds rollback, tenant isolation feeds tenant isolation, and calibration
feeds verification calibration.

Checked-in manifests cover two execution modes:

- Frontier API release checks from the local Mac, using OpenAI or Anthropic
  drivers against routing/synthesis and tool-call-payload evals.
- A Vertex A100 Gemma release check that runs the real script matrix inside one
  custom job, then writes an approval event if the report passes.

## Consequences

- Existing scripts do not need to be rewritten to participate in release gates.
- Default tests stay deterministic and do not require model/API/GPU access.
- Release approval events now carry the full structured report payload, not just
  opaque booleans.
- Frontier and Vertex release checks share the same report and threshold schema,
  so failures are comparable even when the execution backend differs.
- Phase 9b can add Fisher/drift evaluators as another evaluator category without
  changing the CLI shape.

## Rejected Alternatives

- Replace every eval script immediately — too much churn and risks breaking
  already-validated Vertex workflows.
- Make real GPU/API evals default — violates the project’s opt-in validation
  discipline.
- Keep manual booleans only — preserves the old gap where passing evidence is
  not directly tied to release approval.
