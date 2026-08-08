# ADR-024: Release Evaluation Harness

Status: amended by the committee-readiness fail-closed pass.

## Context

VECL-QB has accumulated real validation scripts for routing, tool-call payloads,
ethics, Terraform, TimesFM, persistence, and chain execution. Before Phase 9a,
those scripts were trusted manually: each printed a summary and returned a
process exit code, but ReleaseGate still accepted hand-supplied booleans.

## Decision

Introduce a small `vecl.evaluation` harness with evaluator results, metric
thresholds, release reports, and a `vecl release evaluate|approve|reject` CLI.
`release evaluate` requires an explicit manifest. `ScriptEvaluator` runs the
configured command, parses a named `*_SUMMARY` line, applies thresholds, and
records the evaluator identity/version, exact command, configuration hash, Git
revision and clean-tree state, plus summary/stdout/stderr hashes. Release reports
are content-hashed and revalidated when loaded and again at approval.

Approval fails closed unless every required gate category has passing,
non-synthetic evidence. The required categories are invariants, regression,
adversarial/ethics, rollback, verification calibration, and tenant isolation;
Fisher drift is required when supplied. In-process canned summaries are named
`DemoSummaryEvaluator`, are marked synthetic, and cannot authorize a release.

Release reports map evaluator categories onto the existing ReleaseGate fields:
routing/tool-call/chain results feed regression, ethics feeds adversarial,
rollback feeds rollback, tenant isolation feeds tenant isolation, and calibration
feeds verification calibration.

Checked-in manifests cover two execution modes:

- Frontier API release checks from the local Mac, using OpenAI or Anthropic
  drivers against routing/synthesis and tool-call-payload evals.
- A Vertex A100 Gemma evaluation matrix. Its current aggregate adapter consumes
  in-process summaries and is therefore deliberately unable to write an approval
  event until those adapters emit the same reproducibility evidence as
  `ScriptEvaluator`.

The checked-in Frontier and Vertex manifests cover model-facing regression and
adversarial checks, but do not yet cover every required gate category. They can
produce diagnostic reports; as checked in, they cannot approve a candidate.

## Consequences

- Existing scripts do not need to be rewritten to participate in release gates.
- Deterministic demo evaluators remain available for tests and UI examples, but
  cannot cross the production approval boundary.
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
  discipline and could incur unapproved cost.
- Keep manual booleans only — preserves the old gap where passing evidence is
  not directly tied to release approval.
