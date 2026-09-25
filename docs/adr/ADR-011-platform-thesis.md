# ADR-011: Platform Thesis — Artifact-Producing Agent With Governance And Continual Learning

**Status clarification (2026-09-24): historical platform thesis, retained as research rationale.**
The decision, consequences, and alternatives below describe intended capabilities. Exact rollback,
formally bounded drift, tenant isolation by construction, and a unique continual-learning contribution
are not established guarantees or demonstrated results. Implemented mechanisms provide per-update
slot limits, tokens and linked ledger records, plus optional Fisher-weighted snapshot checks; the
CPU oracle checks scalar arithmetic rather than independently verifying tensor transitions.
See [research status and publication scope](../research-status.md) for current enforcement gaps and
negative learning evidence. Continual learning through specialist tool use remains the objective.

## Context

The project's value proposition needs to be explicit, because subsequent architecture decisions (substrate scope, specialist roster, evaluation strategy, ethics scope, training cadence) all depend on what the system is for. Earlier ADRs describe *how* the system is built; this one fixes *what* it is.

## Historical decision (research targets)

VECL-QB is a continual-learning agent platform whose value is producing *artifacts* — netlists, GDS-II layouts, BAM alignments, STEP CAD files, GeoTIFFs, Parquet tables, rendered PDFs, terraform plans, etc. — that frontier models structurally cannot produce on their own. The platform is governed by chain-hashed provenance, bounded parameter drift, exact rollback, tenant isolation, and an immutable ethics layer.

Three properties define the differentiation:

1. **Artifacts, not text.** Frontier models describe how a Yosys netlist or AlphaFold prediction would be produced; this platform produces the file itself by orchestrating specialist tools.
2. **Sequential tool-chain orchestration.** Multi-tool DAGs (e.g., SystemVerilog → Yosys → OpenROAD → Verilator → timing report) execute with per-step ethics gating, output-as-input dependencies, and provenance chaining across the whole DAG.
3. **Operational properties opaque frontier models cannot offer.** Per-state-change audit reconstruction, exact replay rollback, formally bounded parameter drift (ADR-013), tenant isolation by construction, per-tool trust calibration over outcome data.

The positioning is vertical-agent *platform*, not single-vertical agent and not frontier-model-competitor. The specialist roster spans hardware EDA, geospatial, bioinformatics, CAD/CAM, legal/financial, scientific computing, 3D/graphics, network/infrastructure, statistical/ML, and document/research domains. Stockfish, AlphaFold, and TimesFM are first-wave canonical specialists. The full roster is a long-tail target; the platform is built to make adding a specialist mechanical, not heroic.

## Historical consequences

- A specialist plugin platform (base classes `SubprocessSpecialist`, `LibrarySpecialist`, `ServiceSpecialist`; declarative tool registry; standard claim schemas per artifact type) is required before specialist breadth. Wiring tools by hand is a different product than mechanically-added tools.
- The QB orchestrator evolves from parallel claim collection to DAG-based chain execution. Sequential tool chains become a first-class concept.
- The EthicsKernel stops being a research property and becomes a launch blocker the moment destructive specialists (Terraform `apply`, Ansible playbooks, `kubectl delete`, FFmpeg with file overwrite) enter the roster.
- The EpisodicStore becomes load-bearing for cross-chain memory and pattern reuse at this specialist count. It is not deferred to future-work.
- Artifact storage with content hashing enters the ledger schema. Every produced artifact gets a content-addressed entry; provenance survives across runs.
- The action plan revises from 10 phases to 13. The earlier 10-phase plan is superseded.

## Historical rejected alternatives

- **Single-vertical agent** (legal-only, biotech-only, etc.) — narrower market and competes with funded vertical incumbents (Harvey, Recursion, Higharc, etc.).
- **Frontier-model competitor on raw capability** — uncompetitive scale; the project's resources cannot match Anthropic/OpenAI/Google/DeepSeek pretraining budgets.
- **Engineering tool-use harness with no learning component** — fails to differentiate from existing frameworks (LangChain, LangGraph, MCP server ecosystems). Continual learning of tool-use patterns is the unique technical contribution.
- **Governance-only product without artifact production** — useful but less defensible; the artifact-production claim is what makes the value pitch tangible.
