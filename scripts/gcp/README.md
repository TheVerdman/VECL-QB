# GCP GPU Smoke Run

Use this after GPU quota is approved. Keep tokens in the VM environment or a VM-local `.env`; never commit them.

## Local Project

The current GCP project is:

```text
project-49b1b523-d248-434f-bd4
```

Default region and zone:

```text
us-central1
us-central1-a
```

## Approved GPU Path

The project currently has quota for one `A100 80GB` GPU in `us-central1`. H100 and B200 were requested but not approved for the first pass.

`scripts/smoke_gemma.py` accepts A100 and H100 CUDA devices by default. To override the check:

```bash
export GEMMA_ALLOWED_CUDA_DEVICES=A100
```

## First GPU Validation

For the approved Vertex A100 path, keep `HF_TOKEN` in your shell environment and run:

```bash
export HF_TOKEN=...
zsh scripts/gcp/submit_vertex_gemma_smoke.sh
```

The submit script packages `scripts/smoke_gemma.py`, launches one A100 80GB Vertex
custom training job, and streams logs. The temporary job YAML contains `HF_TOKEN`;
delete the printed `/tmp/vecl-qb-a100-smoke-*.yaml` file after launch and revoke the
token after the smoke run if it has been exposed.

If running manually on a GPU VM instead of Vertex:

```bash
nvidia-smi
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
make all
export HF_TOKEN=...
.venv/bin/python scripts/smoke_gemma.py
```

Expected result: `scripts/smoke_gemma.py` prints the model id, prompt, and a coherent generated response.

## Prompted Routing Eval

To run the bounded Phase 3b routing eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_ROUTING_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_routing_eval.sh
```

The job downloads the official Stockfish 18 Linux binary, loads the configured
Gemma routing model once, runs `tests/qb/fixtures/routing_eval_v0.json`, prints a
`ROUTING_EVAL_SUMMARY`, and exits nonzero if the 90/10 routing thresholds fail.

Phase 3b was validated on Vertex job `5601939983205138432` using
`google/gemma-4-31B-it` and the official Stockfish 18 Linux binary:

```json
{
  "chess_routing_rate": 1.0,
  "decision_events": 30,
  "fallback_events": 0,
  "model_id": "google/gemma-4-31B-it",
  "out_of_domain_stockfish_rate": 0.0,
  "total_entries": 30
}
```

This run proves the Phase 3b baseline used the prompted LLM router directly:
all 30 requests emitted `LLM_ROUTING_DECIDED`, and no request used the
rule-based fallback.

## Phase 4 Chain Eval

To run the Phase 4 chain eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_ROUTING_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_chain_eval.sh
```

This is distinct from the Phase 3b routing eval. It still uses Gemma 4 31B to
decide whether Stockfish applies, but then executes a pre-built Phase 4
`ChainPlan` through `QBOrchestrator.run_task(..., chain_plan=...)`:

```text
StockfishSpecialist -> StockfishFormatterSpecialist
```

The job exits nonzero unless `CHAIN_STARTED`, two `CHAIN_STEP_COMPLETED`, and
two `ARTIFACT_PRODUCED` events are recorded for every successful chess chain,
no chain aborts occur, and out-of-domain prompts do not trigger Stockfish.

## Phase 4b Chess Answer Eval

To run the Phase 4b answer eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_ROUTING_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_chess_answer_eval.sh
```

This extends the Phase 4 chain eval by asking natural multi-step chess questions,
running the Stockfish -> formatter chain, then prompting Gemma again with the
Stockfish claim plus transcript artifact. The job expects the final JSON answer
to cite the engine-backed best move rather than relying on Gemma's own chess
intuition.

Default bounds:

```text
VECL_CHESS_ANSWER_EVAL_MAX_ENTRIES=16
VECL_CHESS_ANSWER_MIN_RATE=0.8
VECL_ROUTING_MAX_NEW_TOKENS=256
```

The job prints `CHESS_ANSWER_EVAL_SUMMARY` and exits nonzero if routing,
chain execution, out-of-domain rejection, or final-answer grounding falls below
the configured thresholds.

## Phase 5 Episodic Embedding Eval

To run the Phase 5 episodic embedding eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_EPISODIC_EMBED_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_episodic_embed_eval.sh
```

This loads the configured base model as a hidden-state embedder, writes a tiny
multi-tenant episodic corpus into embedded Qdrant, and verifies finite normalized
vectors, stable dimensions, same-tenant retrieval, and tenant isolation. The job
prints `EPISODIC_EMBED_EVAL_SUMMARY` and exits nonzero if those checks fail.

To run the Phase 6b ethics-gate eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_ETHICS_EVAL_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_ethics_eval.sh
```

This asks Gemma to convert five natural-language risky/control requests into a structured
single-step action proposal, then executes that proposal through
`ChainExecutor + EthicsKernel + ControlledActionSpecialist`. The specialist is a
sandbox counter. The eval passes only if refused/review cases halt before the
specialist runs, the benign control completes, and the job prints
`ETHICS_EVAL_SUMMARY` with no unexpected outcomes.

## Phase 8 Terraform Eval

To run the Phase 8 Terraform plan-only eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_ROUTING_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_terraform_eval.sh
```

The job downloads the official Terraform CLI, asks Gemma 4 31B to route
Terraform plan/validate requests, executes real `terraform plan` and
`terraform validate` against a tiny local module, and asks Gemma to produce a
final answer from the JSON artifact. Mutation requests (`apply`, `destroy`) are
safe only if they either do not route to Terraform or are halted by
`EthicsKernel` before the specialist is invoked. The job prints
`TERRAFORM_EVAL_SUMMARY` and exits nonzero on specialist-call leaks, routing
fallbacks, failed plan execution, ungrounded answers, or Terraform false
positives on out-of-domain prompts.

To run the Phase 8b.2 combined Terraform + Stockfish + persistence eval:

```bash
export HF_TOKEN=...
export VECL_ROUTING_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_terraform_stockfish_eval.sh
```

This registers both Terraform and Stockfish in the same prompted-router card
set. It verifies Terraform plan/validate, mutation blocking before specialist
execution, chess routing to real Stockfish 18, and SQLite-backed provenance
reopen plus artifact restore. The job prints
`TERRAFORM_STOCKFISH_EVAL_SUMMARY`.

## Phase 9a Release Eval

To run the aggregate Phase 9a release gate on one Vertex A100 job:

```bash
export HF_TOKEN=...
export VECL_ROUTING_MODEL_ID=google/gemma-4-31B-it
zsh scripts/gcp/submit_vertex_release_eval.sh
```

This packages the release harness, real eval scripts, and fixtures into one
Vertex custom job. The job runs the Gemma aggregate evaluator in-process so the
model-driver, payload, routing, ethics, Terraform/Stockfish, and TimesFM checks
reuse one loaded Gemma driver instead of spawning a fresh Python process per
check. It then approves the report into a SQLite ledger if every threshold
passes.

The manifest currently runs:

- `MODEL_DRIVER_EVAL_SUMMARY`
- `TOOL_CALL_PAYLOAD_EVAL_SUMMARY`
- `PHASE7_ROUTING_EVAL_SUMMARY`
- `ETHICS_EVAL_SUMMARY`
- `TERRAFORM_STOCKFISH_EVAL_SUMMARY`
- `TIMESFM_DEMAND_EVAL_SUMMARY`

The job is intentionally larger than the single-phase evals so it can make use
of an already-provisioned A100. It still exits nonzero on the first failed
release report.

For frontier API release checks on the local Mac, use:

```bash
.venv/bin/python -m vecl release evaluate phase9a-openai \
  --manifest configs/release/phase9a-frontier-openai.json

.venv/bin/python -m vecl release evaluate phase9a-anthropic \
  --manifest configs/release/phase9a-frontier-anthropic.json
```

Those manifests run the model-driver and model-authored tool-call payload evals
with `VECL_MODEL_DRIVER` set to the selected provider.

## Phase 9b Fisher Eval

To run the opt-in Fisher diagonal eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_FISHER_MODEL_ID=google/gemma-4-31B-it
export VECL_FISHER_LORA_RANK=8
export VECL_FISHER_LAST_N_LAYERS=8
export VECL_FISHER_BASELINE_SNAPSHOT_URI=gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-lora.npz
export VECL_FISHER_BASELINE_SNAPSHOT_HASH=b733d830f0de1b98be56dad60e92938f20b170e214302362ba06ae9bbf9a428e
zsh scripts/gcp/submit_vertex_fisher_eval.sh
```

This loads Gemma 4 31B once, attaches a LoRA substrate, optionally restores the
canonical `tool_use_v0` baseline, runs a tiny supervised batch covering
tool-call JSON and grounded final-answer examples, computes one Fisher value per
LoRA slot, writes an approved snapshot containing the Fisher diagonal, uploads
the outputs to GCS, and verifies Fisher-weighted drift rejection on an
artificially perturbed candidate. The job prints `FISHER_EVAL_SUMMARY`.

Current canonical Fisher-bearing `tool_use_v0` baseline:

```bash
export VECL_TRAIN_BASELINE_SNAPSHOT_URI=gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-fisher.npz
export VECL_TRAIN_BASELINE_SNAPSHOT_HASH=465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
```

Its audit summary is stored at:

```text
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-fisher-summary.json
```

## Phase 11a Tool-Use Training Eval

To run the opt-in supervised LoRA tool-use training eval on Vertex A100:

```bash
export HF_TOKEN=...
export VECL_TRAIN_MODEL_ID=google/gemma-4-31B-it
export VECL_TRAIN_LORA_RANK=8
export VECL_TRAIN_LAST_N_LAYERS=8
export VECL_TRAIN_MAX_SLOTS=8
export VECL_TRAIN_SAMPLE_COUNT=1024
export VECL_TRAIN_GRADIENT_ACCUMULATION_STEPS=8
export VECL_TRAIN_BASELINE_ID=tool_use_v0
export VECL_TRAIN_HELDOUT_PROBE_COUNT=48
export VECL_TRAIN_HARD_PROBE_COUNT=48
export VECL_TRAIN_INTERFERENCE_THRESHOLD=0.05
zsh scripts/gcp/submit_vertex_tool_use_train.sh
```

This loads Gemma 4 31B once, attaches a wider LoRA substrate, uploads the local
`data/synthetic/v1-hard/exports` corpus split to GCS, samples a stratified
mixed-specialist training batch, evaluates smoke/heldout/hard-heldout CE probes,
creates or restores the named `tool_use_v0` LoRA baseline snapshot, commits the
sparse update through the learning-event monitor with `ewc_lambda=0.0`, uploads
the baseline/before/after LoRA snapshots, per-slot selection diagnostics, and
`summary.json` to GCS, and prints `TOOL_USE_TRAINING_SUMMARY`. By default the
submit script requires the expanded corpus export directory and checks coverage
for all current corpus domains: `blast,cross,eda,stockfish,sympy,terraform,timesfm`.

After the first baseline-creating run, pass
`VECL_TRAIN_BASELINE_SNAPSHOT_URI=gs://.../tool_use_v0-baseline-lora.npz` and
`VECL_TRAIN_BASELINE_SNAPSHOT_HASH=...` to force later runs to restore the exact
same content-addressed baseline instead of generating a fresh deterministic
baseline inside the worker.

Current canonical `tool_use_v0` baseline:

```bash
export VECL_TRAIN_BASELINE_SNAPSHOT_URI=gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-lora.npz
export VECL_TRAIN_BASELINE_SNAPSHOT_HASH=b733d830f0de1b98be56dad60e92938f20b170e214302362ba06ae9bbf9a428e
```

For Phase 11b EWC training, use the Fisher-bearing baseline snapshot above,
then set:

```bash
export VECL_TRAIN_BASELINE_SNAPSHOT_URI=gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-fisher.npz
export VECL_TRAIN_BASELINE_SNAPSHOT_HASH=465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
export VECL_TRAIN_EWC_LAMBDA=0.1
export VECL_TRAIN_EWC_DRIFT_THRESHOLD=1.0
```

The first sparse update from an exact approved baseline has zero EWC penalty at
the starting point; the penalty becomes active for resumed/drifted candidates,
while the post-update Fisher drift report still gates the candidate immediately.

Set `VECL_TRAIN_DOMAIN_MIX` to a comma-separated list such as
`stockfish,sympy,blast` to restrict the proof run. When a focused domain filter
is set, the script also checks non-trained domains for heldout CE regression
greater than `VECL_TRAIN_INTERFERENCE_THRESHOLD`. Override
`VECL_TRAIN_EXPECT_DOMAINS` only when intentionally running a partial corpus
coverage proof.

For a smaller smoke run, override the wider defaults:

```bash
export HF_TOKEN=...
export VECL_TRAIN_MODEL_ID=google/gemma-4-31B-it
export VECL_TRAIN_LORA_RANK=2
export VECL_TRAIN_LAST_N_LAYERS=2
export VECL_TRAIN_MAX_SLOTS=2
export VECL_TRAIN_SAMPLE_COUNT=64
export VECL_TRAIN_HELDOUT_PROBE_COUNT=24
export VECL_TRAIN_HARD_PROBE_COUNT=24
zsh scripts/gcp/submit_vertex_tool_use_train.sh
```

## Known Vertex Environment Fix

The current Vertex PyTorch GPU prebuilt image provides `torch 2.4.x` and
`torchvision 0.19.x`, which are too old for the current Gemma 4 `transformers`
stack. The submit script pins `torch==2.7.1` and `torchvision==0.22.1` inside
the job package so `transformers` does not import against the stale image copies.

The Vertex image is Python 3.10. Runtime code that must execute there imports
`UTC` and `StrEnum` through `vecl._compat`, not directly from Python 3.11+
standard-library aliases. The routing eval package also pins
`numpy>=1.25,<1.28`; upgrading NumPy to 2.x breaks the image's preinstalled
SciPy import path during Transformers model loading.

## Safety

- Keep the billing budget alert active.
- Shut down GPU resources after each run.
- Prefer `google/gemma-4-E4B-it` for the first smoke test.
- Do not commit `.env` files or Hugging Face tokens.
