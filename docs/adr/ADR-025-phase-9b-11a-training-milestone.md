# ADR-025: Phase 9b And Phase 11a Training Milestone

## Context

> Historical record: cloud project and bucket names have been replaced with
> `REDACTED_ARTIFACT_BUCKET`. Object paths, run timestamps, hashes, and outcomes are retained
> so the research sequence remains intelligible without publishing environment identifiers.

On May 21, 2026, VECL-QB crossed from specialist orchestration and release
plumbing into the first real 31B LoRA training update. The work did not change
the platform thesis from ADR-011: Gemma base weights remain frozen, specialists
produce exact artifacts, the model proposes actions, deterministic code
validates them, provenance records every state transition, and learning remains
bounded, sparse, auditable, and reversible.

This ADR records the implementation milestone rather than a single narrow API
decision because several phases became connected in one day:

- roster-wide specialist payload contracts,
- Phase 9b Fisher/drift infrastructure,
- the Phase 11 synthetic tool-use corpus factory,
- Phase 11a supervised tool-use training plumbing,
- real Vertex A100 proof runs on `google/gemma-4-31B-it`.

## Decision

Treat Phase 11a as a proof of the governed sparse-LoRA training loop, not as a
claim of meaningful capability improvement yet. The acceptance bar for this
phase is structural:

1. curated examples are converted to supervised prompt/target text;
2. cross-entropy loss backpropagates through Gemma with LoRA attached;
3. LoRA gradients are extracted at the ADR-010 slot boundary;
4. sparse slot selection uses the existing CPU oracle and monitor discipline;
5. only selected LoRA slots move;
6. base model weights and unselected LoRA slots remain unchanged;
7. before/after snapshots are hash-addressed and uploaded;
8. the learning event commits through provenance;
9. probe losses are measured before and after the update.

Meaningful quality gains, interference analysis, and EWC training penalties are
deferred to larger training runs and Phase 11b.

## What Landed

### Roster-wide specialist contracts

The specialist contract layer now covers the registered roster rather than only
the early chess/math cases. `vecl.qb.specialist_contracts` defines prompt-visible
payload contracts and configuration behavior for current specialists, including
Stockfish, SymPy, BLAST, Terraform, TimesFM, and cross-specialist regression
fixtures. This matters because model-authored tool calls are only research-grade
when the model sees the input language of each specialist and VECL can validate
the proposed payload deterministically.

The practical fix was to make "give me a BLAST configuration" and related
requests first-class contract cases instead of treating every mention of a tool
as an executable job. Configuration artifacts are allowed; unsupported or
underspecified executable payloads are rejected without invoking specialists.

### Phase 9b Fisher infrastructure

Phase 9b implemented slot-level Fisher and drift machinery for the LoRA
substrate:

- `vecl.evaluation.fisher.LoRAFisherEstimate`;
- `LoRADriftReport`;
- Fisher accumulation from raw LoRA gradients;
- Fisher-weighted snapshot drift computation;
- snapshot compatibility for optional Fisher arrays and metadata;
- release-gate fields for `drift_within_bound`, `drift_value`, and
  `drift_threshold`;
- a `fisher_drift` evaluator category;
- `scripts/fisher_eval_gemma.py`;
- `scripts/gcp/submit_vertex_fisher_eval.sh`.

The Fisher unit is the ADR-010 slot: one paired `lora_A` rank row and
`lora_B` rank column. Drift is computed as:

```text
D = sum_i F_i * (||A_i_candidate - A_i_approved||^2
               + ||B_i_candidate - B_i_approved||^2)
```

Release evaluation remains backward-compatible: frontier/API drivers do not
expose gradients or LoRA tensors, so drift is enforced only when a Fisher/drift
report is supplied.

### Synthetic tool-use corpus factory

Phase 11 needed more than a hand-written Stockfish toy set, so a repeatable
corpus factory was added under `vecl.training`:

- schema and JSONL round-tripping;
- deterministic corpus generation;
- optional LLM augmentation with cache/admission tracking;
- validation for schema, deduplication, label distribution, forbidden content,
  metadata completeness, and export eligibility;
- semantic diversity metrics over both the whole corpus and the supervised
  training export;
- scripts for generation, validation, reporting, and export.

The generated `v1-hard` corpus contains 25,000 records across Stockfish, SymPy,
BLAST, Terraform, TimesFM, cross-specialist regression, and real EDA coverage.
The supervised export contains 18,913 eligible examples:

- train: 16,993;
- heldout: 973;
- hard-heldout: 947.

EDA coverage is no longer placeholder-only. The corpus includes Yosys tool-call,
OpenROAD tool-call, Yosys -> OpenROAD chain, final-answer, and hard-negative
examples. The current EDA slice contains 3,043 records and zero placeholder EDA
records.

The corpus is intentionally synthetic and provenance-labeled. LLM-origin records
are admitted only after validation, and real user data is not used. Stockfish
synthetic-example authority is aligned to the Stockfish trust anchor (`0.9`) and
covered by a regression test, so fixture authority does not drift away from the
credentialed specialist authority.

### Phase 11a training loop

`vecl.training.tool_use_loop` now implements the supervised LoRA update cycle:

- `SupervisedToolUseExample`;
- `ToolUseTrainingConfig`;
- `ToolUseTrainingReport`;
- `ToolUseTrainer.run_cycle(...)`;
- ledger-derived slot rarity;
- conservative minimum batch authority;
- gradient-normalized activation;
- before/after LoRA snapshots;
- tensor delta provenance payloads;
- abort-and-restore behavior when failures occur after token creation.

The Phase 11a scoring policy is deliberately simple:

- activation is normalized per-slot gradient magnitude;
- rarity is `1 / sqrt(1 + historical_selection_count_i)`;
- authority is the conservative minimum authority across the curated batch.

This is gradient-greedy sparse SGD with anti-collapse pressure. It is a valid
v0 policy, not a final theory of memory salience.

### Phase 11a.1 hardening

After the first proof run, the probe reporting path was hardened so future
runs cannot accidentally overclaim from noisy CE movement:

- each probe set now retains per-example loss deltas;
- each probe set reports sample count, mean/median delta, relative mean change,
  improved/worsened/tied counts, exact one-sided sign-test p-values, and a
  paired-t statistic with a normal-approximation p-value;
- probe metrics are broken down by corpus domain, category, and task kind;
- focused single-domain training runs produce an interference report for
  non-trained domains;
- the default interference threshold is a 5% heldout CE regression;
- the Vertex launch docs include a larger-slot rarity proof configuration.

These checks remain evidence summaries, not automatic capability claims.

### Vertex 31B proof run

The first real Phase 11a proof run completed successfully on Vertex using
`google/gemma-4-31B-it`.

The run used 64 examples sampled from the `v1-hard` supervised train export and
three probe sets:

- smoke: 8 examples;
- heldout: 24 examples;
- hard-heldout: 24 examples.

The job wrote:

- `before-lora.npz`;
- `after-lora.npz`;
- `summary.json`.

The GCS output prefix was:

```text
gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260521-184707
```

The summary showed:

- model id: `google/gemma-4-31B-it`;
- sample count: 64;
- slot count: 4;
- selected slot count: 2;
- learning event committed: true;
- base weights unchanged: true;
- selected LoRA slots changed: true;
- unselected LoRA slots unchanged: true;
- snapshots uploaded: true;
- EWC lambda: 0.0.

The representative selected-slot tensor delta was nonzero
(`7.53e-05` total norm), confirming gradient flow and a real LoRA tensor update.
The summary also retained per-example `before_losses` and `after_losses` arrays
for each probe set, so later statistical checks can be computed without
rerunning the job.

Probe loss movement was small but positive:

- aggregate mean CE: `3.4670 -> 3.4661`;
- heldout mean CE: `3.4498 -> 3.4482`;
- hard-heldout mean CE: `3.4859 -> 3.4856`;
- smoke mean CE: `3.4621 -> 3.4611`;
- improved examples: 30 of 56 aggregate probe examples.

This is evidence that the loop works, not evidence that the model is now
materially better.

### Phase 11a.2 expanded-slot and seed-repeat proof

The initial proof run selected 2 of 4 available slots, which was enough to
validate the tensor-update boundary but not enough to stress rarity. A follow-up
Vertex run expanded the adapter to 64 slots and selected 8, so only 12.5% of
eligible slots moved.

The first expanded run used:

- Vertex job id: `1571657021848027136`;
- GCS output prefix:
  `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260521-222420`;
- torch seed: `1234`;
- sample seed: `1107`;
- sample count: 128;
- slot count: 64;
- selected slot count: 8;
- dataset hash:
  `49f701e6b7313496a18a3259897ce35682e6cd5f160528105081c127cd0872d2`.

The run preserved the core structural invariants:

- learning event committed: true;
- base weights unchanged: true;
- selected LoRA slots changed: true;
- unselected LoRA slots unchanged: true;
- snapshots uploaded: true;
- EWC lambda: 0.0.

It also exercised roster-wide coverage. The train set, heldout probe, and
hard-heldout probe each covered BLAST, cross-specialist examples, EDA,
Stockfish, SymPy, Terraform, and TimesFM.

The aggregate CE result remained a null:

- aggregate mean CE delta: `+2.72e-05`;
- improved/worsened/tied: `45 / 45 / 14`;
- paired-t normal-approximation p-value: `0.913`.

A second expanded run repeated the same corpus sample/probe selection but changed
the torch seed:

- Vertex job id: `4342144848866836480`;
- GCS output prefix:
  `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260521-225301`;
- torch seed: `4321`;
- sample seed: `1107`;
- dataset hash:
  `49f701e6b7313496a18a3259897ce35682e6cd5f160528105081c127cd0872d2`.

The repeat run preserved the same structural invariants and coverage. Its
aggregate CE movement was also tiny:

- aggregate mean CE delta: `-3.17e-04`;
- improved/worsened/tied: `43 / 36 / 25`;
- paired-t normal-approximation p-value: `0.132`.

The heldout split in the repeat run had a nominal paired-t p-value of `0.034`,
but the magnitude was only `-5.63e-04` CE, or about `-0.017%` relative mean
change. Hard-heldout stayed null (`p = 0.683`). Per-domain movements remained
far below the pre-committed 5% interference threshold. The largest aggregate
per-domain relative movements were on the order of `0.03%`, so these runs set a
useful seed-to-seed noise floor rather than demonstrating durable capability
gain.

The per-domain aggregate CE noise matrix was:

| Domain | Seed 1234 mean CE delta | Seed 4321 mean CE delta | Difference | Largest relative movement |
| --- | ---: | ---: | ---: | ---: |
| BLAST | `+0.000315` | `-0.001165` | `-0.001480` | `0.0379%` |
| Cross | `+0.000376` | `+0.000314` | `-0.000062` | `0.0131%` |
| EDA | `+0.000243` | `-0.000376` | `-0.000618` | `0.0194%` |
| Stockfish | `+0.000051` | `-0.000326` | `-0.000377` | `0.0108%` |
| SymPy | `-0.000561` | `+0.000783` | `+0.001344` | `0.0494%` |
| Terraform | `-0.000255` | `-0.001410` | `-0.001155` | `0.0298%` |
| TimesFM | `-0.000371` | `+0.000079` | `+0.000451` | `0.0150%` |

Directly comparing `after-lora.npz` across the two seeds is not an appropriate
Phase 11b drift threshold because it is dominated by different random LoRA
initialization. The useful seed-noise measurement is the difference between the
actual update vectors:

```text
update_seed_1234 = after_seed_1234 - before_seed_1234
update_seed_4321 = after_seed_4321 - before_seed_4321
```

For these two runs:

- seed 1234 update norm: `1.9248e-04`;
- seed 4321 update norm: `2.2809e-04`;
- distance between update vectors: `3.4053e-04`;
- squared distance between update vectors: `1.1596e-07`;
- changed-slot intersection: `[56, 57, 59, 60, 61]`;
- changed-slot union: `[44, 51, 53, 54, 56, 57, 59, 60, 61, 62, 63]`.

This gives Phase 11b an empirical update-scale floor. It should not be confused
with a Fisher-weighted drift threshold yet; proper EWC calibration still needs
Fisher values and, ideally, one more seed or an approved-snapshot baseline.

### Phase 11a.4: Fixed Baseline and 1024-Example Proof

Phase 11a.4 added the missing approved-baseline shape before Phase 11b EWC:

- a named `tool_use_v0` LoRA baseline snapshot;
- baseline restore before training when a baseline URI/path is configured;
- baseline id/hash/source metadata in reports and learning-event payloads;
- explicit gradient accumulation configuration in training reports;
- per-slot selection diagnostics covering gradient summary, activation, rarity,
  authority, score, rank, selected status, and cutoff margin;
- stable GCS baseline promotion for future runs.

The first `tool_use_v0` baseline is:

```text
uri: gs://REDACTED_ARTIFACT_BUCKET/lora-baselines/tool_use_v0/tool_use_v0-baseline-lora.npz
hash: b733d830f0de1b98be56dad60e92938f20b170e214302362ba06ae9bbf9a428e
model_id: google/gemma-4-31B-it
rank: 8
last_n_layers: 8
torch_seed: 1234
```

The larger Vertex proof then ran from that baseline shape:

```text
job_id: 6756214986625777664
state: JOB_STATE_SUCCEEDED
output: gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260521-235325
sample_count: 1024
slot_count: 64
selected_slot_count: 8
gradient_accumulation_steps: 8
ewc_lambda: 0.0
baseline_restored: true
baseline_snapshot_hash: b733d830f0de1b98be56dad60e92938f20b170e214302362ba06ae9bbf9a428e
before_snapshot_hash: b733d830f0de1b98be56dad60e92938f20b170e214302362ba06ae9bbf9a428e
after_snapshot_hash: af74de4a3c73ecfef7d9a4d82e4b087516803770f24626baa2906e6bb645a02e
selection_diagnostics_hash: f743914792e19199db4bab11299be0ceadddc05add8ff0e894d0ed4793f62e66
```

The structural invariants held:

```text
base_weights_unchanged: true
selected_lora_slots_changed: true
unselected_lora_slots_unchanged: true
learning_event_committed: true
coverage_passed: true
```

Training coverage included all current corpus domains:

```text
blast,cross,eda,stockfish,sympy,terraform,timesfm
```

The CE probe movement remained null-scale evidence:

| Probe | Count | Improved | Worsened | Tied | Mean Delta | Approx p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aggregate | `104` | `48` | `42` | `14` | `-0.000035` | `0.895` |
| heldout | `48` | `23` | `19` | `6` | `+0.000085` | `0.837` |
| hard-heldout | `48` | `21` | `19` | `8` | `-0.000283` | `0.419` |
| smoke | `8` | `4` | `4` | `0` | `+0.000739` | `0.492` |

This strengthens the structural proof and creates the common approved baseline
needed for coherent Phase 11b drift measurement. It is still not a capability
claim.

### Phase 11b entry: Fisher-bearing baseline and EWC wiring

Phase 11b started by making the approved baseline usable for both drift
measurement and training-time regularization:

- `LoRAMemorySubstrate` can compute a differentiable Fisher-weighted drift
  penalty against an approved LoRA snapshot;
- `ToolUseTrainer` accepts `ewc_lambda`, an approved Fisher snapshot, and an
  optional drift threshold;
- training reports and `SPARSE_UPDATE_APPLIED` payloads include task loss,
  total loss, EWC penalty, approved snapshot hash, and post-update drift report;
- the Vertex training script can run with `VECL_TRAIN_EWC_LAMBDA > 0`;
- the Fisher eval script can restore the canonical baseline, accumulate Fisher,
  write an approved Fisher-bearing snapshot, upload it to GCS, and verify
  artificial over-threshold rejection.

The canonical Fisher-bearing `tool_use_v0` baseline is:

```text
uri: gs://REDACTED_ARTIFACT_BUCKET/lora-baselines/tool_use_v0/tool_use_v0-baseline-fisher.npz
hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
summary: gs://REDACTED_ARTIFACT_BUCKET/lora-baselines/tool_use_v0/tool_use_v0-baseline-fisher-summary.json
```

The Fisher Vertex run was:

```text
job_id: 2595223182470283264
state: JOB_STATE_SUCCEEDED
output: gs://REDACTED_ARTIFACT_BUCKET/fisher-eval/20260522-004917
model_id: google/gemma-4-31B-it
torch_seed: 1234
sample_count: 4
slot_count: 64
positive_fisher_slots: 64
baseline_restored: true
baseline_snapshot_hash: b733d830f0de1b98be56dad60e92938f20b170e214302362ba06ae9bbf9a428e
approved_snapshot_hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
drift_value: 0.0
drift_threshold: 1.0
drift_within_bound: true
artificial_over_threshold_rejected: true
outputs_uploaded: true
```

This is still an entry proof, not a full EWC capability run. The first update
from an exact approved baseline has zero EWC penalty at the starting point
because `theta == theta_approved`; the post-update Fisher drift report is the
immediate bound check. The penalty becomes active for resumed or already-drifted
candidates and for longer training loops.

The first bounded Phase 11b training proof then ran from the Fisher-bearing
baseline:

```text
job_id: 1016148563123503104
state: JOB_STATE_SUCCEEDED
output: gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260522-011020
model_id: google/gemma-4-31B-it
sample_count: 128
slot_count: 64
selected_slot_count: 8
gradient_accumulation_steps: 4
baseline_restored: true
baseline_snapshot_hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
ewc_lambda: 0.1
ewc_penalty_value: 0.0
ewc_drift_value: 3.7488226369255254e-10
ewc_drift_threshold: 1.0
ewc_drift_within_bound: true
after_snapshot_hash: adc439a2a936f45e4126f552d3cb09a70b10a2a48fa3b433590d9b58bc6aede0
selection_diagnostics_hash: 15fd839defa214b528f57c294533d9fc2994df1293f29047fa5b8b6324fcba9b
```

The structural invariants held under `ewc_lambda > 0`:

```text
base_weights_unchanged: true
selected_lora_slots_changed: true
unselected_lora_slots_unchanged: true
learning_event_committed: true
coverage_passed: true
snapshots_uploaded: true
selection_diagnostics_uploaded: true
```

Training coverage again included all current corpus domains:

```text
blast,cross,eda,stockfish,sympy,terraform,timesfm
```

The CE probe movement remained null-scale:

| Probe | Count | Improved | Worsened | Tied | Mean Delta | Approx p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aggregate | `72` | `32` | `32` | `8` | `+0.000052` | `0.855` |
| heldout | `32` | `14` | `14` | `4` | `+0.000125` | `0.774` |
| hard-heldout | `32` | `15` | `13` | `4` | `-0.000247` | `0.521` |
| smoke | `8` | `3` | `5` | `0` | `+0.000953` | `0.374` |

This validates the bounded-training path mechanically: Fisher-bearing baseline
restore, EWC configuration, sparse LoRA update, drift report, release-style
bound check, artifact upload, and provenance commit all work together. It still
does not show capability gain.

### Phase 11b.1: Learning candidate lifecycle hardening

Phase 11b.1 closes the gap between drift reporting and bounded update
enforcement. Training now records a candidate lifecycle around each sparse LoRA
attempt:

- `LEARNING_CANDIDATE_PREPARED`;
- `LEARNING_CANDIDATE_EVALUATED`;
- `LEARNING_CANDIDATE_ACCEPTED`;
- `LEARNING_CANDIDATE_REJECTED`;
- `LEARNING_CANDIDATE_RESTORED`.

When a configured Fisher drift bound is exceeded, the trainer:

- keeps the candidate `after-lora.npz` snapshot as a quarantined artifact;
- aborts the pending learning event;
- restores active LoRA tensors to the start snapshot;
- records candidate rejection and restoration provenance;
- does not emit `SPARSE_UPDATE_APPLIED` or `LEARNING_EVENT_COMMITTED`.

Accepted candidates continue to emit `SPARSE_UPDATE_APPLIED` and
`LEARNING_EVENT_COMMITTED`, now with candidate id, start snapshot hash, candidate
snapshot hash, and accepted decision fields. The `LearningEventToken` schema is
unchanged; approved/start/candidate hashes live in the downstream provenance
payloads and training report.

The Vertex training script now supports distinct start and approved snapshots:

```text
VECL_TRAIN_BASELINE_SNAPSHOT_URI          # start snapshot
VECL_TRAIN_EWC_APPROVED_SNAPSHOT_URI      # approved Fisher snapshot
```

This lets future runs train from a candidate while regularizing and measuring
against the approved `tool_use_v0` baseline. It also supports deliberate
rejection proofs via `VECL_TRAIN_EXPECT_REJECTION=1`.

The deliberate over-bound Vertex proof then ran successfully:

- Vertex job id: `6396738456017436672`;
- GCS output prefix:
  `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-181320`;
- model id: `google/gemma-4-31B-it`;
- sample count: `64`;
- slot count: `64`;
- selected slot count: `8`;
- baseline snapshot hash:
  `465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1`;
- EWC lambda: `0.1`;
- Fisher drift threshold: `1e-12`;
- observed Fisher drift: `3.750780446094006e-10`;
- drift within bound: `false`;
- expected rejection: `true`;
- candidate decision: `rejected_drift_exceeded`;
- learning event committed: `false`;
- selected LoRA slots changed in the quarantined candidate: `true`;
- unselected LoRA slots unchanged: `true`;
- active tensors restored: `restored_snapshot_hash == start_snapshot_hash`;
- roster coverage passed across BLAST, cross-specialist regression, EDA,
  Stockfish, SymPy, Terraform, and TimesFM.

This run proves the full GPU/GCS candidate rejection path: the candidate can be
materially written and snapshotted for audit, fail the Fisher gate, avoid sparse
update commit, and restore the active LoRA state before the run exits.

### Phase 11b.2: Fixed-baseline lambda=0 drift floor

After the rejection proof, a 1024-example `lambda=0` run measured the ordinary
training drift floor from the same fixed `tool_use_v0` baseline:

- Vertex job id: `7638148656427696128`;
- GCS output prefix:
  `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-185349`;
- model id: `google/gemma-4-31B-it`;
- sample count: `1024`;
- slot count: `64`;
- selected slot count: `8`;
- baseline snapshot hash:
  `465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1`;
- EWC lambda: `0.0`;
- candidate decision: `accepted`;
- learning event committed: `true`;
- base weights unchanged: `true`;
- selected LoRA slots changed: `true`;
- unselected LoRA slots unchanged: `true`;
- roster coverage passed across BLAST, cross-specialist regression, EDA,
  Stockfish, SymPy, Terraform, and TimesFM.

The run did not emit an inline `ewc_drift_report` because no explicit
`VECL_TRAIN_EWC_APPROVED_SNAPSHOT_URI` was supplied for the `lambda=0` case.
The drift floor was therefore computed offline from the persisted candidate
snapshot against the Fisher-bearing `tool_use_v0` baseline:

```text
Fisher drift:              3.6935913401063406e-10
Unweighted slot delta sum: 3.674775529229044e-08
Positive Fisher slots:     64
Top drift slot:            slot 61, layer 59 mlp.up_proj rank 5
```

The aggregate probe movement remained null-scale:

- aggregate probe examples: `264`;
- improved / worsened / tied: `120 / 110 / 34`;
- aggregate mean CE delta: `-0.0001317407145644679`;
- paired-t normal-approximation p-value: `0.43677598333635714`.

This gives the first fixed-baseline, lambda-free Fisher drift floor for the
1024-example roster-wide configuration. Future `lambda=0` calibration runs
should pass the approved Fisher snapshot explicitly so the drift report is
emitted inline in `summary.json` instead of computed post-hoc.

### Phase 11b.3: Inline-drift size sweep

After the post-hoc 1024 drift-floor run, the training script was hardened so
calibration runs can require inline drift reporting. When
`VECL_TRAIN_REQUIRE_INLINE_DRIFT=1`, the script fails unless both the approved
Fisher snapshot and drift threshold are supplied and the resulting
`summary.json` contains the bound decision.

The tool-call generation probe was also corrected to use the same VECL tool-call
author prompt as the production tool-call path. The first probe version used
the raw corpus prompt and therefore measured a different task from actual
VECL-QB tool-call authoring. The corrected probe includes specialist cards and
payload contracts before asking the model to emit the JSON proposal.

The corrected 1024 anchor and the 2048/4096/8192 sweep then ran on Vertex with:

```text
model_id: google/gemma-4-31B-it
baseline_snapshot_hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
ewc_approved_snapshot_hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
ewc_lambda: 0.0
ewc_drift_threshold: 1.0
inline_drift_required: true
slot_count: 64
selected_slot_count: 8
gradient_accumulation_steps: 4
tool_call_generation_probe_count: 16
```

The job ids and outputs were:

| Sample Count | Job ID | Output Prefix |
| ---: | --- | --- |
| 1024 | `7523447603418103808` | `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-202145` |
| 2048 | `1888881519624192000` | `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-205434` |
| 4096 | `4650432516132438016` | `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-210331` |
| 8192 | `4055957365319532544` | `gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-210420` |

All four jobs succeeded and preserved the structural invariants:

```text
baseline_restored: true
candidate_decision: accepted
learning_event_committed: true
base_weights_unchanged: true
selected_lora_slots_changed: true
unselected_lora_slots_unchanged: true
coverage_passed: true
snapshots_uploaded: true
selection_diagnostics_uploaded: true
```

The sweep results were:

| Samples | Fisher Drift | Mean CE Delta | Improved | Worsened | Tied | Approx p | Exact Tool-Call Before | Exact Tool-Call After | Validation Before | Validation After |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | `3.693591e-10` | `-0.0001317` | `120` | `110` | `34` | `0.437` | `0.375` | `0.375` | `0.625` | `0.625` |
| 2048 | `3.668088e-10` | `-0.0000984` | `118` | `112` | `34` | `0.562` | `0.375` | `0.375` | `0.625` | `0.625` |
| 4096 | `3.670766e-10` | `-0.0000882` | `118` | `112` | `34` | `0.599` | `0.375` | `0.375` | `0.625` | `0.625` |
| 8192 | `3.728856e-10` | `-0.0000997` | `115` | `114` | `35` | `0.557` | `0.375` | `0.375` | `0.625` | `0.625` |

The main finding is negative and useful: increasing examples in the current
single averaged-gradient update does not materially increase Fisher drift or
capability movement. The update magnitude remains roughly `3.7e-10` across the
sweep, and exact tool-call generation stays flat at 6/16 examples. This suggests
the next capability experiment should vary optimization dose (multi-cycle or
multi-step training, and then EWC comparison) rather than simply averaging more
examples into one sparse update.

### Phase 11b.4: Tiny-overfit diagnostic harness

The flat sample-size sweep leaves two live interpretations: the sparse regime
may be under-dosed, or the training path may be mechanically misaligned. Before
spending more GPU time on multi-cycle sparse updates, the trainer now has a
diagnostic path to answer the basic ML debugging question: can Gemma+LoRA
overfit a tiny curated batch at all?

Two changes support that diagnostic:

- `ToolUseTrainer` accepts an optional `training_prompt_builder`, so training
  can use the same prompt shape as runtime evaluation.
- `scripts/tool_use_train_gemma.py` defaults
  `VECL_TRAIN_PROMPT_MODE=tool_call_author`, which wraps `tool_call_json`
  examples with the same VECL tool-call author prompt, specialist cards, and
  payload contracts used by the generation probe.

The script also supports:

```text
VECL_TRAIN_DIAGNOSTIC_MODE=tiny_overfit
VECL_TRAIN_OVERFIT_SAMPLE_COUNT=8
VECL_TRAIN_OVERFIT_STEPS=25
VECL_TRAIN_OVERFIT_LEARNING_RATE=0.01
VECL_TRAIN_OVERFIT_GENERATION_PROBE_COUNT=8
```

This mode restores the configured baseline, selects a tiny tool-call batch,
uses ordinary dense LoRA optimizer steps without committing a learning event,
and prints `TOOL_USE_OVERFIT_SUMMARY`. It is explicitly a diagnostic, not an
approved bounded-learning update. If CE cannot drop on 8-16 examples with the
bound off, larger sparse/EWC experiments should pause while the loss masking,
prompt formatting, update scale, and tokenization path are inspected.

The first attempt enabled before/after generation for all eight examples and
failed with CUDA OOM on A100 80GB. The harness was adjusted to backward each
example separately within each step, clear CUDA cache between phases, and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the Vertex job.

The CE-only rerun then succeeded:

```text
job_id: 2475123527368310784
output: gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-224530
model_id: google/gemma-4-31B-it
baseline_snapshot_hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
training_prompt_mode: tool_call_author
sample_count: 8
steps: 25
learning_rate: 0.01
generation_probe_enabled: false
mean_ce_before: 1.5441702753305435
mean_ce_after: 0.01824237243272364
mean_ce_drop: 1.5259279028978199
relative_mean_change: -0.9881862947861636
improved_examples: 8
worsened_examples: 0
passed: true
```

This rules out the simplest mechanical-bug hypothesis: with dense LoRA updates
and the runtime author prompt, the model can memorize a tiny tool-call batch.
The flat sparse sweep is therefore more likely explained by sparse update dose,
gradient cancellation across tasks, slot capacity/selection, or the difference
between dense optimizer steps and one VECL sparse averaged update.

The matched sparse-overfit diagnostic then ran the same eight-example
Stockfish tool-call batch through repeated VECL sparse updates instead of
dense optimizer steps:

```text
job_id: 8424800847589801984
output: gs://REDACTED_ARTIFACT_BUCKET/tool-use-training/20260525-233642
model_id: google/gemma-4-31B-it
baseline_snapshot_hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
training_prompt_mode: tool_call_author
sample_count: 8
cycles: 25
learning_rate: 0.01
max_slots: 8
gradient_accumulation_steps: 8
learning_commit_count: 25
ledger_event_count: 175
mean_ce_before: 1.5441702753305435
mean_ce_after: 1.5375268906354904
mean_ce_drop: 0.006643384695053101
relative_mean_change: -0.004302235835766897
improved_examples: 8
worsened_examples: 0
snapshots_uploaded: true
passed: false
```

The sparse job returned nonzero because it intentionally required
`mean_ce_drop >= 0.05` to count as a memorization pass. That makes the result
negative but informative: the governed sparse path did move selected LoRA
slots, committed 25 learning events, improved all eight examples by the final
cycle, and preserved snapshots, but it did not approach dense-overfit
memorization. Dense optimizer steps dropped CE by about `1.53`; repeated VECL
sparse updates dropped CE by about `0.0066` under the same prompt shape and
baseline. The remaining investigation is therefore squarely about sparse update
strength, slot/capacity policy, and whether the current sparse tensor update is
too weak or too constrained for meaningful capability movement.

## Consequences

- VECL-QB now has an end-to-end path from curated tool-use examples to a real
  sparse LoRA update on 31B Gemma.
- The base-model immutability claim survived contact with training: only
  selected LoRA slots moved.
- Phase 9b's Fisher/drift measurement path and Phase 11a's update path are now
  separated cleanly: Fisher measures bounded drift; 11a trains with
  `ewc_lambda = 0.0`.
- The corpus factory becomes a standing research asset. Future specialist waves
  should extend it rather than hand-write isolated mini-fixtures.
- The release harness can now evaluate routing/tool-call/synthesis behavior
  across frontier APIs and Vertex Gemma while keeping Fisher enforcement scoped
  to LoRA-capable open/local models.
- Larger-slot sparse selection has now been exercised at 8 of 64 slots. That is
  the first run where ledger-derived rarity and TopK selection have meaningful
  room to matter, though a pure-gradient ablation is still needed to isolate the
  rarity term's effect.
- Phase 11b threshold calibration should compare candidates against a common
  approved snapshot. Cross-seed after-snapshot distance is not a valid drift
  floor when each run initializes LoRA independently.
- Phase 11b can now use `tool_use_v0` as the common approved snapshot for
  Fisher-weighted drift comparisons.
- Phase 11b also has training-time EWC plumbing and a Fisher-bearing approved
  baseline, so subsequent proofs can stress the drift-gated training path
  beyond the first exact-baseline sparse update.
- The first bounded Phase 11b Vertex training proof passed mechanically with
  `ewc_lambda = 0.1`; post-update Fisher drift was nonzero but far below the
  configured threshold.
- Phase 11b.1 converts over-bound drift from a script-level failure into a
  lifecycle decision: rejected candidates are preserved for audit, but they are
  not committed as sparse updates and active tensors are restored.
- The Phase 11b.1 rejection path has now fired on Vertex, not only in local
  tests. This closes the previous gap between measuring an over-bound candidate
  and enforcing the bound before accepting a sparse update.
- The fixed-baseline lambda=0 drift floor for the 1024-example roster-wide run
  is approximately `3.69e-10`, which is close to the earlier 64-example
  over-bound proof's observed drift scale and gives a concrete baseline for
  lambda comparison.
- The fixed-baseline sample-size sweep shows that the current single
  averaged-gradient update has a flat drift scale from 1024 to 8192 examples.
  More examples improve gradient representativeness, but they are not equivalent
  to more optimization steps under the current trainer.
- Tool-call training and tool-call generation probing now share the same
  author-prompt shape by default. This removes one train/eval mismatch from the
  interpretation of future tool-call generation runs.
- The dense-vs-sparse tiny-overfit ablation localizes the learning gap: Gemma
  plus LoRA can memorize the tiny batch, while the current governed sparse path
  only produces a small directional CE improvement under the same conditions.
- Serious multi-hour training should use the real EDA chain coverage so the
  model trains on chain structure and artifact handoff, not just single-step
  calls.

## Known Limits

- The 31B training runs were intentionally small. They validate plumbing and safety
  properties, not durable skill acquisition.
- The probe movement is not statistical evidence of capability gain. An
  even split between improved and worsened examples and mean CE changes around
  `1e-4` to `1e-3`
  should be treated as loop-health signal only until future runs include
  pre-registered significance checks.
- Fisher is wired as a training penalty and passed a bounded Vertex proof, but
  no longer training run has shown a material EWC effect yet. The first
  exact-baseline sparse update starts with a zero penalty; longer, resumed, or
  multi-cycle runs are needed to observe regularization pressure.
- Candidate rejection/restore has now been exercised on Vertex with an
  intentionally tiny threshold. The next unproven path is not simple rejection;
  it is a longer resumed or multi-cycle run where EWC pressure measurably
  changes the drift trajectory.
- The first lambda=0 drift-floor run required post-hoc drift calculation
  because the approved Fisher snapshot was not passed explicitly for inline
  drift reporting. This is a run-configuration issue, not a tensor/artifact
  issue; the persisted snapshots were sufficient to compute the drift exactly.
- The corrected inline-drift sweep fixed that reporting issue for subsequent
  runs, but it did not show capability gain. Exact tool-call generation stayed
  flat across the 1024/2048/4096/8192 sample-count sweep.
- The dense tiny-overfit path drove CE down sharply on Vertex. The matched
  sparse-overfit diagnostic improved all examples but failed its memorization
  threshold. This is not an infrastructure failure; it is evidence that the
  current governed sparse update path is much weaker than dense LoRA optimizer
  steps on the same tiny batch.
- The current activation signal is gradient-derived, so selection is still
  mostly top-K by gradient norm with rarity and authority as modifiers.
- Ledger-derived rarity is implemented and has been exercised at 8 of 64 slots,
  but its independent effect has not yet been separated from gradient magnitude.
  A pure-gradient selection comparison is still needed.
- Yosys/OpenROAD are represented in the corpus and specialists, but EDA
  training examples remain supervised JSONL labels rather than traces generated
  by executing every chain during corpus generation.
- The synthetic corpus is useful for bootstrapping, but it must be supplemented
  by artifact-verified chain traces as the system matures.
- The corpus is now a training trust boundary. Current validators catch schema,
  duplication, distribution, metadata, export eligibility, and simple forbidden
  content issues; they do not fully address subtle label noise, latent shortcut
  features, malicious augmentation, or prompt-injection-like examples.
- Frontier API drivers cannot participate in LoRA/Fisher bounded continual
  learning because they expose no weights, gradients, or exact rollback.

## Next Steps

1. Expand the release/benchmark fixtures around model-authored payloads and
   final synthesis from verified artifacts.
2. Analyze the sparse-overfit diagnostics and selection records to determine
   whether the weak CE movement is caused by update scale, slot budget,
   gradient/slot selection, or the sparse tensor update rule itself.
3. Run a controlled sparse-overfit knob sweep before building larger training
   machinery: learning rate, selected-slot count, and cycle count, all from the
   same fixed baseline and tiny batch.
4. If sparse-overfit becomes capable of memorization, implement a true
   optimization-dose experiment: multi-cycle or multi-step training from the
   fixed baseline, with inline drift reports and the same corrected tool-call
   generation probe.
5. Run a matched fixed-baseline lambda comparison (`lambda = 0` vs. a nonzero
   lambda) over that multi-cycle path or a distinct start snapshot to measure
   whether EWC actually slows Fisher-weighted drift.
6. Decide when to split the shared `tool_use_v0` adapter into domain adapters.
   The default remains one shared adapter until evals show interference or
   specialization pressure.
7. Compare sparse selection against a pure-gradient ranking baseline so the
   rarity term's independent effect can be measured.
8. Add a corpus adversary model and review checklist describing how corpus
   poisoning could enter through generators, LLM augmentation, or future
   observation traces, and which validators or human reviews would catch it.

## Rejected Alternatives

- Claim capability gain from the proof run. Rejected because the probe movement
  is small and the run was designed to validate structure.
- Train long before adding richer deterministic chains. Rejected because the
  current corpus is broad but still light on multi-step artifact handoff.
- Let frontier API outputs become LoRA training targets by default. Rejected
  because API models do not expose bounded-learning substrates and provider
  terms may restrict distillation.
- Collapse Fisher into the 11a training loop. Rejected because proper Fisher is
  per-example expected gradient-square measurement and belongs in Phase 9b/11b,
  not the first supervised update proof.
