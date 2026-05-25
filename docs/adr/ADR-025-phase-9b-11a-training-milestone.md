# ADR-025: Phase 9b And Phase 11a Training Milestone

## Context

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
gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/tool-use-training/20260521-184707
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
  `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/tool-use-training/20260521-222420`;
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
  `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/tool-use-training/20260521-225301`;
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
uri: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-lora.npz
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
output: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/tool-use-training/20260521-235325
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
uri: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-fisher.npz
hash: 465af72564819f03e83a02b46288c0d612e77083e368f86e443027d06336b5b1
summary: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/lora-baselines/tool_use_v0/tool_use_v0-baseline-fisher-summary.json
```

The Fisher Vertex run was:

```text
job_id: 2595223182470283264
state: JOB_STATE_SUCCEEDED
output: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/fisher-eval/20260522-004917
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
output: gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/tool-use-training/20260522-011020
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
2. Exercise Phase 11b rejection and rollback/escalation behavior when the
   Fisher-weighted bound is exceeded.
3. Decide when to split the shared `tool_use_v0` adapter into domain adapters.
   The default remains one shared adapter until evals show interference or
   specialization pressure.
4. Compare sparse selection against a pure-gradient ranking baseline so the
   rarity term's independent effect can be measured.
5. Add a corpus adversary model and review checklist describing how corpus
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
