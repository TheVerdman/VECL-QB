# Research status and publication scope

Scope: the 2026-09-24 release audit at `e3d4c893002a9a34256a000ba4a174377f4742cd`.
This documentation follow-up checked the cited source at that same revision. Probe outcomes and
test counts below are the audit's recorded results, not new executions. GitHub metadata confirms
the repository is public as of this follow-up; the audit's earlier private-visibility snapshot is stale.

## Implemented bounds and audit records

- The [CPU oracle](../vecl/sparse/oracle.py#L18-L53) selects at most `max_slots` eligible slots per
  call, with deterministic TopK. [Tokens](../vecl/runtime/tokens.py#L11-L89) carry event/batch/tenant
  identifiers, non-empty source/root labels, thresholds, expiry and policy versions. These are
  per-update constraints, not lifetime budgets or bounds on behavior, forgetting, or retained skill.
- The [monitor](../vecl/runtime/monitor.py#L66-L157) checks cardinality, eligibility and scalar
  deltas, then recomputes with the same oracle. LoRA slots pair an A row with a B column;
  [flattening](../vecl/substrate/lora_memory.py#L154-L175) supplies norm summaries. The separate
  [tensor update](../vecl/substrate/lora_memory.py#L195-L251) happens after scalar verification
  ([trainer](../vecl/training/tool_use_loop.py#L255-L274)). Tensor hashes/norms are recorded, but
  the oracle does not independently verify the actual before/after tensor transition.
- Optional [Fisher-weighted snapshot distance](../vecl/evaluation/fisher.py#L149-L197) sums
  slot-weighted squared A/B tensor differences. The trainer can reject/restore an over-threshold
  candidate when configured with a Fisher-bearing reference snapshot and threshold. Defaults are
  `ewc_drift_threshold=None` and `ewc_lambda=0.0`; missing snapshot/threshold yields no drift report
  ([configuration](../vecl/training/tool_use_loop.py#L65-L76),
  [drift handling](../vecl/training/tool_use_loop.py#L294-L345),
  [report condition](../vecl/training/tool_use_loop.py#L668-L677)). This does not establish retention.
- The [ledger](../vecl/provenance/ledger.py#L37-L132) checks linked hashes, event uniqueness,
  earlier parents, tenant consistency of event links, and typed references. Those checks do not
  authenticate caller-supplied evidence or prove that an authorized, correct state transition occurred.

## Reproduced enforcement limits

The audit's in-memory probes used mock specialists, fixture strings, and the scalar CPU path;
they did not execute a model or real tool. Every case below retained a hash-valid ledger.

| Probe | Recorded outcome | Source boundary |
| --- | --- | --- |
| Apply, abort, reuse the same unexpired token, commit | Accepted; one committed event after abort | [Monitor](../vecl/runtime/monitor.py#L47-L64) accepts an existing prepared event; [abort](../vecl/runtime/monitor.py#L159-L172) clears pending state only. |
| Apply, commit, reuse the same unexpired token, commit | Accepted; two committed events | [Commit](../vecl/runtime/monitor.py#L116-L157) clears pending state without enforcing terminal token state. |
| Response contains both a wrong request ID and wrong tenant ID | `PASSED` on direct and chain paths | [Direct](../vecl/qb/orchestrator.py#L88-L130) and [chain](../vecl/qb/chain_executor.py#L129-L175) paths do not validate that binding. |
| Claim names nonexistent evidence; caller supplies two invented independent roots | `PASSED` on both paths with a two-root policy | [Evidence nodes](../vecl/qb/claim_graph.py#L17-L37) and [root labels](../vecl/qb/orchestrator.py#L243-L251) come from supplied strings; [verification](../vecl/qb/verifier.py#L52-L57) counts labels. |
| Mock response has `refusal_or_error='timeout'` and no claims | Direct `PASSED`; chain `FAILED` | Direct path ignores refusal/error; [chain path](../vecl/qb/chain_executor.py#L144-L155) aborts. |
| Input requests a different target tenant, with default ethics rules installed | Direct `PASSED`; chain `FAILED` before specialist invocation | Only the [chain path](../vecl/qb/chain_executor.py#L91-L113) applies the configured ethics gate. |

These cases show specific authorization, response-binding, and evidence-enforcement gaps; they do
not establish that every tenant check fails. Ledger parent/reference tenant checks remain implemented.

Two additional source findings limit rollback and reconstruction claims: `mutated=True` is set only
after the tensor updater returns, so a partial mutation followed by an exception can evade restoration;
post-commit probe failures enter the same abort/restore handler
([update](../vecl/training/tool_use_loop.py#L271-L274),
[commit/probe](../vecl/training/tool_use_loop.py#L421-L440),
[handler](../vecl/training/tool_use_loop.py#L484-L518)). Generic cycles also reuse dataset-derived
snapshot paths ([path construction](../vecl/training/tool_use_loop.py#L195-L213)); the historical
sparse-overfit wrapper instead [separates cycle directories](../scripts/tool_use_train_gemma.py#L766-L780).
These findings were source-inspected, not newly fault-injected or tested on tensors.

## Learning evidence remains mixed

The historical [training milestone](adr/ADR-025-phase-9b-11a-training-milestone.md#L643-L776)
preserves the results and original run identifiers/hashes:

| Historical experiment | Recorded result | Limit |
| --- | --- | --- |
| 1,024–8,192-example sweep | Exact tool-call accuracy stayed 6/16 before and after at every size. | More examples in one averaged update did not demonstrate useful tool-use improvement. |
| Dense LoRA tiny overfit, eight examples, 25 steps | CE 1.544 → 0.018; generation probe disabled. | Memorization diagnostic, not fresh-task or retention evidence. |
| Governed sparse tiny overfit, eight examples, 25 cycles | CE 1.544 → 1.538; failed the predefined 0.05 CE-drop threshold despite 25 commits. | Negative result; optimizer and update regimes differ from dense AdamW, so this does not isolate sparsity as the cause. |

Useful continual-learning gain has not been demonstrated. The objective remains verified specialist
experience followed by improvement on fresh tasks with retention and cost measurements; sparse LoRA
is an implementation hypothesis. The generator sets `target_independently_verified=True` alongside
`target_source='deterministic_template'` ([source](../vecl/training/corpus_factory.py#L1551-L1565)).
That flag is template metadata, not evidence of observed specialist execution or real teacher output.

## Publication materials and pending owner decision

The audit parsed all records in seven reachable JSONL blobs with no malformed records. Recorded
provenance is deterministic synthetic generation; no teacher/provider-derived records were observed.
The older v0 schema lacks the newer explicit LLM-origin flags. This follow-up verified the Git
object/path mappings below without re-parsing or regenerating the corpora.

`current` means commit `e3d4c893002a9a34256a000ba4a174377f4742cd`; `baseline` means
`d9a789162f34ccf8f66ed36dfa05cef8c73f061e`, tagged `baseline-learning-protocol-2026-05-25`.

| Snapshot | Path | Records | Git blob |
| --- | --- | ---: | --- |
| current | `data/synthetic/v0-small/corpus.jsonl` | 420 | `bd05ac87a1b3f64ca4f2e55b15038be2c7569415` |
| baseline | `data/synthetic/v0-small/corpus.jsonl` | 420 | `3b7e9886302bb8b53ed7ce7bcf531d30e3e136e5` |
| baseline | `data/synthetic/v0-full/corpus.jsonl` | 5,000 | `ce06322dcd1a6795f7a69b3287164f7486387c4e` |
| baseline | `data/synthetic/v1-hard/corpus.jsonl` | 25,000 | `ce4a7e37b8b27d6b2a6d81f77134aaf353dbd947` |
| baseline | `data/synthetic/v1-hard/exports/train.jsonl` | 16,993 | `15438e04ab63151cd8c2c169555144c8406a966d` |
| baseline | `data/synthetic/v1-hard/exports/heldout.jsonl` | 973 | `dfaf6a09275c07971adf463ea3fc3bbd55368561` |
| baseline | `data/synthetic/v1-hard/exports/hard-heldout.jsonl` | 947 | `74535edec32481afc424cf7cdaddd6566557922d` |

The tracked [archival DOCX](research/VECL-QB-research-product-document.docx) is also included:
blob `cfbc3b696494b4ec4deed99b37f4fcbe5e53ac07` at the current commit, archived at this path by
`f26f7d0ffa7974b965030ef86720104819173ef4`. It is historical rationale, not a current contract.

The baseline predates `LICENSE`, added at `6a6bab069e1c69f84b575b6dd532447748bf372c`.
Its lack of a license file does not by itself establish a publication prohibition. The baseline is
an ancestor of current main, so omitting its tag would not remove ancestral data blobs.

**Pending owner decision:** confirm whether these seven original synthetic data/export blobs, the
archived DOCX, and their reachable history are intended public, and state the intended licensing
scope for those materials and historical snapshots, including whether the current MIT notice is
intended to cover them. Public visibility alone is not recorded here as that decision. Ownership
and scope have not been inferred; no license, notice, attribution, or historical artifact is changed.
Referenced external weights, tools, and any future provider-generated data have separate provenance.

## Check coverage and follow-up boundary

The audit recorded 94 scoped tests passing, Ruff formatting/lint and mypy passing, and validation
of the 420-record fixture. The central LoRA/trainer selection skipped both modules because `peft`
was missing (two skips, pytest exit 5/no tests collected). Those results do not validate actual
tensor transitions. This follow-up is documentation only; tests, probes, training, and corpus
validation were not rerun, and no dependencies were installed.

Offline documentation checks passed: whitespace/diff checks, local links and source-line anchors,
table structure, and Git object/path identities. Before/after file hashes confirmed the saved
project checkout and original audit reports were unchanged.

Separate future fixes, not implemented here or made outreach prerequisites:

- Enforce terminal token states and shared response/refusal/ethics checks; resolve evidence against
  authenticated same-tenant records rather than supplied labels.
- Verify real tensor transitions and transactional restoration; preserve immutable cycle artifacts.
- Add a non-skipping substrate test lane and distinguish template labels from executed-tool evidence.
  Evaluate useful learning with matched baselines and fresh-task/retention/cost measures separately.
