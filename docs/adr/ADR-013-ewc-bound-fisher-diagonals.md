# ADR-013: EWC Bound With Fisher-Diagonal Drift Constraint

## Context

The platform thesis (ADR-011) lists *bounded parameter drift* among the operational properties that distinguish the system from opaque frontier models. The repository today bounds *trust drift* through `BoundedTrustUpdater`, but parameter drift — how far the LoRA substrate has moved from the last release-gate-approved snapshot — is unbounded. Without a formal bound, "bounded drift" is rhetoric, not a measurable property.

The Elastic Weight Consolidation method (Kirkpatrick et al., 2017) provides a well-trodden technique for this exact problem: penalize updates to parameters that were important for previously-mastered tasks, with importance estimated by the diagonal of the Fisher information matrix.

## Decision

The tool-use training loop (Phase 11) implements an EWC-style bound on parameter drift for selected LoRA rows. The mechanism:

1. **After release-gate approval**, compute the Fisher diagonal `F_i` per LoRA row (the rank-paired `lora_A` row and `lora_B` column treated as one slot per ADR-010 semantics) over a held-out evaluation set. Snapshot the approved parameters `θ_i_approved` alongside.
2. **During subsequent sleep-cycle training**, the loss becomes:

   ```
   L_total = L_task + λ · Σ_{i ∈ selected} F_i · (θ_i − θ_i_approved)²
   ```

   where the sum is taken over slots selected by the existing top-t sparse-update oracle. The penalty term applies only to slots being updated; non-selected slots are unaffected.
3. **Before the next release-gate approval**, the gate computes the Fisher-weighted drift `D = Σ_i F_i · (θ_i − θ_i_approved)²` and rejects any candidate snapshot for which `D > τ`, where `τ` is the configured drift bound. Rejection emits `RELEASE_REJECTED` with `D` recorded in the event payload.
4. **Drift is provenance-visible.** The release-gate event payload carries the computed `D`, the configured `τ`, and the snapshot hashes of both the approved baseline and the candidate.

This applies to the LoRA-row substrate as currently implemented. When in-block memory is eventually added (ADR-012), an analogous Fisher-weighted bound on those parameters will be required; the formula generalizes without change.

## Consequences

- "Bounded parameter drift" becomes a measurable property, not a marketing claim. Operators in regulated contexts can be quoted concrete drift numbers per release cycle.
- The release gate gains a concrete pre-approval check beyond its current boolean inputs. ADR-008's `ReleaseGateEvaluation` schema gains a `drift_within_bound: bool` field plus a numeric `drift_value` and `drift_threshold` for evidence.
- Phase 11 of the action plan splits into:
  - **11a** — training-loop plumbing on synthetic data with the EWC penalty term wired but `λ = 0`. Validates the gradient flow, Fisher snapshot, and bound-check infrastructure end-to-end.
  - **11b** — full EWC enabled (`λ > 0`), Fisher computed on real eval data, drift-bound check enforced.
- Fisher diagonal computation adds non-trivial cost to each release-gate cycle: one full-eval-set forward+backward pass per slot in scope. This is acceptable at sleep-cycle cadence (hours to days between approvals) but would not be acceptable at training-step cadence.
- `λ` and `τ` become release-gate-tracked configuration. Changing them is a governance action, not an inline tweak.
- Snapshots stored by `LoRAMemorySubstrate.snapshot()` must include `θ_approved` and `F_diagonal` so the next cycle's penalty term can be computed without re-running the eval.

## Rejected Alternatives

- **No formal drift bound** — the platform thesis's "bounded drift" claim becomes rhetorical, and the release gate stays governance theatre. Rejected because the thesis claim is load-bearing for the operational-seriousness pitch.
- **Hard L2-distance bound without Fisher weighting** — overconstrains unimportant parameters and under-constrains important ones (the standard EWC critique). Rejected because the LoRA substrate has substantial heterogeneity in row importance.
- **Online Fisher computation** (computed continuously during training rather than once per release-gate cycle) — easier in some respects but less stable, and the cost amortization argument inverts (you pay per-step instead of per-cycle). Deferred as a future optimization.
- **Per-layer rather than per-slot Fisher** — coarser than the slot semantics from ADR-010 and loses the locality that the sparse-update oracle relies on.
