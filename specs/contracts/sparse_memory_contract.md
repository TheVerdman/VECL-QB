# Sparse Memory Update Contract

This contract defines production semantics for VECL-QB sparse memory updates. The CPU oracle must implement these rules exactly; accelerated implementations must refine and match the oracle.

## State Variables

- `memory_values`: numeric vector of current memory slot values.
- `gradients`: numeric vector of proposed update gradients.
- `activation`: numeric vector measuring current replay/task activation.
- `rarity`: numeric vector weighting rare or underrepresented evidence.
- `authority`: numeric vector carrying externally governed trust authority per slot.
- `score`: numeric vector computed from activation, rarity, and authority.
- `eligible_slots`: ordered set/list of slot ids that may be selected.
- `selected_slots`: deterministic TopK subset of eligible slots.
- `learning_event_token`: prepared runtime authorization for this update.
- `delta_records`: provenance records for selected slot deltas.

All numeric vectors have the same length `N`, indexed by slot id `i` where `0 <= i < N`.

## Score Rule

Mathematical form:

```text
score[i] = activation[i] * rarity[i] * authority[i]
```

Plain English: a slot becomes attractive for replay only when it is active, rare, and backed by authority. A high activation value alone is not sufficient.

## Eligibility Rule

Mathematical form:

```text
eligible[i] iff
  authority[i] >= min_authority
  and score[i] >= min_score
  and i notin quarantined_slots
```

Plain English: a slot must pass both an authority threshold and a score threshold, and quarantined slots are never eligible.

## Selection Rule

Mathematical form:

```text
selected_slots = deterministic_top_k(eligible_slots, score, max_slots)
```

The deterministic ordering is:

1. Higher `score` first.
2. For equal scores, lower `slot_id` first.

Plain English: TopK is allowed to rank only eligible slots. Ties are stable so repeated runs select the same slots.

## Update Rule

For selected slots:

```text
memory_values'[i] =
  memory_values[i] - learning_rate * authority[i] * gradients[i]
```

For non-selected slots:

```text
memory_values'[i] = memory_values[i]
```

Plain English: authority scales the update magnitude after eligibility has already decided whether the slot is allowed to move.

## Provenance Rule

No update may occur without a prepared, non-expired learning event token. Each selected slot must have a provenance-visible record with:

- event id
- slot id
- old value
- new value
- delta
- score
- authority
- provenance root

The CPU oracle emits a delta record for every selected slot, including zero-delta selections, so selection itself remains auditable.

## Invariants

- `NoUpdateWithoutLearningEvent`: if no valid learning event token exists, `memory_values' = memory_values`.
- `BoundedUpdateSize`: `Len(selected_slots) <= max_slots`.
- `NoIneligibleSlotUpdated`: if `i notin eligible_slots`, then `memory_values'[i] = memory_values[i]`.
- `SelectedSlotsSubsetEligible`: every selected slot is eligible.
- `ZeroAuthorityNoUpdate`: if `authority[i] = 0` and `min_authority > 0`, slot `i` is not updated.
- `ZeroScoreNoUpdate`: if `score[i] = 0` and `min_score > 0`, slot `i` is not updated.
- `DeterministicTieBreaking`: equal scores are ordered by lower slot id.
- `EveryDeltaHasProvenance`: each selected slot has a delta record tied to the learning event and provenance root.
- `NoNaNOrInf`: inputs, scores, and output memory values are finite.
- `HostVerifiesBeforeCommit`: host/runtime verification must pass before committed state becomes visible.

## TopK Is Not Trust Suppression

TopK alone is insufficient for provenance stability because rank order is invariant under positive scalar multiplication:

```text
TopK(alpha * base_scores) = TopK(base_scores), for alpha > 0
```

Plain English: multiplying all scores by a small positive authority value preserves their rank. A low-authority adversarial source can still dominate TopK if no threshold blocks eligibility. Therefore trust suppression requires thresholded eligibility and authority-scaled update magnitude, not ranking alone.
