from __future__ import annotations

import numpy as np

from vecl.sparse.topk import deterministic_top_k
from vecl.sparse.types import DeltaRecord, SparseMemoryInputs, SparseUpdateResult


def sparse_update_oracle(
    inputs: SparseMemoryInputs, event_id: str, provenance_root: str
) -> SparseUpdateResult:
    """CPU reference semantics for sparse memory updates.

    A DeltaRecord is emitted for every selected slot, including zero-delta
    selections, so selection itself remains provenance-visible.
    """

    if not event_id:
        raise ValueError("event_id must be non-empty")
    if not provenance_root:
        raise ValueError("provenance_root must be non-empty")

    scores = inputs.activation * inputs.rarity * inputs.authority
    if not np.all(np.isfinite(scores)):
        raise ValueError("computed scores must not contain NaN or Inf")

    eligible_slots = [
        slot
        for slot, score in enumerate(scores)
        if inputs.authority[slot] >= inputs.min_authority
        and score >= inputs.min_score
        and slot not in inputs.quarantined_slots
    ]
    selected_slots = deterministic_top_k(scores, eligible_slots, inputs.max_slots)

    new_values = inputs.memory_values.copy()
    records: list[DeltaRecord] = []
    for slot in selected_slots:
        old_value = float(new_values[slot])
        delta = -float(inputs.learning_rate * inputs.authority[slot] * inputs.gradients[slot])
        new_value = old_value + delta
        new_values[slot] = new_value
        records.append(
            DeltaRecord(
                event_id=event_id,
                slot_id=slot,
                old_value=old_value,
                new_value=float(new_value),
                delta=float(delta),
                score=float(scores[slot]),
                authority=float(inputs.authority[slot]),
                provenance_root=provenance_root,
            )
        )

    return SparseUpdateResult(
        new_memory_values=new_values,
        scores=scores,
        eligible_slots=eligible_slots,
        selected_slots=selected_slots,
        delta_records=records,
    )
