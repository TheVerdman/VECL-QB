# ADR-010: LoRA Slot Semantics For The Sparse Update Oracle

## Context

Phase 1 introduced `LoRAMemorySubstrate` (`vecl/substrate/lora_memory.py`) as the concrete substrate the sparse update oracle operates on. The oracle's input schema (`SparseMemoryInputs`) was designed earlier for a hypothetical 1-D memory value vector and expects:

- `memory_values: NDArray[float64]` (1-D)
- `gradients: NDArray[float64]` (1-D, same length)
- `activation`, `rarity`, `authority`: 1-D summaries used in `score_i = a_i · r_i · α_i`
- `selected_slots`: top-k by score subject to eligibility thresholds

A LoRA "row" is multi-dimensional (`lora_A[k, :]` has shape `(in_features,)`, paired `lora_B[:, k]` has shape `(out_features,)`). The substrate had to decide how multi-dimensional LoRA parameters map onto the oracle's 1-D vector contract without changing the existing schema (which is depended on by the existing sparse update tests, monitor, ledger, and sleep cycle code).

This decision is load-bearing for ADR-013 (per-slot Fisher diagonals) and ADR-012 (in-block memory's eventual coexistence with the LoRA substrate).

## Decision

**One VECL slot = one rank index `k` in one target module**, binding `lora_A[k, :]` (row) and `lora_B[:, k]` (column) as a single addressable unit. The pair is updated together; provenance treats them as one slot.

- `slot_count = rank × num_target_modules`. For E4B with `rank=8` and the last 8 layers' `mlp.up_proj` targeted, this is 64 slots.
- Slot ordering is deterministic by `(layer_index, module_name, rank_index)`.
- `flatten()` returns a 1-D `float64` array of **per-slot summary scalars** — currently `sqrt(||lora_A_row||² + ||lora_B_column||²)`. This summary is *not* the slot's underlying state; it is a scalar used by the oracle for selection and provenance.
- `gradient_summaries(raw_gradients)` returns the same shape from raw gradient tensors, using the same norm.
- The oracle's existing math (`score = activation × rarity × authority`, top-k, eligibility thresholds) runs unchanged on these per-slot summaries.
- `apply_update(result, raw_gradients, learning_rate)` is where the substrate applies the **real tensor update** to selected slots: `a_row.sub_(lr · authority · grad_a)` and `b_column.sub_(lr · authority · grad_b)`. The oracle's `new_memory_values` arithmetic is *not* applied to the underlying weights.
- `LoRATensorDeltaRecord` captures per-slot delta norms and before/after tensor hashes (SHA-256) for provenance — richer than the oracle's `DeltaRecord`.
- Snapshots persist the raw `lora_A` and `lora_B` tensors plus deterministic metadata (model class, adapter name, rank, target modules, layer indices, slot count, per-slot metadata). Restore validates metadata compatibility before copying tensors.

**The oracle ranks slots; the substrate applies the real update.** This separation preserves the `SparseMemoryInputs` schema while letting the substrate carry parameter state that does not fit in a 1-D vector.

## Consequences

- `SparseMemoryInputs.memory_values` for this substrate is semantically a per-slot summary, not slot state. Callers must use `apply_update` for the real state change; treating the oracle's `result.new_memory_values` as the new substrate state would be incorrect.
- `SparseUpdateMonitor.commit_learning_event` stores `result.new_memory_values.copy()` as `committed_memory_values`. For the LoRA substrate, that captured value is a summary snapshot, not substrate state. True substrate state is captured via the substrate's `snapshot()` method, and the ledger references it via tensor hashes in `LoRATensorDeltaRecord.after_tensor_hash`. ADR-013's Fisher persistence rides alongside these snapshots.
- ADR-013's Fisher diagonal computation maps naturally onto this slot definition: `F_i` is one scalar per `(lora_A row, lora_B column)` pair, since the pair is the smallest unit the substrate updates.
- Adding new target modules (`down_proj`, `gate_proj`, attention projections) just adds more slots in the same shape; no schema or oracle change required.
- When in-block memory is eventually added per ADR-012, it will have its own slot definition (one slot per K/V row in the in-block layer). The runtime must therefore accept `Sequence[Substrate]` and route ranking/updating per substrate, rather than assume a single LoRA substrate.
- Slot count grows linearly with rank and number of targeted modules; remains tractable at the scales we care about (≤ a few hundred slots for E4B and 31B configurations).

## Rejected Alternatives

- **Scalar row scales** — Codex's initial Phase 1 plan: each slot = one scalar coefficient on a Kaiming-random frozen direction, applied as `row = scale · base_direction`. Collapses the learnable subspace per slot from `row_dim` to 1, defeats half of LoRA's parameterization (`lora_A` rows can no longer learn direction), and reduces gradients to projections onto a random direction. Rejected: too expressive a sacrifice for a schema-only benefit.
- **Per-element slots** — each scalar element of `lora_A` and `lora_B` is its own slot. Gives extreme fine-grain but explodes slot count (millions for E4B); per-element provenance is overkill; not what SMFT means by a slot.
- **MLP-row surgery (textbook SMFT)** — slots are rows of Gemma's actual MLP weights, updated directly. Closest to the Meta paper's setup. Rejected per ADR-009 — out of scope for the LoRA substrate path; rollback safety and base-model stability arguments dominate. Re-examinable when ADR-012's deferred research is taken up.
- **Vector-valued sparse slots** (changing `SparseMemoryInputs` so `memory_values[i]` is itself a tensor) — would let the oracle operate directly on row tensors. Rejected: invasive schema change, breaks every existing sparse-memory test and the CPU oracle's vector math, propagates 2-D thinking through the entire stack for a localized substrate concern.
