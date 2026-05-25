from __future__ import annotations

import json
import os

import numpy as np
import pytest

from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs
from vecl.substrate.lora_memory import LoRAMemorySubstrate, LoRARawGradient

torch = pytest.importorskip("torch")
pytest.importorskip("peft")


class _TinyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)


class _TinyBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.mlp = _TinyMLP(hidden_size, intermediate_size)


class _TinyBackbone(torch.nn.Module):
    def __init__(self, layer_count: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_TinyBlock(hidden_size, intermediate_size) for _ in range(layer_count)]
        )


class _TinyGemmaShapedModel(torch.nn.Module):
    def __init__(
        self, layer_count: int = 10, hidden_size: int = 5, intermediate_size: int = 7
    ) -> None:
        super().__init__()
        self.config = {"model_type": "gemma"}
        self.model = _TinyBackbone(layer_count, hidden_size, intermediate_size)

    def forward(self, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        del input_ids
        return torch.zeros(1)


def _make_substrate(
    *, layer_count: int = 10, rank: int = 3, last_n_layers: int = 8
) -> LoRAMemorySubstrate:
    torch.manual_seed(1234)
    model = _TinyGemmaShapedModel(layer_count=layer_count)
    return LoRAMemorySubstrate.attach(model, rank=rank, last_n_layers=last_n_layers)


def _raw_gradients(
    substrate: LoRAMemorySubstrate, *, fill: float = 1.0
) -> dict[int, LoRARawGradient]:
    gradients: dict[int, LoRARawGradient] = {}
    for slot_id, slot in substrate._slot_by_id.items():  # noqa: SLF001 - test inspects slots.
        a_row, b_column = substrate._slot_tensors(slot)  # noqa: SLF001 - test inspects tensors.
        gradients[slot_id] = LoRARawGradient(
            slot_id=slot_id,
            lora_a_row=torch.full_like(a_row, fill),
            lora_b_column=torch.full_like(b_column, fill),
        )
    return gradients


def _oracle_result(
    substrate: LoRAMemorySubstrate,
    raw_gradients: dict[int, LoRARawGradient],
    *,
    max_slots: int = 3,
    learning_rate: float = 0.2,
) -> tuple[SparseMemoryInputs, object]:
    memory_values = substrate.flatten()
    gradient_values = substrate.gradient_summaries(raw_gradients)
    inputs = SparseMemoryInputs(
        memory_values=memory_values,
        gradients=gradient_values,
        activation=gradient_values,
        rarity=np.ones(substrate.slot_count),
        authority=np.linspace(0.25, 1.0, substrate.slot_count),
        learning_rate=learning_rate,
        min_authority=0.0,
        min_score=0.0,
        max_slots=max_slots,
        quarantined_slots=set(),
    )
    return inputs, sparse_update_oracle(inputs, "evt-lora", "root-lora")


def _slot_hashes(substrate: LoRAMemorySubstrate) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    copies: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for slot_id, slot in substrate._slot_by_id.items():  # noqa: SLF001 - test inspects slots.
        a_row, b_column = substrate._slot_tensors(slot)  # noqa: SLF001 - test inspects tensors.
        copies[slot_id] = (a_row.detach().clone(), b_column.detach().clone())
    return copies


def test_attach_discovers_last_eight_up_proj_layers_and_freezes_base_weights() -> None:
    substrate = _make_substrate(rank=3)

    assert substrate.slot_count == 8 * 3
    assert sorted({metadata.layer_index for metadata in substrate.slot_metadata}) == list(
        range(2, 10)
    )
    assert [metadata.slot_id for metadata in substrate.slot_metadata] == list(
        range(substrate.slot_count)
    )
    assert all(metadata.module_name.endswith("mlp.up_proj") for metadata in substrate.slot_metadata)

    for name, parameter in substrate.model.named_parameters():
        if "lora_A.vecl_sparse" in name or "lora_B.vecl_sparse" in name:
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad


def test_from_peft_model_reconstructs_deterministic_slot_metadata() -> None:
    first = _make_substrate(rank=2)
    second = LoRAMemorySubstrate.from_peft_model(first.model)
    third = _make_substrate(rank=2)

    assert second.slot_metadata == first.slot_metadata
    assert third.slot_metadata == first.slot_metadata
    inner_model = first.model.base_model.model
    assert LoRAMemorySubstrate.from_peft_model(inner_model).slot_metadata == first.slot_metadata


def test_flatten_and_gradient_summaries_are_float64_finite_vectors() -> None:
    substrate = _make_substrate(rank=2, last_n_layers=4)
    raw_gradients = _raw_gradients(substrate, fill=0.5)

    values = substrate.flatten()
    gradients = substrate.gradient_summaries(raw_gradients)

    assert values.dtype == np.float64
    assert gradients.dtype == np.float64
    assert values.shape == (8,)
    assert gradients.shape == (8,)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))
    assert np.all(gradients > 0)


def test_oracle_selects_slots_and_apply_update_mutates_only_selected_components() -> None:
    substrate = _make_substrate(rank=2, last_n_layers=3)
    raw_gradients = _raw_gradients(substrate, fill=0.25)
    before = _slot_hashes(substrate)
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in substrate.model.named_parameters()
        if "lora_A" not in name and "lora_B" not in name
    }

    inputs, result = _oracle_result(substrate, raw_gradients, max_slots=2, learning_rate=0.3)
    del inputs
    delta_records = substrate.apply_update(result, raw_gradients, learning_rate=0.3)

    assert [record.slot_id for record in delta_records] == result.selected_slots
    selected = set(result.selected_slots)
    after = _slot_hashes(substrate)
    for slot_id, (before_a, before_b) in before.items():
        after_a, after_b = after[slot_id]
        if slot_id in selected:
            assert not torch.allclose(before_a, after_a)
            assert not torch.allclose(before_b, after_b)
        else:
            assert torch.allclose(before_a, after_a)
            assert torch.allclose(before_b, after_b)

    for name, parameter in substrate.model.named_parameters():
        if name in base_before:
            assert torch.allclose(base_before[name], parameter.detach())


def test_snapshot_restore_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    substrate = _make_substrate(rank=2, last_n_layers=2)
    snapshot_path = tmp_path / "lora_snapshot.npz"
    before = substrate.flatten()
    substrate.snapshot(snapshot_path)
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        assert np.allclose(snapshot["fisher_diagonal"], np.zeros(substrate.slot_count))
        fisher_metadata = json.loads(str(snapshot["fisher_metadata_json"].item()))
        assert fisher_metadata["fisher_status"] == "not_accumulated"
        assert fisher_metadata["sample_count"] == 0

    raw_gradients = _raw_gradients(substrate, fill=0.2)
    _inputs, result = _oracle_result(substrate, raw_gradients, max_slots=2)
    substrate.apply_update(result, raw_gradients, learning_rate=0.2)
    assert not np.allclose(before, substrate.flatten())

    substrate.restore(snapshot_path)
    assert np.allclose(before, substrate.flatten())


def test_snapshot_can_store_optional_fisher_metadata(tmp_path) -> None:  # type: ignore[no-untyped-def]
    substrate = _make_substrate(rank=2, last_n_layers=2)
    snapshot_path = tmp_path / "lora_snapshot_with_fisher.npz"
    fisher = np.linspace(0.1, 1.0, substrate.slot_count)

    substrate.snapshot(
        snapshot_path,
        fisher_diagonal=fisher,
        fisher_metadata={"sample_count": 2, "dataset_hash": "dataset", "loss_sum": 1.5},
    )

    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        assert np.allclose(snapshot["fisher_diagonal"], fisher)
        fisher_metadata = json.loads(str(snapshot["fisher_metadata_json"].item()))
        assert fisher_metadata["sample_count"] == 2
        assert fisher_metadata["fisher_status"] == "accumulated"
    substrate.restore(snapshot_path)


def test_fisher_weighted_drift_penalty_is_differentiable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    substrate = _make_substrate(rank=2, last_n_layers=2)
    snapshot_path = tmp_path / "approved_with_fisher.npz"
    substrate.snapshot(
        snapshot_path,
        fisher_diagonal=np.ones(substrate.slot_count),
        fisher_metadata={"sample_count": 1, "dataset_hash": "dataset", "loss_sum": 1.0},
    )
    assert (
        float(substrate.fisher_weighted_drift_penalty(snapshot_path).detach().cpu().item()) == 0.0
    )

    first_lora_a = next(
        parameter for name, parameter in substrate.model.named_parameters() if "lora_A" in name
    )
    with torch.no_grad():
        first_lora_a.add_(0.01)
    penalty = substrate.fisher_weighted_drift_penalty(snapshot_path)

    assert float(penalty.detach().cpu().item()) > 0.0
    penalty.backward()
    assert first_lora_a.grad is not None
    assert bool(torch.isfinite(first_lora_a.grad).all())


def test_fisher_weighted_drift_penalty_rejects_unaccumulated_fisher(tmp_path) -> None:  # type: ignore[no-untyped-def]
    substrate = _make_substrate(rank=2, last_n_layers=2)
    snapshot_path = tmp_path / "unaccumulated.npz"
    substrate.snapshot(snapshot_path)

    with pytest.raises(ValueError, match="not accumulated"):
        substrate.fisher_weighted_drift_penalty(snapshot_path)


def test_raw_gradients_from_current_grads_returns_slot_shaped_gradients() -> None:
    substrate = _make_substrate(rank=2, last_n_layers=2)
    loss = None
    for name, parameter in substrate.model.named_parameters():
        if "lora_A.vecl_sparse" in name or "lora_B.vecl_sparse" in name:
            term = parameter.square().sum()
            loss = term if loss is None else loss + term
    assert loss is not None
    loss.backward()

    raw_gradients = substrate.raw_gradients_from_current_grads()
    summaries = substrate.gradient_summaries(raw_gradients)

    assert sorted(raw_gradients) == list(range(substrate.slot_count))
    assert summaries.shape == (substrate.slot_count,)
    assert np.all(np.isfinite(summaries))


def test_restore_rejects_incompatible_rank_and_layer_selection(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = _make_substrate(rank=2, last_n_layers=4)
    snapshot_path = tmp_path / "lora_snapshot.npz"
    source.snapshot(snapshot_path)

    with pytest.raises(ValueError, match="metadata"):
        _make_substrate(rank=3, last_n_layers=4).restore(snapshot_path)
    with pytest.raises(ValueError, match="metadata"):
        _make_substrate(rank=2, last_n_layers=2).restore(snapshot_path)


def test_restore_rejects_tensor_shape_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    substrate = _make_substrate(rank=2, last_n_layers=2)
    snapshot_path = tmp_path / "lora_snapshot.npz"
    bad_snapshot_path = tmp_path / "bad_lora_snapshot.npz"
    substrate.snapshot(snapshot_path)

    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        arrays = {name: snapshot[name] for name in snapshot.files}
        arrays["module_0_lora_a"] = arrays["module_0_lora_a"][:1]
        metadata = json.loads(str(arrays["metadata_json"].item()))
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(bad_snapshot_path, **arrays)

    with pytest.raises(ValueError, match="shape mismatch"):
        substrate.restore(bad_snapshot_path)


def test_apply_update_rejects_missing_or_invalid_selected_gradients() -> None:
    substrate = _make_substrate(rank=2, last_n_layers=2)
    raw_gradients = _raw_gradients(substrate)
    _inputs, result = _oracle_result(substrate, raw_gradients, max_slots=1)

    with pytest.raises(ValueError, match="missing raw gradient"):
        substrate.apply_update(result, {}, learning_rate=0.1)

    selected_slot = result.selected_slots[0]
    invalid = dict(raw_gradients)
    invalid[selected_slot] = LoRARawGradient(
        slot_id=selected_slot,
        lora_a_row=torch.ones(1),
        lora_b_column=raw_gradients[selected_slot].lora_b_column,
    )
    with pytest.raises(ValueError, match="LoRA A gradient shape mismatch"):
        substrate.apply_update(result, invalid, learning_rate=0.1)


@pytest.mark.skipif(
    os.environ.get("VECL_RUN_GEMMA_TESTS") != "1",
    reason="set VECL_RUN_GEMMA_TESTS=1, HF_TOKEN, and VECL_GEMMA_MODEL_ID to run Gemma LoRA test",
)
def test_opt_in_real_gemma_lora_slot_discovery() -> None:
    transformers = pytest.importorskip("transformers")
    model_id = os.environ.get("VECL_GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN is required for opt-in Gemma integration test")
    model_cls = transformers.AutoModelForImageTextToText
    model = model_cls.from_pretrained(model_id, token=token, device_map="auto")

    substrate = LoRAMemorySubstrate.attach(model, rank=2)

    assert substrate.slot_count > 0
    assert all(metadata.module_name.endswith("mlp.up_proj") for metadata in substrate.slot_metadata)
    assert substrate.slot_metadata == LoRAMemorySubstrate.from_peft_model(model).slot_metadata
