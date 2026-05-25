from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vecl.evaluation.fisher import (
    accumulate_lora_fisher,
    compute_lora_snapshot_drift,
    load_snapshot_fisher,
    validate_fisher_diagonal,
)
from vecl.substrate.lora_memory import LoRARawGradient as SubstrateLoRARawGradient


def test_accumulate_lora_fisher_from_raw_gradient_batches() -> None:
    estimate = accumulate_lora_fisher(
        [
            {
                0: SubstrateLoRARawGradient(0, np.array([1.0, 2.0]), np.array([3.0])),
                1: SubstrateLoRARawGradient(1, np.array([0.5]), np.array([1.5, 2.0])),
            },
            {
                0: SubstrateLoRARawGradient(0, np.array([2.0, 0.0]), np.array([1.0])),
                1: SubstrateLoRARawGradient(1, np.array([1.0]), np.array([1.0, 1.0])),
            },
        ],
        slot_count=2,
        dataset_items=[{"id": "a"}, {"id": "b"}],
        loss_sum=3.5,
    )

    assert estimate.sample_count == 2
    assert estimate.loss_sum == 3.5
    assert np.allclose(
        estimate.slot_fisher, [(1 + 4 + 9 + 4 + 1) / 2, (0.25 + 2.25 + 4 + 1 + 1 + 1) / 2]
    )


def test_fisher_validation_rejects_bad_vectors() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_fisher_diagonal([1.0, float("nan")], slot_count=2)
    with pytest.raises(ValueError, match="non-negative"):
        validate_fisher_diagonal([1.0, -1.0], slot_count=2)
    with pytest.raises(ValueError, match="shape"):
        validate_fisher_diagonal([1.0], slot_count=2)


def test_compute_lora_snapshot_drift_zero_and_positive(tmp_path: Path) -> None:
    approved = tmp_path / "approved.npz"
    candidate_same = tmp_path / "candidate-same.npz"
    candidate_changed = tmp_path / "candidate-changed.npz"
    _write_snapshot(approved, a_value=1.0, b_value=2.0, fisher=[0.5])
    _write_snapshot(candidate_same, a_value=1.0, b_value=2.0)
    _write_snapshot(candidate_changed, a_value=3.0, b_value=2.0)

    assert load_snapshot_fisher(approved) is not None
    same = compute_lora_snapshot_drift(approved, candidate_same, drift_threshold=0.0)
    changed = compute_lora_snapshot_drift(approved, candidate_changed, drift_threshold=0.1)

    assert same.drift_value == 0.0
    assert same.drift_within_bound
    assert changed.drift_value == 4.0
    assert not changed.drift_within_bound
    assert changed.metadata["top_slots"][0]["slot_id"] == 0


def test_compute_lora_snapshot_drift_rejects_missing_fisher(tmp_path: Path) -> None:
    approved = tmp_path / "approved.npz"
    candidate = tmp_path / "candidate.npz"
    _write_snapshot(approved, a_value=1.0, b_value=2.0)
    _write_snapshot(candidate, a_value=1.0, b_value=2.0)

    with pytest.raises(ValueError, match="fisher_diagonal"):
        compute_lora_snapshot_drift(approved, candidate, drift_threshold=1.0)


def test_unaccumulated_fisher_marker_is_not_a_real_estimate(tmp_path: Path) -> None:
    approved = tmp_path / "approved.npz"
    _write_snapshot(
        approved, a_value=1.0, b_value=2.0, fisher=[0.0], fisher_status="not_accumulated"
    )

    assert load_snapshot_fisher(approved) is None


def _write_snapshot(
    path: Path,
    *,
    a_value: float,
    b_value: float,
    fisher: list[float] | None = None,
    fisher_status: str = "accumulated",
) -> None:
    metadata = {
        "model_class_name": "Tiny",
        "adapter_name": "vecl_sparse",
        "rank": 1,
        "target_modules": ["model.layers.0.mlp.up_proj"],
        "layer_indices": [0],
        "slot_count": 1,
        "slot_metadata": [
            {
                "slot_id": 0,
                "adapter_name": "vecl_sparse",
                "module_name": "model.layers.0.mlp.up_proj",
                "layer_index": 0,
                "rank_index": 0,
                "lora_a_shape": [1, 2],
                "lora_b_shape": [3, 1],
            }
        ],
    }
    arrays = {
        "metadata_json": np.array(json.dumps(metadata)),
        "module_0_name": np.array("model.layers.0.mlp.up_proj"),
        "module_0_lora_a": np.full((1, 2), a_value, dtype=np.float64),
        "module_0_lora_b": np.full((3, 1), b_value, dtype=np.float64),
    }
    if fisher is not None:
        arrays["fisher_diagonal"] = np.asarray(fisher, dtype=np.float64)
        arrays["fisher_metadata_json"] = np.array(
            json.dumps(
                {
                    "sample_count": 1,
                    "dataset_hash": "dataset",
                    "loss_sum": 1.0,
                    "fisher_status": fisher_status,
                }
            )
        )
    np.savez_compressed(path, **arrays)
