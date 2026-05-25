import numpy as np
import pytest

from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs


def _inputs(**overrides: object) -> SparseMemoryInputs:
    values: dict[str, object] = {
        "memory_values": np.array([1.0, 1.0, 1.0]),
        "gradients": np.array([1.0, 2.0, 3.0]),
        "activation": np.array([1.0, 10.0, 1.0]),
        "rarity": np.array([1.0, 1.0, 1.0]),
        "authority": np.array([1.0, 0.01, 0.5]),
        "learning_rate": 0.1,
        "min_authority": 0.1,
        "min_score": 0.1,
        "max_slots": 2,
        "quarantined_slots": set(),
    }
    values.update(overrides)
    return SparseMemoryInputs(**values)


def test_zero_authority_produces_no_update_when_threshold_positive() -> None:
    result = sparse_update_oracle(
        _inputs(authority=np.array([0.0, 0.0, 0.0]), min_authority=0.1), "evt", "root"
    )
    assert result.selected_slots == []
    assert np.allclose(result.new_memory_values, [1.0, 1.0, 1.0])


def test_zero_score_produces_no_update_when_min_score_positive() -> None:
    result = sparse_update_oracle(_inputs(activation=np.zeros(3)), "evt", "root")
    assert result.selected_slots == []


def test_low_authority_high_activation_blocked_below_threshold() -> None:
    result = sparse_update_oracle(_inputs(), "evt", "root")
    assert 1 not in result.selected_slots


def test_authority_scales_update_magnitude() -> None:
    result = sparse_update_oracle(
        _inputs(authority=np.array([1.0, 1.0, 0.5]), max_slots=3), "evt", "root"
    )
    assert np.isclose(result.new_memory_values[0], 0.9)
    assert np.isclose(result.new_memory_values[2], 0.85)


def test_only_selected_slots_change_and_bounded() -> None:
    inputs = _inputs(max_slots=1)
    result = sparse_update_oracle(inputs, "evt", "root")
    changed = set(np.where(~np.isclose(inputs.memory_values, result.new_memory_values))[0])
    assert changed.issubset(set(result.selected_slots))
    assert len(changed) <= inputs.max_slots


def test_every_changed_slot_has_delta_record() -> None:
    inputs = _inputs(max_slots=2)
    result = sparse_update_oracle(inputs, "evt", "root")
    changed = set(np.where(~np.isclose(inputs.memory_values, result.new_memory_values))[0])
    assert changed.issubset({record.slot_id for record in result.delta_records})


def test_refuses_missing_event_or_provenance() -> None:
    with pytest.raises(ValueError):
        sparse_update_oracle(_inputs(), "", "root")
    with pytest.raises(ValueError):
        sparse_update_oracle(_inputs(), "evt", "")
