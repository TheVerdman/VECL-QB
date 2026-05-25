import numpy as np
import pytest

from vecl.sparse.types import SparseMemoryInputs


def _valid_inputs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "memory_values": np.array([1.0, 2.0]),
        "gradients": np.array([0.1, 0.2]),
        "activation": np.array([1.0, 1.0]),
        "rarity": np.array([1.0, 1.0]),
        "authority": np.array([0.5, 0.5]),
        "learning_rate": 0.1,
        "min_authority": 0.1,
        "min_score": 0.1,
        "max_slots": 1,
        "quarantined_slots": set(),
    }
    values.update(overrides)
    return values


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        SparseMemoryInputs(**_valid_inputs(gradients=np.array([1.0])))


def test_nan_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        SparseMemoryInputs(**_valid_inputs(activation=np.array([np.nan, 1.0])))


def test_inf_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        SparseMemoryInputs(**_valid_inputs(rarity=np.array([np.inf, 1.0])))


def test_invalid_max_slots_rejected() -> None:
    with pytest.raises(ValueError, match="max_slots"):
        SparseMemoryInputs(**_valid_inputs(max_slots=-1))


def test_invalid_quarantine_index_rejected() -> None:
    with pytest.raises(ValueError, match="quarantined"):
        SparseMemoryInputs(**_valid_inputs(quarantined_slots={2}))
