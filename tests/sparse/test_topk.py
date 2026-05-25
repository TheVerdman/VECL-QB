import numpy as np
import pytest

from vecl.sparse.topk import deterministic_top_k


def test_ordinary_ranking() -> None:
    assert deterministic_top_k(np.array([0.2, 0.7, 0.4]), [0, 1, 2], 2) == [1, 2]


def test_tie_breaking_lower_slot_id() -> None:
    assert deterministic_top_k(np.array([0.5, 0.5, 0.4]), [1, 0, 2], 2) == [0, 1]


def test_k_larger_than_eligible_set() -> None:
    assert deterministic_top_k(np.array([0.2, 0.7]), [1], 5) == [1]


def test_k_zero() -> None:
    assert deterministic_top_k(np.array([0.2, 0.7]), [0, 1], 0) == []


def test_empty_eligible_set() -> None:
    assert deterministic_top_k(np.array([0.2, 0.7]), [], 1) == []


def test_ineligible_high_score_not_selected() -> None:
    assert deterministic_top_k(np.array([100.0, 1.0]), [1], 1) == [1]


def test_deterministic_repeated_calls() -> None:
    scores = np.array([0.5, 0.5, 0.5])
    assert deterministic_top_k(scores, [2, 1, 0], 2) == deterministic_top_k(scores, [0, 1, 2], 2)


def test_nan_inf_scores_rejected() -> None:
    with pytest.raises(ValueError):
        deterministic_top_k(np.array([np.nan]), [0], 1)
    with pytest.raises(ValueError):
        deterministic_top_k(np.array([np.inf]), [0], 1)
