import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.topk import deterministic_top_k
from vecl.sparse.types import SparseMemoryInputs


@given(
    st.lists(
        st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=8,
    ),
    st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_without_threshold_positive_scaling_preserves_topk(
    base_scores: list[float], alpha: float
) -> None:
    # Architectural lesson: TopK is a ranking operation, not a trust-suppression mechanism.
    eligible = list(range(len(base_scores)))
    k = min(3, len(base_scores))
    assert deterministic_top_k(base_scores, eligible, k) == deterministic_top_k(
        np.asarray(base_scores) * alpha, eligible, k
    )


def test_threshold_blocks_low_authority_source() -> None:
    inputs = SparseMemoryInputs(
        memory_values=np.array([1.0]),
        gradients=np.array([1.0]),
        activation=np.array([1000.0]),
        rarity=np.array([1000.0]),
        authority=np.array([0.001]),
        learning_rate=0.1,
        min_authority=0.1,
        min_score=0.0,
        max_slots=1,
        quarantined_slots=set(),
    )
    assert sparse_update_oracle(inputs, "evt", "root").selected_slots == []


def test_authority_scaled_gradient_norm_tends_toward_zero() -> None:
    base = dict(
        memory_values=np.array([1.0]),
        gradients=np.array([10.0]),
        activation=np.array([1.0]),
        rarity=np.array([1.0]),
        learning_rate=1.0,
        min_authority=0.0,
        min_score=0.0,
        max_slots=1,
        quarantined_slots=set(),
    )
    high = SparseMemoryInputs(authority=np.array([0.1]), **base)
    low = SparseMemoryInputs(authority=np.array([0.0001]), **base)
    high_result = sparse_update_oracle(high, "evt-high", "root")
    low_result = sparse_update_oracle(low, "evt-low", "root")
    assert np.linalg.norm(low_result.new_memory_values - low.memory_values) < np.linalg.norm(
        high_result.new_memory_values - high.memory_values
    )


def test_thresholded_eligibility_empty_below_threshold() -> None:
    inputs = SparseMemoryInputs(
        memory_values=np.array([1.0, 1.0]),
        gradients=np.array([1.0, 1.0]),
        activation=np.array([100.0, 100.0]),
        rarity=np.array([1.0, 1.0]),
        authority=np.array([0.01, 0.02]),
        learning_rate=0.1,
        min_authority=0.1,
        min_score=0.0,
        max_slots=2,
        quarantined_slots=set(),
    )
    result = sparse_update_oracle(inputs, "evt", "root")
    assert result.eligible_slots == []
    assert result.selected_slots == []
