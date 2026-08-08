from datetime import datetime

from vecl._compat import UTC
from vecl.trust.updater import BoundedTrustUpdater, LearnedTrustState, TrustUpdateObservation


def _state(value: float = 0.5) -> LearnedTrustState:
    return LearnedTrustState("source", value, datetime.now(UTC), 0)


def test_adversarial_repeated_self_confirmation_does_not_increase_effective_trust() -> None:
    updater = BoundedTrustUpdater()
    state = _state(0.5)
    observations = [
        TrustUpdateObservation("source", 1.0, "PASSED", 0.0, self_reported=True) for _ in range(20)
    ]
    new_state = updater.update(0.5, state, observations)
    assert updater.effective_trust(0.5, new_state.learned_trust) == 0.5


def test_independent_verification_can_increase_within_cap() -> None:
    updater = BoundedTrustUpdater(max_total_drift=0.2)
    new_state = updater.update(
        0.5,
        _state(0.5),
        [TrustUpdateObservation("source", 1.0, "PASSED", 0.0, independent_confirmation_count=2)],
    )
    assert 0.5 < new_state.learned_trust <= 0.7


def test_contradiction_lowers_learned_trust() -> None:
    updater = BoundedTrustUpdater()
    new_state = updater.update(
        0.5,
        _state(0.5),
        [TrustUpdateObservation("source", 0.0, "FAILED", 1.0, independent_confirmation_count=1)],
    )
    assert new_state.learned_trust < 0.5


def test_bounded_drift_per_update() -> None:
    updater = BoundedTrustUpdater(max_drift_per_update=0.01)
    new_state = updater.update(
        0.5,
        _state(0.5),
        [TrustUpdateObservation("source", 10.0, "PASSED", 0.0, independent_confirmation_count=1)],
    )
    assert new_state.learned_trust <= 0.51
