import json

import numpy as np
import pytest

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.checkpoints import CheckpointStore, RollbackService
from vecl.runtime.metrics import rollback_metrics, tenant_crossing_metric
from vecl.runtime.release_gate import ReleaseGate
from vecl.runtime.traces import replay_trace_with_oracle, validate_trace


def _prepared_update(ledger: ProvenanceLedger, event_id: str, delta: float, source_id: str = "s"):
    ledger.append(
        ProvenanceEvent(
            EventType.LEARNING_EVENT_PREPARED,
            "t",
            "test",
            {"event_id": event_id},
            event_id=event_id,
        )
    )
    return ledger.append(
        ProvenanceEvent(
            EventType.SPARSE_UPDATE_APPLIED,
            "t",
            "test",
            {
                "learning_event_id": event_id,
                "source_ids": [source_id],
                "eligible_slots": [0],
                "selected_slots": [0],
                "delta_records": [{"slot_id": 0, "delta": delta}],
            },
        )
    )


def test_rollback_by_event_and_source_preserve_unrelated_events() -> None:
    ledger = ProvenanceLedger()
    store = CheckpointStore()
    store.create_checkpoint(np.array([1.0]), "t", "c0")
    event_a = _prepared_update(ledger, "learn-a", -0.1, "a")
    _prepared_update(ledger, "learn-b", -0.2, "b")
    service = RollbackService(ledger, store)
    by_event = service.rollback_by_event(event_a.event_id, "t", "c0")
    assert np.allclose(by_event.memory_values, [0.8])
    by_source = service.rollback_by_source("a", "t", "c0")
    assert np.allclose(by_source.memory_values, [0.8])
    assert any(event.event_type == EventType.ROLLBACK_PERFORMED for event in ledger.events())


def test_approximate_rollback_cannot_be_mistaken_for_exact() -> None:
    ledger = ProvenanceLedger()
    store = CheckpointStore()
    store.create_checkpoint(np.array([1.0]), "t", "c0")
    service = RollbackService(ledger, store)
    with pytest.raises(ValueError, match="approximate"):
        service.rollback_by_event("e", "t", "c0", approximate=True)
    assert not service.rollback_approximate("t", "toy").exact


def test_trace_validation_ordering_and_selection() -> None:
    assert not validate_trace(
        {
            "events": [
                {
                    "event_type": "SparseUpdateApplied",
                    "payload": {"learning_event_id": "missing"},
                }
            ]
        }
    ).valid
    assert not validate_trace(
        {
            "events": [
                {
                    "event_type": "LearningEventPrepared",
                    "event_id": "e",
                    "payload": {"event_id": "e"},
                },
                {"event_type": "LearningEventCommitted", "payload": {"learning_event_id": "e"}},
            ]
        }
    ).valid
    assert not validate_trace(
        {
            "events": [
                {
                    "event_type": "LearningEventPrepared",
                    "event_id": "e",
                    "payload": {"event_id": "e"},
                },
                {
                    "event_type": "SparseUpdateApplied",
                    "payload": {
                        "learning_event_id": "e",
                        "selected_slots": [1],
                        "eligible_slots": [0],
                    },
                },
            ]
        }
    ).valid


def test_replay_trace_with_oracle_reproduces_memory_hash() -> None:
    expected_memory = [0.9]
    trace = {
        "events": [
            {"event_type": "LearningEventPrepared", "event_id": "e", "payload": {"event_id": "e"}},
            {
                "event_type": "SparseUpdateApplied",
                "event_id": "u",
                "payload": {
                    "learning_event_id": "e",
                    "selected_slots": [0],
                    "eligible_slots": [0],
                    "memory_hash": stable_hash(expected_memory),
                    "oracle_inputs": {
                        "memory_values": [1.0],
                        "gradients": [1.0],
                        "activation": [1.0],
                        "rarity": [1.0],
                        "authority": [1.0],
                        "learning_rate": 0.1,
                        "min_authority": 0.1,
                        "min_score": 0.1,
                        "max_slots": 1,
                    },
                },
            },
            {"event_type": "LearningEventCommitted", "payload": {"learning_event_id": "e"}},
        ]
    }
    result = replay_trace_with_oracle(json.dumps(trace))
    assert result.reproduced


def test_release_gate_blocks_failed_adversarial_simulation_and_emits_events() -> None:
    ledger = ProvenanceLedger()
    gate = ReleaseGate(ledger, "t")
    failed = gate.evaluate_candidate_checkpoint(
        "c1",
        invariant_tests_pass=True,
        regression_tests_pass=True,
        adversarial_simulation_within_bound=False,
        rollback_test_pass=True,
        verification_calibration_not_worse=True,
        no_tenant_isolation_failure=True,
    )
    with pytest.raises(ValueError):
        gate.approve(failed)
    rejected = gate.reject(failed, "adversarial bound failed")
    assert rejected.event_type == EventType.RELEASE_REJECTED

    passed = gate.evaluate_candidate_checkpoint(
        "c2",
        invariant_tests_pass=True,
        regression_tests_pass=True,
        adversarial_simulation_within_bound=True,
        rollback_test_pass=True,
        verification_calibration_not_worse=True,
        no_tenant_isolation_failure=True,
    )
    assert gate.approve(passed).event_type == EventType.RELEASE_APPROVED


def test_release_gate_defaults_drift_pass_and_blocks_exceeded_drift() -> None:
    ledger = ProvenanceLedger()
    gate = ReleaseGate(ledger, "t")
    no_drift_report = gate.evaluate_candidate_checkpoint(
        "c1",
        invariant_tests_pass=True,
        regression_tests_pass=True,
        adversarial_simulation_within_bound=True,
        rollback_test_pass=True,
        verification_calibration_not_worse=True,
        no_tenant_isolation_failure=True,
    )
    exceeded = gate.evaluate_candidate_checkpoint(
        "c2",
        invariant_tests_pass=True,
        regression_tests_pass=True,
        adversarial_simulation_within_bound=True,
        rollback_test_pass=True,
        verification_calibration_not_worse=True,
        no_tenant_isolation_failure=True,
        drift_within_bound=False,
        drift_value=2.0,
        drift_threshold=1.0,
        drift_report={"drift_value": 2.0, "drift_threshold": 1.0},
    )

    assert no_drift_report.passed
    assert no_drift_report.drift_within_bound
    assert not exceeded.passed
    assert exceeded.drift_value == 2.0
    with pytest.raises(ValueError):
        gate.approve(exceeded)
    rejected = gate.reject(exceeded, "drift exceeded")
    assert rejected.payload["evaluation"]["drift_report"]["drift_value"] == 2.0


def test_metrics_helpers() -> None:
    assert rollback_metrics(np.array([1.0]), np.array([1.1]))["rollback_error_norm"] > 0
    assert tenant_crossing_metric(2)["tenant_crossing_attempt_count"] == 2.0
