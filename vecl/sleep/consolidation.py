from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from vecl._compat import UTC
from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.runtime.tokens import create_learning_event_token
from vecl.sleep.replay import ReplayBatchBuilder, ReplayCandidate, ReplayPolicy
from vecl.sparse.types import SparseMemoryInputs


@dataclass(frozen=True)
class SleepCycleReport:
    batch_id: str
    selected_evidence_ids: list[str]
    selected_slots: list[int]
    delta_record_ids: list[str]
    verification_status: str
    committed: bool
    abort_reason: str | None = None


class SleepCycleConsolidator:
    def __init__(
        self, monitor: SparseUpdateMonitor, builder: ReplayBatchBuilder | None = None
    ) -> None:
        self.monitor = monitor
        self.builder = builder or ReplayBatchBuilder()

    def run_cycle(
        self,
        candidates: list[ReplayCandidate],
        policy: ReplayPolicy,
        memory_values: np.ndarray,
        learning_rate: float,
        min_score: float,
        policy_version: str = "policy-v0",
        trust_policy_version: str = "trust-v0",
        force_tamper: bool = False,
    ) -> SleepCycleReport:
        batch = self.builder.build_batch(candidates, policy)
        self.monitor.ledger.append(
            ProvenanceEvent(
                event_type=EventType.REPLAY_BATCH_PREPARED,
                tenant_id=batch.tenant_id,
                actor="sleep-consolidator",
                payload={
                    "batch_id": batch.batch_id,
                    "evidence_ids": [candidate.evidence_id for candidate in batch.candidates],
                    "source_hashes": [candidate.source_id for candidate in batch.candidates],
                },
            )
        )
        size = len(memory_values)
        activation = np.zeros(size)
        rarity = np.zeros(size)
        authority = np.zeros(size)
        gradients = np.zeros(size)
        for slot, candidate in enumerate(batch.candidates[:size]):
            activation[slot] = candidate.activation
            rarity[slot] = candidate.rarity
            authority[slot] = candidate.authority
            gradients[slot] = 1.0
        token = create_learning_event_token(
            batch_id=batch.batch_id,
            tenant_id=batch.tenant_id,
            source_set_hash=stable_hash(
                sorted(candidate.source_id for candidate in batch.candidates)
            ),
            provenance_root_hash=stable_hash(batch.batch_id),
            min_authority=policy.min_authority,
            min_score=min_score,
            max_slots=min(policy.max_batch_size, size),
            now=datetime.now(UTC),
            policy_version=policy_version,
            trust_policy_version=trust_policy_version,
        )
        inputs = SparseMemoryInputs(
            memory_values=memory_values,
            gradients=gradients,
            activation=activation,
            rarity=rarity,
            authority=authority,
            learning_rate=learning_rate,
            min_authority=policy.min_authority,
            min_score=min_score,
            max_slots=min(policy.max_batch_size, size),
            quarantined_slots=set(),
        )
        try:
            result = self.monitor.apply_sparse_update(token, inputs)
            if force_tamper and result.selected_slots:
                result.selected_slots.append(size - 1)
            self.monitor.verify_sparse_update(token, inputs, result)
            self.monitor.commit_learning_event(token)
        except Exception as exc:  # noqa: BLE001 - report must preserve abort reason
            self.monitor.abort_learning_event(token, str(exc))
            return SleepCycleReport(
                batch_id=batch.batch_id,
                selected_evidence_ids=[candidate.evidence_id for candidate in batch.candidates],
                selected_slots=[],
                delta_record_ids=[],
                verification_status="FAILED",
                committed=False,
                abort_reason=str(exc),
            )
        return SleepCycleReport(
            batch_id=batch.batch_id,
            selected_evidence_ids=[candidate.evidence_id for candidate in batch.candidates],
            selected_slots=result.selected_slots,
            delta_record_ids=[
                f"{record.event_id}:{record.slot_id}" for record in result.delta_records
            ],
            verification_status="PASSED",
            committed=True,
        )
