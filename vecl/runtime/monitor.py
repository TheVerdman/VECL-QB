from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from vecl._compat import UTC
from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.tokens import LearningEventToken, validate_learning_event_token
from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs, SparseUpdateResult


@dataclass
class SparseUpdateMonitor:
    ledger: ProvenanceLedger
    actor: str = "sparse-monitor"
    committed_memory_values: np.ndarray | None = None
    _pending: dict[str, tuple[LearningEventToken, SparseMemoryInputs, SparseUpdateResult]] = field(
        default_factory=dict
    )

    def prepare_learning_event(self, token: LearningEventToken) -> ProvenanceEvent:
        validate_learning_event_token(token, datetime.now(UTC), token.tenant_id)
        event = ProvenanceEvent(
            event_type=EventType.LEARNING_EVENT_PREPARED,
            event_id=token.event_id,
            tenant_id=token.tenant_id,
            actor=self.actor,
            payload={
                "event_id": token.event_id,
                "batch_id": token.batch_id,
                "source_set_hash": token.source_set_hash,
                "provenance_root_hash": token.provenance_root_hash,
                "min_authority": token.min_authority,
                "min_score": token.min_score,
                "max_slots": token.max_slots,
                "policy_version": token.policy_version,
                "trust_policy_version": token.trust_policy_version,
            },
        )
        return self.ledger.append(event)

    def apply_sparse_update(
        self, token: LearningEventToken, inputs: SparseMemoryInputs
    ) -> SparseUpdateResult:
        validate_learning_event_token(token, datetime.now(UTC), token.tenant_id)
        if (
            inputs.min_authority != token.min_authority
            or inputs.min_score != token.min_score
            or inputs.max_slots != token.max_slots
        ):
            raise ValueError("token thresholds do not match update inputs")
        if self.ledger.get(token.event_id) is None:
            self.prepare_learning_event(token)
        result = sparse_update_oracle(inputs, token.event_id, token.provenance_root_hash)
        self._pending[token.event_id] = (token, inputs, result)
        return result

    def verify_sparse_update(
        self, token: LearningEventToken, inputs: SparseMemoryInputs, result: SparseUpdateResult
    ) -> bool:
        if (
            inputs.min_authority != token.min_authority
            or inputs.min_score != token.min_score
            or inputs.max_slots != token.max_slots
        ):
            raise ValueError("token thresholds do not match update thresholds")
        if not set(result.selected_slots).issubset(set(result.eligible_slots)):
            raise ValueError("selected_slots must be a subset of eligible_slots")
        changed_slots = set(
            np.where(~np.isclose(inputs.memory_values, result.new_memory_values))[0]
        )
        if not changed_slots.issubset(set(result.selected_slots)):
            raise ValueError("changed slots must be a subset of selected_slots")
        if len(changed_slots) > token.max_slots:
            raise ValueError("changed slot count exceeds max_slots")
        if changed_slots - set(result.eligible_slots):
            raise ValueError("ineligible slot changed")
        if not np.all(np.isfinite(result.new_memory_values)) or not np.all(
            np.isfinite(result.scores)
        ):
            raise ValueError("update result contains NaN or Inf")
        if any(not delta.event_id for delta in result.delta_records):
            raise ValueError("every delta must have an event_id")
        delta_slots = {delta.slot_id for delta in result.delta_records}
        if not changed_slots.issubset(delta_slots):
            raise ValueError("every changed slot must have a DeltaRecord")
        return True

    def commit_learning_event(
        self, token: LearningEventToken, extra_payload: dict[str, Any] | None = None
    ) -> list[ProvenanceEvent]:
        if token.event_id not in self._pending:
            raise ValueError("no pending update for token")
        token, inputs, result = self._pending[token.event_id]
        self.verify_sparse_update(token, inputs, result)
        applied = self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.SPARSE_UPDATE_APPLIED,
                tenant_id=token.tenant_id,
                actor=self.actor,
                parent_event_ids=[token.event_id],
                payload={
                    "learning_event_id": token.event_id,
                    "selected_slots": result.selected_slots,
                    "eligible_slots": result.eligible_slots,
                    "delta_records": [delta.__dict__ for delta in result.delta_records],
                    "memory_hash": _array_hash(result.new_memory_values),
                    **(extra_payload or {}),
                },
            )
        )
        committed = self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.LEARNING_EVENT_COMMITTED,
                tenant_id=token.tenant_id,
                actor=self.actor,
                parent_event_ids=[applied.event_id],
                payload={
                    "learning_event_id": token.event_id,
                    "sparse_update_event_id": applied.event_id,
                },
            )
        )
        self.committed_memory_values = result.new_memory_values.copy()
        del self._pending[token.event_id]
        return [applied, committed]

    def abort_learning_event(
        self, token: LearningEventToken, reason: str, payload: dict[str, Any] | None = None
    ) -> ProvenanceEvent:
        self._pending.pop(token.event_id, None)
        return self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.LEARNING_EVENT_ABORTED,
                tenant_id=token.tenant_id,
                actor=self.actor,
                parent_event_ids=[token.event_id] if self.ledger.get(token.event_id) else [],
                payload={"learning_event_id": token.event_id, "reason": reason, **(payload or {})},
            )
        )


def _array_hash(values: np.ndarray) -> str:
    from vecl.provenance.events import stable_hash

    return stable_hash([float(value) for value in values])
