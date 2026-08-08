from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from vecl._compat import UTC, StrEnum


class EventType(StrEnum):
    EVIDENCE_INGESTED = "EvidenceIngested"
    REPLAY_BATCH_PREPARED = "ReplayBatchPrepared"
    TOOL_USE_TRAINING_BATCH_PREPARED = "ToolUseTrainingBatchPrepared"
    LEARNING_CANDIDATE_PREPARED = "LearningCandidatePrepared"
    LEARNING_CANDIDATE_EVALUATED = "LearningCandidateEvaluated"
    LEARNING_CANDIDATE_ACCEPTED = "LearningCandidateAccepted"
    LEARNING_CANDIDATE_REJECTED = "LearningCandidateRejected"
    LEARNING_CANDIDATE_RESTORED = "LearningCandidateRestored"
    LEARNING_EVENT_PREPARED = "LearningEventPrepared"
    SPARSE_UPDATE_APPLIED = "SparseUpdateApplied"
    LEARNING_EVENT_COMMITTED = "LearningEventCommitted"
    LEARNING_EVENT_ABORTED = "LearningEventAborted"
    SOURCE_QUARANTINED = "SourceQuarantined"
    ROLLBACK_PERFORMED = "RollbackPerformed"
    ARTIFACT_PRODUCED = "ArtifactProduced"
    EPISODIC_ENTRY_WRITTEN = "EpisodicEntryWritten"
    CHAIN_STARTED = "ChainStarted"
    CHAIN_STEP_STARTED = "ChainStepStarted"
    CHAIN_STEP_COMPLETED = "ChainStepCompleted"
    CHAIN_STEP_FAILED = "ChainStepFailed"
    CHAIN_ABORTED = "ChainAborted"
    LLM_ROUTING_DECIDED = "LLMRoutingDecided"
    ROUTING_FALLBACK = "RoutingFallback"
    LLM_TOOL_CALL_PROPOSED = "LLMToolCallProposed"
    ETHICS_RULE_INSTALLED = "EthicsRuleInstalled"
    ETHICS_RULE_REVOKED = "EthicsRuleRevoked"
    CHAIN_STEP_REFUSED_BY_ETHICS = "ChainStepRefusedByEthics"
    CHAIN_STEP_AWAITING_REVIEW = "ChainStepAwaitingReview"
    RELEASE_APPROVED = "ReleaseApproved"
    RELEASE_REJECTED = "ReleaseRejected"


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProvenanceEvent:
    event_type: EventType
    tenant_id: str
    actor: str
    payload: dict[str, Any]
    parent_event_ids: list[str] = field(default_factory=list)
    event_id: str = field(default_factory=lambda: f"evt-{uuid4()}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload_hash: str = ""
    chain_hash: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if not self.actor:
            raise ValueError("actor must be non-empty")
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        payload_hash = self.payload_hash or stable_hash(self.payload)
        object.__setattr__(self, "payload_hash", payload_hash)
        object.__setattr__(self, "parent_event_ids", list(self.parent_event_ids))

    def with_chain_hash(self, previous_chain_hash: str) -> ProvenanceEvent:
        chain_hash = stable_hash(
            {
                "previous_chain_hash": previous_chain_hash,
                "event_id": self.event_id,
                "event_type": self.event_type.value,
                "tenant_id": self.tenant_id,
                "actor": self.actor,
                "payload_hash": self.payload_hash,
                "parent_event_ids": self.parent_event_ids,
                "timestamp": self.timestamp.isoformat(),
            }
        )
        return ProvenanceEvent(
            event_type=self.event_type,
            tenant_id=self.tenant_id,
            actor=self.actor,
            payload=self.payload,
            parent_event_ids=self.parent_event_ids,
            event_id=self.event_id,
            timestamp=self.timestamp,
            payload_hash=self.payload_hash,
            chain_hash=chain_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "tenant_id": self.tenant_id,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "parent_event_ids": list(self.parent_event_ids),
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "chain_hash": self.chain_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProvenanceEvent:
        return cls(
            event_type=EventType(str(payload["event_type"])),
            tenant_id=str(payload["tenant_id"]),
            actor=str(payload["actor"]),
            payload=dict(payload["payload"]),
            parent_event_ids=list(payload.get("parent_event_ids", [])),
            event_id=str(payload["event_id"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            payload_hash=str(payload.get("payload_hash", "")),
            chain_hash=str(payload.get("chain_hash", "")),
        )
