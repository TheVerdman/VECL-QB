from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
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
    FINAL_RESPONSE_RECORDED = "FinalResponseRecorded"
    RELEASE_APPROVED = "ReleaseApproved"
    RELEASE_REJECTED = "ReleaseRejected"


class _FrozenMapping(Mapping[str, Any]):
    """Recursively immutable mapping used for committed event payloads."""

    __slots__ = ("_values",)
    _values: Mapping[str, Any]

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("committed event payloads are immutable")

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False

    def __repr__(self) -> str:
        return repr(dict(self._values))


class _FrozenSequence(Sequence[Any]):
    """Tuple-backed sequence that still compares naturally with JSON lists."""

    __slots__ = ("_values",)
    _values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any]) -> None:
        object.__setattr__(self, "_values", tuple(values))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("committed event payloads are immutable")

    def __getitem__(self, index: int | slice) -> Any:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes, bytearray)):
            return list(self) == list(other)
        return False

    def __repr__(self) -> str:
        return repr(list(self._values))


def _freeze(value: Any) -> Any:
    if isinstance(value, _FrozenMapping | _FrozenSequence):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("event payload mappings must use string keys")
        return _FrozenMapping({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _FrozenSequence([_freeze(item) for item in value])
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, bytes | bytearray):
        return f"bytes:{bytes(value).hex()}"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("event payload numbers must be finite")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("event payload datetimes must be timezone-aware")
        return value.isoformat()
    raise ValueError(f"unsupported event payload value type: {type(value).__name__}")


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    if isinstance(value, set | frozenset):
        plain_items = [_plain_value(item) for item in value]
        return sorted(
            plain_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), default=str),
        )
    return deepcopy(value)


def stable_hash(value: Any) -> str:
    raw = json.dumps(
        _plain_value(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProvenanceEvent:
    event_type: EventType
    tenant_id: str
    actor: str
    payload: Mapping[str, Any]
    parent_event_ids: Sequence[str] = field(default_factory=tuple)
    event_id: str = field(default_factory=lambda: f"evt-{uuid4()}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload_hash: str = ""
    chain_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if not isinstance(self.actor, str) or not self.actor:
            raise ValueError("actor must be non-empty")
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be a timezone-aware datetime")
        if isinstance(self.parent_event_ids, str | bytes) or not isinstance(
            self.parent_event_ids, Sequence
        ):
            raise ValueError("parent_event_ids must be a sequence of strings")
        parent_ids = list(self.parent_event_ids)
        if any(not isinstance(parent_id, str) or not parent_id for parent_id in parent_ids):
            raise ValueError("parent_event_ids must contain non-empty strings")
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("parent_event_ids must be unique")
        frozen_payload = _freeze(self.payload)
        computed_payload_hash = stable_hash(frozen_payload)
        if self.payload_hash and self.payload_hash != computed_payload_hash:
            raise ValueError("payload hash does not match event payload")
        object.__setattr__(self, "payload", frozen_payload)
        object.__setattr__(self, "payload_hash", computed_payload_hash)
        object.__setattr__(
            self,
            "parent_event_ids",
            _FrozenSequence(parent_ids),
        )

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
            "payload": _plain_value(self.payload),
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
