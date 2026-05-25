from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vecl.provenance.events import EventType, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs


@dataclass(frozen=True)
class TraceValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayResult:
    reproduced: bool
    final_memory_hash: str
    expected_memory_hash: str | None
    errors: list[str] = field(default_factory=list)


def export_trace(ledger: ProvenanceLedger, event_ids: list[str]) -> str:
    events = [ledger.require(event_id).to_dict() for event_id in event_ids]
    return json.dumps({"events": events}, sort_keys=True, indent=2)


def validate_trace(trace: str | dict[str, Any]) -> TraceValidationResult:
    data = json.loads(trace) if isinstance(trace, str) else trace
    prepared: set[str] = set()
    updated: set[str] = set()
    errors: list[str] = []
    for event in data.get("events", []):
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        learning_event_id = payload.get("learning_event_id") or payload.get("event_id")
        if event_type == EventType.LEARNING_EVENT_PREPARED.value:
            prepared.add(event.get("event_id", learning_event_id))
        elif event_type == "SparseScoresComputed":
            if learning_event_id not in prepared:
                errors.append("scores computed before prepare")
        elif event_type == "SparseSlotsSelected":
            selected = set(payload.get("selected_slots", []))
            eligible = set(payload.get("eligible_slots", []))
            if not selected.issubset(eligible):
                errors.append("selected ineligible slot")
        elif event_type == EventType.SPARSE_UPDATE_APPLIED.value:
            if learning_event_id not in prepared:
                errors.append("update before prepare")
            selected = set(payload.get("selected_slots", []))
            eligible = set(payload.get("eligible_slots", selected))
            if not selected.issubset(eligible):
                errors.append("selected ineligible slot")
            updated.add(learning_event_id)
        elif event_type == EventType.LEARNING_EVENT_COMMITTED.value:
            if learning_event_id not in updated:
                errors.append("commit before update")
    return TraceValidationResult(valid=not errors, errors=errors)


def replay_trace_with_oracle(trace: str | dict[str, Any]) -> ReplayResult:
    data = json.loads(trace) if isinstance(trace, str) else trace
    validation = validate_trace(data)
    if not validation.valid:
        return ReplayResult(False, "", None, validation.errors)
    memory: np.ndarray | None = None
    expected_hash: str | None = None
    errors: list[str] = []
    for event in data.get("events", []):
        if event.get("event_type") != EventType.SPARSE_UPDATE_APPLIED.value:
            continue
        payload = event.get("payload", {})
        if "oracle_inputs" in payload:
            raw = payload["oracle_inputs"]
            inputs = SparseMemoryInputs(
                memory_values=np.asarray(raw["memory_values"], dtype=np.float64),
                gradients=np.asarray(raw["gradients"], dtype=np.float64),
                activation=np.asarray(raw["activation"], dtype=np.float64),
                rarity=np.asarray(raw["rarity"], dtype=np.float64),
                authority=np.asarray(raw["authority"], dtype=np.float64),
                learning_rate=float(raw["learning_rate"]),
                min_authority=float(raw["min_authority"]),
                min_score=float(raw["min_score"]),
                max_slots=int(raw["max_slots"]),
                quarantined_slots=set(raw.get("quarantined_slots", [])),
            )
            result = sparse_update_oracle(
                inputs,
                payload.get("learning_event_id", event["event_id"]),
                raw.get("provenance_root", "trace-root"),
            )
            memory = result.new_memory_values
        elif memory is not None:
            for record in payload.get("delta_records", []):
                memory[int(record["slot_id"])] += float(record["delta"])
        expected_hash = payload.get("memory_hash", expected_hash)
    if memory is None:
        return ReplayResult(False, "", expected_hash, ["trace contains no replayable update"])
    final_hash = stable_hash([float(value) for value in memory])
    return ReplayResult(final_hash == expected_hash, final_hash, expected_hash, errors)
