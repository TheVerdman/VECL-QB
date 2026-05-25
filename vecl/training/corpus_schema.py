from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vecl.provenance.events import stable_hash
from vecl.training.tool_use_loop import SupervisedToolUseExample

CORPUS_DATASET_VERSION = "tool_use_v0"
CORPUS_DATASET_VERSION_V1_HARD = "tool_use_v1_hard"
VALID_DATASET_VERSIONS = frozenset({CORPUS_DATASET_VERSION, CORPUS_DATASET_VERSION_V1_HARD})
CORPUS_TENANT_ID = "vecl-synthetic-tool-use-v0"
CORPUS_TENANT_ID_V1_HARD = "vecl-synthetic-tool-use-v1-hard"

CorpusTaskKind = Literal[
    "tool_call_json",
    "final_answer",
    "no_tool_json",
    "refusal",
    "review",
    "placeholder",
]
CorpusSplit = Literal["train", "heldout", "hard-heldout"]
TargetFormat = Literal["json", "text"]
DifficultyTag = Literal[
    "easy",
    "paraphrase",
    "ambiguous",
    "negative",
    "cross_specialist",
    "chain",
    "ethics_boundary",
    "adversarial_payload",
]

EXECUTABLE_TASK_KINDS = frozenset({"tool_call_json", "final_answer"})
JSON_TASK_KINDS = frozenset({"tool_call_json", "no_tool_json"})
VALID_TASK_KINDS = frozenset(
    {"tool_call_json", "final_answer", "no_tool_json", "refusal", "review", "placeholder"}
)
VALID_SPLITS = frozenset({"train", "heldout", "hard-heldout"})
VALID_TARGET_FORMATS = frozenset({"json", "text"})
VALID_DIFFICULTY_TAGS = frozenset(
    {
        "easy",
        "paraphrase",
        "ambiguous",
        "negative",
        "cross_specialist",
        "chain",
        "ethics_boundary",
        "adversarial_payload",
    }
)


@dataclass(frozen=True)
class ToolUseCorpusRecord:
    dataset_version: str
    example_id: str
    tenant_id: str
    domain: str
    category: str
    split: CorpusSplit
    prompt: str
    target_text: str
    task_kind: CorpusTaskKind
    target_format: TargetFormat
    executable: bool
    specialist_id: str | None
    task_type: str | None
    source_id: str
    source_ids: tuple[str, ...]
    authority: float
    artifact_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset_version not in VALID_DATASET_VERSIONS:
            raise ValueError(f"dataset_version must be one of {sorted(VALID_DATASET_VERSIONS)}")
        required = {
            "example_id": self.example_id,
            "tenant_id": self.tenant_id,
            "domain": self.domain,
            "category": self.category,
            "prompt": self.prompt,
            "target_text": self.target_text,
            "source_id": self.source_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"required corpus fields are empty: {', '.join(missing)}")
        if self.task_kind not in VALID_TASK_KINDS:
            raise ValueError(f"invalid task_kind: {self.task_kind}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {self.split}")
        if self.target_format not in VALID_TARGET_FORMATS:
            raise ValueError(f"invalid target_format: {self.target_format}")
        if self.executable and self.task_kind not in EXECUTABLE_TASK_KINDS:
            raise ValueError("only tool_call_json and final_answer records may be executable")
        if self.task_kind in JSON_TASK_KINDS and self.target_format != "json":
            raise ValueError(f"{self.task_kind} records must use JSON targets")
        if not np.isfinite(self.authority) or self.authority <= 0:
            raise ValueError("authority must be finite and positive")
        if not self.source_ids:
            raise ValueError("source_ids must be non-empty")
        if self.source_id not in self.source_ids:
            raise ValueError("source_id must appear in source_ids")
        if self.executable and self.task_kind == "tool_call_json":
            if not self.specialist_id or not self.task_type:
                raise ValueError(
                    "executable tool_call_json records need specialist_id and task_type"
                )
        object.__setattr__(self, "source_ids", tuple(str(item) for item in self.source_ids))
        object.__setattr__(self, "artifact_ids", tuple(str(item) for item in self.artifact_ids))
        object.__setattr__(self, "claim_ids", tuple(str(item) for item in self.claim_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_obj(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["authority"] = float(self.authority)
        payload["source_ids"] = list(self.source_ids)
        payload["artifact_ids"] = list(self.artifact_ids)
        payload["claim_ids"] = list(self.claim_ids)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> ToolUseCorpusRecord:
        return cls(
            dataset_version=str(payload["dataset_version"]),
            example_id=str(payload["example_id"]),
            tenant_id=str(payload["tenant_id"]),
            domain=str(payload["domain"]),
            category=str(payload["category"]),
            split=str(payload["split"]),  # type: ignore[arg-type]
            prompt=str(payload["prompt"]),
            target_text=str(payload["target_text"]),
            task_kind=str(payload["task_kind"]),  # type: ignore[arg-type]
            target_format=str(payload["target_format"]),  # type: ignore[arg-type]
            executable=bool(payload["executable"]),
            specialist_id=_optional_str(payload.get("specialist_id")),
            task_type=_optional_str(payload.get("task_type")),
            source_id=str(payload["source_id"]),
            source_ids=tuple(str(item) for item in payload["source_ids"]),
            authority=float(payload["authority"]),
            artifact_ids=tuple(str(item) for item in payload.get("artifact_ids", ())),
            claim_ids=tuple(str(item) for item in payload.get("claim_ids", ())),
            metadata=dict(payload.get("metadata") or {}),
        )

    def target_json(self) -> dict[str, Any]:
        parsed = json.loads(self.target_text)
        if not isinstance(parsed, dict):
            raise ValueError("target JSON must be an object")
        return parsed

    def to_supervised_example(self) -> SupervisedToolUseExample:
        if not self.executable:
            raise ValueError("record is not executable and cannot be converted")
        if self.task_kind not in {"tool_call_json", "final_answer"}:
            raise ValueError(f"unsupported executable task_kind: {self.task_kind}")
        metadata = {
            **self.metadata,
            "corpus_dataset_version": self.dataset_version,
            "corpus_domain": self.domain,
            "corpus_category": self.category,
            "corpus_split": self.split,
            "corpus_executable": self.executable,
            "corpus_source_ids": list(self.source_ids),
        }
        return SupervisedToolUseExample(
            example_id=self.example_id,
            tenant_id=self.tenant_id,
            source_id=self.source_id,
            authority=float(self.authority),
            prompt=self.prompt,
            target_text=self.target_text,
            task_kind=self.task_kind,  # type: ignore[arg-type]
            artifact_ids=list(self.artifact_ids),
            claim_ids=list(self.claim_ids),
            metadata=metadata,
        )


@dataclass(frozen=True)
class LLMRawCacheEntry:
    provider: str
    model_id: str
    prompt_hash: str
    prompt: str
    raw_output: str
    parsed_output: Any
    seed: int
    topic_id: str
    validator_outcome: str
    estimated_cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    latency_ms: int | None = None
    accepted_example_ids: tuple[str, ...] = ()
    rejected_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider or not self.model_id or not self.prompt_hash or not self.topic_id:
            raise ValueError("LLM cache provider, model_id, prompt_hash, and topic_id are required")
        object.__setattr__(
            self, "accepted_example_ids", tuple(str(item) for item in self.accepted_example_ids)
        )
        object.__setattr__(
            self, "rejected_reasons", tuple(str(item) for item in self.rejected_reasons)
        )
        object.__setattr__(self, "usage", dict(self.usage))

    def to_json_obj(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["accepted_example_ids"] = list(self.accepted_example_ids)
        payload["rejected_reasons"] = list(self.rejected_reasons)
        return payload

    @classmethod
    def from_json_obj(cls, payload: dict[str, Any]) -> LLMRawCacheEntry:
        return cls(
            provider=str(payload["provider"]),
            model_id=str(payload["model_id"]),
            prompt_hash=str(payload["prompt_hash"]),
            prompt=str(payload["prompt"]),
            raw_output=str(payload["raw_output"]),
            parsed_output=payload.get("parsed_output"),
            seed=int(payload["seed"]),
            topic_id=str(payload["topic_id"]),
            validator_outcome=str(payload["validator_outcome"]),
            estimated_cost_usd=_optional_float(payload.get("estimated_cost_usd")),
            usage=dict(payload.get("usage") or {}),
            finish_reason=_optional_str(payload.get("finish_reason")),
            latency_ms=_optional_int(payload.get("latency_ms")),
            accepted_example_ids=tuple(
                str(item) for item in payload.get("accepted_example_ids", ())
            ),
            rejected_reasons=tuple(str(item) for item in payload.get("rejected_reasons", ())),
        )


def stable_example_id(record: ToolUseCorpusRecord | dict[str, Any]) -> str:
    if isinstance(record, ToolUseCorpusRecord):
        payload = record.to_json_obj()
    else:
        payload = dict(record)
    material = {
        "dataset_version": payload["dataset_version"],
        "domain": payload["domain"],
        "category": payload["category"],
        "prompt": payload["prompt"],
        "target_text": payload["target_text"],
        "task_kind": payload["task_kind"],
        "target_format": payload["target_format"],
        "executable": payload["executable"],
        "specialist_id": payload.get("specialist_id"),
        "task_type": payload.get("task_type"),
        "source_ids": sorted(payload["source_ids"]),
    }
    prefix = "tu-v1h-" if payload["dataset_version"] == CORPUS_DATASET_VERSION_V1_HARD else "tu-v0-"
    return prefix + stable_hash(material)[:20]


def stable_split_for_id(
    example_id: str,
    heldout_modulus: int = 10,
    hard_heldout_modulus: int = 0,
) -> CorpusSplit:
    bucket = int(stable_hash({"split": example_id})[:8], 16)
    if hard_heldout_modulus > 0 and bucket % hard_heldout_modulus == 0:
        return "hard-heldout"
    return "heldout" if bucket % heldout_modulus == 0 else "train"


def stable_split_for_record(record: ToolUseCorpusRecord | dict[str, Any]) -> CorpusSplit:
    if isinstance(record, ToolUseCorpusRecord):
        dataset_version = record.dataset_version
        example_id = record.example_id
    else:
        dataset_version = str(record["dataset_version"])
        example_id = str(record["example_id"])
    hard_modulus = 20 if dataset_version == CORPUS_DATASET_VERSION_V1_HARD else 0
    return stable_split_for_id(example_id, hard_heldout_modulus=hard_modulus)


def corpus_hash(records: list[ToolUseCorpusRecord]) -> str:
    return stable_hash(
        [record.to_json_obj() for record in sorted(records, key=lambda item: item.example_id)]
    )


def write_corpus_jsonl(records: list[ToolUseCorpusRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.example_id):
            handle.write(json.dumps(record.to_json_obj(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_corpus_jsonl(path: str | Path) -> list[ToolUseCorpusRecord]:
    records: list[ToolUseCorpusRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            records.append(ToolUseCorpusRecord.from_json_obj(payload))
    return records


def write_llm_cache_jsonl(entries: list[LLMRawCacheEntry], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_json_obj(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def append_llm_cache_jsonl(entry: LLMRawCacheEntry, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_json_obj(), sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_llm_cache_jsonl(
    path: str | Path, *, allow_truncated_tail: bool = False
) -> list[LLMRawCacheEntry]:
    entries: list[LLMRawCacheEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if allow_truncated_tail and line_number == len(lines):
                    break
                raise ValueError(f"invalid LLM cache JSONL at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"LLM cache line {line_number} must contain an object")
            entries.append(LLMRawCacheEntry.from_json_obj(payload))
    return entries


def supervised_examples_from_corpus(
    records: list[ToolUseCorpusRecord],
) -> list[SupervisedToolUseExample]:
    return [record.to_supervised_example() for record in records if record.executable]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError("expected int, float, str, or None")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError("expected int, float, str, or None")
