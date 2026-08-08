from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vecl._compat import UTC
from vecl.provenance.events import stable_hash

GATE_FIELDS = (
    "invariant_tests_pass",
    "regression_tests_pass",
    "adversarial_simulation_within_bound",
    "rollback_test_pass",
    "verification_calibration_not_worse",
    "no_tenant_isolation_failure",
    "drift_within_bound",
)

GATE_CATEGORY_MAP = {
    "invariant": "invariant_tests_pass",
    "routing": "regression_tests_pass",
    "tool_call": "regression_tests_pass",
    "chain": "regression_tests_pass",
    "regression": "regression_tests_pass",
    "ethics": "adversarial_simulation_within_bound",
    "adversarial": "adversarial_simulation_within_bound",
    "rollback": "rollback_test_pass",
    "calibration": "verification_calibration_not_worse",
    "tenant_isolation": "no_tenant_isolation_failure",
    "fisher": "drift_within_bound",
    "drift": "drift_within_bound",
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
_REQUIRED_GATE_FIELDS = tuple(field for field in GATE_FIELDS if field != "drift_within_bound")


@dataclass(frozen=True)
class MetricThreshold:
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    equals: Any | None = None

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric name must be non-empty")
        if self.min_value is not None:
            object.__setattr__(self, "min_value", float(self.min_value))
        if self.max_value is not None:
            object.__setattr__(self, "max_value", float(self.max_value))
        if self.min_value is not None and not math.isfinite(self.min_value):
            raise ValueError("minimum metric threshold must be finite")
        if self.max_value is not None and not math.isfinite(self.max_value):
            raise ValueError("maximum metric threshold must be finite")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("minimum metric threshold must not exceed maximum")
        if isinstance(self.equals, float) and not math.isfinite(self.equals):
            raise ValueError("equality metric threshold must be finite")

    def evaluate(self, summary: Mapping[str, Any]) -> list[str]:
        value = _lookup_metric(summary, self.metric)
        reasons: list[str] = []
        if self.equals is not None and value != self.equals:
            reasons.append(f"{self.metric} expected {self.equals!r}, got {value!r}")
        if self.min_value is not None or self.max_value is not None:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                reasons.append(f"{self.metric} must be finite: {value}")
            else:
                if self.min_value is not None and numeric_value < self.min_value:
                    reasons.append(f"{self.metric} below {self.min_value}: {value}")
                if self.max_value is not None and numeric_value > self.max_value:
                    reasons.append(f"{self.metric} above {self.max_value}: {value}")
        return reasons

    def to_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "equals": self.equals,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MetricThreshold:
        return cls(
            metric=str(payload["metric"]),
            min_value=_optional_float(payload.get("min_value")),
            max_value=_optional_float(payload.get("max_value")),
            equals=payload.get("equals"),
        )


@dataclass(frozen=True)
class EvaluationEvidence:
    """Reproducibility metadata emitted by an evaluator that actually ran."""

    kind: str
    evaluator_id: str
    evaluator_version: str
    candidate_id: str
    configuration: dict[str, Any]
    configuration_hash: str
    git_revision: str
    command: tuple[str, ...]
    artifact_hashes: dict[str, str]
    working_tree_clean: bool
    synthetic: bool = False
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration",
            json.loads(
                json.dumps(self.configuration, sort_keys=True, default=str, allow_nan=False)
            ),
        )
        object.__setattr__(self, "command", tuple(str(part) for part in self.command))
        object.__setattr__(
            self,
            "artifact_hashes",
            {str(name): str(digest) for name, digest in self.artifact_hashes.items()},
        )

    def validation_reasons(self, summary: Mapping[str, Any]) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.synthetic or self.kind == "synthetic_demo":
            reasons.append("synthetic/demo evidence is not eligible for release approval")
        if self.kind not in {"command", "internal", "synthetic_demo"}:
            reasons.append(f"unsupported evidence kind: {self.kind!r}")
        if not self.evaluator_id.strip() or not self.evaluator_version.strip():
            reasons.append("evaluator identity and version must be non-empty")
        if not self.candidate_id.strip():
            reasons.append("evidence candidate_id must be non-empty")
        if not self.configuration:
            reasons.append("evaluator configuration must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.configuration_hash):
            reasons.append("configuration_hash must be a SHA-256 digest")
        elif self.configuration_hash != stable_hash(self.configuration):
            reasons.append("configuration_hash does not match evaluator configuration")
        if not _GIT_REVISION_PATTERN.fullmatch(self.git_revision):
            reasons.append("git_revision must be a full commit digest")
        if not self.working_tree_clean:
            reasons.append("evaluation working tree was not clean")
        if not self.command or any(not part.strip() for part in self.command):
            reasons.append("evidence command must be non-empty")
        if not self.artifact_hashes:
            reasons.append("evidence must include artifact hashes")
        for name, digest in self.artifact_hashes.items():
            if not name.strip() or not _SHA256_PATTERN.fullmatch(digest):
                reasons.append(f"artifact hash {name!r} is malformed")
        if self.artifact_hashes.get("summary") != stable_hash(summary):
            reasons.append("summary artifact hash does not match evaluation summary")
        try:
            json.dumps(summary, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            reasons.append("evaluation summary must contain finite JSON values")
        if self.generated_at.tzinfo is None:
            reasons.append("evidence generated_at must be timezone-aware")
        return tuple(reasons)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "candidate_id": self.candidate_id,
            "configuration": json.loads(json.dumps(self.configuration, sort_keys=True)),
            "configuration_hash": self.configuration_hash,
            "git_revision": self.git_revision,
            "command": list(self.command),
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "working_tree_clean": self.working_tree_clean,
            "synthetic": self.synthetic,
            "generated_at": self.generated_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvaluationEvidence:
        command = payload.get("command")
        configuration = payload.get("configuration")
        artifact_hashes = payload.get("artifact_hashes")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            raise ValueError("evidence command must be a sequence")
        if not isinstance(artifact_hashes, Mapping):
            raise ValueError("evidence artifact_hashes must be a mapping")
        if not isinstance(configuration, Mapping):
            raise ValueError("evidence configuration must be a mapping")
        return cls(
            kind=str(payload.get("kind") or ""),
            evaluator_id=str(payload.get("evaluator_id") or ""),
            evaluator_version=str(payload.get("evaluator_version") or ""),
            candidate_id=str(payload.get("candidate_id") or ""),
            configuration=dict(configuration),
            configuration_hash=str(payload.get("configuration_hash") or ""),
            git_revision=str(payload.get("git_revision") or ""),
            command=tuple(str(part) for part in command),
            artifact_hashes={str(name): str(digest) for name, digest in artifact_hashes.items()},
            working_tree_clean=_require_bool(payload, "working_tree_clean"),
            synthetic=_require_bool(payload, "synthetic"),
            generated_at=datetime.fromisoformat(str(payload["generated_at"])),
        )


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    category: str
    passed: bool
    summary: dict[str, Any]
    reasons: tuple[str, ...] = ()
    thresholds: tuple[MetricThreshold, ...] = ()
    evidence: EvaluationEvidence | None = None

    def approval_reasons(self, expected_candidate_id: str | None = None) -> tuple[str, ...]:
        if self.evidence is None:
            return ("missing evaluator evidence",)
        reasons = list(self.evidence.validation_reasons(self.summary))
        if self.evidence.evaluator_id != self.name:
            reasons.append("evaluator evidence identity does not match result name")
        if (
            expected_candidate_id is not None
            and self.evidence.candidate_id != expected_candidate_id
        ):
            reasons.append("evaluator evidence candidate does not match release candidate")
        configured_thresholds = self.evidence.configuration.get("thresholds")
        expected_thresholds = [threshold.to_payload() for threshold in self.thresholds]
        if configured_thresholds != expected_thresholds:
            reasons.append("evaluator evidence thresholds do not match result thresholds")
        if self.evidence.configuration.get("category") != self.category:
            reasons.append("evaluator evidence category does not match result category")
        if self.evidence.kind == "command":
            if self.evidence.configuration.get("command") != list(self.evidence.command):
                reasons.append("evaluator evidence command does not match configuration")
            for artifact_name in ("stdout", "stderr"):
                if artifact_name not in self.evidence.artifact_hashes:
                    reasons.append(f"command evidence is missing {artifact_name} hash")
        for threshold in self.thresholds:
            try:
                reasons.extend(threshold.evaluate(self.summary))
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(str(exc))
        if self.passed and self.reasons:
            reasons.append("passed evaluator result contains failure reasons")
        return tuple(reasons)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "thresholds": [threshold.to_payload() for threshold in self.thresholds],
            "evidence": self.evidence.to_payload() if self.evidence is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvaluationResult:
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            raise ValueError("evaluation result passed must be a boolean")
        evidence_payload = payload.get("evidence")
        if evidence_payload is not None and not isinstance(evidence_payload, Mapping):
            raise ValueError("evaluation evidence must be a mapping")
        return cls(
            name=str(payload["name"]),
            category=str(payload["category"]),
            passed=passed,
            summary=dict(payload.get("summary") or {}),
            reasons=tuple(str(reason) for reason in payload.get("reasons", [])),
            thresholds=tuple(
                MetricThreshold.from_payload(threshold)
                for threshold in payload.get("thresholds", [])
            ),
            evidence=(
                EvaluationEvidence.from_payload(evidence_payload)
                if evidence_payload is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ReleaseEvaluationReport:
    candidate_id: str
    results: tuple[EvaluationResult, ...]
    manifest_hash: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def gate_checks(self) -> dict[str, bool]:
        checks = {field: False for field in GATE_FIELDS}
        checks["drift_within_bound"] = True
        seen_fields: set[str] = set()
        for result in self.results:
            field = GATE_CATEGORY_MAP.get(result.category)
            if field is None:
                continue
            checks[field] = (
                result.passed if field not in seen_fields else checks[field] and result.passed
            )
            seen_fields.add(field)
        return checks

    def approval_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.candidate_id.strip():
            reasons.append("candidate_id must be non-empty")
        if self.generated_at.tzinfo is None:
            reasons.append("release report generated_at must be timezone-aware")
        if not self.passed:
            reasons.append("one or more evaluators failed")
        if not _SHA256_PATTERN.fullmatch(self.manifest_hash):
            reasons.append("missing or malformed evaluation manifest hash")
        names = [result.name for result in self.results]
        if any(not name.strip() for name in names):
            reasons.append("evaluator names must be non-empty")
        if len(names) != len(set(names)):
            reasons.append("evaluator names must be unique")
        revisions = {
            result.evidence.git_revision for result in self.results if result.evidence is not None
        }
        if len(revisions) > 1:
            reasons.append("all evaluator evidence must come from the same git revision")
        for result in self.results:
            reasons.extend(
                f"{result.name}: {reason}" for reason in result.approval_reasons(self.candidate_id)
            )
            if result.category not in GATE_CATEGORY_MAP:
                reasons.append(f"{result.name}: unsupported gate category {result.category!r}")
        checks = self.gate_checks()
        for field_name in _REQUIRED_GATE_FIELDS:
            if not checks[field_name]:
                reasons.append(f"required gate evidence missing or failed: {field_name}")
        return tuple(reasons)

    @property
    def approval_eligible(self) -> bool:
        return not self.approval_reasons()

    @property
    def report_hash(self) -> str:
        return stable_hash(self._core_payload())

    def _core_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generated_at": self.generated_at.isoformat(),
            "manifest_hash": self.manifest_hash,
            "results": [result.to_payload() for result in self.results],
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._core_payload(),
            "passed": self.passed,
            "gate_checks": self.gate_checks(),
            "approval_eligible": self.approval_eligible,
            "approval_reasons": list(self.approval_reasons()),
            "report_hash": self.report_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReleaseEvaluationReport:
        results_payload = payload.get("results")
        if not isinstance(results_payload, Sequence) or isinstance(results_payload, (str, bytes)):
            raise ValueError("release report results must be a sequence")
        report = cls(
            candidate_id=str(payload["candidate_id"]),
            generated_at=datetime.fromisoformat(str(payload["generated_at"])),
            manifest_hash=str(payload.get("manifest_hash") or ""),
            results=tuple(EvaluationResult.from_payload(result) for result in results_payload),
        )
        supplied_hash = payload.get("report_hash")
        if not isinstance(supplied_hash, str) or supplied_hash != report.report_hash:
            raise ValueError("release report hash does not match report contents")
        if "passed" in payload and _require_bool(payload, "passed") != report.passed:
            raise ValueError("release report passed field does not match results")
        if "approval_eligible" in payload and (
            _require_bool(payload, "approval_eligible") != report.approval_eligible
        ):
            raise ValueError("release report approval_eligible field is inconsistent")
        supplied_checks = payload.get("gate_checks")
        if supplied_checks is not None and supplied_checks != report.gate_checks():
            raise ValueError("release report gate checks are inconsistent")
        supplied_reasons = payload.get("approval_reasons")
        if supplied_reasons is not None and supplied_reasons != list(report.approval_reasons()):
            raise ValueError("release report approval reasons are inconsistent")
        return report


def _lookup_metric(summary: Mapping[str, Any], metric: str) -> Any:
    current: Any = summary
    for part in metric.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise KeyError(f"metric {metric!r} not found")
    return current


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _require_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def thresholds_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> tuple[MetricThreshold, ...]:
    return tuple(MetricThreshold.from_payload(payload) for payload in payloads)
