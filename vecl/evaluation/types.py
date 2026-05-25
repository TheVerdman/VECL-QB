from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vecl._compat import UTC

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


@dataclass(frozen=True)
class MetricThreshold:
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    equals: Any | None = None

    def evaluate(self, summary: Mapping[str, Any]) -> list[str]:
        value = _lookup_metric(summary, self.metric)
        reasons: list[str] = []
        if self.equals is not None and value != self.equals:
            reasons.append(f"{self.metric} expected {self.equals!r}, got {value!r}")
        if self.min_value is not None and float(value) < self.min_value:
            reasons.append(f"{self.metric} below {self.min_value}: {value}")
        if self.max_value is not None and float(value) > self.max_value:
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
class EvaluationResult:
    name: str
    category: str
    passed: bool
    summary: dict[str, Any]
    reasons: tuple[str, ...] = ()
    thresholds: tuple[MetricThreshold, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "thresholds": [threshold.to_payload() for threshold in self.thresholds],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvaluationResult:
        return cls(
            name=str(payload["name"]),
            category=str(payload["category"]),
            passed=bool(payload["passed"]),
            summary=dict(payload.get("summary") or {}),
            reasons=tuple(str(reason) for reason in payload.get("reasons", [])),
            thresholds=tuple(
                MetricThreshold.from_payload(threshold)
                for threshold in payload.get("thresholds", [])
            ),
        )


@dataclass(frozen=True)
class ReleaseEvaluationReport:
    candidate_id: str
    results: tuple[EvaluationResult, ...]
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def gate_checks(self) -> dict[str, bool]:
        checks = {field: True for field in GATE_FIELDS}
        seen_fields: set[str] = set()
        for result in self.results:
            field = GATE_CATEGORY_MAP.get(result.category, "regression_tests_pass")
            checks[field] = checks[field] and result.passed
            seen_fields.add(field)
        for field in GATE_FIELDS:
            if field == "drift_within_bound":
                continue
            if field not in seen_fields:
                checks[field] = self.passed
        return checks

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generated_at": self.generated_at.isoformat(),
            "passed": self.passed,
            "gate_checks": self.gate_checks(),
            "results": [result.to_payload() for result in self.results],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReleaseEvaluationReport:
        return cls(
            candidate_id=str(payload["candidate_id"]),
            generated_at=datetime.fromisoformat(str(payload["generated_at"])),
            results=tuple(
                EvaluationResult.from_payload(result) for result in payload.get("results", [])
            ),
        )


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


def thresholds_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> tuple[MetricThreshold, ...]:
    return tuple(MetricThreshold.from_payload(payload) for payload in payloads)
