from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from vecl.evaluation.fisher import compute_lora_snapshot_drift
from vecl.evaluation.types import EvaluationResult, MetricThreshold, thresholds_from_payloads


class ReleaseEvaluator(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def category(self) -> str: ...

    def evaluate(self) -> EvaluationResult: ...


@dataclass(frozen=True)
class SummaryEvaluator:
    name: str
    category: str
    summary: dict[str, Any]
    thresholds: tuple[MetricThreshold, ...]

    def evaluate(self) -> EvaluationResult:
        reasons: list[str] = []
        for threshold in self.thresholds:
            try:
                reasons.extend(threshold.evaluate(self.summary))
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(str(exc))
        return EvaluationResult(
            name=self.name,
            category=self.category,
            passed=not reasons,
            summary=dict(self.summary),
            reasons=tuple(reasons),
            thresholds=self.thresholds,
        )


@dataclass(frozen=True)
class ScriptEvaluator:
    name: str
    category: str
    command: tuple[str, ...]
    summary_prefix: str
    thresholds: tuple[MetricThreshold, ...]
    timeout_seconds: float = 900.0
    env: dict[str, str] | None = None

    def evaluate(self) -> EvaluationResult:
        completed = subprocess.run(
            list(self.command),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            env={**os.environ, **(self.env or {})},
        )
        summary, reasons = _parse_summary_line(completed.stdout, self.summary_prefix)
        if completed.returncode != 0:
            reasons.append(f"command exited {completed.returncode}")
        for threshold in self.thresholds:
            try:
                reasons.extend(threshold.evaluate(summary))
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(str(exc))
        return EvaluationResult(
            name=self.name,
            category=self.category,
            passed=not reasons,
            summary={
                **summary,
                "command": list(self.command),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            },
            reasons=tuple(reasons),
            thresholds=self.thresholds,
        )


@dataclass(frozen=True)
class FisherDriftEvaluator:
    name: str
    category: str
    approved_snapshot: str
    candidate_snapshot: str
    drift_threshold: float
    thresholds: tuple[MetricThreshold, ...]

    def evaluate(self) -> EvaluationResult:
        try:
            report = compute_lora_snapshot_drift(
                self.approved_snapshot,
                self.candidate_snapshot,
                drift_threshold=self.drift_threshold,
            )
            summary = report.to_payload()
            reasons = [] if report.drift_within_bound else ["Fisher-weighted drift exceeded bound"]
        except (OSError, ValueError, KeyError) as exc:
            summary = {
                "approved_snapshot": self.approved_snapshot,
                "candidate_snapshot": self.candidate_snapshot,
                "drift_threshold": self.drift_threshold,
            }
            reasons = [str(exc)]
        for threshold in self.thresholds:
            try:
                reasons.extend(threshold.evaluate(summary))
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(str(exc))
        return EvaluationResult(
            name=self.name,
            category=self.category,
            passed=not reasons,
            summary=summary,
            reasons=tuple(reasons),
            thresholds=self.thresholds,
        )


def evaluator_from_config(config: Mapping[str, Any]) -> ReleaseEvaluator:
    evaluator_type = str(config.get("type") or "summary")
    thresholds = thresholds_from_payloads(config.get("thresholds", []))
    if evaluator_type == "summary":
        return SummaryEvaluator(
            name=str(config["name"]),
            category=str(config.get("category") or "regression"),
            summary=dict(config.get("summary") or {}),
            thresholds=thresholds,
        )
    if evaluator_type == "script":
        return ScriptEvaluator(
            name=str(config["name"]),
            category=str(config.get("category") or "regression"),
            command=tuple(str(part) for part in config["command"]),
            summary_prefix=str(config["summary_prefix"]),
            thresholds=thresholds,
            timeout_seconds=float(config.get("timeout_seconds", 900.0)),
            env={str(key): str(value) for key, value in dict(config.get("env") or {}).items()},
        )
    if evaluator_type == "fisher_drift":
        return FisherDriftEvaluator(
            name=str(config["name"]),
            category=str(config.get("category") or "drift"),
            approved_snapshot=str(config["approved_snapshot"]),
            candidate_snapshot=str(config["candidate_snapshot"]),
            drift_threshold=float(config["drift_threshold"]),
            thresholds=thresholds,
        )
    raise ValueError(f"unsupported evaluator type: {evaluator_type}")


def evaluators_from_manifest(payload: Mapping[str, Any]) -> tuple[ReleaseEvaluator, ...]:
    return tuple(evaluator_from_config(config) for config in payload.get("evaluators", []))


def _parse_summary_line(stdout: str, summary_prefix: str) -> tuple[dict[str, Any], list[str]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(summary_prefix):
            raw = line.removeprefix(summary_prefix).strip()
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {}, ["summary JSON was not an object"]
            return payload, []
    return {}, [f"missing summary prefix: {summary_prefix}"]
