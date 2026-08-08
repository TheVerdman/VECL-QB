from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vecl.evaluation.fisher import compute_lora_snapshot_drift
from vecl.evaluation.types import (
    EvaluationEvidence,
    EvaluationResult,
    MetricThreshold,
    thresholds_from_payloads,
)
from vecl.provenance.events import stable_hash


class ReleaseEvaluator(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def category(self) -> str: ...

    def evaluate(self, candidate_id: str) -> EvaluationResult: ...


@dataclass(frozen=True)
class DemoSummaryEvaluator:
    """Threshold evaluator for demos/tests; intentionally ineligible for approval."""

    name: str
    category: str
    summary: dict[str, Any]
    thresholds: tuple[MetricThreshold, ...]
    version: str = "demo-v1"

    def evaluate(self, candidate_id: str) -> EvaluationResult:
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
            evidence=_evidence(
                kind="synthetic_demo",
                evaluator_id=self.name,
                evaluator_version=self.version,
                candidate_id=candidate_id,
                configuration={
                    "category": self.category,
                    "thresholds": [threshold.to_payload() for threshold in self.thresholds],
                },
                command=("demo-summary", self.name),
                summary=self.summary,
                synthetic=True,
            ),
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
    version: str = "1"

    def evaluate(self, candidate_id: str) -> EvaluationResult:
        evaluator_env = {
            **os.environ,
            **(self.env or {}),
            "VECL_RELEASE_CANDIDATE_ID": candidate_id,
        }
        completed = subprocess.run(
            list(self.command),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            env=evaluator_env,
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
            evidence=_evidence(
                kind="command",
                evaluator_id=self.name,
                evaluator_version=self.version,
                candidate_id=candidate_id,
                configuration={
                    "category": self.category,
                    "command": list(self.command),
                    "summary_prefix": self.summary_prefix,
                    "thresholds": [threshold.to_payload() for threshold in self.thresholds],
                    "timeout_seconds": self.timeout_seconds,
                    "env": _safe_environment_configuration(self.env or {}),
                    "candidate_env_var": "VECL_RELEASE_CANDIDATE_ID",
                },
                command=self.command,
                summary={
                    **summary,
                    "command": list(self.command),
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                },
                artifact_hashes={
                    "stdout": _sha256_bytes(completed.stdout.encode()),
                    "stderr": _sha256_bytes(completed.stderr.encode()),
                },
            ),
        )


@dataclass(frozen=True)
class FisherDriftEvaluator:
    name: str
    category: str
    approved_snapshot: str
    candidate_snapshot: str
    drift_threshold: float
    thresholds: tuple[MetricThreshold, ...]
    version: str = "1"

    def evaluate(self, candidate_id: str) -> EvaluationResult:
        artifact_hashes: dict[str, str] = {}
        try:
            report = compute_lora_snapshot_drift(
                self.approved_snapshot,
                self.candidate_snapshot,
                drift_threshold=self.drift_threshold,
            )
            summary = report.to_payload()
            artifact_hashes = {
                "approved_snapshot": _sha256_file(Path(self.approved_snapshot)),
                "candidate_snapshot": _sha256_file(Path(self.candidate_snapshot)),
            }
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
            evidence=_evidence(
                kind="internal",
                evaluator_id=self.name,
                evaluator_version=self.version,
                candidate_id=candidate_id,
                configuration={
                    "category": self.category,
                    "approved_snapshot": self.approved_snapshot,
                    "candidate_snapshot": self.candidate_snapshot,
                    "drift_threshold": self.drift_threshold,
                    "thresholds": [threshold.to_payload() for threshold in self.thresholds],
                },
                command=("internal:fisher_drift",),
                summary=summary,
                artifact_hashes=artifact_hashes,
            ),
        )


def evaluator_from_config(config: Mapping[str, Any]) -> ReleaseEvaluator:
    evaluator_type = str(config.get("type") or "")
    thresholds = thresholds_from_payloads(config.get("thresholds", []))
    if evaluator_type == "demo_summary":
        return DemoSummaryEvaluator(
            name=str(config["name"]),
            category=str(config.get("category") or "regression"),
            summary=dict(config.get("summary") or {}),
            thresholds=thresholds,
            version=str(config.get("version") or "demo-v1"),
        )
    if evaluator_type == "summary":
        raise ValueError("summary evaluators are ambiguous; use demo_summary for non-release demos")
    if evaluator_type == "script":
        return ScriptEvaluator(
            name=str(config["name"]),
            category=str(config.get("category") or "regression"),
            command=tuple(str(part) for part in config["command"]),
            summary_prefix=str(config["summary_prefix"]),
            thresholds=thresholds,
            timeout_seconds=float(config.get("timeout_seconds", 900.0)),
            env={str(key): str(value) for key, value in dict(config.get("env") or {}).items()},
            version=str(config.get("version") or "1"),
        )
    if evaluator_type == "fisher_drift":
        return FisherDriftEvaluator(
            name=str(config["name"]),
            category=str(config.get("category") or "drift"),
            approved_snapshot=str(config["approved_snapshot"]),
            candidate_snapshot=str(config["candidate_snapshot"]),
            drift_threshold=float(config["drift_threshold"]),
            thresholds=thresholds,
            version=str(config.get("version") or "1"),
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


def _evidence(
    *,
    kind: str,
    evaluator_id: str,
    evaluator_version: str,
    candidate_id: str,
    configuration: Mapping[str, Any],
    command: tuple[str, ...],
    summary: Mapping[str, Any],
    artifact_hashes: Mapping[str, str] | None = None,
    synthetic: bool = False,
) -> EvaluationEvidence:
    git_revision, working_tree_clean = _git_state()
    return EvaluationEvidence(
        kind=kind,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        candidate_id=candidate_id,
        configuration=dict(configuration),
        configuration_hash=stable_hash(configuration),
        git_revision=git_revision,
        command=command,
        artifact_hashes={
            **dict(artifact_hashes or {}),
            "summary": stable_hash(summary),
        },
        working_tree_clean=working_tree_clean,
        synthetic=synthetic,
    )


def _git_state() -> tuple[str, bool]:
    repository_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or status.returncode != 0:
        return "", False
    return revision.stdout.strip().lower(), not status.stdout.strip()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_environment_configuration(environment: Mapping[str, str]) -> dict[str, str]:
    sensitive_fragments = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PRIVATE_KEY", "API_KEY")
    return {
        key: "<redacted>"
        if any(fragment in key.upper() for fragment in sensitive_fragments)
        else value
        for key, value in sorted(environment.items())
    }
