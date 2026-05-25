from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from vecl.qb.router import SpecialistCard
from vecl.qb.tool_call import (
    ToolCallValidationConfig,
    ToolCallValidationError,
    parse_tool_call_response,
    validate_tool_call,
)
from vecl.specialists.terraform_specialist import (
    TERRAFORM_ALLOWED_OPERATIONS,
    TERRAFORM_DISABLED_OPERATIONS,
)
from vecl.training.corpus_factory import CATEGORY_PLANS
from vecl.training.corpus_metrics import llm_cache_stats
from vecl.training.corpus_schema import (
    CORPUS_DATASET_VERSION_V1_HARD,
    JSON_TASK_KINDS,
    VALID_DIFFICULTY_TAGS,
    LLMRawCacheEntry,
    ToolUseCorpusRecord,
    corpus_hash,
    read_corpus_jsonl,
    stable_example_id,
    stable_split_for_record,
    supervised_examples_from_corpus,
)

SYNTHETIC_TERRAFORM_CONFIG_DIR = "/tmp/vecl-qb-synthetic-terraform-fixture"

IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class CorpusValidationIssue:
    severity: IssueSeverity
    check: str
    message: str
    example_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusValidationResult:
    valid: bool
    dataset_hash: str
    total_records: int
    executable_records: int
    supervised_records: int
    split_counts: dict[str, int]
    domain_counts: dict[str, int]
    category_counts: dict[str, int]
    task_kind_counts: dict[str, int]
    issues: tuple[CorpusValidationIssue, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [issue.to_payload() for issue in self.issues]
        return payload


def validate_corpus_path(
    path: str | Path,
    *,
    require_category_coverage: bool = True,
    expected_min_records: int | None = None,
    expected_max_records: int | None = None,
) -> CorpusValidationResult:
    return validate_corpus_records(
        read_corpus_jsonl(resolve_corpus_jsonl(path)),
        require_category_coverage=require_category_coverage,
        expected_min_records=expected_min_records,
        expected_max_records=expected_max_records,
    )


def validate_corpus_records(
    records: list[ToolUseCorpusRecord],
    *,
    require_category_coverage: bool = True,
    expected_min_records: int | None = None,
    expected_max_records: int | None = None,
) -> CorpusValidationResult:
    issues: list[CorpusValidationIssue] = []
    if not records:
        issues.append(_issue("schema", "corpus must contain at least one record"))
    if expected_min_records is not None and len(records) < expected_min_records:
        issues.append(
            _issue("record_count", f"record count {len(records)} is below {expected_min_records}")
        )
    if expected_max_records is not None and len(records) > expected_max_records:
        issues.append(
            _issue("record_count", f"record count {len(records)} exceeds {expected_max_records}")
        )

    ids: dict[str, ToolUseCorpusRecord] = {}
    prompt_target_pairs: dict[tuple[str, str], str] = {}
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        issues.extend(validate_single_record(record))
        if record.example_id in ids:
            issues.append(
                _issue(
                    "stable_ids",
                    f"duplicate example_id also used by {ids[record.example_id].category}",
                    record.example_id,
                )
            )
        ids[record.example_id] = record
        pair = (_normalize(record.prompt), _normalize(record.target_text))
        if pair in prompt_target_pairs:
            issues.append(
                _issue(
                    "dedupe",
                    f"duplicate prompt+target pair also used by {prompt_target_pairs[pair]}",
                    record.example_id,
                )
            )
        prompt_target_pairs[pair] = record.example_id
        prompt_splits[_normalize(record.prompt)].add(record.split)

    for prompt, splits in sorted(prompt_splits.items()):
        if len(splits) > 1:
            example_ids = sorted(
                record.example_id for record in records if _normalize(record.prompt) == prompt
            )
            issues.append(
                _issue(
                    "split_leakage",
                    f"prompt appears in train and heldout: {example_ids}",
                    example_ids[0] if example_ids else None,
                )
            )

    if len(records) >= 10:
        split_counts = Counter(record.split for record in records)
        if split_counts.get("train", 0) == 0 or split_counts.get("heldout", 0) == 0:
            issues.append(_issue("split_integrity", "train and heldout splits must be non-empty"))
        if any(record.dataset_version == CORPUS_DATASET_VERSION_V1_HARD for record in records):
            if split_counts.get("hard-heldout", 0) == 0:
                issues.append(
                    _issue("split_integrity", "v1-hard corpora must include hard-heldout records")
                )

    if require_category_coverage:
        expected_categories = {plan.name for plan in CATEGORY_PLANS}
        actual_categories = {record.category for record in records}
        missing = sorted(expected_categories - actual_categories)
        if missing:
            issues.append(
                _issue("category_coverage", "missing required categories: " + ", ".join(missing))
            )

    executable_records = sum(1 for record in records if record.executable)
    supervised = supervised_examples_from_corpus(records)
    issue_tuple = tuple(issues)
    return CorpusValidationResult(
        valid=not any(issue.severity == "error" for issue in issue_tuple),
        dataset_hash=corpus_hash(records) if records else "",
        total_records=len(records),
        executable_records=executable_records,
        supervised_records=len(supervised),
        split_counts=dict(sorted(Counter(record.split for record in records).items())),
        domain_counts=dict(sorted(Counter(record.domain for record in records).items())),
        category_counts=dict(sorted(Counter(record.category for record in records).items())),
        task_kind_counts=dict(sorted(Counter(record.task_kind for record in records).items())),
        issues=issue_tuple,
    )


def validate_single_record(record: ToolUseCorpusRecord) -> list[CorpusValidationIssue]:
    issues: list[CorpusValidationIssue] = []
    expected_id = stable_example_id(record)
    if record.example_id != expected_id:
        issues.append(
            _issue(
                "stable_ids",
                f"example_id is {record.example_id}; expected {expected_id}",
                record.example_id,
            )
        )
    expected_split = stable_split_for_record(record)
    if record.split != expected_split:
        issues.append(
            _issue(
                "split_integrity",
                f"split is {record.split}; expected {expected_split}",
                record.example_id,
            )
        )
    issues.extend(_metadata_issues(record))
    issues.extend(_safety_issues(record))
    if record.task_kind in JSON_TASK_KINDS:
        issues.extend(_json_target_issues(record))
    if "negative" in record.category or record.task_kind == "no_tool_json":
        issues.extend(_negative_issues(record))
    if record.domain == "terraform" or record.specialist_id == "terraform":
        issues.extend(_terraform_issues(record))
    if record.executable and record.task_kind == "tool_call_json":
        issues.extend(_tool_contract_issues(record))
    if record.executable:
        try:
            record.to_supervised_example()
        except ValueError as exc:
            issues.append(_issue("conversion", str(exc), record.example_id))
    return issues


def single_record_acceptor(record: ToolUseCorpusRecord) -> tuple[bool, list[str]]:
    issues = validate_single_record(record)
    errors = [f"{issue.check}: {issue.message}" for issue in issues if issue.severity == "error"]
    return not errors, errors


def resolve_corpus_jsonl(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_dir():
        return raw / "corpus.jsonl"
    return raw


def corpus_markdown_report(result: CorpusValidationResult, *, title: str) -> str:
    return corpus_markdown_report_with_metadata(result, title=title)


def corpus_markdown_report_with_metadata(
    result: CorpusValidationResult,
    *,
    title: str,
    llm_cache_entries: list[LLMRawCacheEntry] | None = None,
    metadata: dict[str, Any] | None = None,
    example_records: list[ToolUseCorpusRecord] | None = None,
) -> str:
    llm_stats = llm_cache_stats(llm_cache_entries or [])
    duplicate_stats = dict((metadata or {}).get("duplicate_stats") or {})
    cost_by_provider = dict(
        (metadata or {}).get("cost_by_provider") or llm_stats["cost_by_provider"]
    )
    lines = [
        f"# {title}",
        "",
        f"- Valid: {result.valid}",
        f"- Dataset hash: `{result.dataset_hash}`",
        f"- Total records: {result.total_records}",
        f"- Executable records: {result.executable_records}",
        f"- Supervised-convertible records: {result.supervised_records}",
        f"- Validation errors: {sum(1 for issue in result.issues if issue.severity == 'error')}",
        f"- Validation warnings: {sum(1 for issue in result.issues if issue.severity == 'warning')}",
        "",
        "## Splits",
        "",
        _markdown_table(("split", "count"), result.split_counts.items()),
        "",
        "## Domains",
        "",
        _markdown_table(("domain", "count"), result.domain_counts.items()),
        "",
        "## Categories",
        "",
        _markdown_table(("category", "count"), result.category_counts.items()),
        "",
        "## Task Kinds",
        "",
        _markdown_table(("task_kind", "count"), result.task_kind_counts.items()),
        "",
        "## Notes",
        "",
        "- LLM augmentation is optional and disabled by default; this report records only admitted records.",
        "- Terraform records are constrained to plan/validate or non-executable refusal/review examples.",
        "- Yosys/OpenROAD records use deterministic local fixture payloads; Docker-backed execution remains opt-in.",
    ]
    if llm_cache_entries is not None or metadata:
        difficulty_counts = dict((metadata or {}).get("difficulty_counts") or {})
        semantic_metrics = dict((metadata or {}).get("semantic_diversity") or {})
        export_metrics = dict((metadata or {}).get("training_export_diversity") or {})
        lines.extend(
            [
                "",
                "## LLM Augmentation",
                "",
                f"- Cache entries: {llm_stats['cache_entries']}",
                f"- Accepted LLM records: {llm_stats['accepted_records']}",
                f"- Admitted LLM records file: `{(metadata or {}).get('llm_admitted_path', '')}`",
                f"- Rejected LLM reasons: {llm_stats['rejected_reasons']}",
                f"- Estimated cost by provider: `{json.dumps(cost_by_provider, sort_keys=True)}`",
                f"- Duplicate records removed: {duplicate_stats.get('duplicate_records_removed', 0)}",
                f"- Duplicate rate: {duplicate_stats.get('duplicate_rate', 0.0)}",
            ]
        )
        if difficulty_counts:
            lines.extend(
                [
                    "",
                    "## Difficulty Tags",
                    "",
                    _markdown_table(("tag", "count"), difficulty_counts.items()),
                ]
            )
        if semantic_metrics:
            lines.extend(
                [
                    "",
                    "## Semantic Diversity",
                    "",
                    _markdown_table(("metric", "value"), semantic_metrics.items()),
                ]
            )
        if export_metrics:
            lines.extend(
                [
                    "",
                    "## Training Export Diversity",
                    "",
                    _markdown_table(("metric", "value"), export_metrics.items()),
                ]
            )
    if example_records:
        lines.extend(["", "## Examples", ""])
        for record in example_records[:8]:
            tags = ",".join(str(tag) for tag in record.metadata.get("difficulty_tags", []))
            prompt = record.prompt.replace("\n", " ")[:180]
            lines.append(
                f"- `{record.example_id}` {record.category} [{record.split}; {tags}]: {prompt}"
            )
    if result.issues:
        lines.extend(["", "## Issues", ""])
        for issue in result.issues[:50]:
            location = f" `{issue.example_id}`" if issue.example_id else ""
            lines.append(f"- {issue.severity.upper()} {issue.check}{location}: {issue.message}")
        if len(result.issues) > 50:
            lines.append(f"- ... {len(result.issues) - 50} additional issues omitted")
    return "\n".join(lines) + "\n"


def _metadata_issues(record: ToolUseCorpusRecord) -> list[CorpusValidationIssue]:
    required = (
        "generator_name",
        "seed",
        "topic_id",
        "trust_anchor_source_id",
        "trust_anchor_kind",
        "synthetic_only",
        "production_user_data",
        "validator_authoritative",
    )
    issues: list[CorpusValidationIssue] = []
    for key in required:
        if key not in record.metadata:
            issues.append(_issue("metadata", f"metadata missing {key}", record.example_id))
    if record.metadata.get("synthetic_only") is not True:
        issues.append(_issue("metadata", "synthetic_only must be true", record.example_id))
    if record.metadata.get("production_user_data") is not False:
        issues.append(_issue("metadata", "production_user_data must be false", record.example_id))
    if record.metadata.get("validator_authoritative") is not True:
        issues.append(_issue("metadata", "validator_authoritative must be true", record.example_id))
    if str(record.metadata.get("trust_anchor_source_id") or "") not in record.source_ids:
        issues.append(
            _issue(
                "metadata",
                "trust_anchor_source_id must appear in source_ids",
                record.example_id,
            )
        )
    if record.dataset_version == CORPUS_DATASET_VERSION_V1_HARD:
        tags = record.metadata.get("difficulty_tags")
        if not isinstance(tags, list) or not tags:
            issues.append(
                _issue("metadata", "v1-hard records require difficulty_tags", record.example_id)
            )
        else:
            invalid = sorted({str(tag) for tag in tags} - VALID_DIFFICULTY_TAGS)
            if invalid:
                issues.append(
                    _issue(
                        "metadata",
                        "invalid difficulty_tags: " + ", ".join(invalid),
                        record.example_id,
                    )
                )
        for key in (
            "llm_origin",
            "llm_generated_target",
            "target_independently_verified",
            "target_source",
            "training_export_eligible",
        ):
            if key not in record.metadata:
                issues.append(
                    _issue("metadata", f"v1-hard metadata missing {key}", record.example_id)
                )
        if record.metadata.get("llm_origin") is True:
            for key in ("llm_provider", "llm_model_id", "llm_prompt_hash", "seed_example_id"):
                if key not in record.metadata:
                    issues.append(
                        _issue("metadata", f"LLM-origin record missing {key}", record.example_id)
                    )
    return issues


def _json_target_issues(record: ToolUseCorpusRecord) -> list[CorpusValidationIssue]:
    try:
        payload = record.target_json()
    except (json.JSONDecodeError, ValueError) as exc:
        return [_issue("json_target", str(exc), record.example_id)]
    issues: list[CorpusValidationIssue] = []
    if record.task_kind == "tool_call_json":
        for key in ("specialist_id", "task_type", "input_payload"):
            if key not in payload:
                issues.append(
                    _issue("json_target", f"target JSON missing {key}", record.example_id)
                )
        if payload.get("specialist_id") != record.specialist_id:
            issues.append(_issue("json_target", "specialist_id mismatch", record.example_id))
        if payload.get("task_type") != record.task_type:
            issues.append(_issue("json_target", "task_type mismatch", record.example_id))
        if not isinstance(payload.get("input_payload"), dict):
            issues.append(
                _issue("json_target", "input_payload must be an object", record.example_id)
            )
    if record.task_kind == "no_tool_json":
        if payload.get("specialist_id") is not None or payload.get("task_type") is not None:
            issues.append(
                _issue(
                    "negative", "no-tool target must keep specialist/task null", record.example_id
                )
            )
    return issues


def _negative_issues(record: ToolUseCorpusRecord) -> list[CorpusValidationIssue]:
    issues: list[CorpusValidationIssue] = []
    if record.executable:
        issues.append(
            _issue("negative", "negative/no-tool records must not be executable", record.example_id)
        )
    if record.task_kind == "no_tool_json":
        try:
            payload = record.target_json()
        except (json.JSONDecodeError, ValueError):
            return issues
        if payload.get("specialist_id") is not None:
            issues.append(
                _issue("negative", "negative target selected a specialist", record.example_id)
            )
    return issues


def _terraform_issues(record: ToolUseCorpusRecord) -> list[CorpusValidationIssue]:
    issues: list[CorpusValidationIssue] = []
    if record.task_kind == "tool_call_json":
        try:
            payload = record.target_json()
        except (json.JSONDecodeError, ValueError):
            return issues
        if payload.get("specialist_id") == "terraform":
            input_payload = payload.get("input_payload") or {}
            operation = _terraform_operation(input_payload)
            if operation not in TERRAFORM_ALLOWED_OPERATIONS:
                issues.append(
                    _issue(
                        "terraform",
                        f"terraform mutation is never allowed: {operation}",
                        record.example_id,
                    )
                )
            if input_payload.get("config_dir") is not None:
                issues.append(
                    _issue(
                        "terraform",
                        "model target must not supply Terraform config_dir",
                        record.example_id,
                    )
                )
    if record.task_kind in {"refusal", "review"} or "mutation" in record.category:
        if record.executable:
            issues.append(
                _issue(
                    "terraform",
                    "mutation refusal/review records must be non-executable",
                    record.example_id,
                )
            )
        target = record.target_text.lower()
        refusal_markers = ("cannot", "must not", "plan-only", "human review", "no automatic")
        if not any(marker in target for marker in refusal_markers):
            issues.append(
                _issue(
                    "terraform", "mutation target must refuse or require review", record.example_id
                )
            )
    return issues


def _tool_contract_issues(record: ToolUseCorpusRecord) -> list[CorpusValidationIssue]:
    try:
        proposal = parse_tool_call_response(record.target_text)
        validate_tool_call(
            proposal,
            cards=_specialist_cards(),
            config=ToolCallValidationConfig(
                tenant_id=record.tenant_id,
                request_id=record.example_id,
                terraform_config_dir=SYNTHETIC_TERRAFORM_CONFIG_DIR,
            ),
        )
    except (ValueError, ToolCallValidationError) as exc:
        return [_issue("specialist_contract", str(exc), record.example_id)]
    return []


def _safety_issues(record: ToolUseCorpusRecord) -> list[CorpusValidationIssue]:
    text = "\n".join(
        [record.prompt, record.target_text, json.dumps(record.metadata, sort_keys=True)]
    )
    issues: list[CorpusValidationIssue] = []
    for name, pattern in _secret_patterns().items():
        if pattern.search(text):
            issues.append(_issue("safety", f"possible {name} in record text", record.example_id))
    if re.search(r"(?i)\b(exfiltrate|malware|credential theft|phishing kit)\b", text):
        issues.append(_issue("safety", "unsafe instruction phrase detected", record.example_id))
    return issues


def _specialist_cards() -> dict[str, SpecialistCard]:
    return {
        "stockfish": SpecialistCard(
            "stockfish",
            {"chess_eval"},
            {"description": "Stockfish chess analysis"},
            cost_hint=1.0,
            latency_hint=1.0,
            version="stockfish-18",
            effective_trust=0.9,
            description="Analyze chess FENs.",
        ),
        "sympy": SpecialistCard(
            "sympy",
            {"symbolic_math", "sequence_math"},
            {"description": "SymPy symbolic math"},
            cost_hint=0.2,
            latency_hint=0.3,
            version="sympy-1.14",
            effective_trust=0.95,
            description="Exact symbolic math.",
        ),
        "blast": SpecialistCard(
            "blast",
            {"sequence_alignment"},
            {"description": "BLAST local alignment"},
            cost_hint=0.8,
            latency_hint=1.0,
            version="blast-2.17",
            effective_trust=0.9,
            description="Local fixture sequence alignment.",
        ),
        "terraform": SpecialistCard(
            "terraform",
            {"infrastructure_plan"},
            {"description": "Terraform plan-only"},
            cost_hint=0.5,
            latency_hint=0.6,
            version="terraform-cli",
            effective_trust=0.95,
            description="Plan and validate only.",
        ),
        "timesfm": SpecialistCard(
            "timesfm",
            {"demand_forecast", "tool_request"},
            {"description": "TimesFM demand forecast"},
            cost_hint=0.7,
            latency_hint=1.2,
            version="timesfm-2.5",
            effective_trust=0.9,
            description="Demand forecasting.",
        ),
        "yosys": SpecialistCard(
            "yosys",
            {"hardware_synthesis", "eda_flow"},
            {"description": "Yosys hardware synthesis"},
            cost_hint=0.8,
            latency_hint=1.0,
            version="yosys-cli",
            effective_trust=0.9,
            description="Synthesize local Verilog RTL.",
        ),
        "openroad": SpecialistCard(
            "openroad",
            {"physical_design", "eda_flow"},
            {"description": "OpenROAD physical design"},
            cost_hint=1.2,
            latency_hint=1.5,
            version="openroad-cli",
            effective_trust=0.9,
            description="Analyze local synthesized netlists.",
        ),
    }


def _terraform_operation(input_payload: object) -> str:
    if not isinstance(input_payload, dict):
        return ""
    raw = (
        str(input_payload.get("operation") or input_payload.get("command") or "plan")
        .strip()
        .lower()
    )
    if raw.startswith("terraform "):
        raw = raw.removeprefix("terraform ").strip()
    operation = raw.split()[0] if raw else "plan"
    return operation if operation not in TERRAFORM_DISABLED_OPERATIONS else operation


def _secret_patterns() -> dict[str, re.Pattern[str]]:
    return {
        "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "api key": re.compile(r"(?i)\b(api[_-]?key|password|secret|bearer)\s*[:=]\s*\S+"),
        "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "aws key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    }


def _markdown_table(headers: tuple[str, str], rows: Any) -> str:
    materialized = [(str(key), str(value)) for key, value in rows]
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | ---: |"]
    lines.extend(f"| {key} | {value} |" for key, value in materialized)
    return "\n".join(lines)


def _issue(check: str, message: str, example_id: str | None = None) -> CorpusValidationIssue:
    return CorpusValidationIssue("error", check, message, example_id)


def _normalize(value: str) -> str:
    return " ".join(value.strip().split())
