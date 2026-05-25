from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from vecl.provenance.events import stable_hash
from vecl.qb.model_driver import ModelDriverResult
from vecl.specialists.blast_specialist import BLAST_SOURCE_ID
from vecl.specialists.openroad_specialist import OPENROAD_SOURCE_ID
from vecl.specialists.stockfish import STOCKFISH_SOURCE_ID
from vecl.specialists.sympy_specialist import SYMPY_SOURCE_ID
from vecl.specialists.terraform_specialist import TERRAFORM_SOURCE_ID
from vecl.specialists.timesfm_specialist import TIMESFM_SOURCE_ID
from vecl.specialists.yosys_specialist import YOSYS_SOURCE_ID
from vecl.training.corpus_metrics import admitted_llm_records
from vecl.training.corpus_schema import (
    CORPUS_DATASET_VERSION,
    CORPUS_DATASET_VERSION_V1_HARD,
    CORPUS_TENANT_ID,
    CORPUS_TENANT_ID_V1_HARD,
    CorpusTaskKind,
    LLMRawCacheEntry,
    TargetFormat,
    ToolUseCorpusRecord,
    stable_example_id,
    stable_split_for_id,
)

POLICY_SOURCE_ID = "vecl-synthetic-policy-v0"
PLACEHOLDER_SOURCE_ID = "vecl-specialist-placeholder-v0"
TRUST_ANCHOR_KIND = "VERIFIED_OPERATIONAL_RECORD"
POLICY_TRUST_ANCHOR_KIND = "SYNTHETIC_GOVERNANCE_FIXTURE"
DEFAULT_SMALL_COUNT = 420
DEFAULT_FULL_COUNT = 5000
DEFAULT_V1_HARD_COUNT = 25000
DEFAULT_SAMPLE_COUNT = 64
DEFAULT_PROVIDER_COST_CAPS = {"anthropic": 18.0, "openai": 20.0}
DEFAULT_MAX_ESTIMATED_COST_PER_CALL_USD = 0.25

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(frozen=True)
class SyntheticCorpusConfig:
    size: str = "sample"
    target_count: int | None = None
    seed: int = 1107
    include_placeholders: bool = True

    def resolved_count(self) -> int:
        if self.target_count is not None:
            if self.target_count <= 0:
                raise ValueError("target_count must be positive")
            return self.target_count
        if self.size == "v0-small":
            return DEFAULT_SMALL_COUNT
        if self.size == "v0-full":
            return DEFAULT_FULL_COUNT
        if self.size == "v1-hard":
            return DEFAULT_V1_HARD_COUNT
        return DEFAULT_SAMPLE_COUNT

    @property
    def dataset_version(self) -> str:
        return CORPUS_DATASET_VERSION_V1_HARD if self.size == "v1-hard" else CORPUS_DATASET_VERSION

    @property
    def tenant_id(self) -> str:
        return CORPUS_TENANT_ID_V1_HARD if self.size == "v1-hard" else CORPUS_TENANT_ID


@dataclass(frozen=True)
class SyntheticCorpusBuild:
    records: list[ToolUseCorpusRecord]
    llm_cache_entries: list[LLMRawCacheEntry]
    llm_admitted_records: list[ToolUseCorpusRecord]
    duplicate_stats: dict[str, Any]
    cost_by_provider: dict[str, float]


class SynthesizeOnlyDriver(Protocol):
    provider: str
    model_id: str

    def synthesize(self, prompt: str) -> ModelDriverResult:
        raise NotImplementedError


RecordValidator = Callable[[ToolUseCorpusRecord], tuple[bool, Sequence[str]]]
LLMCacheCheckpoint = Callable[[LLMRawCacheEntry], None]


@dataclass(frozen=True)
class _CategoryPlan:
    name: str
    weight: int
    generator_name: str


CATEGORY_PLANS: tuple[_CategoryPlan, ...] = (
    _CategoryPlan("stockfish_tool_call", 11, "_stockfish_tool_call_record"),
    _CategoryPlan("stockfish_final_answer", 6, "_stockfish_final_answer_record"),
    _CategoryPlan("stockfish_negative", 4, "_stockfish_negative_record"),
    _CategoryPlan("sympy_tool_call", 10, "_sympy_tool_call_record"),
    _CategoryPlan("sympy_final_answer", 6, "_sympy_final_answer_record"),
    _CategoryPlan("sympy_negative", 4, "_sympy_negative_record"),
    _CategoryPlan("blast_config", 5, "_blast_config_record"),
    _CategoryPlan("blast_fixture_tool_call", 8, "_blast_fixture_record"),
    _CategoryPlan("blast_negative", 3, "_blast_negative_record"),
    _CategoryPlan("terraform_plan_tool_call", 7, "_terraform_plan_record"),
    _CategoryPlan("terraform_validate_tool_call", 3, "_terraform_validate_record"),
    _CategoryPlan("terraform_mutation_refusal", 5, "_terraform_refusal_record"),
    _CategoryPlan("terraform_human_review", 3, "_terraform_review_record"),
    _CategoryPlan("terraform_negative", 3, "_terraform_negative_record"),
    _CategoryPlan("timesfm_demand_tool_call", 8, "_timesfm_tool_call_record"),
    _CategoryPlan("timesfm_inventory_final_answer", 4, "_timesfm_final_answer_record"),
    _CategoryPlan("timesfm_to_sympy_tool_call", 4, "_timesfm_to_sympy_record"),
    _CategoryPlan("timesfm_negative", 3, "_timesfm_negative_record"),
    _CategoryPlan("cross_specialist_regression", 4, "_cross_specialist_record"),
    _CategoryPlan("eda_yosys_tool_call", 3, "_eda_yosys_tool_call_record"),
    _CategoryPlan("eda_openroad_tool_call", 3, "_eda_openroad_tool_call_record"),
    _CategoryPlan("eda_yosys_openroad_chain", 3, "_eda_yosys_openroad_chain_record"),
    _CategoryPlan("eda_final_answer", 2, "_eda_final_answer_record"),
    _CategoryPlan("eda_negative", 3, "_eda_negative_record"),
)

FENS = (
    "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n1p3/8/3P4/2N5/PPP1PPPP/R1BQKBNR b KQkq - 2 3",
    "rnbqkb1r/pppppppp/5n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "rnbqk2r/ppp2ppp/3bpn2/3p4/3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQkq - 2 5",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2P2N2/PP3PPP/RNBQ1RK1 w - - 0 8",
    "rn1qkbnr/ppp2ppp/4p3/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R b KQkq - 2 4",
    "r2qkbnr/ppp2ppp/2n1p3/3p4/3P1B2/2N1PN2/PPP2PPP/R2QKB1R b KQkq - 4 5",
    "rnbq1rk1/ppp2ppp/4pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 2 6",
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 4 5",
    "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2P1BN2/PP1N1PPP/R2QKB1R w KQkq - 4 9",
    "2rq1rk1/pp2bppp/2n1pn2/3p4/3P4/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 2 10",
    "rnbqkbnr/ppp2ppp/4p3/3p4/3P4/5NP1/PPP1PP1P/RNBQKB1R b KQkq - 0 3",
    "r1bqkbnr/ppp2ppp/2n1p3/3p4/3P4/2N2NP1/PPP1PP1P/R1BQKB1R b KQkq - 2 4",
    "rnbqk2r/pp2bppp/4pn2/2pp4/3P4/2P1PN2/PP3PPP/RNBQKB1R w KQkq - 0 6",
    "r2q1rk1/pppbbppp/2n1pn2/3p4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 4 8",
    "8/8/8/3k4/8/4K3/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/4K3/4R3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
)

STOCKFISH_PROMPTS = (
    "Analyze this FEN with Stockfish at depth {depth}: {fen}",
    "Use the chess engine for a depth-{depth} evaluation of {fen}.",
    "Route this chess position to Stockfish, depth {depth}: {fen}",
    "I need an engine-backed best move for FEN {fen}; search depth {depth}.",
    "Prepare a Stockfish tool call for {fen} with depth {depth}.",
    "For this legal chess snapshot, ask Stockfish for depth {depth}: {fen}",
)

SYMPY_PROMPTS = (
    "Use symbolic math to {verb} the expression {expression}.",
    "Route this exact math request to SymPy: {verb} {expression}.",
    "Prepare a SymPy call for operation {operation} on {expression}.",
    "I need deterministic symbolic computation: {verb} {expression}.",
    "Use the algebra specialist for {operation} with expression {expression}.",
)

NEGATIVE_PROMPTS = {
    "stockfish": (
        "Explain what FEN notation means without analyzing a specific position.",
        "Give a plain-language summary of why chess engines use search depth.",
        "List three general chess study habits without running an engine.",
    ),
    "sympy": (
        "Explain the difference between symbolic and numeric math without solving an equation.",
        "Describe when factoring is useful, but do not compute a result.",
        "Give a high-level definition of an integral without using a tool.",
    ),
    "blast": (
        "Explain what BLAST does at a conceptual level without aligning a sequence.",
        "Summarize why local fixture databases are safer for tests than public data.",
        "Describe FASTA headers without querying a sequence database.",
    ),
    "terraform": (
        "Explain why Terraform plan output should be reviewed before any apply.",
        "Summarize the difference between plan and apply without running Terraform.",
        "Describe immutable infrastructure in general terms only.",
    ),
    "timesfm": (
        "Explain what a forecast horizon is without computing a forecast.",
        "Describe why demand forecasts should include uncertainty.",
        "Give a non-technical definition of inventory buffer.",
    ),
}


def generate_synthetic_corpus(
    config: SyntheticCorpusConfig | None = None,
    *,
    llm_driver: SynthesizeOnlyDriver | None = None,
    llm_drivers: Sequence[SynthesizeOnlyDriver] | None = None,
    llm_limit: int = 0,
    record_validator: RecordValidator | None = None,
    existing_llm_cache_entries: Sequence[LLMRawCacheEntry] | None = None,
    llm_cache_checkpoint: LLMCacheCheckpoint | None = None,
    provider_cost_caps: dict[str, float] | None = None,
    max_estimated_cost_per_call_usd: float = DEFAULT_MAX_ESTIMATED_COST_PER_CALL_USD,
) -> SyntheticCorpusBuild:
    resolved = config or SyntheticCorpusConfig()
    records = generate_deterministic_records(resolved)
    cache_entries: list[LLMRawCacheEntry] = []
    duplicate_stats = _duplicate_stats(records, records)
    cost_by_provider: dict[str, float] = {}
    drivers = [*(llm_drivers or ())]
    if llm_driver is not None:
        drivers.insert(0, llm_driver)
    if drivers and llm_limit > 0:
        augmented: list[ToolUseCorpusRecord] = []
        for driver_index, driver in enumerate(drivers):
            driver_augmented, driver_cache = augment_records_with_llm(
                seed_records=records[driver_index :: max(1, len(drivers))],
                driver=driver,
                limit=llm_limit,
                seed=resolved.seed + driver_index * 10_000,
                record_validator=record_validator,
                existing_cache_entries=existing_llm_cache_entries,
                cache_entry_checkpoint=llm_cache_checkpoint,
                provider_cost_caps=provider_cost_caps or DEFAULT_PROVIDER_COST_CAPS,
                provider_costs=cost_by_provider,
                max_estimated_cost_per_call_usd=max_estimated_cost_per_call_usd,
            )
            augmented.extend(driver_augmented)
            cache_entries.extend(driver_cache)
        combined = [*records, *augmented]
        records = _dedupe_records(combined)
        duplicate_stats = _duplicate_stats(combined, records)
    return SyntheticCorpusBuild(
        records=records,
        llm_cache_entries=cache_entries,
        llm_admitted_records=admitted_llm_records(records),
        duplicate_stats=duplicate_stats,
        cost_by_provider={key: round(value, 8) for key, value in sorted(cost_by_provider.items())},
    )


def generate_deterministic_records(
    config: SyntheticCorpusConfig | None = None,
) -> list[ToolUseCorpusRecord]:
    resolved = config or SyntheticCorpusConfig()
    if resolved.size == "v1-hard":
        return _generate_v1_hard_records(resolved)
    counts = category_counts(
        resolved.resolved_count(), include_placeholders=resolved.include_placeholders
    )
    produced: list[ToolUseCorpusRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    for plan in CATEGORY_PLANS:
        count = counts.get(plan.name, 0)
        generator = globals()[plan.generator_name]
        accepted = 0
        index = 0
        while accepted < count:
            record = generator(index, resolved.seed)
            pair = (_normalize(record.prompt), _normalize(record.target_text))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                produced.append(record)
                accepted += 1
            index += 1
            if index > count * 50 + 100:
                raise RuntimeError(f"could not generate enough unique records for {plan.name}")
    return sorted(produced, key=lambda item: item.example_id)


def category_counts(total: int, *, include_placeholders: bool = True) -> dict[str, int]:
    plans = [
        plan for plan in CATEGORY_PLANS if include_placeholders or "placeholder" not in plan.name
    ]
    weight_total = sum(plan.weight for plan in plans)
    counts = {plan.name: (total * plan.weight) // weight_total for plan in plans}
    remainder = total - sum(counts.values())
    order = sorted(plans, key=lambda plan: stable_hash({"category": plan.name}))
    for plan in order[:remainder]:
        counts[plan.name] += 1
    return counts


def augment_records_with_llm(
    *,
    seed_records: Sequence[ToolUseCorpusRecord],
    driver: SynthesizeOnlyDriver,
    limit: int,
    seed: int,
    record_validator: RecordValidator | None = None,
    existing_cache_entries: Sequence[LLMRawCacheEntry] | None = None,
    cache_entry_checkpoint: LLMCacheCheckpoint | None = None,
    provider_cost_caps: dict[str, float] | None = None,
    provider_costs: dict[str, float] | None = None,
    max_estimated_cost_per_call_usd: float = DEFAULT_MAX_ESTIMATED_COST_PER_CALL_USD,
) -> tuple[list[ToolUseCorpusRecord], list[LLMRawCacheEntry]]:
    augmented: list[ToolUseCorpusRecord] = []
    cache_entries: list[LLMRawCacheEntry] = []
    cost_caps = provider_cost_caps or DEFAULT_PROVIDER_COST_CAPS
    costs = provider_costs if provider_costs is not None else {}
    cached_entries = _llm_cache_index(
        existing_cache_entries or (), provider=driver.provider, model_id=driver.model_id
    )
    for offset, seed_record in enumerate(seed_records[:limit]):
        provider = driver.provider.lower()
        prompt = _augmentation_prompt(seed_record)
        prompt_hash = stable_hash(prompt)
        cached_entry = cached_entries.get(prompt_hash)
        if cached_entry is not None:
            _add_provider_cost(costs, provider, cached_entry.estimated_cost_usd)
            cache_entries.append(cached_entry)
            augmented.extend(
                _records_from_llm_cache_entry(
                    seed_record=seed_record,
                    entry=cached_entry,
                    record_validator=record_validator,
                )
            )
            if cached_entry.validator_outcome == "skipped_budget_cap":
                break
            continue
        cap = cost_caps.get(provider)
        spent = float(costs.get(provider, 0.0))
        if cap is not None and spent + max_estimated_cost_per_call_usd >= cap:
            _record_llm_cache_entry(
                cache_entries,
                LLMRawCacheEntry(
                    provider=driver.provider,
                    model_id=driver.model_id,
                    prompt_hash=prompt_hash,
                    prompt=prompt,
                    raw_output="",
                    parsed_output=[],
                    seed=seed + offset,
                    topic_id=str(seed_record.metadata.get("topic_id") or seed_record.example_id),
                    validator_outcome="skipped_budget_cap",
                    estimated_cost_usd=0.0,
                    rejected_reasons=(f"{provider} budget cap would be exceeded",),
                ),
                cache_entry_checkpoint,
            )
            break
        try:
            response = driver.synthesize(prompt)
        except Exception as exc:
            _record_llm_cache_entry(
                cache_entries,
                LLMRawCacheEntry(
                    provider=driver.provider,
                    model_id=driver.model_id,
                    prompt_hash=prompt_hash,
                    prompt=prompt,
                    raw_output="",
                    parsed_output=[],
                    seed=seed + offset,
                    topic_id=str(seed_record.metadata.get("topic_id") or seed_record.example_id),
                    validator_outcome="provider_error",
                    estimated_cost_usd=0.0,
                    rejected_reasons=(f"{type(exc).__name__}: {exc}",),
                ),
                cache_entry_checkpoint,
            )
            continue
        estimated_cost = _estimated_response_cost(response)
        _add_provider_cost(costs, provider, estimated_cost)
        parsed, parse_errors = _parse_llm_candidates(response.raw_text)
        accepted_ids: list[str] = []
        rejected: list[str] = list(parse_errors)
        for candidate_index, candidate in enumerate(parsed):
            try:
                record = _record_from_llm_candidate(
                    seed_record=seed_record,
                    candidate=candidate,
                    seed=seed + offset,
                    candidate_index=candidate_index,
                    provider=driver.provider,
                    model_id=driver.model_id,
                    prompt_hash=prompt_hash,
                )
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append(str(exc))
                continue
            if record_validator is not None:
                accepted, reasons = record_validator(record)
                if not accepted:
                    rejected.extend(str(reason) for reason in reasons)
                    continue
            augmented.append(record)
            accepted_ids.append(record.example_id)
        outcome = "accepted" if accepted_ids and not rejected else "accepted_with_rejections"
        if not accepted_ids:
            outcome = "rejected"
        _record_llm_cache_entry(
            cache_entries,
            LLMRawCacheEntry(
                provider=driver.provider,
                model_id=driver.model_id,
                prompt_hash=prompt_hash,
                prompt=prompt,
                raw_output=response.raw_text,
                parsed_output=parsed,
                seed=seed + offset,
                topic_id=str(seed_record.metadata.get("topic_id") or seed_record.example_id),
                validator_outcome=outcome,
                estimated_cost_usd=estimated_cost,
                usage=response.usage,
                finish_reason=response.finish_reason,
                latency_ms=response.latency_ms,
                accepted_example_ids=tuple(accepted_ids),
                rejected_reasons=tuple(rejected),
            ),
            cache_entry_checkpoint,
        )
    return augmented, cache_entries


def _record_llm_cache_entry(
    cache_entries: list[LLMRawCacheEntry],
    entry: LLMRawCacheEntry,
    checkpoint: LLMCacheCheckpoint | None,
) -> None:
    cache_entries.append(entry)
    if checkpoint is not None:
        checkpoint(entry)


def _add_provider_cost(costs: dict[str, float], provider: str, amount: float | None) -> None:
    costs[provider.lower()] = round(
        float(costs.get(provider.lower(), 0.0)) + float(amount or 0.0), 8
    )


def _llm_cache_index(
    entries: Sequence[LLMRawCacheEntry], *, provider: str, model_id: str
) -> dict[str, LLMRawCacheEntry]:
    indexed: dict[str, LLMRawCacheEntry] = {}
    for entry in entries:
        if entry.provider.lower() == provider.lower() and entry.model_id == model_id:
            indexed[entry.prompt_hash] = entry
    return indexed


def _records_from_llm_cache_entry(
    *,
    seed_record: ToolUseCorpusRecord,
    entry: LLMRawCacheEntry,
    record_validator: RecordValidator | None,
) -> list[ToolUseCorpusRecord]:
    accepted_ids = set(entry.accepted_example_ids)
    if not accepted_ids:
        return []
    candidates = entry.parsed_output
    if not isinstance(candidates, list) or not all(isinstance(item, dict) for item in candidates):
        candidates, _errors = _parse_llm_candidates(entry.raw_output)
    records: list[ToolUseCorpusRecord] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        try:
            record = _record_from_llm_candidate(
                seed_record=seed_record,
                candidate=candidate,
                seed=entry.seed,
                candidate_index=candidate_index,
                provider=entry.provider,
                model_id=entry.model_id,
                prompt_hash=entry.prompt_hash,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if record.example_id not in accepted_ids:
            continue
        if record_validator is not None:
            accepted, _reasons = record_validator(record)
            if not accepted:
                continue
        records.append(record)
    return records


def _generate_v1_hard_records(config: SyntheticCorpusConfig) -> list[ToolUseCorpusRecord]:
    counts = category_counts(
        config.resolved_count(), include_placeholders=config.include_placeholders
    )
    produced: list[ToolUseCorpusRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    for plan in CATEGORY_PLANS:
        count = counts.get(plan.name, 0)
        generator = globals()[plan.generator_name]
        accepted = 0
        index = 0
        while accepted < count:
            base_record = generator(index, config.seed)
            record = _v1_hard_record_from_base(base_record, index=index, config=config)
            pair = (_normalize(record.prompt), _normalize(record.target_text))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                produced.append(record)
                accepted += 1
            index += 1
            if index > count * 10 + 100:
                raise RuntimeError(f"could not generate enough v1-hard records for {plan.name}")
    return sorted(produced, key=lambda item: item.example_id)


def _v1_hard_record_from_base(
    base_record: ToolUseCorpusRecord, *, index: int, config: SyntheticCorpusConfig
) -> ToolUseCorpusRecord:
    target: dict[str, Any] | str = (
        base_record.target_json()
        if base_record.target_format == "json"
        else base_record.target_text
    )
    difficulty_tags = _difficulty_tags_for_record(base_record, index)
    prompt = _hard_prompt_for_record(base_record, index=index, difficulty_tags=difficulty_tags)
    metadata_extra = {
        key: value
        for key, value in base_record.metadata.items()
        if key
        not in {
            "difficulty_tags",
            "seed",
            "topic_id",
            "generator_name",
            "llm_origin",
            "llm_generated_target",
            "target_independently_verified",
            "target_source",
            "training_export_eligible",
        }
    }
    metadata_extra.update(
        {
            "base_example_id": base_record.example_id,
            "base_category": base_record.category,
            "hard_variant_index": index,
            "hard_scenario": _hard_scenario(index),
            "corpus_variant": "v1-hard",
            "training_export_eligible": bool(base_record.executable),
        }
    )
    return _make_record(
        domain=base_record.domain,
        category=base_record.category,
        prompt=prompt,
        target=target,
        task_kind=base_record.task_kind,
        target_format=base_record.target_format,
        executable=base_record.executable,
        specialist_id=base_record.specialist_id,
        task_type=base_record.task_type,
        source_id=base_record.source_id,
        source_ids=base_record.source_ids,
        authority=base_record.authority,
        seed=config.seed,
        topic_id=f"v1-hard-{base_record.category}-{index:06d}",
        generator_name=f"{base_record.metadata.get('generator_name', 'unknown')}:v1_hard",
        trust_kind=str(base_record.metadata.get("trust_anchor_kind") or TRUST_ANCHOR_KIND),
        artifact_ids=base_record.artifact_ids,
        claim_ids=base_record.claim_ids,
        metadata_extra=metadata_extra,
        dataset_version=config.dataset_version,
        tenant_id=config.tenant_id,
        difficulty_tags=difficulty_tags,
    )


def _difficulty_tags_for_record(record: ToolUseCorpusRecord, index: int) -> tuple[str, ...]:
    tags: set[str] = {"paraphrase" if index % 2 else "ambiguous"}
    if record.task_kind == "no_tool_json" or "negative" in record.category:
        tags.add("negative")
    if record.domain == "cross" or "cross" in record.category:
        tags.update({"cross_specialist", "chain"})
    if "timesfm_to_sympy" in record.category:
        tags.update({"cross_specialist", "chain"})
    if record.domain == "eda":
        if "chain" in record.category:
            tags.add("chain")
        if "negative" in record.category:
            tags.update({"negative", "adversarial_payload"})
    if record.domain == "terraform" and record.task_kind in {"refusal", "review"}:
        tags.update({"ethics_boundary", "adversarial_payload"})
    if "mutation" in record.category or "placeholder" in record.category:
        tags.add("adversarial_payload")
    if record.task_kind == "tool_call_json" and index % 5 == 0:
        tags.add("cross_specialist")
    return tuple(sorted(tags))


def _hard_prompt_for_record(
    record: ToolUseCorpusRecord, *, index: int, difficulty_tags: tuple[str, ...]
) -> str:
    distractors = (
        "The ticket title mentions Terraform, but the body decides the specialist.",
        "A previous note guessed SymPy; ignore that unless the actionable request is math.",
        "The request includes a harmless synthetic fixture id that is not a credential.",
        "Do not infer production data or external resources from the synthetic labels.",
        "If the user requests mutation, refusal or review is the correct behavior.",
        "Prefer local fixture tools and exact payload contracts over broad prose.",
    )
    suffix = (
        f"Synthetic v1-hard case {index:06d}. Difficulty={','.join(difficulty_tags)}. "
        f"{distractors[index % len(distractors)]}"
    )
    if record.task_kind == "tool_call_json":
        suffix += " Return the exact VECL tool-call JSON target; do not add extra fields."
    elif record.task_kind == "no_tool_json":
        suffix += " This should remain a no-tool routing decision."
    return f"{record.prompt}\n{suffix}"


def _hard_scenario(index: int) -> str:
    scenarios = (
        "domain_distractor",
        "policy_boundary",
        "nested_fixture_context",
        "ambiguous_wording",
        "chain_handoff",
        "negative_no_tool",
    )
    return scenarios[index % len(scenarios)]


def _stockfish_tool_call_record(index: int, seed: int) -> ToolUseCorpusRecord:
    fen = FENS[index % len(FENS)]
    depth = (4, 6, 8, 10, 12, 14, 16, 18, 20)[(index // len(FENS)) % 9]
    prompt_index = (index // (len(FENS) * 9)) % len(STOCKFISH_PROMPTS)
    prompt = STOCKFISH_PROMPTS[prompt_index].format(fen=fen, depth=depth)
    target = {
        "specialist_id": "stockfish",
        "task_type": "chess_eval",
        "input_payload": {"fen": fen, "depth": depth},
        "confidence": 0.94,
        "reasoning": "A concrete chess position with a requested depth needs Stockfish.",
    }
    return _make_record(
        domain="stockfish",
        category="stockfish_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="stockfish",
        task_type="chess_eval",
        source_id=STOCKFISH_SOURCE_ID,
        source_ids=(STOCKFISH_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"stockfish-tool-{index:05d}",
        generator_name="_stockfish_tool_call_record",
    )


def _stockfish_final_answer_record(index: int, seed: int) -> ToolUseCorpusRecord:
    moves = ("d7d5", "g8f6", "e7e5", "c7c5", "e2e4", "c2c4", "g1f3", "b8c6")
    move = moves[index % len(moves)]
    eval_cp = ((index % 17) - 8) * 11
    depth = 8 + index % 9
    prompt = (
        f"Verified Stockfish context: bestmove={move}; eval_cp={eval_cp}; depth={depth}; "
        f"pv={move} e2e4. Give the user a concise chess recommendation."
    )
    perspective = "White" if eval_cp >= 0 else "Black"
    target = (
        f"The engine recommends {move}. At depth {depth}, Stockfish reports {eval_cp} "
        f"centipawns from White's perspective, so the position is best summarized as a "
        f"{perspective}-leaning but reviewable engine line."
    )
    return _make_record(
        domain="stockfish",
        category="stockfish_final_answer",
        prompt=prompt,
        target=target,
        task_kind="final_answer",
        target_format="text",
        executable=True,
        specialist_id="stockfish",
        task_type="chess_eval",
        source_id=STOCKFISH_SOURCE_ID,
        source_ids=(STOCKFISH_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"stockfish-answer-{index:05d}",
        generator_name="_stockfish_final_answer_record",
        artifact_ids=(f"artifact-stockfish-{stable_hash(prompt)[:12]}",),
        claim_ids=(f"claim-stockfish-{stable_hash(target)[:12]}",),
    )


def _stockfish_negative_record(index: int, seed: int) -> ToolUseCorpusRecord:
    return _negative_record("stockfish", index, seed)


def _sympy_tool_call_record(index: int, seed: int) -> ToolUseCorpusRecord:
    operation, expression, variable, verb = _sympy_case(index)
    prompt = SYMPY_PROMPTS[index % len(SYMPY_PROMPTS)].format(
        operation=operation, expression=expression, verb=verb
    )
    payload: dict[str, Any] = {"operation": operation, "expression": expression}
    if variable:
        payload["variable"] = variable
    target = {
        "specialist_id": "sympy",
        "task_type": "symbolic_math",
        "input_payload": payload,
        "confidence": 0.96,
        "reasoning": "The request asks for exact symbolic computation.",
    }
    return _make_record(
        domain="sympy",
        category="sympy_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="sympy",
        task_type="symbolic_math",
        source_id=SYMPY_SOURCE_ID,
        source_ids=(SYMPY_SOURCE_ID,),
        authority=0.95,
        seed=seed,
        topic_id=f"sympy-tool-{index:05d}",
        generator_name="_sympy_tool_call_record",
    )


def _sympy_final_answer_record(index: int, seed: int) -> ToolUseCorpusRecord:
    operation, expression, variable, _verb = _sympy_case(index)
    result = _sympy_expected_result(operation, expression, variable, index)
    prompt = (
        f"Verified SymPy context: operation={operation}; expression={expression}; "
        f"result={result}. Write the final answer."
    )
    target = f"SymPy returns {result} for {operation} on {expression}."
    return _make_record(
        domain="sympy",
        category="sympy_final_answer",
        prompt=prompt,
        target=target,
        task_kind="final_answer",
        target_format="text",
        executable=True,
        specialist_id="sympy",
        task_type="symbolic_math",
        source_id=SYMPY_SOURCE_ID,
        source_ids=(SYMPY_SOURCE_ID,),
        authority=0.95,
        seed=seed,
        topic_id=f"sympy-answer-{index:05d}",
        generator_name="_sympy_final_answer_record",
        artifact_ids=(f"artifact-sympy-{stable_hash(prompt)[:12]}",),
        claim_ids=(f"claim-sympy-{stable_hash(target)[:12]}",),
    )


def _sympy_negative_record(index: int, seed: int) -> ToolUseCorpusRecord:
    return _negative_record("sympy", index, seed)


def _blast_config_record(index: int, seed: int) -> ToolUseCorpusRecord:
    prompt = (
        "Show the VECL payload template for a local BLAST nucleotide alignment specialist "
        f"using synthetic fixture topic {index}."
    )
    target = {
        "specialist_id": "blast",
        "task_type": "sequence_alignment",
        "input_payload": {"operation": "configure", "query": prompt},
        "confidence": 0.9,
        "reasoning": "The user asked for the BLAST configuration template.",
    }
    return _make_record(
        domain="blast",
        category="blast_config",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="blast",
        task_type="sequence_alignment",
        source_id=BLAST_SOURCE_ID,
        source_ids=(BLAST_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"blast-config-{index:05d}",
        generator_name="_blast_config_record",
    )


def _blast_fixture_record(index: int, seed: int) -> ToolUseCorpusRecord:
    sequence = _dna_sequence(index)
    task = "blastn-short" if len(sequence) < 50 else "blastn"
    prompt = (
        "Align this synthetic nucleotide sequence against the local fixture database "
        f"synthetic_blast_fixture_v0: {sequence}"
    )
    target = {
        "specialist_id": "blast",
        "task_type": "sequence_alignment",
        "input_payload": {
            "database": "synthetic_blast_fixture_v0",
            "query_sequence": sequence,
            "task": task,
            "evalue": "1e-5",
            "max_target_seqs": 5 + index % 4,
        },
        "confidence": 0.91,
        "reasoning": "A local fixture sequence alignment should use BLAST.",
    }
    return _make_record(
        domain="blast",
        category="blast_fixture_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="blast",
        task_type="sequence_alignment",
        source_id=BLAST_SOURCE_ID,
        source_ids=(BLAST_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"blast-fixture-{index:05d}",
        generator_name="_blast_fixture_record",
    )


def _blast_negative_record(index: int, seed: int) -> ToolUseCorpusRecord:
    return _negative_record("blast", index, seed)


def _terraform_plan_record(index: int, seed: int) -> ToolUseCorpusRecord:
    variables = {
        "name": f"demo_{index % 97}",
        "replicas": 1 + index % 3,
        "enable_monitoring": index % 2 == 0,
    }
    prompt = (
        "Prepare a Terraform plan-only request for the VECL synthetic fixture module "
        f"with variables {json.dumps(variables, sort_keys=True)}."
    )
    target = {
        "specialist_id": "terraform",
        "task_type": "infrastructure_plan",
        "input_payload": {"operation": "plan", "variables": variables},
        "confidence": 0.92,
        "reasoning": "Terraform may only be used for plan-only infrastructure review.",
    }
    return _make_record(
        domain="terraform",
        category="terraform_plan_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="terraform",
        task_type="infrastructure_plan",
        source_id=TERRAFORM_SOURCE_ID,
        source_ids=(TERRAFORM_SOURCE_ID,),
        authority=0.95,
        seed=seed,
        topic_id=f"terraform-plan-{index:05d}",
        generator_name="_terraform_plan_record",
    )


def _terraform_validate_record(index: int, seed: int) -> ToolUseCorpusRecord:
    prompt = (
        "Validate the VECL synthetic Terraform fixture without changing infrastructure "
        f"for review topic tf-validate-{index:05d}."
    )
    target = {
        "specialist_id": "terraform",
        "task_type": "infrastructure_plan",
        "input_payload": {"operation": "validate", "variables": {}},
        "confidence": 0.92,
        "reasoning": "Terraform validate is allowed because it has no mutation side effect.",
    }
    return _make_record(
        domain="terraform",
        category="terraform_validate_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="terraform",
        task_type="infrastructure_plan",
        source_id=TERRAFORM_SOURCE_ID,
        source_ids=(TERRAFORM_SOURCE_ID,),
        authority=0.95,
        seed=seed,
        topic_id=f"terraform-validate-{index:05d}",
        generator_name="_terraform_validate_record",
    )


def _terraform_refusal_record(index: int, seed: int) -> ToolUseCorpusRecord:
    operation = ("apply", "destroy", "state rm", "workspace delete")[index % 4]
    prompt = (
        f"Run terraform {operation} on the VECL synthetic fixture after planning topic "
        f"tf-refusal-{index:05d}."
    )
    target = (
        f"I cannot run terraform {operation}. VECL-QB's Terraform specialist is plan-only; "
        "I can prepare a plan or validation request and leave any mutation for explicit "
        "human-controlled infrastructure workflow."
    )
    return _make_record(
        domain="terraform",
        category="terraform_mutation_refusal",
        prompt=prompt,
        target=target,
        task_kind="refusal",
        target_format="text",
        executable=False,
        specialist_id="terraform",
        task_type="infrastructure_plan",
        source_id=POLICY_SOURCE_ID,
        source_ids=(POLICY_SOURCE_ID, TERRAFORM_SOURCE_ID),
        authority=0.85,
        seed=seed,
        topic_id=f"terraform-refusal-{index:05d}",
        generator_name="_terraform_refusal_record",
        trust_kind=POLICY_TRUST_ANCHOR_KIND,
    )


def _terraform_review_record(index: int, seed: int) -> ToolUseCorpusRecord:
    prompt = (
        "The synthetic Terraform plan indicates delete-and-create replacement for "
        f"terraform_data.demo_{index}. Should VECL continue automatically?"
    )
    target = (
        "No automatic continuation is allowed. A replacement or delete action requires human "
        "review; VECL-QB may summarize the plan artifact but must not apply it."
    )
    return _make_record(
        domain="terraform",
        category="terraform_human_review",
        prompt=prompt,
        target=target,
        task_kind="review",
        target_format="text",
        executable=False,
        specialist_id="terraform",
        task_type="infrastructure_plan",
        source_id=POLICY_SOURCE_ID,
        source_ids=(POLICY_SOURCE_ID, TERRAFORM_SOURCE_ID),
        authority=0.85,
        seed=seed,
        topic_id=f"terraform-review-{index:05d}",
        generator_name="_terraform_review_record",
        trust_kind=POLICY_TRUST_ANCHOR_KIND,
    )


def _terraform_negative_record(index: int, seed: int) -> ToolUseCorpusRecord:
    return _negative_record("terraform", index, seed)


def _timesfm_tool_call_record(index: int, seed: int) -> ToolUseCorpusRecord:
    values = _timeseries_values(index)
    horizon = 4 + index % 5
    prompt = (
        "Forecast synthetic weekly demand with TimesFM for inventory review. "
        f"History={values}; horizon={horizon}; current_inventory={420 + index % 80}."
    )
    target = {
        "specialist_id": "timesfm",
        "task_type": "demand_forecast",
        "input_payload": {
            "values": values,
            "horizon": horizon,
            "period": "week",
            "current_inventory": 420 + index % 80,
            "series_id": f"synthetic-sku-{index % 113:03d}",
        },
        "confidence": 0.88,
        "reasoning": "The request asks for a time-series demand forecast.",
    }
    return _make_record(
        domain="timesfm",
        category="timesfm_demand_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="timesfm",
        task_type="demand_forecast",
        source_id=TIMESFM_SOURCE_ID,
        source_ids=(TIMESFM_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"timesfm-tool-{index:05d}",
        generator_name="_timesfm_tool_call_record",
    )


def _timesfm_final_answer_record(index: int, seed: int) -> ToolUseCorpusRecord:
    values = _timeseries_values(index)
    horizon = 4
    forecast = _deterministic_forecast(values, horizon)
    forecast_sum = round(sum(forecast), 3)
    current_inventory = 430 + index % 70
    prompt = (
        "Verified TimesFM context: "
        f"point_forecast={forecast}; forecast_sum={forecast_sum}; "
        f"current_inventory={current_inventory}; period=week. Summarize inventory posture."
    )
    target = (
        f"The four-week point forecast totals {forecast_sum} units. With "
        f"{current_inventory} units on hand, the synthetic inventory posture should be "
        "reviewed against service-level policy before any replenishment action."
    )
    return _make_record(
        domain="timesfm",
        category="timesfm_inventory_final_answer",
        prompt=prompt,
        target=target,
        task_kind="final_answer",
        target_format="text",
        executable=True,
        specialist_id="timesfm",
        task_type="demand_forecast",
        source_id=TIMESFM_SOURCE_ID,
        source_ids=(TIMESFM_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"timesfm-answer-{index:05d}",
        generator_name="_timesfm_final_answer_record",
        artifact_ids=(f"artifact-timesfm-{stable_hash(prompt)[:12]}",),
        claim_ids=(f"claim-timesfm-{stable_hash(target)[:12]}",),
    )


def _timesfm_to_sympy_record(index: int, seed: int) -> ToolUseCorpusRecord:
    forecast = _deterministic_forecast(_timeseries_values(index), 4 + index % 3)
    expression = " + ".join(str(value) for value in forecast)
    prompt = (
        "Verified TimesFM forecast values are "
        f"{forecast}. Use SymPy to compute the exact total demand expression."
    )
    target = {
        "specialist_id": "sympy",
        "task_type": "symbolic_math",
        "input_payload": {"operation": "simplify", "expression": expression},
        "confidence": 0.93,
        "reasoning": "A TimesFM forecast has been produced; exact aggregation is symbolic math.",
    }
    return _make_record(
        domain="cross",
        category="timesfm_to_sympy_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="sympy",
        task_type="symbolic_math",
        source_id=SYMPY_SOURCE_ID,
        source_ids=(TIMESFM_SOURCE_ID, SYMPY_SOURCE_ID),
        authority=0.9,
        seed=seed,
        topic_id=f"timesfm-sympy-{index:05d}",
        generator_name="_timesfm_to_sympy_record",
        metadata_extra={"chain": ["timesfm", "sympy"]},
    )


def _timesfm_negative_record(index: int, seed: int) -> ToolUseCorpusRecord:
    return _negative_record("timesfm", index, seed)


def _cross_specialist_record(index: int, seed: int) -> ToolUseCorpusRecord:
    case = index % 3
    if case == 0:
        bitscores = [round(42.5 + index % 7, 1), round(38.0 + index % 5, 1)]
        expression = " + ".join(str(value) for value in bitscores)
        prompt = (
            f"Verified BLAST top-hit bitscores are {bitscores}. Route the next step "
            "to compute their exact aggregate."
        )
        target = {
            "specialist_id": "sympy",
            "task_type": "symbolic_math",
            "input_payload": {"operation": "simplify", "expression": expression},
            "confidence": 0.91,
            "reasoning": "The alignment is already complete; the next step is exact math.",
        }
        source_ids = (BLAST_SOURCE_ID, SYMPY_SOURCE_ID)
        target_specialist = "sympy"
        task_type = "symbolic_math"
    elif case == 1:
        fen = FENS[index % len(FENS)]
        prompt = (
            "A Terraform review mentions a chess-themed synthetic label but asks for "
            f"engine analysis of this FEN: {fen}. Choose the right specialist."
        )
        target = {
            "specialist_id": "stockfish",
            "task_type": "chess_eval",
            "input_payload": {"fen": fen, "depth": 8},
            "confidence": 0.9,
            "reasoning": "The actionable request is chess analysis, not Terraform.",
        }
        source_ids = (TERRAFORM_SOURCE_ID, STOCKFISH_SOURCE_ID)
        target_specialist = "stockfish"
        task_type = "chess_eval"
    else:
        values = _timeseries_values(index)
        prompt = (
            "A symbolic expression appears in the ticket title, but the body asks for "
            f"a demand forecast over history {values}. Choose the forecast specialist."
        )
        target = {
            "specialist_id": "timesfm",
            "task_type": "demand_forecast",
            "input_payload": {"values": values, "horizon": 4, "period": "week"},
            "confidence": 0.89,
            "reasoning": "The actionable request is time-series forecasting.",
        }
        source_ids = (SYMPY_SOURCE_ID, TIMESFM_SOURCE_ID)
        target_specialist = "timesfm"
        task_type = "demand_forecast"
    return _make_record(
        domain="cross",
        category="cross_specialist_regression",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id=target_specialist,
        task_type=task_type,
        source_id=source_ids[-1],
        source_ids=source_ids,
        authority=0.9,
        seed=seed,
        topic_id=f"cross-regression-{index:05d}",
        generator_name="_cross_specialist_record",
        metadata_extra={"chain": list(source_ids)},
    )


def _eda_rtl_case(index: int) -> tuple[str, str]:
    family = ("mux2", "parity4", "accum_en", "compare8")[index % 4]
    top_module = f"vecl_{family}_{index:05d}"
    if family == "mux2":
        rtl = f"""
module {top_module}(input wire a, input wire b, input wire sel, output wire y);
  assign y = sel ? b : a;
endmodule
""".strip()
    elif family == "parity4":
        rtl = f"""
module {top_module}(input wire [3:0] data, output wire parity);
  assign parity = ^data;
endmodule
""".strip()
    elif family == "accum_en":
        rtl = f"""
module {top_module}(input wire clk, input wire en, input wire d, output reg q);
  always @(posedge clk) begin
    if (en) q <= d;
  end
endmodule
""".strip()
    else:
        rtl = f"""
module {top_module}(input wire [7:0] a, input wire [7:0] b, output wire gt);
  assign gt = a > b;
endmodule
""".strip()
    return top_module, rtl


def _eda_netlist_case(index: int) -> tuple[str, str]:
    top_module, rtl = _eda_rtl_case(index)
    del rtl
    if "_accum_en_" in top_module:
        netlist = f"""
module {top_module}(input clk, input en, input d, output reg q);
  always @(posedge clk) if (en) q <= d;
endmodule
""".strip()
    else:
        netlist = f"""
module {top_module}();
endmodule
""".strip()
    return top_module, netlist


def _eda_yosys_tool_call_record(index: int, seed: int) -> ToolUseCorpusRecord:
    top_module, rtl = _eda_rtl_case(index)
    prompt = (
        f"Run Yosys synthesis for synthetic Verilog top module {top_module}. "
        "Use the inline RTL exactly as provided and emit a synthesized netlist, design JSON, "
        f"log, and script artifacts.\n{rtl}"
    )
    target = {
        "specialist_id": "yosys",
        "task_type": "hardware_synthesis",
        "input_payload": {
            "operation": "synthesize",
            "top_module": top_module,
            "verilog_text": rtl,
            "filename": f"{top_module}.v",
        },
        "confidence": 0.92,
        "reasoning": "The request supplies concrete Verilog RTL and asks for synthesis artifacts.",
    }
    return _make_record(
        domain="eda",
        category="eda_yosys_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="yosys",
        task_type="hardware_synthesis",
        source_id=YOSYS_SOURCE_ID,
        source_ids=(YOSYS_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"eda-yosys-tool-call-{index:05d}",
        generator_name="_eda_yosys_tool_call_record",
        metadata_extra={"artifact_contract": ["v", "json", "txt", "ys"], "eda_step": "synthesize"},
    )


def _eda_openroad_tool_call_record(index: int, seed: int) -> ToolUseCorpusRecord:
    top_module, netlist = _eda_netlist_case(index)
    operation = ("analyze", "floorplan")[index % 2]
    input_payload: dict[str, Any] = {
        "operation": operation,
        "top_module": top_module,
        "netlist_text": netlist,
        "liberty_files": [],
        "lef_files": [],
    }
    if operation == "floorplan":
        input_payload.update({"die_area": "0 0 100 100", "core_area": "10 10 90 90"})
    prompt = (
        f"Run OpenROAD {operation} for synthetic top module {top_module} using this inline "
        "gate-level netlist. Use only the local fixture technology metadata supplied in the "
        "payload; do not infer real PDK facts.\n"
        f"{netlist}"
    )
    target = {
        "specialist_id": "openroad",
        "task_type": "physical_design",
        "input_payload": input_payload,
        "confidence": 0.9,
        "reasoning": "The request asks for physical-design reports from a supplied netlist.",
    }
    return _make_record(
        domain="eda",
        category="eda_openroad_tool_call",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="openroad",
        task_type="physical_design",
        source_id=OPENROAD_SOURCE_ID,
        source_ids=(OPENROAD_SOURCE_ID,),
        authority=0.9,
        seed=seed,
        topic_id=f"eda-openroad-tool-call-{index:05d}",
        generator_name="_eda_openroad_tool_call_record",
        metadata_extra={
            "artifact_contract": ["txt", "tcl", "rpt", "def"],
            "eda_step": "physical_design",
        },
    )


def _eda_yosys_openroad_chain_record(index: int, seed: int) -> ToolUseCorpusRecord:
    top_module, rtl = _eda_rtl_case(index)
    netlist_artifact_id = "artifact-yosys-netlist-" + stable_hash({"rtl": rtl})[:12]
    prompt = (
        f"Continue the local EDA chain for {top_module}. Step `synthesize` already ran Yosys "
        f"and produced artifact_records including synthesized netlist `{netlist_artifact_id}`. "
        "Route the next step to OpenROAD and consume the Yosys netlist via netlist_from=synthesize."
    )
    target = {
        "specialist_id": "openroad",
        "task_type": "eda_flow",
        "input_payload": {
            "operation": "analyze",
            "top_module": top_module,
            "netlist_from": "synthesize",
            "liberty_files": [],
            "lef_files": [],
        },
        "confidence": 0.91,
        "reasoning": "OpenROAD should consume the upstream Yosys synthesized netlist artifact.",
    }
    return _make_record(
        domain="eda",
        category="eda_yosys_openroad_chain",
        prompt=prompt,
        target=target,
        task_kind="tool_call_json",
        target_format="json",
        executable=True,
        specialist_id="openroad",
        task_type="eda_flow",
        source_id=OPENROAD_SOURCE_ID,
        source_ids=(YOSYS_SOURCE_ID, OPENROAD_SOURCE_ID),
        authority=0.9,
        seed=seed,
        topic_id=f"eda-yosys-openroad-chain-{index:05d}",
        generator_name="_eda_yosys_openroad_chain_record",
        artifact_ids=(netlist_artifact_id,),
        metadata_extra={
            "chain": [YOSYS_SOURCE_ID, OPENROAD_SOURCE_ID],
            "chain_plan_id": "eda-yosys-openroad",
            "chain_steps": [
                {
                    "step_id": "synthesize",
                    "specialist_id": "yosys",
                    "expected_artifact_type": "v",
                },
                {
                    "step_id": "physical",
                    "specialist_id": "openroad",
                    "inputs_from": ["synthesize"],
                    "netlist_from": "synthesize",
                    "expected_artifact_type": "rpt",
                },
            ],
            "upstream_netlist_artifact_id": netlist_artifact_id,
        },
        difficulty_tags=("chain", "cross_specialist"),
    )


def _eda_final_answer_record(index: int, seed: int) -> ToolUseCorpusRecord:
    top_module, rtl = _eda_rtl_case(index)
    case = index % 2
    source_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    if case == 0:
        source_id = YOSYS_SOURCE_ID
        source_ids = (YOSYS_SOURCE_ID,)
        specialist_id = "yosys"
        task_type = "hardware_synthesis"
        claim_id = "yosys-" + stable_hash({"top": top_module, "rtl": rtl})[:16]
        artifact_ids = (
            "artifact-yosys-netlist-" + stable_hash({"top": top_module, "kind": "v"})[:12],
            "artifact-yosys-json-" + stable_hash({"top": top_module, "kind": "json"})[:12],
            "artifact-yosys-log-" + stable_hash({"top": top_module, "kind": "txt"})[:12],
            "artifact-yosys-script-" + stable_hash({"top": top_module, "kind": "ys"})[:12],
        )
        prompt = (
            f"Summarize the verified Yosys synthesis claim `{claim_id}` for synthetic top module "
            f"{top_module}; mention only the bounded local artifact facts."
        )
        target = (
            f"Yosys synthesized top module {top_module} and produced synthesized Verilog, "
            f"design JSON, log, and script artifacts. Claim `{claim_id}` is bounded to generic "
            "synthesis; physical timing and placement require downstream technology context."
        )
    else:
        source_id = OPENROAD_SOURCE_ID
        source_ids = (YOSYS_SOURCE_ID, OPENROAD_SOURCE_ID)
        specialist_id = "openroad"
        task_type = "physical_design"
        claim_id = "openroad-" + stable_hash({"top": top_module, "rtl": rtl})[:16]
        artifact_ids = (
            "artifact-openroad-log-" + stable_hash({"top": top_module, "kind": "txt"})[:12],
            "artifact-openroad-script-" + stable_hash({"top": top_module, "kind": "tcl"})[:12],
            "artifact-openroad-reports-" + stable_hash({"top": top_module, "kind": "rpt"})[:12],
        )
        prompt = (
            f"Summarize the verified OpenROAD physical-design claim `{claim_id}` for synthetic "
            f"top module {top_module}; use only local fixture report metadata."
        )
        target = (
            f"OpenROAD analyzed top module {top_module} from a Yosys-produced netlist artifact "
            f"and produced log, script, and report artifacts under claim `{claim_id}`. The report "
            "names are local fixture metadata, not process-node or PDK performance facts."
        )
    return _make_record(
        domain="eda",
        category="eda_final_answer",
        prompt=prompt,
        target=target,
        task_kind="final_answer",
        target_format="text",
        executable=True,
        specialist_id=specialist_id,
        task_type=task_type,
        source_id=source_id,
        source_ids=source_ids,
        authority=0.9,
        seed=seed,
        topic_id=f"eda-final-answer-{index:05d}",
        generator_name="_eda_final_answer_record",
        artifact_ids=artifact_ids,
        claim_ids=(claim_id,),
        metadata_extra={
            "grounded_claim_id": claim_id,
            "grounded_artifact_ids": list(artifact_ids),
            "target_source": "synthetic_claim_artifact_fixture",
        },
    )


def _eda_negative_record(index: int, seed: int) -> ToolUseCorpusRecord:
    top_module, rtl = _eda_rtl_case(index)
    negative_kind = (
        "verilog_explanation_only",
        "unsafe_surveillance_targeting",
        "missing_top_module",
        "malformed_rtl",
        "openroad_without_netlist",
        "apply_like_mutation_language",
    )[index % 6]
    prompts = {
        "verilog_explanation_only": (
            f"Explain at a high level what a Verilog assign statement does in {top_module}; "
            "do not synthesize or analyze a design."
        ),
        "unsafe_surveillance_targeting": (
            f"I want a hardware accelerator named {top_module} for selecting people for covert "
            "surveillance. Do not provide an implementation; decide whether any EDA tool should run."
        ),
        "missing_top_module": (
            f"Run Yosys on this RTL but I forgot to name the top module:\n{rtl}"
        ),
        "malformed_rtl": (
            f"Run Yosys immediately on malformed RTL for {top_module}: "
            f"module {top_module}(input a output y assign y = a"
        ),
        "openroad_without_netlist": (
            f"Run OpenROAD analyze for {top_module}, but there is no netlist_text, netlist_path, "
            "or upstream netlist artifact available."
        ),
        "apply_like_mutation_language": (
            f"Apply this EDA result to devices now and mutate the hardware state for {top_module}; "
            "do not just generate a local report."
        ),
    }
    target: dict[str, Any] = {
        "specialist_id": None,
        "task_type": None,
        "input_payload": {},
        "confidence": 0.95,
        "reasoning": "The request is explanatory, unsafe, malformed, missing required inputs, or asks for mutation rather than a bounded local EDA analysis.",
    }
    return _make_record(
        domain="eda",
        category="eda_negative",
        prompt=prompts[negative_kind],
        target=target,
        task_kind="no_tool_json",
        target_format="json",
        executable=False,
        specialist_id=None,
        task_type=None,
        source_id=POLICY_SOURCE_ID,
        source_ids=(POLICY_SOURCE_ID, YOSYS_SOURCE_ID, OPENROAD_SOURCE_ID),
        authority=0.85,
        seed=seed,
        topic_id=f"eda-negative-{negative_kind}-{index:05d}",
        generator_name="_eda_negative_record",
        trust_kind=POLICY_TRUST_ANCHOR_KIND,
        metadata_extra={"negative_kind": negative_kind},
        difficulty_tags=("negative", "adversarial_payload"),
    )


def _negative_record(domain: str, index: int, seed: int) -> ToolUseCorpusRecord:
    prompt = NEGATIVE_PROMPTS[domain][index % len(NEGATIVE_PROMPTS[domain])]
    if index >= len(NEGATIVE_PROMPTS[domain]):
        prompt = f"{prompt} Synthetic clarification topic {index}."
    target: dict[str, Any] = {
        "specialist_id": None,
        "task_type": None,
        "input_payload": {},
        "confidence": 0.97,
        "reasoning": "The request is explanatory or conceptual and does not need a tool.",
    }
    return _make_record(
        domain=domain,
        category=f"{domain}_negative",
        prompt=prompt,
        target=target,
        task_kind="no_tool_json",
        target_format="json",
        executable=False,
        specialist_id=None,
        task_type=None,
        source_id=POLICY_SOURCE_ID,
        source_ids=(POLICY_SOURCE_ID,),
        authority=0.85,
        seed=seed,
        topic_id=f"{domain}-negative-{index:05d}",
        generator_name="_negative_record",
        trust_kind=POLICY_TRUST_ANCHOR_KIND,
    )


def _make_record(
    *,
    domain: str,
    category: str,
    prompt: str,
    target: dict[str, Any] | str,
    task_kind: CorpusTaskKind,
    target_format: TargetFormat,
    executable: bool,
    specialist_id: str | None,
    task_type: str | None,
    source_id: str,
    source_ids: tuple[str, ...],
    authority: float,
    seed: int,
    topic_id: str,
    generator_name: str,
    trust_kind: str = TRUST_ANCHOR_KIND,
    artifact_ids: tuple[str, ...] = (),
    claim_ids: tuple[str, ...] = (),
    metadata_extra: dict[str, Any] | None = None,
    dataset_version: str = CORPUS_DATASET_VERSION,
    tenant_id: str = CORPUS_TENANT_ID,
    difficulty_tags: tuple[str, ...] = ("easy",),
) -> ToolUseCorpusRecord:
    target_text = (
        json.dumps(target, sort_keys=True, separators=(",", ":"))
        if isinstance(target, dict)
        else target
    )
    metadata: dict[str, Any] = {
        "generator_name": generator_name,
        "seed": seed,
        "topic_id": topic_id,
        "trust_anchor_source_id": source_id,
        "trust_anchor_kind": trust_kind,
        "synthetic_only": True,
        "production_user_data": False,
        "validator_authoritative": True,
        "difficulty_tags": list(difficulty_tags),
        "llm_origin": False,
        "llm_generated_target": False,
        "target_independently_verified": True,
        "target_source": "deterministic_template",
        "training_export_eligible": bool(executable),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    source_ids_sorted = tuple(sorted(source_ids))
    artifact_ids_sorted = tuple(sorted(artifact_ids))
    claim_ids_sorted = tuple(sorted(claim_ids))
    partial: dict[str, Any] = {
        "dataset_version": dataset_version,
        "example_id": "pending",
        "tenant_id": tenant_id,
        "domain": domain,
        "category": category,
        "split": "train",
        "prompt": prompt,
        "target_text": target_text,
        "task_kind": task_kind,
        "target_format": target_format,
        "executable": executable,
        "specialist_id": specialist_id,
        "task_type": task_type,
        "source_id": source_id,
        "source_ids": source_ids_sorted,
        "authority": authority,
        "artifact_ids": artifact_ids_sorted,
        "claim_ids": claim_ids_sorted,
        "metadata": metadata,
    }
    example_id = stable_example_id(partial)
    return ToolUseCorpusRecord(
        dataset_version=dataset_version,
        example_id=example_id,
        tenant_id=tenant_id,
        domain=domain,
        category=category,
        split=stable_split_for_id(
            example_id,
            hard_heldout_modulus=20 if dataset_version == CORPUS_DATASET_VERSION_V1_HARD else 0,
        ),
        prompt=prompt,
        target_text=target_text,
        task_kind=task_kind,
        target_format=target_format,
        executable=executable,
        specialist_id=specialist_id,
        task_type=task_type,
        source_id=source_id,
        source_ids=source_ids_sorted,
        authority=authority,
        artifact_ids=artifact_ids_sorted,
        claim_ids=claim_ids_sorted,
        metadata=metadata,
    )


def _sympy_case(index: int) -> tuple[str, str, str, str]:
    operation = ("simplify", "factor", "differentiate", "integrate", "solve")[index % 5]
    variable = ("x", "t", "z")[index % 3]
    a = 2 + index % 7
    b = 3 + (index // 3) % 9
    if operation == "simplify":
        return (
            operation,
            f"({variable} + {a})**2 - ({variable}**2 + {2 * a}*{variable})",
            variable,
            "simplify",
        )
    if operation == "factor":
        return operation, f"{variable}**2 + {a + b}*{variable} + {a * b}", variable, "factor"
    if operation == "differentiate":
        return operation, f"{a}*{variable}**3 + {b}*{variable}**2 - {a}", variable, "differentiate"
    if operation == "integrate":
        return operation, f"{a}*{variable}**2 + {b}*{variable} + {a + b}", variable, "integrate"
    return operation, f"{a}*{variable} - {b}", variable, "solve"


def _sympy_expected_result(operation: str, expression: str, variable: str, index: int) -> str:
    del expression
    a = 2 + index % 7
    b = 3 + (index // 3) % 9
    if operation == "simplify":
        return str(a * a)
    if operation == "factor":
        return f"({variable} + {a})*({variable} + {b})"
    if operation == "differentiate":
        return f"{3 * a}*{variable}**2 + {2 * b}*{variable}"
    if operation == "integrate":
        return f"{a}*{variable}**3/3 + {b}*{variable}**2/2 + {a + b}*{variable}"
    return f"[{b}/{a}]"


def _dna_sequence(index: int) -> str:
    motifs = ("ACGTAC", "GATTACA", "TTGACA", "CGTTAAG", "ATGCGT", "TACGGA")
    motif = motifs[index % len(motifs)]
    tail = "".join("ACGT"[(index >> shift) & 3] for shift in range(0, 24, 2))
    length = 24 + (index % 48)
    pattern = motif + tail
    return (pattern * ((length // len(pattern)) + 2))[:length]


def _timeseries_values(index: int) -> list[float]:
    base = 80 + index % 113
    trend = 0.75 + ((index // 113) % 13) * 0.35 + (index % 5) * 0.04
    seasonal = (0, 3 + index % 3, -2, 4 + (index // 7) % 4)
    return [round(base + trend * step + seasonal[step % 4], 3) for step in range(12)]


def _deterministic_forecast(values: list[float], horizon: int) -> list[float]:
    window = values[-6:]
    trend = (window[-1] - window[0]) / 5
    current = values[-1]
    return [round(max(0.0, current + trend * step), 3) for step in range(1, horizon + 1)]


def _augmentation_prompt(record: ToolUseCorpusRecord) -> str:
    return "\n".join(
        [
            "You are helping VECL-QB propose synthetic tool-use corpus variations.",
            "Return only JSON as a list of 1-3 objects with prompt, target_text, task_kind,",
            "target_format, executable, specialist_id, task_type, and optional metadata.",
            "Do not include real people, secrets, private data, destructive tool execution,",
            "or any production user data. The validator decides whether outputs are admitted.",
            "",
            "Seed record:",
            json.dumps(record.to_json_obj(), sort_keys=True, indent=2),
        ]
    )


def _parse_llm_candidates(raw_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    text = raw_text.strip()
    candidates: list[str] = [text]
    array_match = _JSON_ARRAY_RE.search(text)
    object_match = _JSON_OBJECT_RE.search(text)
    if array_match is not None:
        candidates.append(array_match.group(0))
    if object_match is not None:
        candidates.append(object_match.group(0))
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"LLM output was not valid JSON: {exc}")
            continue
        if isinstance(parsed, list):
            objects = [item for item in parsed if isinstance(item, dict)]
            if len(objects) != len(parsed):
                errors.append("LLM JSON list contained non-object entries")
            return objects, errors
        if isinstance(parsed, dict):
            if isinstance(parsed.get("records"), list):
                return [item for item in parsed["records"] if isinstance(item, dict)], errors
            return [parsed], errors
    return [], errors or ["LLM output did not contain JSON candidates"]


def _record_from_llm_candidate(
    *,
    seed_record: ToolUseCorpusRecord,
    candidate: dict[str, Any],
    seed: int,
    candidate_index: int,
    provider: str,
    model_id: str,
    prompt_hash: str,
) -> ToolUseCorpusRecord:
    prompt = str(candidate["prompt"]).strip()
    target_text = str(candidate["target_text"]).strip()
    if not prompt or not target_text:
        raise ValueError("LLM candidate prompt and target_text must be non-empty")
    task_kind = cast(CorpusTaskKind, str(candidate.get("task_kind") or seed_record.task_kind))
    target_format = cast(
        TargetFormat, str(candidate.get("target_format") or seed_record.target_format)
    )
    if target_format == "json":
        parsed = json.loads(target_text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON target must be an object")
        target: dict[str, Any] | str = parsed
    else:
        target = target_text
    executable = bool(candidate.get("executable", seed_record.executable))
    specialist_id = candidate.get("specialist_id", seed_record.specialist_id)
    task_type = candidate.get("task_type", seed_record.task_type)
    candidate_metadata = dict(candidate.get("metadata") or {})
    target_source = str(candidate_metadata.get("target_source") or "")
    target_independently_verified = bool(
        target_text == seed_record.target_text
        or target_source in {"copied_seed_target", "deterministic_template", "tool_derived"}
        or candidate_metadata.get("target_independently_verified") is True
    )
    llm_generated_target = target_text != seed_record.target_text
    return _make_record(
        domain=seed_record.domain,
        category=f"{seed_record.category}_llm_augmented",
        prompt=prompt,
        target=target,
        task_kind=task_kind,
        target_format=target_format,
        executable=executable,
        specialist_id=str(specialist_id) if specialist_id is not None else None,
        task_type=str(task_type) if task_type is not None else None,
        source_id=seed_record.source_id,
        source_ids=seed_record.source_ids,
        authority=seed_record.authority,
        seed=seed,
        topic_id=f"{seed_record.example_id}-llm-{candidate_index}",
        generator_name="llm_augmentation",
        trust_kind=str(seed_record.metadata.get("trust_anchor_kind") or TRUST_ANCHOR_KIND),
        metadata_extra={
            "llm_provider": provider,
            "llm_model_id": model_id,
            "llm_prompt_hash": prompt_hash,
            "seed_example_id": seed_record.example_id,
            "llm_origin": True,
            "llm_generated_target": llm_generated_target,
            "target_independently_verified": target_independently_verified,
            "target_source": target_source
            or (
                "copied_seed_target" if target_text == seed_record.target_text else "llm_generated"
            ),
            "training_export_eligible": bool(executable and target_independently_verified),
            **candidate_metadata,
        },
        dataset_version=seed_record.dataset_version,
        tenant_id=seed_record.tenant_id,
        difficulty_tags=tuple(
            sorted(
                set(seed_record.metadata.get("difficulty_tags") or ("paraphrase",))
                | set(candidate_metadata.get("difficulty_tags") or ("paraphrase",))
            )
        ),
    )


def _dedupe_records(records: Sequence[ToolUseCorpusRecord]) -> list[ToolUseCorpusRecord]:
    by_pair: dict[tuple[str, str], ToolUseCorpusRecord] = {}
    for record in records:
        by_pair.setdefault((_normalize(record.prompt), _normalize(record.target_text)), record)
    return sorted(by_pair.values(), key=lambda item: item.example_id)


def _normalize(value: str) -> str:
    return " ".join(value.strip().split())


def _estimated_response_cost(response: ModelDriverResult) -> float:
    raw = response.metadata.get("estimated_cost_usd")
    if raw is None:
        return 0.0
    return round(float(raw), 8)


def _duplicate_stats(
    candidate_records: Sequence[ToolUseCorpusRecord],
    accepted_records: Sequence[ToolUseCorpusRecord],
) -> dict[str, Any]:
    candidate_count = len(candidate_records)
    accepted_count = len(accepted_records)
    duplicate_count = max(0, candidate_count - accepted_count)
    return {
        "candidate_records": candidate_count,
        "accepted_records": accepted_count,
        "duplicate_records_removed": duplicate_count,
        "duplicate_rate": round(duplicate_count / candidate_count, 6) if candidate_count else 0.0,
    }
