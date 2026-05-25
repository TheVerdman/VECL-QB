from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from vecl.training.corpus_schema import LLMRawCacheEntry, ToolUseCorpusRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def semantic_diversity_metrics(records: list[ToolUseCorpusRecord]) -> dict[str, Any]:
    prompts = [record.prompt for record in records]
    targets = [record.target_text for record in records]
    prompt_tokens = [_tokens(prompt) for prompt in prompts]
    target_tokens = [_tokens(target) for target in targets]
    prompt_bigrams = [_ngrams(tokens, 2) for tokens in prompt_tokens]
    prompt_trigrams = [_ngrams(tokens, 3) for tokens in prompt_tokens]
    target_bigrams = [_ngrams(tokens, 2) for tokens in target_tokens]
    target_trigrams = [_ngrams(tokens, 3) for tokens in target_tokens]
    signature_counts = Counter(_semantic_signature(record) for record in records)
    difficulty_counts = Counter(
        tag for record in records for tag in record.metadata.get("difficulty_tags", [])
    )
    prompt_similarity = _sampled_pairwise_jaccard(prompt_tokens)
    target_similarity = _sampled_pairwise_jaccard(target_tokens)
    target_counts = Counter(targets)
    return {
        "record_count": len(records),
        "unique_prompt_ratio": _ratio(len(set(prompts)), len(prompts)),
        "unique_target_ratio": _ratio(len(set(targets)), len(targets)),
        "top_target_share": _ratio(
            max(target_counts.values()) if target_counts else 0, len(records)
        ),
        "token_type_token_ratio": _type_token_ratio(prompt_tokens),
        "bigram_type_token_ratio": _type_token_ratio(prompt_bigrams),
        "trigram_type_token_ratio": _type_token_ratio(prompt_trigrams),
        "target_token_type_token_ratio": _type_token_ratio(target_tokens),
        "target_bigram_type_token_ratio": _type_token_ratio(target_bigrams),
        "target_trigram_type_token_ratio": _type_token_ratio(target_trigrams),
        "domain_entropy": _normalized_entropy(record.domain for record in records),
        "category_entropy": _normalized_entropy(record.category for record in records),
        "task_kind_entropy": _normalized_entropy(record.task_kind for record in records),
        "difficulty_entropy": _normalized_entropy(difficulty_counts.elements()),
        "semantic_signature_count": len(signature_counts),
        "semantic_signature_entropy": _normalized_entropy(signature_counts.elements()),
        "top_semantic_signature_share": _ratio(
            max(signature_counts.values()) if signature_counts else 0, len(records)
        ),
        "sampled_mean_prompt_jaccard": prompt_similarity["mean"],
        "sampled_p95_prompt_jaccard": prompt_similarity["p95"],
        "sampled_mean_target_jaccard": target_similarity["mean"],
        "sampled_p95_target_jaccard": target_similarity["p95"],
        "sampled_pair_count": prompt_similarity["pairs"],
    }


def llm_cache_stats(entries: list[LLMRawCacheEntry]) -> dict[str, Any]:
    cost_by_provider: dict[str, float] = {}
    accepted = 0
    rejected = 0
    outcomes = Counter(entry.validator_outcome for entry in entries)
    for entry in entries:
        provider = entry.provider.lower()
        cost_by_provider[provider] = round(
            cost_by_provider.get(provider, 0.0) + float(entry.estimated_cost_usd or 0.0), 8
        )
        accepted += len(entry.accepted_example_ids)
        rejected += len(entry.rejected_reasons)
    return {
        "cache_entries": len(entries),
        "accepted_records": accepted,
        "rejected_reasons": rejected,
        "validator_outcomes": dict(sorted(outcomes.items())),
        "cost_by_provider": dict(sorted(cost_by_provider.items())),
    }


def admitted_llm_records(records: list[ToolUseCorpusRecord]) -> list[ToolUseCorpusRecord]:
    return [record for record in records if record.metadata.get("llm_origin") is True]


def training_export_records(
    records: list[ToolUseCorpusRecord], *, include_llm_unverified_targets: bool = False
) -> list[ToolUseCorpusRecord]:
    return [
        record
        for record in records
        if _default_export_eligible(
            record, include_llm_unverified_targets=include_llm_unverified_targets
        )
    ]


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _default_export_eligible(
    record: ToolUseCorpusRecord, *, include_llm_unverified_targets: bool
) -> bool:
    if not record.executable:
        return False
    if record.metadata.get("training_export_eligible") is False:
        return False
    return not (
        record.metadata.get("llm_origin") is True
        and record.metadata.get("llm_generated_target") is True
        and record.metadata.get("target_independently_verified") is not True
        and not include_llm_unverified_targets
    )


def _ngrams(tokens: list[str], size: int) -> list[str]:
    if len(tokens) < size:
        return []
    return [" ".join(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1)]


def _type_token_ratio(token_lists: list[list[str]]) -> float:
    flattened = [token for tokens in token_lists for token in tokens]
    return _ratio(len(set(flattened)), len(flattened))


def _normalized_entropy(values: Iterable[object]) -> float:
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(entropy / math.log2(len(counts)), 6)


def _semantic_signature(record: ToolUseCorpusRecord) -> str:
    tags = ",".join(sorted(str(tag) for tag in record.metadata.get("difficulty_tags", [])))
    chain = ",".join(str(item) for item in record.metadata.get("chain", []))
    return "|".join(
        [
            record.domain,
            record.category,
            record.task_kind,
            str(record.specialist_id or "none"),
            str(record.task_type or "none"),
            tags,
            chain,
        ]
    )


def _sampled_pairwise_jaccard(
    token_lists: list[list[str]], *, max_pairs: int = 5000
) -> dict[str, Any]:
    if len(token_lists) < 2:
        return {"mean": 0.0, "p95": 0.0, "pairs": 0}
    step = max(1, len(token_lists) // max_pairs)
    similarities: list[float] = []
    for index in range(0, len(token_lists) - 1, step):
        left = set(token_lists[index])
        right = set(token_lists[(index * 7919 + 104729) % len(token_lists)])
        if index == (index * 7919 + 104729) % len(token_lists):
            right = set(token_lists[(index + 1) % len(token_lists)])
        similarities.append(_jaccard(left, right))
        if len(similarities) >= max_pairs:
            break
    similarities.sort()
    if not similarities:
        return {"mean": 0.0, "p95": 0.0, "pairs": 0}
    p95_index = min(len(similarities) - 1, int(0.95 * (len(similarities) - 1)))
    return {
        "mean": round(sum(similarities) / len(similarities), 6),
        "p95": round(similarities[p95_index], 6),
        "pairs": len(similarities),
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
