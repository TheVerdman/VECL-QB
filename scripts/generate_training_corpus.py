#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.qb.model_driver import resolve_model_driver  # noqa: E402
from vecl.training.corpus_factory import (  # noqa: E402
    DEFAULT_PROVIDER_COST_CAPS,
    SyntheticCorpusConfig,
    generate_synthetic_corpus,
)
from vecl.training.corpus_metrics import (  # noqa: E402
    semantic_diversity_metrics,
    training_export_records,
)
from vecl.training.corpus_schema import (  # noqa: E402
    append_llm_cache_jsonl,
    corpus_hash,
    read_llm_cache_jsonl,
    write_corpus_jsonl,
    write_llm_cache_jsonl,
)
from vecl.training.corpus_validation import (  # noqa: E402
    single_record_acceptor,
    validate_corpus_records,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate VECL-QB synthetic tool-use corpus.")
    parser.add_argument(
        "--size", choices=("sample", "v0-small", "v0-full", "v1-hard"), default="sample"
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--llm-augment", action="store_true")
    parser.add_argument("--llm-limit", type=int, default=0)
    parser.add_argument("--llm-driver", action="append", default=[])
    parser.add_argument(
        "--checkpoint-cache",
        nargs="?",
        const="",
        type=str,
        default=None,
        help=(
            "Append each LLM raw cache entry as it completes and resume existing entries. "
            "With no path, uses OUTPUT_DIR/llm_raw_cache.jsonl."
        ),
    )
    parser.add_argument(
        "--openai-cap-usd", type=float, default=DEFAULT_PROVIDER_COST_CAPS["openai"]
    )
    parser.add_argument(
        "--anthropic-cap-usd", type=float, default=DEFAULT_PROVIDER_COST_CAPS["anthropic"]
    )
    parser.add_argument("--max-estimated-cost-per-call-usd", type=float, default=0.25)
    parser.add_argument("--no-placeholders", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or ROOT / "data" / "synthetic" / args.size
    drivers = []
    if args.llm_augment:
        selected_drivers = args.llm_driver or _env_llm_drivers()
        drivers = [_resolve_optional_driver(name) for name in selected_drivers]
        drivers = [driver for driver in drivers if driver is not None]

    checkpoint_cache_path = _resolve_checkpoint_cache_path(args.checkpoint_cache, output_dir)
    existing_checkpoint_entries = []
    if checkpoint_cache_path is not None and checkpoint_cache_path.exists():
        existing_checkpoint_entries = read_llm_cache_jsonl(
            checkpoint_cache_path, allow_truncated_tail=True
        )

    config = SyntheticCorpusConfig(
        size=args.size,
        target_count=args.count,
        seed=args.seed,
        include_placeholders=not args.no_placeholders,
    )
    build = generate_synthetic_corpus(
        config,
        llm_drivers=drivers,
        llm_limit=args.llm_limit if args.llm_augment else 0,
        record_validator=single_record_acceptor,
        existing_llm_cache_entries=existing_checkpoint_entries,
        llm_cache_checkpoint=(
            (lambda entry: append_llm_cache_jsonl(entry, checkpoint_cache_path))
            if checkpoint_cache_path is not None and args.llm_augment
            else None
        ),
        provider_cost_caps={"openai": args.openai_cap_usd, "anthropic": args.anthropic_cap_usd},
        max_estimated_cost_per_call_usd=args.max_estimated_cost_per_call_usd,
    )
    validation = validate_corpus_records(
        build.records, require_category_coverage=not args.no_placeholders
    )
    if not validation.valid:
        print(
            "TRAINING_CORPUS_GENERATION_SUMMARY "
            + json.dumps({"valid": False, "validation": validation.to_payload()}, sort_keys=True),
            flush=True,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    metadata_path = output_dir / "metadata.json"
    cache_path = output_dir / "llm_raw_cache.jsonl"
    admitted_path = output_dir / "llm_admitted.jsonl"
    write_corpus_jsonl(build.records, corpus_path)
    if build.llm_cache_entries:
        write_llm_cache_jsonl(build.llm_cache_entries, cache_path)
    elif cache_path.exists():
        cache_path.unlink()
    if build.llm_admitted_records:
        write_corpus_jsonl(build.llm_admitted_records, admitted_path)
    elif admitted_path.exists():
        admitted_path.unlink()

    metadata = {
        "size": args.size,
        "target_count": config.resolved_count(),
        "actual_count": len(build.records),
        "seed": args.seed,
        "dataset_hash": corpus_hash(build.records),
        "llm_augmentation_used": bool(build.llm_cache_entries),
        "llm_cache_entries": len(build.llm_cache_entries),
        "llm_admitted_records": len(build.llm_admitted_records),
        "llm_checkpoint_cache_path": str(checkpoint_cache_path or ""),
        "llm_checkpoint_entries_loaded": len(existing_checkpoint_entries),
        "cost_by_provider": build.cost_by_provider,
        "duplicate_stats": build.duplicate_stats,
        "difficulty_counts": _difficulty_counts(build.records),
        "semantic_diversity": semantic_diversity_metrics(build.records),
        "training_export_diversity": semantic_diversity_metrics(
            training_export_records(build.records)
        ),
        "corpus_path": str(corpus_path),
        "llm_raw_cache_path": str(cache_path) if build.llm_cache_entries else "",
        "llm_admitted_path": str(admitted_path) if build.llm_admitted_records else "",
        "validation": validation.to_payload(),
    }
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "TRAINING_CORPUS_GENERATION_SUMMARY "
        + json.dumps(
            {
                "valid": True,
                "output_dir": str(output_dir),
                "corpus_path": str(corpus_path),
                "records": len(build.records),
                "dataset_hash": metadata["dataset_hash"],
                "llm_augmentation_used": metadata["llm_augmentation_used"],
                "llm_admitted_records": metadata["llm_admitted_records"],
                "llm_checkpoint_cache_path": metadata["llm_checkpoint_cache_path"],
                "llm_checkpoint_entries_loaded": metadata["llm_checkpoint_entries_loaded"],
                "cost_by_provider": metadata["cost_by_provider"],
                "duplicate_stats": metadata["duplicate_stats"],
                "semantic_diversity": metadata["semantic_diversity"],
                "training_export_diversity": metadata["training_export_diversity"],
                "difficulty_counts": metadata["difficulty_counts"],
                "split_counts": validation.split_counts,
                "domain_counts": validation.domain_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _resolve_checkpoint_cache_path(raw: str | None, output_dir: Path) -> Path | None:
    if raw is None:
        return None
    if raw == "":
        return output_dir / "llm_raw_cache.jsonl"
    return Path(raw)


def _env_llm_drivers() -> list[str]:
    configured = os.environ.get("VECL_SYNTHETIC_LLM_DRIVER", "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return ["openai", "anthropic"]


def _resolve_optional_driver(name: str) -> object | None:
    try:
        return resolve_model_driver(name)
    except ValueError:
        return None


def _difficulty_counts(records: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        metadata = getattr(record, "metadata", {})
        for tag in metadata.get("difficulty_tags", []):
            key = str(tag)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
