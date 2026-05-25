#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.training.corpus_metrics import (  # noqa: E402
    semantic_diversity_metrics,
    training_export_records,
)
from vecl.training.corpus_schema import (  # noqa: E402
    LLMRawCacheEntry,
    read_corpus_jsonl,
    read_llm_cache_jsonl,
)
from vecl.training.corpus_validation import (  # noqa: E402
    corpus_markdown_report_with_metadata,
    validate_corpus_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a markdown report for a synthetic corpus.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-category-coverage", action="store_true")
    args = parser.parse_args(argv)

    result = validate_corpus_path(
        args.path, require_category_coverage=not args.no_category_coverage
    )
    records = read_corpus_jsonl(args.path / "corpus.jsonl" if args.path.is_dir() else args.path)
    metadata = _read_metadata(args.path)
    metadata.setdefault("semantic_diversity", semantic_diversity_metrics(records))
    metadata.setdefault(
        "training_export_diversity",
        semantic_diversity_metrics(training_export_records(records)),
    )
    cache_entries = _read_cache(args.path)
    examples = records[:8]
    name = args.path.name if args.path.is_dir() else args.path.stem
    title = args.title or f"VECL-QB Synthetic Corpus {name}"
    output = args.output or ROOT / "reports" / f"synthetic_corpus_{name.replace('-', '_')}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        corpus_markdown_report_with_metadata(
            result,
            title=title,
            llm_cache_entries=cache_entries,
            metadata=metadata,
            example_records=examples,
        ),
        encoding="utf-8",
    )
    print(
        "TRAINING_CORPUS_REPORT_SUMMARY "
        + json.dumps(
            {
                "valid": result.valid,
                "output": str(output),
                "records": result.total_records,
                "dataset_hash": result.dataset_hash,
                "errors": sum(1 for issue in result.issues if issue.severity == "error"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.valid else 1


def _read_metadata(path: Path) -> dict[str, object]:
    metadata_path = path / "metadata.json" if path.is_dir() else path.with_name("metadata.json")
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _read_cache(path: Path) -> list[LLMRawCacheEntry]:
    cache_path = (
        path / "llm_raw_cache.jsonl" if path.is_dir() else path.with_name("llm_raw_cache.jsonl")
    )
    if not cache_path.exists():
        return []
    return read_llm_cache_jsonl(cache_path)


if __name__ == "__main__":
    raise SystemExit(main())
