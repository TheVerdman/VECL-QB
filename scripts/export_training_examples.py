#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.training.corpus_schema import ToolUseCorpusRecord, read_corpus_jsonl  # noqa: E402
from vecl.training.corpus_validation import validate_corpus_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export supervised training examples from a validated corpus."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-llm-unverified-targets", action="store_true")
    parser.add_argument("--no-category-coverage", action="store_true")
    args = parser.parse_args(argv)

    validation = validate_corpus_path(
        args.path, require_category_coverage=not args.no_category_coverage
    )
    if not validation.valid:
        print(
            "TRAINING_EXAMPLE_EXPORT_SUMMARY "
            + json.dumps({"valid": False, "validation": validation.to_payload()}, sort_keys=True),
            flush=True,
        )
        return 1

    corpus_path = args.path / "corpus.jsonl" if args.path.is_dir() else args.path
    records = read_corpus_jsonl(corpus_path)
    output_dir = args.output_dir or (
        args.path / "exports" if args.path.is_dir() else args.path.parent / "exports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "heldout": [],
        "hard-heldout": [],
    }
    skipped = Counter()
    for record in records:
        reason = _skip_reason(
            record, include_llm_unverified_targets=args.include_llm_unverified_targets
        )
        if reason:
            skipped[reason] += 1
            continue
        exported_by_split[record.split].append(record.to_supervised_example().to_payload())

    paths: dict[str, str] = {}
    for split, payloads in exported_by_split.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for payload in sorted(payloads, key=lambda item: str(item["example_id"])):
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        paths[split] = str(path)

    summary = {
        "valid": True,
        "input": str(corpus_path),
        "output_dir": str(output_dir),
        "paths": paths,
        "export_counts": {split: len(payloads) for split, payloads in exported_by_split.items()},
        "skipped": dict(sorted(skipped.items())),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print("TRAINING_EXAMPLE_EXPORT_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _skip_reason(
    record: ToolUseCorpusRecord, *, include_llm_unverified_targets: bool
) -> str | None:
    if not record.executable:
        return "non_executable"
    if record.metadata.get("training_export_eligible") is False:
        return "metadata_export_ineligible"
    if (
        record.metadata.get("llm_origin") is True
        and record.metadata.get("llm_generated_target") is True
        and record.metadata.get("target_independently_verified") is not True
        and not include_llm_unverified_targets
    ):
        return "llm_unverified_target"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
