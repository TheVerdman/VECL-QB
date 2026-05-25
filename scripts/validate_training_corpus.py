#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.training.corpus_validation import validate_corpus_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a VECL-QB synthetic tool-use corpus.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--min-records", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-category-coverage", action="store_true")
    args = parser.parse_args(argv)

    result = validate_corpus_path(
        args.path,
        require_category_coverage=not args.no_category_coverage,
        expected_min_records=args.min_records,
        expected_max_records=args.max_records,
    )
    print(
        "TRAINING_CORPUS_VALIDATION_SUMMARY " + json.dumps(result.to_payload(), sort_keys=True),
        flush=True,
    )
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
