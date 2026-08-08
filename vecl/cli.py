from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vecl.evaluation.evaluators import evaluators_from_manifest
from vecl.evaluation.release import (
    approve_release_report,
    load_release_report,
    reject_release_report,
    run_release_evaluation,
    save_release_report,
)
from vecl.provenance.events import stable_hash
from vecl.provenance.ledger import ProvenanceLedger, SqliteProvenanceLedger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vecl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    release = subparsers.add_parser("release")
    release_subparsers = release.add_subparsers(dest="release_command", required=True)

    evaluate = release_subparsers.add_parser("evaluate")
    evaluate.add_argument("candidate_id")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    approve = release_subparsers.add_parser("approve")
    approve.add_argument("report_path", type=Path)
    approve.add_argument("--ledger-path", type=Path)
    approve.add_argument("--tenant-id", default="release")

    reject = release_subparsers.add_parser("reject")
    reject.add_argument("report_path", type=Path)
    reject.add_argument("--reason", required=True)
    reject.add_argument("--ledger-path", type=Path)
    reject.add_argument("--tenant-id", default="release")

    args = parser.parse_args(argv)
    if args.command == "release" and args.release_command == "evaluate":
        return _evaluate(args)
    if args.command == "release" and args.release_command == "approve":
        return _approve(args)
    if args.command == "release" and args.release_command == "reject":
        return _reject(args)
    raise RuntimeError("unreachable command dispatch")


def _evaluate(args: argparse.Namespace) -> int:
    manifest = _load_json(args.manifest)
    evaluators = evaluators_from_manifest(manifest)
    report = run_release_evaluation(
        args.candidate_id,
        evaluators,
        manifest_hash=stable_hash(manifest),
    )
    payload = report.to_payload()
    if args.output:
        save_release_report(report, args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0 if report.approval_eligible else 1


def _approve(args: argparse.Namespace) -> int:
    report = load_release_report(args.report_path)
    with _ledger(args.ledger_path) as ledger:
        event = approve_release_report(report, ledger, args.tenant_id)
        print(json.dumps(event.to_dict(), sort_keys=True))
    return 0


def _reject(args: argparse.Namespace) -> int:
    report = load_release_report(args.report_path)
    with _ledger(args.ledger_path) as ledger:
        event = reject_release_report(report, ledger, args.tenant_id, args.reason)
        print(json.dumps(event.to_dict(), sort_keys=True))
    return 0


def _ledger(path: Path | None) -> Any:
    return SqliteProvenanceLedger(path) if path is not None else _InMemoryLedgerContext()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


class _InMemoryLedgerContext:
    def __init__(self) -> None:
        self.ledger = ProvenanceLedger()

    def __enter__(self) -> ProvenanceLedger:
        return self.ledger

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
