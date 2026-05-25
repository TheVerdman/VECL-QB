#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID, gemma_route_once
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.prompted_router import PromptedLLMRouter
from vecl.qb.router import SpecialistCard
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.blast_specialist import BLASTSpecialist
from vecl.specialists.sympy_specialist import SymPySpecialist

DEFAULT_BLAST_URL = (
    "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/"
    "ncbi-blast-2.17.0+-x64-linux.tar.gz"
)
PHASE7_SUBJECTS = {
    "subject1": "ATGCGTACGTAGCTAGCTAGCTAG",
    "subject2": "TTTTCCCCAAAAGGGGTTTTCCCC",
}


def main() -> int:
    fixture_path = _default_fixture_path()
    entries = json.loads(fixture_path.read_text())
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required for the Phase 7 Gemma routing eval.", flush=True)
        return 2
    summary = run_phase7_routing_eval(
        entries=entries,
        inference_fn=gemma_route_once,
        model_id=model_id,
    )
    print("PHASE7_ROUTING_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["sympy_routing_success_rate"] < 0.9:
        return 1
    if summary["blast_routing_success_rate"] < 0.9:
        return 1
    if summary["out_of_domain_false_positive_rate"] > 0.1:
        return 1
    if summary["decision_events"] != summary["total_entries"]:
        return 1
    if summary["fallback_events"] != 0:
        return 1
    return 0


def run_phase7_routing_eval(
    *,
    entries: Sequence[dict[str, Any]],
    inference_fn: Callable[[str], str],
    model_id: str,
    artifact_root: Path | None = None,
    blastn_binary: Path | None = None,
    makeblastdb_binary: Path | None = None,
) -> dict[str, Any]:
    if blastn_binary is None or makeblastdb_binary is None:
        blastn_binary, makeblastdb_binary = ensure_blast_plus()
    artifact_root = artifact_root or Path(
        os.environ.get("VECL_ARTIFACT_STORE", "/tmp/vecl-phase7-routing-artifacts")
    )
    db_prefix = create_phase7_blast_db(
        artifact_root / "blast-db",
        makeblastdb_binary=makeblastdb_binary,
    )
    store = ContentAddressedStore(artifact_root / "artifacts")
    ledger = ProvenanceLedger()
    router = PromptedLLMRouter(ledger=ledger, inference_fn=inference_fn, model_id=model_id)
    sympy = SymPySpecialist(task_types=("tool_request",), artifact_store=store)
    blast = BLASTSpecialist(
        task_types=("tool_request",),
        binary=blastn_binary,
        artifact_store=store,
    )
    router.register_specialist(sympy_card(), sympy)
    router.register_specialist(blast_card(), blast)
    orchestrator = QBOrchestrator(router, ledger)

    counters: dict[str, int] = {
        "sympy_total": 0,
        "sympy_success": 0,
        "blast_total": 0,
        "blast_success": 0,
        "out_total": 0,
        "out_false_positive": 0,
    }
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        payload = dict(entry["input_payload"])
        if payload.get("query_sequence") and not payload.get("database"):
            payload["database"] = str(db_prefix)
            payload.setdefault("task", "blastn-short")
        before_artifacts = len(ledger.find_by_type(EventType.ARTIFACT_PRODUCED))
        result = orchestrator.run_task(
            tenant_id="phase7-routing-eval",
            task_type=str(entry["task_type"]),
            input_payload=payload,
        )
        after_artifacts = len(ledger.find_by_type(EventType.ARTIFACT_PRODUCED))
        artifacts_added = after_artifacts - before_artifacts
        expected = entry["expected_specialist_id"]
        actual = result.specialist_ids[0] if result.specialist_ids else None
        success = actual == expected and (expected is None or artifacts_added > 0)
        if expected == "sympy":
            counters["sympy_total"] += 1
            counters["sympy_success"] += int(success)
        elif expected == "blast":
            counters["blast_total"] += 1
            counters["blast_success"] += int(success)
        else:
            counters["out_total"] += 1
            counters["out_false_positive"] += int(actual is not None)
        row = {
            "index": index,
            "id": entry["id"],
            "expected": expected,
            "actual": actual,
            "verification_status": result.verification_status,
            "artifacts_added": artifacts_added,
            "success": success,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    decision_events = ledger.find_by_type(EventType.LLM_ROUTING_DECIDED)
    fallback_events = ledger.find_by_type(EventType.ROUTING_FALLBACK)
    artifact_events = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    summary = {
        **counters,
        "model_id": model_id,
        "blastn_binary": str(blastn_binary),
        "makeblastdb_binary": str(makeblastdb_binary),
        "blast_database": str(db_prefix),
        "total_entries": len(entries),
        "decision_events": len(decision_events),
        "fallback_events": len(fallback_events),
        "artifact_events": len(artifact_events),
        "sympy_routing_success_rate": _rate(counters["sympy_success"], counters["sympy_total"]),
        "blast_routing_success_rate": _rate(counters["blast_success"], counters["blast_total"]),
        "out_of_domain_false_positive_rate": _rate(
            counters["out_false_positive"], counters["out_total"]
        ),
        "rows": rows,
    }
    return summary


def sympy_card() -> SpecialistCard:
    return SpecialistCard(
        "sympy",
        {"tool_request"},
        {
            "description": (
                "Use for exact symbolic math, algebra, calculus, equation solving, "
                "simplification, factoring, integration, and differentiation."
            )
        },
        cost_hint=0.1,
        latency_hint=0.1,
        version="sympy-1.14",
        effective_trust=0.95,
        description=(
            "Exact symbolic mathematics with SymPy; not for biological sequence alignment."
        ),
    )


def blast_card() -> SpecialistCard:
    return SpecialistCard(
        "blast",
        {"tool_request"},
        {
            "description": (
                "Use for nucleotide or protein sequence alignment against a local BLAST "
                "database and tabular top-hit reporting."
            )
        },
        cost_hint=0.6,
        latency_hint=0.5,
        version="blast-2.17",
        effective_trust=0.9,
        description="Local NCBI BLAST+ sequence alignment; not for algebra or calculus.",
    )


def create_phase7_blast_db(root: Path, *, makeblastdb_binary: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fasta_path = root / "subjects.fasta"
    db_prefix = root / "phase7_subjects"
    fasta_path.write_text(
        "".join(f">{name}\n{sequence}\n" for name, sequence in sorted(PHASE7_SUBJECTS.items()))
    )
    completed = subprocess.run(
        [
            str(makeblastdb_binary),
            "-in",
            str(fasta_path),
            "-dbtype",
            "nucl",
            "-out",
            str(db_prefix),
            "-parse_seqids",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "makeblastdb failed")
    return db_prefix


def ensure_blast_plus() -> tuple[Path, Path]:
    blastn = _resolve_existing_binary("BLASTN_BINARY", "blastn")
    makeblastdb = _resolve_existing_binary("MAKEBLASTDB_BINARY", "makeblastdb")
    if blastn is not None and makeblastdb is not None:
        return blastn, makeblastdb

    root = Path(tempfile.gettempdir()) / "vecl-ncbi-blast-2.17"
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "ncbi-blast-2.17.0-linux.tar.gz"
    if not archive_path.exists():
        url = os.environ.get("BLAST_DOWNLOAD_URL", DEFAULT_BLAST_URL)
        print(f"Downloading NCBI BLAST+ from {url}", flush=True)
        urllib.request.urlretrieve(url, archive_path)  # noqa: S310 - official URL, env-overridable.
    with tarfile.open(archive_path) as archive:
        archive.extractall(root)
    downloaded_blastn = _find_downloaded_binary(root, "blastn")
    downloaded_makeblastdb = _find_downloaded_binary(root, "makeblastdb")
    return downloaded_blastn, downloaded_makeblastdb


def _resolve_existing_binary(env_name: str, binary_name: str) -> Path | None:
    raw = os.environ.get(env_name)
    candidates = [Path(raw)] if raw else []
    resolved = shutil.which(binary_name)
    if resolved:
        candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _find_downloaded_binary(root: Path, name: str) -> Path:
    for path in sorted(root.rglob(name)):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o755)
            return path
    raise RuntimeError(f"{name} was not found in downloaded BLAST+ archive")


def _default_fixture_path() -> Path:
    env_path = os.environ.get("VECL_PHASE7_ROUTING_EVAL_FIXTURE")
    if env_path:
        return Path(env_path)
    packaged = Path(__file__).with_name("phase7_routing_eval_v0.json")
    if packaged.exists():
        return packaged
    return Path(__file__).parents[1] / "tests" / "qb" / "fixtures" / "phase7_routing_eval_v0.json"


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
