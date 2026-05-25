#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.prompted_router import PromptedLLMRouter
from vecl.qb.router import SpecialistCard
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import StockfishSpecialist

DEFAULT_STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_18/stockfish-ubuntu-x86-64-avx2.tar"
)


def main() -> int:
    fixture_path = Path(
        os.environ.get(
            "VECL_ROUTING_EVAL_FIXTURE", Path(__file__).with_name("routing_eval_v0.json")
        )
    )
    entries = json.loads(fixture_path.read_text())
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required for the Gemma routing eval.", flush=True)
        return 2

    stockfish_binary = ensure_stockfish_18()
    artifact_root = Path(os.environ.get("VECL_ARTIFACT_STORE", "/tmp/vecl-routing-artifacts"))
    ledger = ProvenanceLedger()
    router = PromptedLLMRouter(ledger=ledger, model_id=model_id)
    specialist = StockfishSpecialist(
        binary=stockfish_binary,
        working_directory=stockfish_binary.parent,
        artifact_store=ContentAddressedStore(artifact_root),
    )
    router.register_specialist(
        SpecialistCard(
            "stockfish",
            {"chess_eval"},
            {"description": "Analyzes chess positions from FEN and returns best moves."},
            cost_hint=1.0,
            latency_hint=1.0,
            version="stockfish-18",
            effective_trust=0.9,
            description="Analyze chess positions with Stockfish.",
        ),
        specialist,
    )
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"chess_eval"}),
    )

    chess_total = 0
    chess_stockfish = 0
    out_total = 0
    out_stockfish = 0
    rows = []
    for index, entry in enumerate(entries, start=1):
        result = orchestrator.run_task(
            tenant_id="routing-eval",
            task_type=entry["task_type"],
            input_payload=entry["input_payload"],
        )
        routed_to_stockfish = "stockfish" in result.specialist_ids
        if entry["expected_specialist_id"] == "stockfish":
            chess_total += 1
            chess_stockfish += int(routed_to_stockfish)
        else:
            out_total += 1
            out_stockfish += int(routed_to_stockfish)
        row = {
            "index": index,
            "id": entry["id"],
            "expected": entry["expected_specialist_id"],
            "specialist_ids": result.specialist_ids,
            "verification_status": result.verification_status,
            "routed_to_stockfish": routed_to_stockfish,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    decision_events = ledger.find_by_type(EventType.LLM_ROUTING_DECIDED)
    fallback_events = ledger.find_by_type(EventType.ROUTING_FALLBACK)
    chess_rate = chess_stockfish / chess_total
    out_rate = out_stockfish / out_total
    summary = {
        "model_id": model_id,
        "stockfish_binary": str(stockfish_binary),
        "chess_routing_rate": chess_rate,
        "out_of_domain_stockfish_rate": out_rate,
        "decision_events": len(decision_events),
        "fallback_events": len(fallback_events),
        "total_entries": len(entries),
    }
    print("ROUTING_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if chess_rate < 0.9 or out_rate > 0.1:
        return 1
    if len(decision_events) != len(entries) or fallback_events:
        return 1
    return 0


def ensure_stockfish_18() -> Path:
    existing = os.environ.get("STOCKFISH_BINARY") or shutil.which("stockfish")
    if existing:
        candidate = Path(existing)
        if _is_stockfish_18(candidate):
            return candidate

    root = Path(tempfile.gettempdir()) / "vecl-stockfish-18"
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "stockfish-18.tar"
    if not archive_path.exists():
        url = os.environ.get("STOCKFISH_DOWNLOAD_URL", DEFAULT_STOCKFISH_URL)
        print(f"Downloading Stockfish 18 from {url}", flush=True)
        urllib.request.urlretrieve(url, archive_path)  # noqa: S310 - official URL, overridable by env.
    with tarfile.open(archive_path) as archive:
        archive.extractall(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and "stockfish" in path.name.lower():
            path.chmod(path.stat().st_mode | 0o755)
            if _is_stockfish_18(path):
                return path
    raise RuntimeError(
        "could not find a working Stockfish 18 binary in the official release tarball"
    )


def _is_stockfish_18(path: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(path)],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "id name Stockfish 18" in completed.stdout


if __name__ == "__main__":
    raise SystemExit(main())
