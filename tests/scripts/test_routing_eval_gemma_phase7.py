from __future__ import annotations

import json
from pathlib import Path

from scripts.routing_eval_gemma_phase7 import (
    _default_fixture_path,
    blast_card,
    create_phase7_blast_db,
    run_phase7_routing_eval,
    sympy_card,
)
from vecl.specialists.blast_specialist import resolve_blast_binary


def test_phase7_cards_describe_distinct_specialists() -> None:
    sympy = sympy_card()
    blast = blast_card()

    assert sympy.specialist_id == "sympy"
    assert blast.specialist_id == "blast"
    assert "symbolic" in sympy.description.lower()
    assert "sequence" in blast.description.lower()


def test_create_phase7_blast_db_uses_real_makeblastdb(tmp_path: Path) -> None:
    db_prefix = create_phase7_blast_db(
        tmp_path,
        makeblastdb_binary=resolve_blast_binary("makeblastdb"),
    )

    assert db_prefix.with_suffix(".ndb").exists() or db_prefix.with_suffix(".nin").exists()


def test_phase7_eval_with_mock_inference_routes_and_executes_real_tools(
    tmp_path: Path,
) -> None:
    entries = json.loads(_default_fixture_path().read_text())

    summary = run_phase7_routing_eval(
        entries=entries,
        inference_fn=_mock_phase7_inference,
        model_id="mock-gemma-31b",
        artifact_root=tmp_path,
        blastn_binary=resolve_blast_binary("blastn"),
        makeblastdb_binary=resolve_blast_binary("makeblastdb"),
    )

    assert summary["sympy_routing_success_rate"] == 1.0
    assert summary["blast_routing_success_rate"] == 1.0
    assert summary["out_of_domain_false_positive_rate"] == 0.0
    assert summary["decision_events"] == len(entries)
    assert summary["fallback_events"] == 0
    assert summary["artifact_events"] == 10
    assert all(row["success"] for row in summary["rows"])


def _mock_phase7_inference(prompt: str) -> str:
    request = prompt.split("Request:", 1)[1]
    if any(
        marker in request
        for marker in [
            "Simplify",
            "Solve",
            "Differentiate",
            "Integrate",
            "Factor",
            "sin(x)",
            "x**2",
        ]
    ):
        return json.dumps(
            {
                "tool": "sympy",
                "confidence": 0.98,
                "reasoning": "The request asks for symbolic mathematics.",
            }
        )
    if any(
        marker in request
        for marker in ["BLAST", "DNA", "nucleotide", "homologs", "sequence alignment"]
    ):
        return json.dumps(
            {
                "tool": "blast",
                "confidence": 0.97,
                "reasoning": "The request asks for biological sequence alignment.",
            }
        )
    return json.dumps(
        {
            "tool": None,
            "confidence": 0.91,
            "reasoning": "No registered specialist is appropriate.",
        }
    )
