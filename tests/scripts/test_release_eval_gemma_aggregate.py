from __future__ import annotations

from pathlib import Path

from scripts.release_eval_gemma_aggregate import (
    EVAL_ORDER,
    build_report_from_summaries,
    load_aggregate_specs,
)


def test_aggregate_specs_load_from_vertex_manifest() -> None:
    manifest = Path(__file__).parents[2] / "configs" / "release" / "phase9a-vertex-gemma.json"

    specs = load_aggregate_specs(manifest)

    assert tuple(specs) == EVAL_ORDER
    assert specs["gemma_model_driver_eval"].category == "routing"
    assert specs["phase6b_ethics_eval"].category == "ethics"


def test_aggregate_report_scores_manifest_thresholds() -> None:
    manifest = Path(__file__).parents[2] / "configs" / "release" / "phase9a-vertex-gemma.json"
    specs = load_aggregate_specs(manifest)

    report = build_report_from_summaries(
        candidate_id="candidate",
        specs=specs,
        summaries=_passing_summaries(),
    )

    assert report.passed
    assert report.gate_checks()["regression_tests_pass"]
    assert report.gate_checks()["adversarial_simulation_within_bound"]


def test_aggregate_report_fails_bad_summary() -> None:
    manifest = Path(__file__).parents[2] / "configs" / "release" / "phase9a-vertex-gemma.json"
    specs = load_aggregate_specs(manifest)
    summaries = _passing_summaries()
    summaries["phase7b_timesfm_demand_eval"] = {
        **summaries["phase7b_timesfm_demand_eval"],
        "fallback_events": 1,
    }

    report = build_report_from_summaries(
        candidate_id="candidate",
        specs=specs,
        summaries=summaries,
    )

    assert not report.passed
    failed = [result for result in report.results if not result.passed]
    assert failed[0].name == "phase7b_timesfm_demand_eval"
    assert failed[0].reasons == ("fallback_events expected 0, got 1",)


def _passing_summaries() -> dict[str, dict[str, object]]:
    return {
        "gemma_model_driver_eval": {
            "route_success_rate": 1.0,
            "synthesis_success_rate": 1.0,
            "route_parse_failures": 0,
            "synthesis_parse_failures": 0,
        },
        "gemma_tool_call_payload_eval": {
            "proposal_success_rate": 1.0,
            "validation_success_rate": 1.0,
            "execution_success_rate": 1.0,
            "synthesis_success_rate": 1.0,
        },
        "phase7_sympy_blast_routing_eval": {
            "sympy_routing_success_rate": 1.0,
            "blast_routing_success_rate": 1.0,
            "out_of_domain_false_positive_rate": 0.0,
            "fallback_events": 0,
        },
        "phase6b_ethics_eval": {
            "parse_failures": 0,
            "unexpected_outcomes": [],
            "blocked_specialist_calls": 0,
        },
        "phase8b2_terraform_stockfish_persistence_eval": {
            "plan_chain_success_rate": 1.0,
            "answer_success_rate": 1.0,
            "mutation_safe_rate": 1.0,
            "stockfish_routing_success_rate": 1.0,
            "out_of_domain_terraform_rate": 0.0,
            "blocked_specialist_calls": 0,
            "fallback_events": 0,
        },
        "phase7b_timesfm_demand_eval": {
            "timesfm_routing_rate": 1.0,
            "timesfm_chain_success_rate": 1.0,
            "timesfm_answer_success_rate": 1.0,
            "out_of_domain_timesfm_rate": 0.0,
            "fallback_events": 0,
        },
    }
