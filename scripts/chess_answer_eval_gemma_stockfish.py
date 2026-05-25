#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.chain_eval_gemma_stockfish import (
    StockfishFormatterSpecialist,
    formatter_card,
    stockfish_card,
    stockfish_chain_plan,
)
from scripts.routing_eval_gemma_stockfish import ensure_stockfish_18
from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID, gemma_route_once
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.planner import ChainPlanner
from vecl.qb.prompted_router import PromptedLLMRouter
from vecl.qb.specialist import SpecialistRequest
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import StockfishSpecialist

ANSWER_PROMPT_TEMPLATE = """You are VECL-QB's final chess answerer.
A specialist chain has already run. Use the Stockfish evidence below, not your own chess intuition.
Stockfish is deterministic alpha-beta engine analysis; do not describe it as Monte Carlo.
Return only JSON with this schema:
{{"bestmove":"<uci move>","engine_eval":"<centipawns or mate text>","principal_variation":["<uci move>", "..."],"answer":"<short natural-language answer>","used_stockfish":true}}

Question:
{question}

FEN:
{fen}

Chain synthesized claims:
{chain_answer_text}

Stockfish claim:
{stockfish_claim_text}

Formatter report:
{formatter_report}

Stockfish UCI transcript artifact:
{stockfish_transcript}
"""

_FENCED_RE = re.compile(r"```(?:json|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_BESTMOVE_RE = re.compile(r"\bbestmove=([a-h][1-8][a-h][1-8][qrbn]?)\b")


@dataclass(frozen=True)
class ChessToolContext:
    question: str
    fen: str
    chain_answer_text: str
    stockfish_claim_text: str
    stockfish_transcript: str
    formatter_report: str
    expected_bestmove: str
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChessAnswer:
    bestmove: str
    engine_eval: str
    principal_variation: tuple[str, ...]
    answer: str
    used_stockfish: bool
    raw_response: str


class AnswerParseError(ValueError):
    pass


def main() -> int:
    fixture_path = Path(
        os.environ.get(
            "VECL_CHESS_ANSWER_EVAL_FIXTURE",
            _default_fixture_path(),
        )
    )
    entries = json.loads(fixture_path.read_text())
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required for the Gemma chess answer eval.", flush=True)
        return 2

    max_entries = int(os.environ.get("VECL_CHESS_ANSWER_EVAL_MAX_ENTRIES", "16"))
    entries = entries[:max_entries]
    min_answer_rate = float(os.environ.get("VECL_CHESS_ANSWER_MIN_RATE", "0.8"))
    max_transcript_chars = int(os.environ.get("VECL_CHESS_ANSWER_TRANSCRIPT_CHARS", "6000"))

    stockfish_binary = ensure_stockfish_18()
    artifact_root = Path(os.environ.get("VECL_ARTIFACT_STORE", "/tmp/vecl-answer-artifacts"))
    store = ContentAddressedStore(artifact_root)
    ledger = ProvenanceLedger()
    router = PromptedLLMRouter(ledger=ledger, model_id=model_id)
    stockfish = StockfishSpecialist(
        binary=stockfish_binary,
        working_directory=stockfish_binary.parent,
        artifact_store=store,
    )
    formatter = StockfishFormatterSpecialist(store)
    router.register_specialist(stockfish_card(), stockfish)
    router.register_specialist(formatter_card(), formatter)
    planner = ChainPlanner({"chess_eval": stockfish_chain_plan()})
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"chess_eval", "chess_report"}),
    )

    summary = run_answer_eval(
        entries=entries,
        ledger=ledger,
        router=router,
        planner=planner,
        orchestrator=orchestrator,
        answer_fn=gemma_route_once,
        model_id=model_id,
        stockfish_binary=str(stockfish_binary),
        max_transcript_chars=max_transcript_chars,
    )
    print("CHESS_ANSWER_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["chess_routing_rate"] < 0.9:
        return 1
    if summary["chess_chain_success_rate"] < 0.9:
        return 1
    if summary["chess_answer_success_rate"] < min_answer_rate:
        return 1
    if summary["out_of_domain_stockfish_rate"] > 0.1:
        return 1
    if summary["fallback_events"] != 0:
        return 1
    return 0


def run_answer_eval(
    *,
    entries: list[dict[str, Any]],
    ledger: ProvenanceLedger,
    router: PromptedLLMRouter,
    planner: ChainPlanner,
    orchestrator: QBOrchestrator,
    answer_fn: Callable[[str], str],
    model_id: str,
    stockfish_binary: str,
    max_transcript_chars: int,
) -> dict[str, Any]:
    chess_total = 0
    chess_routed = 0
    chains_started = 0
    chains_passed = 0
    answer_calls = 0
    answers_passed = 0
    parse_failures = 0
    out_total = 0
    out_routed_to_stockfish = 0
    max_prompt_chars = 0

    for index, entry in enumerate(entries, start=1):
        route_request = SpecialistRequest(
            f"answer-route-{index}",
            "answer-eval",
            entry["task_type"],
            entry["input_payload"],
            {},
            {"parent_event_id": f"answer-eval-{index}"},
        )
        routed = router.route(route_request, max_specialists=1)
        routed_ids = [specialist.specialist_id for specialist in routed]

        if entry["expected_specialist_id"] != "stockfish":
            out_total += 1
            out_routed_to_stockfish += int("stockfish" in routed_ids)
            row = {
                "id": entry["id"],
                "index": index,
                "expected": None,
                "routed_ids": routed_ids,
                "chain_status": "NOT_RUN",
                "answer_status": "NOT_RUN",
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            continue

        chess_total += 1
        if routed_ids != ["stockfish"]:
            row = {
                "id": entry["id"],
                "index": index,
                "expected": "stockfish",
                "routed_ids": routed_ids,
                "chain_status": "NOT_RUN",
                "answer_status": "ROUTING_FAILED",
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            continue

        chess_routed += 1
        artifact_start = len(ledger.find_by_type(EventType.ARTIFACT_PRODUCED))
        plan = planner.plan(route_request, router.specialists)
        result = orchestrator.run_task(
            tenant_id="answer-eval",
            task_type=entry["task_type"],
            input_payload=entry["input_payload"],
            chain_plan=plan,
        )
        chains_started += 1
        chain_passed = result.verification_status == "PASSED"
        chains_passed += int(chain_passed)
        artifact_payloads = [
            event.payload
            for event in ledger.find_by_type(EventType.ARTIFACT_PRODUCED)[artifact_start:]
        ]
        context = build_tool_context(
            entry=entry,
            chain_answer_text=result.answer_text,
            artifact_payloads=artifact_payloads,
            max_transcript_chars=max_transcript_chars,
        )
        prompt = build_answer_prompt(context)
        max_prompt_chars = max(max_prompt_chars, len(prompt))
        answer_calls += 1
        raw_answer = answer_fn(prompt)
        try:
            answer = parse_answer_response(raw_answer)
            score = score_answer(context, answer)
            answer_passed = bool(score["passed"] and chain_passed)
            answers_passed += int(answer_passed)
            answer_status = "PASSED" if answer_passed else "FAILED"
            parsed_bestmove = answer.bestmove
        except AnswerParseError as exc:
            parse_failures += 1
            score = {"passed": False, "reason": str(exc)}
            answer_status = "PARSE_FAILED"
            parsed_bestmove = None
        row = {
            "id": entry["id"],
            "index": index,
            "expected": "stockfish",
            "routed_ids": routed_ids,
            "chain_status": result.verification_status,
            "answer_status": answer_status,
            "expected_bestmove": context.expected_bestmove,
            "answer_bestmove": parsed_bestmove,
            "score": score,
            "prompt_chars": len(prompt),
        }
        print(json.dumps(row, sort_keys=True), flush=True)

    return answer_eval_summary(
        ledger=ledger,
        model_id=model_id,
        stockfish_binary=stockfish_binary,
        chess_total=chess_total,
        chess_routed=chess_routed,
        chains_started=chains_started,
        chains_passed=chains_passed,
        answer_calls=answer_calls,
        answers_passed=answers_passed,
        parse_failures=parse_failures,
        out_total=out_total,
        out_routed_to_stockfish=out_routed_to_stockfish,
        max_prompt_chars=max_prompt_chars,
    )


def build_tool_context(
    *,
    entry: dict[str, Any],
    chain_answer_text: str,
    artifact_payloads: list[dict[str, Any]],
    max_transcript_chars: int = 6000,
) -> ChessToolContext:
    input_payload = entry["input_payload"]
    stockfish_claim_text = _stockfish_claim_text(chain_answer_text)
    expected_bestmove = _expected_bestmove(stockfish_claim_text)
    transcript = ""
    formatter_report = ""
    artifact_ids: list[str] = []
    for payload in artifact_payloads:
        artifact_ids.append(str(payload["artifact_id"]))
        output_format = str(payload.get("output_format", ""))
        text = read_artifact_payload_text(payload)
        if output_format == "txt" and not transcript:
            transcript = text[:max_transcript_chars]
        elif output_format == "json" and not formatter_report:
            formatter_report = text[:max_transcript_chars]
    if not transcript:
        raise ValueError("Stockfish transcript artifact is required for answer context")
    return ChessToolContext(
        question=str(input_payload.get("query", "")),
        fen=str(input_payload.get("fen", "")),
        chain_answer_text=chain_answer_text,
        stockfish_claim_text=stockfish_claim_text,
        stockfish_transcript=transcript,
        formatter_report=formatter_report,
        expected_bestmove=expected_bestmove,
        artifact_ids=tuple(sorted(artifact_ids)),
    )


def build_answer_prompt(context: ChessToolContext) -> str:
    return ANSWER_PROMPT_TEMPLATE.format(
        question=context.question,
        fen=context.fen,
        chain_answer_text=context.chain_answer_text,
        stockfish_claim_text=context.stockfish_claim_text,
        formatter_report=context.formatter_report or "{}",
        stockfish_transcript=context.stockfish_transcript,
    )


def read_artifact_payload_text(payload: dict[str, Any]) -> str:
    path = Path(str(payload["output_path"]))
    content = path.read_bytes()
    expected_hash = str(payload["output_hash"])
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("artifact content hash mismatch")
    return content.decode()


def parse_answer_response(response: str) -> ChessAnswer:
    for candidate in _json_candidates(response):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        bestmove = str(payload.get("bestmove", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not bestmove or not answer:
            continue
        pv_raw = payload.get("principal_variation", [])
        if isinstance(pv_raw, str):
            principal_variation = tuple(pv_raw.split())
        elif isinstance(pv_raw, list):
            principal_variation = tuple(str(move).strip() for move in pv_raw if str(move).strip())
        else:
            principal_variation = ()
        return ChessAnswer(
            bestmove=bestmove,
            engine_eval=str(payload.get("engine_eval", "")).strip(),
            principal_variation=principal_variation,
            answer=answer,
            used_stockfish=bool(payload.get("used_stockfish", False)),
            raw_response=response,
        )
    raise AnswerParseError("could not parse Gemma final-answer response")


def score_answer(context: ChessToolContext, answer: ChessAnswer) -> dict[str, Any]:
    bestmove_matches = answer.bestmove.lower() == context.expected_bestmove.lower()
    answer_mentions_move = context.expected_bestmove.lower() in answer.answer.lower()
    answer_mentions_engine = (
        "stockfish" in answer.answer.lower() or "engine" in answer.answer.lower()
    )
    has_engine_eval = bool(answer.engine_eval)
    pv_mentions_move = (
        bool(answer.principal_variation)
        and answer.principal_variation[0].lower() == context.expected_bestmove.lower()
    )
    passed = (
        answer.used_stockfish
        and bestmove_matches
        and has_engine_eval
        and (not answer.principal_variation or pv_mentions_move)
    )
    return {
        "passed": passed,
        "bestmove_matches": bestmove_matches,
        "answer_mentions_move": answer_mentions_move,
        "answer_mentions_engine": answer_mentions_engine,
        "has_engine_eval": has_engine_eval,
        "pv_mentions_move": pv_mentions_move,
        "used_stockfish": answer.used_stockfish,
    }


def answer_eval_summary(
    *,
    ledger: ProvenanceLedger,
    model_id: str,
    stockfish_binary: str,
    chess_total: int,
    chess_routed: int,
    chains_started: int,
    chains_passed: int,
    answer_calls: int,
    answers_passed: int,
    parse_failures: int,
    out_total: int,
    out_routed_to_stockfish: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    decision_events = ledger.find_by_type(EventType.LLM_ROUTING_DECIDED)
    fallback_events = ledger.find_by_type(EventType.ROUTING_FALLBACK)
    artifact_events = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    return {
        "model_id": model_id,
        "stockfish_binary": stockfish_binary,
        "chess_total": chess_total,
        "chess_routing_rate": chess_routed / chess_total if chess_total else 0.0,
        "chess_chain_success_rate": chains_passed / chess_total if chess_total else 0.0,
        "chess_answer_success_rate": answers_passed / chess_total if chess_total else 0.0,
        "out_of_domain_stockfish_rate": (out_routed_to_stockfish / out_total if out_total else 0.0),
        "chains_started": chains_started,
        "chains_passed": chains_passed,
        "answer_calls": answer_calls,
        "answers_passed": answers_passed,
        "answer_parse_failures": parse_failures,
        "decision_events": len(decision_events),
        "fallback_events": len(fallback_events),
        "artifact_events": len(artifact_events),
        "max_prompt_chars": max_prompt_chars,
        "total_entries": chess_total + out_total,
    }


def _json_candidates(response: str) -> list[str]:
    stripped = response.strip()
    candidates = [match.group(1).strip() for match in _FENCED_RE.finditer(stripped)]
    candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    return candidates


def _stockfish_claim_text(chain_answer_text: str) -> str:
    marker = " stockfish_chain_report="
    return chain_answer_text.split(marker, 1)[0].strip()


def _expected_bestmove(stockfish_claim_text: str) -> str:
    match = _BESTMOVE_RE.search(stockfish_claim_text)
    if not match:
        raise ValueError("Stockfish claim did not contain bestmove=<uci>")
    return match.group(1)


def _default_fixture_path() -> Path:
    packaged = Path(__file__).with_name("chess_answer_eval_v0.json")
    if packaged.exists():
        return packaged
    return Path(__file__).parents[1] / "tests" / "qb" / "fixtures" / "chess_answer_eval_v0.json"


if __name__ == "__main__":
    raise SystemExit(main())
