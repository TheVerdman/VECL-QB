from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from vecl.provenance.events import EventType, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.planner import ChainPlan, ChainPlanner, ChainStep, topological_steps
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.blast_specialist import BLASTSpecialist, resolve_blast_binary
from vecl.specialists.stockfish import StockfishSpecialist
from vecl.specialists.sympy_specialist import SymPySpecialist
from vecl.specialists.timesfm_specialist import DeterministicTimesFMRunner, TimesFMSpecialist

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SUBJECT_SEQUENCE = "ATGCGTACGTAGCTAGCTAGCTAG"
DEMAND_HISTORY = [100, 104, 108, 112, 116, 120, 124, 128]


class JsonProducerSpecialist(Specialist):
    specialist_id = "producer"

    def __init__(self, store: ContentAddressedStore) -> None:
        self.store = store

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "synthetic_chain"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        value = str(request.input_payload.get("value", "unset"))
        record = self.store.write_text(
            f'{{"value":"{value}"}}',
            producer_specialist_id=self.specialist_id,
            producer_version="v1",
            input_hash=stable_hash({"value": value}),
            output_format="json",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            "claim-producer",
            self.specialist_id,
            f"produced={value}",
            "json",
            0.95,
            evidence_ids=[stable_hash({"value": value})],
            artifact_ids=[record.artifact_id],
        )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
            cost_metadata={"artifact_records": [record.to_payload()]},
        )


class JsonConsumerSpecialist(Specialist):
    specialist_id = "consumer"

    def __init__(self, store: ContentAddressedStore) -> None:
        self.store = store
        self.seen_payload: dict[str, Any] | None = None

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "synthetic_chain"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        self.seen_payload = dict(request.input_payload)
        upstream = request.input_payload["upstream"]["produce"]
        upstream_text = upstream["claims"][0]["claim_text"]
        record = self.store.write_text(
            f"report:{upstream_text}",
            producer_specialist_id=self.specialist_id,
            producer_version="v1",
            input_hash=stable_hash({"upstream": upstream}),
            output_format="txt",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            "claim-consumer",
            self.specialist_id,
            f"formatted {upstream_text}",
            "report",
            0.9,
            evidence_ids=[stable_hash({"upstream_claim_id": upstream["claim_ids"][0]})],
            artifact_ids=[record.artifact_id],
        )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
            cost_metadata={"artifact_records": [record.to_payload()]},
        )


class FailingSpecialist(Specialist):
    specialist_id = "failing"

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "synthetic_chain"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [],
            request.tenant_id,
            refusal_or_error="synthetic failure",
        )


class StockfishFormatterSpecialist(Specialist):
    specialist_id = "stockfish-formatter"

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "chess_eval"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        upstream = request.input_payload["upstream"]["analyze"]
        claim_text = upstream["claims"][0]["claim_text"]
        claim = SpecialistClaim(
            "claim-stockfish-format",
            self.specialist_id,
            f"json_report={{'analysis':'{claim_text}'}}",
            "chess_report",
            0.85,
            evidence_ids=[stable_hash({"upstream_claim_id": upstream["claim_ids"][0]})],
            artifact_ids=upstream["artifact_ids"],
        )
        return SpecialistResponse(
            request.request_id, self.specialist_id, [claim], request.tenant_id
        )


def _card(specialist_id: str, task_type: str = "synthetic_chain") -> SpecialistCard:
    return SpecialistCard(
        specialist_id,
        {task_type},
        {"trust_anchor_id": f"anchor-{specialist_id}"},
        cost_hint=1.0,
        latency_hint=1.0,
        version="v1",
        effective_trust=0.8,
    )


def _synthetic_plan(second_specialist_id: str = "consumer") -> ChainPlan:
    return ChainPlan(
        "synthetic-two-step",
        "synthetic_chain",
        (
            ChainStep("produce", "producer", expected_artifact_type="json"),
            ChainStep(
                "consume",
                second_specialist_id,
                inputs_from=("produce",),
                expected_artifact_type="txt",
            ),
        ),
    )


def _tiny_blast_db(tmp_path: Path) -> Path:
    fasta_path = tmp_path / "subjects.fasta"
    db_prefix = tmp_path / "blast-db" / "tiny_sequences"
    db_prefix.parent.mkdir()
    fasta_path.write_text(
        "\n".join(
            [
                ">subject1 known synthetic sequence",
                SUBJECT_SEQUENCE,
                ">subject2 unrelated synthetic sequence",
                "TTTTCCCCAAAAGGGGTTTTCCCC",
                "",
            ]
        )
    )
    completed = subprocess.run(
        [
            str(resolve_blast_binary("makeblastdb")),
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
    assert completed.returncode == 0, completed.stderr
    return db_prefix


def test_chain_planner_returns_registered_template_and_validates_specialists() -> None:
    plan = _synthetic_plan()
    planner = ChainPlanner({"synthetic_chain": plan})
    request = SpecialistRequest("req", "tenant", "synthetic_chain", {}, {}, {})

    assert planner.plan(request, {"producer", "consumer"}) == plan
    assert [step.step_id for step in topological_steps(plan)] == ["produce", "consume"]
    with pytest.raises(ValueError, match="unregistered specialists"):
        planner.plan(request, {"producer"})


def test_two_step_chain_records_step_and_artifact_provenance(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    producer = JsonProducerSpecialist(store)
    consumer = JsonConsumerSpecialist(store)
    router = QBRouter()
    router.register_specialist(_card("producer"), producer)
    router.register_specialist(_card("consumer"), consumer)
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"json", "report"}),
    )

    result = orchestrator.run_task(
        tenant_id="tenant",
        task_type="synthetic_chain",
        input_payload={"value": "alpha"},
        chain_plan=_synthetic_plan(),
    )

    assert result.verification_status == "PASSED"
    assert result.specialist_ids == ["producer", "consumer"]
    assert consumer.seen_payload is not None
    assert consumer.seen_payload["upstream"]["produce"]["claim_ids"] == ["claim-producer"]
    chain_started = ledger.find_by_type(EventType.CHAIN_STARTED)[0]
    completed = ledger.find_by_type(EventType.CHAIN_STEP_COMPLETED)
    artifacts = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    assert chain_started.parent_event_ids == [result.provenance_event_ids[0]]
    assert [event.payload["step_id"] for event in completed] == ["produce", "consume"]
    assert len(artifacts) == 2
    assert artifacts[0].parent_event_ids == [completed[0].event_id]
    assert artifacts[1].parent_event_ids == [completed[1].event_id]
    assert ledger.find_by_type(EventType.CHAIN_ABORTED) == []


def test_failed_chain_aborts_and_preserves_prior_artifact(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    producer = JsonProducerSpecialist(store)
    failing = FailingSpecialist()
    router = QBRouter()
    router.register_specialist(_card("producer"), producer)
    router.register_specialist(_card("failing"), failing)
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(router, ledger)

    result = orchestrator.run_task(
        tenant_id="tenant",
        task_type="synthetic_chain",
        input_payload={"value": "beta"},
        chain_plan=_synthetic_plan("failing"),
    )

    artifacts = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    failed = ledger.find_by_type(EventType.CHAIN_STEP_FAILED)[0]
    aborted = ledger.find_by_type(EventType.CHAIN_ABORTED)[0]
    assert result.verification_status == "FAILED"
    assert "synthetic failure" in result.answer_text
    assert len(artifacts) == 1
    assert Path(str(artifacts[0].payload["output_path"])).exists()
    assert failed.payload["step_id"] == "consume"
    assert aborted.parent_event_ids == [failed.event_id]


def test_real_stockfish_chain_can_feed_synthetic_postprocessor(tmp_path: Path) -> None:
    stockfish = StockfishSpecialist(artifact_store=ContentAddressedStore(tmp_path))
    formatter = StockfishFormatterSpecialist()
    router = QBRouter()
    router.register_specialist(_card("stockfish", "chess_eval"), stockfish)
    router.register_specialist(_card("stockfish-formatter", "chess_eval"), formatter)
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"chess_eval", "chess_report"}),
    )
    plan = ChainPlan(
        "stockfish-format",
        "chess_eval",
        (
            ChainStep(
                "analyze", "stockfish", parameters={"depth": 4}, expected_artifact_type="txt"
            ),
            ChainStep("format", "stockfish-formatter", inputs_from=("analyze",)),
        ),
    )

    result = orchestrator.run_task(
        tenant_id="tenant-chess",
        task_type="chess_eval",
        input_payload={"fen": STARTING_FEN},
        chain_plan=plan,
    )

    assert result.verification_status == "PASSED"
    assert result.specialist_ids == ["stockfish", "stockfish-formatter"]
    assert len(ledger.find_by_type(EventType.CHAIN_STEP_COMPLETED)) == 2
    assert len(ledger.find_by_type(EventType.ARTIFACT_PRODUCED)) == 1


def test_real_blast_chain_feeds_sympy_summary(tmp_path: Path) -> None:
    db_prefix = _tiny_blast_db(tmp_path)
    store = ContentAddressedStore(tmp_path / "artifacts")
    blast = BLASTSpecialist(artifact_store=store)
    sympy = SymPySpecialist(
        task_types=("sequence_alignment",),
        artifact_store=store,
    )
    router = QBRouter()
    router.register_specialist(_card("blast", "sequence_alignment"), blast)
    router.register_specialist(_card("sympy", "sequence_alignment"), sympy)
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"sequence_alignment", "symbolic_math"}),
    )
    plan = ChainPlan(
        "blast-sympy-score",
        "sequence_alignment",
        (
            ChainStep(
                "align",
                "blast",
                parameters={"max_target_seqs": 1, "task": "blastn-short"},
                expected_artifact_type="tsv",
            ),
            ChainStep(
                "score",
                "sympy",
                inputs_from=("align",),
                parameters={
                    "operation": "simplify",
                    "expression_from": "blast_bitscore_sum",
                },
                expected_artifact_type="tex",
            ),
        ),
    )

    result = orchestrator.run_task(
        tenant_id="tenant-bio",
        task_type="sequence_alignment",
        input_payload={"database": str(db_prefix), "query_sequence": SUBJECT_SEQUENCE},
        chain_plan=plan,
    )

    artifacts = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    assert result.verification_status == "PASSED"
    assert result.specialist_ids == ["blast", "sympy"]
    assert len(ledger.find_by_type(EventType.CHAIN_STEP_COMPLETED)) == 2
    assert len(artifacts) == 2
    assert [event.payload["output_format"] for event in artifacts] == ["tsv", "tex"]


def test_timesfm_chain_feeds_sympy_cumulative_demand(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "artifacts")
    timesfm = TimesFMSpecialist(
        runner=DeterministicTimesFMRunner(),
        task_types=("demand_forecast",),
        artifact_store=store,
    )
    sympy = SymPySpecialist(
        task_types=("demand_forecast",),
        artifact_store=store,
    )
    router = QBRouter()
    router.register_specialist(_card("timesfm", "demand_forecast"), timesfm)
    router.register_specialist(_card("sympy", "demand_forecast"), sympy)
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"time_series_forecast", "symbolic_math"}),
    )
    plan = ChainPlan(
        "timesfm-sympy-demand",
        "demand_forecast",
        (
            ChainStep("forecast", "timesfm", expected_artifact_type="json"),
            ChainStep(
                "cumulative",
                "sympy",
                inputs_from=("forecast",),
                parameters={
                    "operation": "simplify",
                    "expression_from": "timesfm_forecast_sum",
                },
                expected_artifact_type="tex",
            ),
        ),
    )

    result = orchestrator.run_task(
        tenant_id="tenant-demand",
        task_type="demand_forecast",
        input_payload={
            "query": "Forecast demand for inventory planning.",
            "series_id": "sku-123",
            "history": DEMAND_HISTORY,
            "horizon": 4,
            "period": "week",
        },
        chain_plan=plan,
    )

    artifacts = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    assert result.verification_status == "PASSED"
    assert result.specialist_ids == ["timesfm", "sympy"]
    assert len(ledger.find_by_type(EventType.CHAIN_STEP_COMPLETED)) == 2
    assert [event.payload["output_format"] for event in artifacts] == ["json", "csv", "tex"]
    assert "result=552.000000000000" in result.answer_text
