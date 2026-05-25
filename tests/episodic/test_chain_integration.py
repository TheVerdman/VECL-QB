from __future__ import annotations

from pathlib import Path

from vecl.episodic.store import DeterministicEmbedder, EpisodicStore
from vecl.provenance.events import EventType, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.planner import ChainPlan, ChainStep
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore


class EpisodicProducer(Specialist):
    specialist_id = "episodic-producer"

    def __init__(self, store: ContentAddressedStore) -> None:
        self.store = store

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "episodic_chain"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        value = str(request.input_payload["value"])
        record = self.store.write_text(
            f"producer artifact {value}",
            producer_specialist_id=self.specialist_id,
            producer_version="v1",
            input_hash=stable_hash({"value": value}),
            output_format="txt",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            "claim-producer",
            self.specialist_id,
            f"produced chess analysis {value}",
            "producer_claim",
            0.9,
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


class EpisodicConsumer(Specialist):
    specialist_id = "episodic-consumer"

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "episodic_chain"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        upstream = request.input_payload["upstream"]["produce"]
        claim = SpecialistClaim(
            "claim-consumer",
            self.specialist_id,
            f"formatted {upstream['claims'][0]['claim_text']}",
            "consumer_claim",
            0.9,
            evidence_ids=[stable_hash({"upstream": upstream["claim_ids"][0]})],
            artifact_ids=upstream["artifact_ids"],
        )
        return SpecialistResponse(
            request.request_id, self.specialist_id, [claim], request.tenant_id
        )


def test_chain_executor_writes_episodic_entries_and_links_artifacts(tmp_path: Path) -> None:
    artifact_store = ContentAddressedStore(tmp_path / "artifacts")
    episodic_store = EpisodicStore.local(
        tmp_path / "episodes",
        embedder=DeterministicEmbedder(dimensions=32),
    )
    router = QBRouter()
    router.register_specialist(_card("episodic-producer"), EpisodicProducer(artifact_store))
    router.register_specialist(_card("episodic-consumer"), EpisodicConsumer())
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"producer_claim", "consumer_claim"}),
        episodic_store=episodic_store,
    )

    result = orchestrator.run_task(
        tenant_id="tenant-episodic",
        task_type="episodic_chain",
        input_payload={"value": "alpha"},
        chain_plan=ChainPlan(
            "episodic-two-step",
            "episodic_chain",
            (
                ChainStep("produce", "episodic-producer", expected_artifact_type="txt"),
                ChainStep("consume", "episodic-consumer", inputs_from=("produce",)),
            ),
        ),
    )

    completed = ledger.find_by_type(EventType.CHAIN_STEP_COMPLETED)
    episodic_events = ledger.find_by_type(EventType.EPISODIC_ENTRY_WRITTEN)
    artifacts = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    assert result.verification_status == "PASSED"
    assert len(completed) == 2
    assert len(episodic_events) == 2
    assert episodic_events[0].parent_event_ids == [completed[0].event_id]
    assert episodic_events[1].parent_event_ids == [completed[1].event_id]
    assert len(artifacts) == 1
    assert artifacts[0].parent_event_ids == [episodic_events[0].event_id]
    assert artifacts[0].payload["episodic_entry_id"] == episodic_events[0].payload["entry_id"]
    assert result.provenance_event_ids == [
        event.event_id for event in ledger.query_by_tenant("tenant-episodic")
    ]

    retrieved = episodic_store.retrieve_text(
        "formatted chess analysis", k=5, tenant_id="tenant-episodic"
    )
    assert [entry.metadata["step_id"] for entry in retrieved] == ["consume", "produce"]
    assert all(entry.tenant_id == "tenant-episodic" for entry in retrieved)


def _card(specialist_id: str) -> SpecialistCard:
    return SpecialistCard(
        specialist_id,
        {"episodic_chain"},
        {"trust_anchor_id": f"anchor-{specialist_id}"},
        cost_hint=1.0,
        latency_hint=1.0,
        version="v1",
        effective_trust=0.8,
    )
