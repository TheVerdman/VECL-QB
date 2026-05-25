# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.claim_graph import ClaimGraph
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.qb.verifier import VerificationPolicy
from vecl.runtime.metrics import claim_graph_metrics


class _BaseFreightSpecialist(Specialist):
    specialist_id = "base"
    claim_type = "base"
    text = "base"

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "freight_decision"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [
                SpecialistClaim(
                    f"{self.specialist_id}-claim",
                    self.specialist_id,
                    self.text,
                    self.claim_type,
                    0.82,
                    evidence_ids=[f"evidence-{self.specialist_id}"],
                    source_ids=[f"source-{self.specialist_id}"],
                )
            ],
            request.tenant_id,
        )


class CarrierQualificationSpecialist(_BaseFreightSpecialist):
    specialist_id = "carrier-qualification"
    claim_type = "carrier_qualification"
    text = "Carrier is qualified for pharma freight."


class TemperatureControlSpecialist(_BaseFreightSpecialist):
    specialist_id = "temperature-control"
    claim_type = "temperature_control"
    text = "Carrier has validated cold-chain temperature control."

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        if request.input_payload.get("missing_temperature_qualification"):
            text = "Reject: carrier is missing temperature qualification."
        else:
            text = self.text
        claim = SpecialistClaim(
            "temperature-control-claim",
            self.specialist_id,
            text,
            self.claim_type,
            0.85,
            evidence_ids=["temperature-cert"],
            source_ids=["qa-system"],
        )
        return SpecialistResponse(
            request.request_id, self.specialist_id, [claim], request.tenant_id
        )


class LaneHistorySpecialist(_BaseFreightSpecialist):
    specialist_id = "lane-history"
    claim_type = "lane_history"
    text = "Lane history is reliable for the delivery deadline."


class RateSpecialist(_BaseFreightSpecialist):
    specialist_id = "rate"
    claim_type = "rate"
    text = "Accept: carrier has the lowest compliant-looking rate."


class ComplianceSpecialist(_BaseFreightSpecialist):
    specialist_id = "compliance"
    claim_type = "compliance"
    text = "Compliance documents are present."


def build_orchestrator() -> QBOrchestrator:
    router = QBRouter()
    for specialist in [
        CarrierQualificationSpecialist(),
        TemperatureControlSpecialist(),
        LaneHistorySpecialist(),
        RateSpecialist(),
        ComplianceSpecialist(),
    ]:
        router.register_specialist(
            SpecialistCard(
                specialist.specialist_id,
                {"freight_decision"},
                {},
                cost_hint=1.0,
                latency_hint=1.0,
                version="toy-v0",
                effective_trust=0.9,
            ),
            specialist,
        )
    return QBOrchestrator(
        router,
        ProvenanceLedger(),
        policy=VerificationPolicy(
            required_claim_types={"carrier_qualification", "temperature_control", "compliance"}
        ),
    )


def run_case(name: str, bad_temperature: bool = False) -> None:
    result = build_orchestrator().run_task(
        tenant_id="magellan-demo",
        task_type="freight_decision",
        input_payload={
            "lane": "Boston -> Chicago",
            "temperature_requirement": "2-8C",
            "carrier": "ToyCarrier",
            "deadline": "Friday 17:00",
            "missing_temperature_qualification": bad_temperature,
            "contradictions": [("rate-claim", "temperature-control-claim")]
            if bad_temperature
            else [],
        },
    )
    print(f"\n{name}")
    print("Auditable recommendation:")
    print(result.answer_text)
    print("Verification status:", result.verification_status)
    print("Claim graph:", result.claim_graph_hash)
    print("Provenance event ids:", ", ".join(result.provenance_event_ids))
    print("Metrics:", claim_graph_metrics(ClaimGraph(), [result.verification_status]))


def main() -> None:
    run_case("Accepted case")
    run_case("Bad case: rate advantage but missing temperature qualification", bad_temperature=True)


if __name__ == "__main__":
    main()
