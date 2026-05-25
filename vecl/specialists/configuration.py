from __future__ import annotations

import json
from typing import Any

from vecl.provenance.events import stable_hash
from vecl.qb.specialist import SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.qb.specialist_contracts import config_payload_for_specialist, is_config_request
from vecl.specialists.artifacts import ContentAddressedStore


def maybe_specialist_config_response(
    *,
    specialist_id: str,
    version: str,
    source_id: str,
    request: SpecialistRequest,
    artifact_store: ContentAddressedStore,
) -> SpecialistResponse | None:
    if not is_config_request(request.input_payload):
        return None
    payload = config_payload_for_specialist(specialist_id, request.input_payload)
    parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
    text = json.dumps(payload, sort_keys=True, indent=2)
    record = artifact_store.write_text(
        text,
        producer_specialist_id=specialist_id,
        producer_version=version,
        input_hash=stable_hash({"request": request.input_payload, "config": payload}),
        output_format="json",
        parent_event_id=parent_event_id,
    )
    claim_payload: dict[str, Any] = {
        "specialist_id": specialist_id,
        "operation": "configure",
        "artifact_id": record.artifact_id,
    }
    claim = SpecialistClaim(
        claim_id=f"{specialist_id}-config-" + stable_hash(claim_payload)[:16],
        specialist_id=specialist_id,
        claim_text="specialist_config="
        + json.dumps(claim_payload, sort_keys=True, separators=(",", ":")),
        claim_type="specialist_config",
        confidence=0.99,
        evidence_ids=[stable_hash(request.input_payload)],
        source_ids=[source_id],
        artifact_ids=[record.artifact_id],
        assumptions=["configuration template artifact; specialist execution was not run"],
    )
    return SpecialistResponse(
        request.request_id,
        specialist_id,
        [claim],
        request.tenant_id,
        cost_metadata={
            "artifact_records": [record.to_payload()],
            "operation": "configure",
            "execution_skipped": True,
        },
    )
