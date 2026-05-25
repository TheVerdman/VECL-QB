from __future__ import annotations

import json
from pathlib import Path

from vecl.qb.specialist import SpecialistRequest
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.timesfm_specialist import (
    TIMESFM_SOURCE_ID,
    DeterministicTimesFMRunner,
    TimesFMSpecialist,
    create_timesfm_trust_anchor,
)

DEMAND_HISTORY = [100, 104, 108, 112, 116, 120, 124, 128]


def _request(payload: dict[str, object]) -> SpecialistRequest:
    return SpecialistRequest(
        "req-timesfm",
        "tenant-demand",
        "demand_forecast",
        payload,
        {},
        {"parent_event_id": "evt-parent"},
    )


def test_timesfm_forecast_writes_json_and_csv_artifacts(tmp_path: Path) -> None:
    specialist = TimesFMSpecialist(
        runner=DeterministicTimesFMRunner(),
        artifact_store=ContentAddressedStore(tmp_path),
    )

    response = specialist.run(
        _request(
            {
                "series_id": "sku-123",
                "history": DEMAND_HISTORY,
                "horizon": 4,
                "period": "week",
            }
        )
    )

    assert response.refusal_or_error is None
    claim = response.claims[0]
    assert claim.claim_type == "time_series_forecast"
    assert claim.source_ids == [TIMESFM_SOURCE_ID]
    payload = json.loads(claim.claim_text.removeprefix("timesfm_forecast="))
    assert payload["series"][0]["series_id"] == "sku-123"
    assert payload["series"][0]["point_forecast"] == [132.0, 136.0, 140.0, 144.0]
    records = response.cost_metadata["artifact_records"]  # type: ignore[index]
    assert [record["output_format"] for record in records] == ["json", "csv"]
    assert "point_forecast" in Path(str(records[0]["output_path"])).read_text()
    assert "series_id,step,point_forecast" in Path(str(records[1]["output_path"])).read_text()


def test_timesfm_rejects_too_short_history(tmp_path: Path) -> None:
    specialist = TimesFMSpecialist(
        runner=DeterministicTimesFMRunner(),
        artifact_store=ContentAddressedStore(tmp_path),
    )

    response = specialist.run(_request({"history": [1, 2, 3], "horizon": 2}))

    assert response.claims == []
    assert response.refusal_or_error == "at least 8 historical observations are required"


def test_timesfm_trust_anchor() -> None:
    anchor = create_timesfm_trust_anchor()

    assert anchor.source_id == TIMESFM_SOURCE_ID
    assert anchor.root_kind.value == "VERIFIED_OPERATIONAL_RECORD"
    assert anchor.trust_value == 0.9
    assert anchor.credential_hash
