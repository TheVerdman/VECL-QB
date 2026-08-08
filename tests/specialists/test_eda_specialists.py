from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.chain_executor import ChainExecutor
from vecl.qb.planner import ChainPlan, ChainStep
from vecl.qb.specialist import SpecialistRequest
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore
from vecl.specialists.openroad_specialist import (
    OPENROAD_SOURCE_ID,
    OpenROADSpecialist,
    create_openroad_trust_anchor,
    resolve_openroad_binary,
)
from vecl.specialists.yosys_specialist import (
    YOSYS_SOURCE_ID,
    YosysSpecialist,
    create_yosys_trust_anchor,
    resolve_yosys_binary,
)

TINY_RTL = """
module vecl_topk_helper(input wire [7:0] a, input wire [7:0] b, output wire [7:0] y);
  assign y = (a > b) ? a : b;
endmodule
"""

TINY_OPENROAD_NETLIST = """
module vecl_topk_helper();
endmodule
"""


def _request(
    task_type: str, payload: dict[str, object], *, parent_event_id: str = "evt-parent"
) -> SpecialistRequest:
    return SpecialistRequest(
        "req-eda",
        "tenant-eda",
        task_type,
        payload,
        {},
        {"parent_event_id": parent_event_id},
    )


def test_yosys_specialist_writes_netlist_json_log_and_script_artifacts(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "artifacts")
    specialist = YosysSpecialist(binary=_fake_yosys(tmp_path), artifact_store=store)

    response = specialist.run(
        _request(
            "hardware_synthesis",
            {
                "operation": "synthesize",
                "top_module": "vecl_topk_helper",
                "verilog_text": TINY_RTL,
            },
        )
    )

    assert response.refusal_or_error is None
    claim = response.claims[0]
    assert claim.claim_type == "hardware_synthesis"
    assert claim.source_ids == [YOSYS_SOURCE_ID]
    payload = json.loads(claim.claim_text.removeprefix("yosys_synthesis="))
    assert payload["top_module"] == "vecl_topk_helper"
    assert "Number of cells" in "\n".join(payload["stat_excerpt"])
    records = [
        ArtifactRecord.from_payload(raw)
        for raw in response.cost_metadata["artifact_records"]  # type: ignore[index]
    ]
    assert {record.output_format for record in records} == {"v", "json", "txt", "ys"}
    netlist = next(record for record in records if record.output_format == "v")
    assert "module vecl_topk_helper" in store.restore(netlist).decode()


def test_openroad_specialist_consumes_inline_netlist_and_writes_reports(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "artifacts")
    specialist = OpenROADSpecialist(binary=_fake_openroad(tmp_path), artifact_store=store)

    response = specialist.run(
        _request(
            "physical_design",
            {
                "operation": "analyze",
                "top_module": "vecl_topk_helper",
                "netlist_text": TINY_RTL,
            },
        )
    )

    assert response.refusal_or_error is None
    claim = response.claims[0]
    assert claim.claim_type == "physical_design"
    assert claim.source_ids == [OPENROAD_SOURCE_ID]
    payload = json.loads(claim.claim_text.removeprefix("openroad_physical_design="))
    assert payload["netlist_origin"]["kind"] == "inline"
    assert payload["report_names"] == ["area", "checks", "design"]
    records = [
        ArtifactRecord.from_payload(raw)
        for raw in response.cost_metadata["artifact_records"]  # type: ignore[index]
    ]
    assert {"txt", "tcl", "rpt"} <= {record.output_format for record in records}


def test_yosys_to_openroad_chain_hands_off_netlist_artifact(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "artifacts")
    yosys = YosysSpecialist(binary=_fake_yosys(tmp_path), artifact_store=store)
    openroad = OpenROADSpecialist(binary=_fake_openroad(tmp_path), artifact_store=store)
    ledger = ProvenanceLedger()
    parent = ledger.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant-eda", "test", {"request_id": "req"})
    )
    executor = ChainExecutor(
        ledger=ledger,
        specialists={"yosys": yosys, "openroad": openroad},
    )
    plan = ChainPlan(
        "eda-yosys-openroad",
        "eda_flow",
        (
            ChainStep("synthesize", "yosys", expected_artifact_type="v"),
            ChainStep(
                "physical",
                "openroad",
                inputs_from=("synthesize",),
                parameters={"operation": "analyze", "netlist_from": "synthesize"},
                expected_artifact_type="rpt",
            ),
        ),
    )

    result = executor.execute(
        plan,
        _request(
            "eda_flow",
            {
                "top_module": "vecl_topk_helper",
                "verilog_text": TINY_RTL,
            },
            parent_event_id=parent.event_id,
        ),
    )

    assert result.aborted is False
    assert [response.specialist_id for response in result.responses] == ["yosys", "openroad"]
    openroad_claim = result.responses[-1].claims[0]
    payload = json.loads(openroad_claim.claim_text.removeprefix("openroad_physical_design="))
    assert payload["netlist_origin"]["kind"] == "upstream_artifact"
    assert payload["netlist_origin"]["step_id"] == "synthesize"


def test_eda_trust_anchors_bind_binary_and_version(tmp_path: Path) -> None:
    yosys_anchor = create_yosys_trust_anchor(_fake_yosys(tmp_path), "Yosys 0.53")
    openroad_anchor = create_openroad_trust_anchor(_fake_openroad(tmp_path), "OpenROAD 2.0")

    assert yosys_anchor.source_id == YOSYS_SOURCE_ID
    assert yosys_anchor.trust_value == 0.9
    assert yosys_anchor.credential_hash
    assert openroad_anchor.source_id == OPENROAD_SOURCE_ID
    assert openroad_anchor.trust_value == 0.9
    assert openroad_anchor.credential_hash


def test_real_yosys_synthesizes_tiny_rtl_when_installed(tmp_path: Path) -> None:
    try:
        binary = resolve_yosys_binary()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    store = ContentAddressedStore(tmp_path / "artifacts")
    specialist = YosysSpecialist(binary=binary, artifact_store=store)

    response = specialist.run(
        _request(
            "hardware_synthesis",
            {
                "operation": "synthesize",
                "top_module": "vecl_topk_helper",
                "verilog_text": TINY_RTL,
            },
        )
    )

    assert response.refusal_or_error is None
    assert response.claims[0].claim_type == "hardware_synthesis"


def test_real_openroad_analyzes_tiny_netlist_when_enabled(tmp_path: Path) -> None:
    if os.environ.get("VECL_RUN_OPENROAD_TESTS") != "1":
        pytest.skip("set VECL_RUN_OPENROAD_TESTS=1 to run real OpenROAD integration")
    try:
        binary = resolve_openroad_binary()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    store = ContentAddressedStore(tmp_path / "artifacts")
    specialist = OpenROADSpecialist(binary=binary, artifact_store=store)
    tech_payload = _openroad_test_tech_payload()
    if tech_payload is None:
        pytest.skip("set VECL_OPENROAD_TEST_TECH_DIR to run real OpenROAD analyze")

    response = specialist.run(
        _request(
            "physical_design",
            {
                "operation": "analyze",
                "top_module": "vecl_topk_helper",
                "netlist_text": TINY_OPENROAD_NETLIST,
                **tech_payload,
            },
        )
    )

    assert response.refusal_or_error is None
    assert response.claims[0].claim_type == "physical_design"


def _fake_yosys(tmp_path: Path) -> Path:
    path = tmp_path / "yosys"
    path.write_text(
        """#!/bin/sh
cat > synthesized.v <<'EOF'
module vecl_topk_helper(input [7:0] a, input [7:0] b, output [7:0] y);
  assign y = a > b ? a : b;
endmodule
EOF
cat > design.json <<'EOF'
{"modules":{"vecl_topk_helper":{"cells":{"cmp":{"type":"$gt"}}}}}
EOF
echo "=== vecl_topk_helper ==="
echo "Number of cells: 1"
exit 0
""",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)
    return path


def _fake_openroad(tmp_path: Path) -> Path:
    path = tmp_path / "openroad"
    path.write_text(
        """#!/bin/sh
cat > area.rpt <<'EOF'
Design area 42.0 u^2
EOF
cat > checks.rpt <<'EOF'
No timing paths found in synthetic fixture
EOF
cat > design.rpt <<'EOF'
Units: synthetic
EOF
echo "OpenROAD synthetic fixture completed"
exit 0
""",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)
    return path


def _openroad_test_tech_payload() -> dict[str, object] | None:
    root_raw = os.environ.get("VECL_OPENROAD_TEST_TECH_DIR")
    if not root_raw:
        return None
    root = Path(root_raw).expanduser().resolve()
    liberty = root / "lib" / "NangateOpenCellLibrary_typical.lib"
    lef_files = [
        root / "lef" / "NangateOpenCellLibrary.tech.lef",
        root / "lef" / "NangateOpenCellLibrary.macro.lef",
    ]
    if not liberty.is_file() or not all(path.is_file() for path in lef_files):
        return None
    return {
        "liberty_files": [str(liberty)],
        "lef_files": [str(path) for path in lef_files],
    }
