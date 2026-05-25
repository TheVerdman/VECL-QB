from __future__ import annotations

from copy import deepcopy
from typing import Any

CONFIG_OPERATIONS = frozenset({"configure", "config", "configuration", "schema", "template"})


SPECIALIST_PAYLOAD_CONTRACTS: dict[str, dict[str, Any]] = {
    "stockfish": {
        "task_types": ["chess_eval"],
        "execute": {
            "required": {
                "fen": "six-field Forsyth-Edwards Notation string",
                "depth": "positive integer search depth when the user requests one",
            },
            "optional": {"movetime_ms": "positive integer time budget in milliseconds"},
            "output_artifacts": ["txt UCI transcript"],
        },
        "configure": {
            "operation": "configure",
            "template": {"fen": "<six-field FEN>", "depth": 12, "movetime_ms": None},
        },
    },
    "sympy": {
        "task_types": ["symbolic_math", "sequence_math"],
        "execute": {
            "required": {
                "operation": "one of differentiate, factor, integrate, plot, simplify, solve",
                "expression": "exact symbolic expression from the user request",
            },
            "optional": {"variable": "symbol variable when needed"},
            "output_artifacts": ["tex", "optional png for plots"],
        },
        "configure": {
            "operation": "configure",
            "template": {"operation": "simplify", "expression": "(x + 1)^2"},
        },
    },
    "blast": {
        "task_types": ["sequence_alignment"],
        "execute": {
            "required": {
                "database": "local BLAST database path or name supplied by the caller/runtime",
                "query_sequence_or_fasta": "query_sequence, query_sequences, or query_fasta",
            },
            "optional": {
                "task": "blastn-short or blastn",
                "evalue": "string or number e-value threshold",
                "max_target_seqs": "positive integer hit cap",
            },
            "output_artifacts": ["tsv BLAST format 6 results"],
        },
        "configure": {
            "operation": "configure",
            "template": {
                "database": "<local-blast-db>",
                "query_fasta": ">query1\\nACGTACGTACGT",
                "task": "blastn-short",
                "evalue": "1e-5",
                "max_target_seqs": 10,
            },
        },
    },
    "terraform": {
        "task_types": ["infrastructure_plan"],
        "execute": {
            "required": {"operation": "plan or validate"},
            "runtime_controlled": {
                "config_dir": "do not invent; VECL supplies this from trusted runtime config"
            },
            "optional": {"variables": "JSON object of Terraform variable values"},
            "forbidden": ["apply", "destroy", "state mutation", "workspace mutation"],
            "output_artifacts": ["json Terraform plan or validate output"],
        },
        "configure": {
            "operation": "configure",
            "template": {"operation": "plan", "variables": {"name": "demo"}},
        },
    },
    "timesfm": {
        "task_types": ["demand_forecast", "tool_request"],
        "execute": {
            "required": {
                "values_or_series": "values, history, or series of numeric observations",
                "horizon": "positive integer forecast horizon",
            },
            "optional": {
                "period": "semantic interval label such as day, week, or month",
                "current_inventory": "caller-supplied stock level for downstream inventory math",
            },
            "output_artifacts": ["json forecast payload", "csv forecast table"],
        },
        "configure": {
            "operation": "configure",
            "template": {
                "values": [120, 128, 131, 129, 136, 142, 145, 151],
                "horizon": 4,
                "period": "week",
                "current_inventory": 600,
            },
        },
    },
    "yosys": {
        "task_types": ["hardware_synthesis", "eda_flow"],
        "execute": {
            "required": {
                "top_module": "Verilog top module name",
                "rtl_source": "one of verilog_text, verilog_sources, or rtl_files",
            },
            "optional": {
                "operation": "synthesize",
                "filename": "filename for inline verilog_text, default input.v",
            },
            "output_artifacts": [
                "v synthesized Verilog netlist",
                "json Yosys design JSON",
                "txt Yosys log/stat report",
                "ys Yosys script",
            ],
        },
        "configure": {
            "operation": "configure",
            "template": {
                "operation": "synthesize",
                "top_module": "vecl_topk_helper",
                "verilog_text": "module vecl_topk_helper(input logic clk); endmodule",
            },
        },
    },
    "openroad": {
        "task_types": ["physical_design", "eda_flow"],
        "execute": {
            "required": {
                "top_module": "linked top module name",
                "netlist": "netlist_path, netlist_text, or upstream Yosys netlist artifact",
            },
            "optional": {
                "operation": "analyze, floorplan, or place_route",
                "netlist_from": "upstream chain step id that produced the netlist",
                "liberty_files": "technology liberty file path or list",
                "lef_files": "technology LEF file path or list",
                "sdc_file": "constraints file path",
                "die_area": "OpenROAD die area string for floorplan/place_route",
                "core_area": "OpenROAD core area string for floorplan/place_route",
            },
            "output_artifacts": [
                "txt OpenROAD log",
                "tcl OpenROAD script",
                "rpt reports",
                "optional def floorplan/layout",
            ],
        },
        "configure": {
            "operation": "configure",
            "template": {
                "operation": "analyze",
                "top_module": "vecl_topk_helper",
                "netlist_from": "synthesize",
                "liberty_files": [],
                "lef_files": [],
            },
        },
    },
}


def contract_for_specialist(specialist_id: str) -> dict[str, Any] | None:
    contract = SPECIALIST_PAYLOAD_CONTRACTS.get(specialist_id)
    return deepcopy(contract) if contract is not None else None


def payload_contracts_for_specialists(
    specialist_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    return {
        specialist_id: deepcopy(SPECIALIST_PAYLOAD_CONTRACTS[specialist_id])
        for specialist_id in sorted(specialist_ids)
        if specialist_id in SPECIALIST_PAYLOAD_CONTRACTS
    }


def is_config_request(payload: dict[str, Any]) -> bool:
    operation = (
        str(
            payload.get("operation")
            or payload.get("command")
            or payload.get("mode")
            or payload.get("intent")
            or ""
        )
        .strip()
        .lower()
    )
    return operation in CONFIG_OPERATIONS or bool(payload.get("configuration_template"))


def config_payload_for_specialist(
    specialist_id: str, request_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    contract = contract_for_specialist(specialist_id)
    if contract is None:
        raise ValueError(f"no payload contract is registered for specialist: {specialist_id}")
    request_payload = request_payload or {}
    return {
        "specialist_id": specialist_id,
        "operation": "configure",
        "contract": contract,
        "template": contract["configure"]["template"],
        "caller_query": str(request_payload.get("query") or ""),
    }
