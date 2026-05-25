from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from vecl._compat import UTC
from vecl.provenance.events import stable_hash
from vecl.qb.specialist import SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore
from vecl.specialists.base import LibrarySpecialist
from vecl.specialists.configuration import maybe_specialist_config_response
from vecl.trust.anchors import TrustAnchor, TrustRootKind

SYMPY_SOURCE_ID = "sympy-v1.14"
SUPPORTED_OPERATIONS = {
    "differentiate",
    "factor",
    "integrate",
    "plot",
    "simplify",
    "solve",
}


@dataclass(frozen=True)
class SymPyResult:
    operation: str
    expression: str
    result_text: str
    latex_text: str
    png_content: bytes | None
    limitations: tuple[str, ...] = ()


class SymPySpecialist(LibrarySpecialist):
    def __init__(
        self,
        specialist_id: str = "sympy",
        task_types: set[str] | tuple[str, ...] = ("symbolic_math", "sequence_math"),
        *,
        version: str = "1.14",
        artifact_store: ContentAddressedStore | Path | str | None = None,
    ) -> None:
        super().__init__(specialist_id, task_types, version=f"sympy-{version}")
        if isinstance(artifact_store, ContentAddressedStore):
            self.artifact_store = artifact_store
        else:
            root = artifact_store or os.environ.get("VECL_ARTIFACT_STORE")
            self.artifact_store = ContentAddressedStore(
                root or Path(os.environ.get("TMPDIR", "/tmp")) / "vecl-qb-artifacts"
            )

    def call(self, request: SpecialistRequest) -> SpecialistResponse:
        if not self.can_handle(request):
            return SpecialistResponse(
                request.request_id,
                self.specialist_id,
                [],
                request.tenant_id,
                refusal_or_error=f"unsupported task_type: {request.task_type}",
            )
        try:
            config_response = maybe_specialist_config_response(
                specialist_id=self.specialist_id,
                version=self.version,
                source_id=SYMPY_SOURCE_ID,
                request=request,
                artifact_store=self.artifact_store,
            )
            if config_response is not None:
                return config_response
            result = _run_sympy(request)
            records = self._write_artifacts(request, result)
            claim = self._claim_from_result(request, result, records)
        except (ValueError, TypeError) as exc:
            return SpecialistResponse(
                request.request_id,
                self.specialist_id,
                [],
                request.tenant_id,
                refusal_or_error=str(exc),
            )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
            cost_metadata={
                "artifact_records": [record.to_payload() for record in records],
                "operation": result.operation,
                "expression": result.expression,
            },
        )

    def _write_artifacts(
        self, request: SpecialistRequest, result: SymPyResult
    ) -> list[ArtifactRecord]:
        parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
        input_hash = stable_hash(
            {
                "operation": result.operation,
                "expression": result.expression,
                "request": request.input_payload,
            }
        )
        records = [
            self.artifact_store.write_text(
                result.latex_text,
                producer_specialist_id=self.specialist_id,
                producer_version=self.version,
                input_hash=input_hash,
                output_format="tex",
                parent_event_id=parent_event_id,
            )
        ]
        if result.png_content is not None:
            records.append(
                self.artifact_store.write_bytes(
                    result.png_content,
                    producer_specialist_id=self.specialist_id,
                    producer_version=self.version,
                    input_hash=input_hash,
                    output_format="png",
                    parent_event_id=parent_event_id,
                )
            )
        return records

    def _claim_from_result(
        self,
        request: SpecialistRequest,
        result: SymPyResult,
        records: list[ArtifactRecord],
    ) -> SpecialistClaim:
        artifact_ids = [record.artifact_id for record in records]
        claim_id = (
            "sympy-"
            + stable_hash(
                {
                    "operation": result.operation,
                    "expression": result.expression,
                    "result": result.result_text,
                    "artifact_ids": artifact_ids,
                }
            )[:16]
        )
        return SpecialistClaim(
            claim_id=claim_id,
            specialist_id=self.specialist_id,
            claim_text=(
                f"operation={result.operation}; expression={result.expression}; "
                f"result={result.result_text}"
            ),
            claim_type="symbolic_math",
            confidence=0.95,
            evidence_ids=[stable_hash(request.input_payload)],
            source_ids=[SYMPY_SOURCE_ID],
            artifact_ids=artifact_ids,
            limitations=list(result.limitations),
        )


def create_sympy_trust_anchor(
    version_name: str = "SymPy 1.14",
    *,
    issued_at: datetime | None = None,
) -> TrustAnchor:
    issued = issued_at or datetime.now(UTC)
    credential_hash = stable_hash({"package": "sympy", "version_name": version_name})
    return TrustAnchor(
        anchor_id=f"sympy-v1.14-{credential_hash[:12]}",
        source_id=SYMPY_SOURCE_ID,
        root_kind=TrustRootKind.VERIFIED_OPERATIONAL_RECORD,
        trust_value=0.95,
        issued_by="vecl-qb",
        issued_at=issued,
        expires_at=issued + timedelta(days=365),
        credential_hash=credential_hash,
    )


def _run_sympy(request: SpecialistRequest) -> SymPyResult:
    import sympy as sp

    operation = str(request.input_payload.get("operation") or "simplify").strip().lower()
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported SymPy operation: {operation}")
    expression_text = _expression_from_request(request)
    expression = sp.sympify(expression_text)
    variable = _symbol_for_request(request, expression)

    if operation == "simplify":
        result = sp.simplify(expression)
    elif operation == "factor":
        result = sp.factor(expression)
    elif operation == "differentiate":
        result = sp.diff(expression, variable)
    elif operation == "integrate":
        result = sp.integrate(expression, variable)
    elif operation == "solve":
        result = sp.solve(expression, variable)
    else:
        result = expression

    latex_body = _latex_document(
        operation=operation,
        expression_latex=sp.latex(expression),
        result_latex=sp.latex(result),
    )
    png_content, limitations = _maybe_plot_png(request, operation, expression, variable)
    return SymPyResult(
        operation=operation,
        expression=expression_text,
        result_text=sp.sstr(result),
        latex_text=latex_body,
        png_content=png_content,
        limitations=limitations,
    )


def _expression_from_request(request: SpecialistRequest) -> str:
    expression = request.input_payload.get("expression")
    if expression is not None and str(expression).strip():
        return str(expression)
    expression_from = str(request.input_payload.get("expression_from") or "").strip()
    if expression_from == "blast_bitscore_sum":
        return _blast_bitscore_sum_expression(request.input_payload)
    if expression_from == "timesfm_forecast_sum":
        return _timesfm_forecast_expression(request.input_payload, metric="forecast_sum")
    if expression_from == "timesfm_forecast_max":
        return _timesfm_forecast_expression(request.input_payload, metric="forecast_max")
    raise ValueError("expression is required")


def _blast_bitscore_sum_expression(input_payload: dict[str, Any]) -> str:
    upstream = input_payload.get("upstream")
    if not isinstance(upstream, dict):
        raise ValueError("upstream BLAST output is required for blast_bitscore_sum")
    for step_payload in upstream.values():
        claims = step_payload.get("claims") if isinstance(step_payload, dict) else None
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("claim_text") or "")
            if text.startswith("blast_hits="):
                hits = json.loads(text.removeprefix("blast_hits="))
                bitscores = [str(hit["bitscore"]) for hit in hits.get("top_hits", [])]
                return " + ".join(bitscores) if bitscores else "0"
    raise ValueError("upstream BLAST claim did not contain blast_hits JSON")


def _timesfm_forecast_expression(input_payload: dict[str, Any], *, metric: str) -> str:
    upstream = input_payload.get("upstream")
    if not isinstance(upstream, dict):
        raise ValueError("upstream TimesFM output is required")
    for step_payload in upstream.values():
        claims = step_payload.get("claims") if isinstance(step_payload, dict) else None
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("claim_text") or "")
            if text.startswith("timesfm_forecast="):
                forecast = json.loads(text.removeprefix("timesfm_forecast="))
                series = forecast.get("series", [])
                if not series:
                    return "0"
                primary = series[0]
                if metric in primary:
                    return str(primary[metric])
                values = primary.get("point_forecast", [])
                if metric == "forecast_sum":
                    return " + ".join(str(value) for value in values) if values else "0"
                if metric == "forecast_max":
                    return str(max(float(value) for value in values)) if values else "0"
    raise ValueError("upstream TimesFM claim did not contain forecast JSON")


def _symbol_for_request(request: SpecialistRequest, expression: Any) -> Any:
    import sympy as sp

    variable = request.input_payload.get("variable")
    if variable:
        return sp.Symbol(str(variable))
    symbols = sorted(expression.free_symbols, key=lambda symbol: symbol.name)
    if symbols:
        return symbols[0]
    return sp.Symbol("x")


def _latex_document(*, operation: str, expression_latex: str, result_latex: str) -> str:
    return "\n".join(
        [
            "% Generated by VECL-QB SymPySpecialist",
            "\\documentclass{article}",
            "\\usepackage{amsmath}",
            "\\begin{document}",
            f"\\textbf{{Operation:}} {operation}",
            "\\[",
            expression_latex,
            "\\]",
            "\\[",
            result_latex,
            "\\]",
            "\\end{document}",
            "",
        ]
    )


def _maybe_plot_png(
    request: SpecialistRequest, operation: str, expression: Any, variable: Any
) -> tuple[bytes | None, tuple[str, ...]]:
    if operation != "plot" and not bool(request.input_payload.get("plot_png")):
        return None, ()
    try:
        import sympy as sp
    except ImportError:
        return None, ("SymPy plotting unavailable",)
    try:
        x_min = float(request.input_payload.get("x_min", -5))
        x_max = float(request.input_payload.get("x_max", 5))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sympy-plot.png"
            plot = sp.plot(expression, (variable, x_min, x_max), show=False)
            plot.save(str(output_path))
            plot.close()
            return output_path.read_bytes(), ()
    except Exception as exc:
        return None, (f"PNG plot unavailable: {exc}",)
