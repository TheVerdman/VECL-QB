from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from vecl._compat import UTC
from vecl._paths import environment_directory
from vecl.provenance.events import stable_hash
from vecl.qb.specialist import SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore
from vecl.specialists.base import ServiceSpecialist
from vecl.specialists.configuration import maybe_specialist_config_response
from vecl.trust.anchors import TrustAnchor, TrustRootKind

TIMESFM_SOURCE_ID = "timesfm-v2.5-200m-pytorch"
DEFAULT_TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"


@dataclass(frozen=True)
class ForecastSeries:
    series_id: str
    history: tuple[float, ...]
    horizon: int
    point_forecast: tuple[float, ...]
    q10: tuple[float, ...]
    q50: tuple[float, ...]
    q90: tuple[float, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "series_id": self.series_id,
            "history_length": len(self.history),
            "history_tail": list(self.history[-8:]),
            "horizon": self.horizon,
            "point_forecast": list(self.point_forecast),
            "q10": list(self.q10),
            "q50": list(self.q50),
            "q90": list(self.q90),
            "forecast_sum": sum(self.point_forecast),
            "forecast_max": max(self.point_forecast),
            "forecast_min": min(self.point_forecast),
            "last_observed": self.history[-1],
            "recent_trend": _recent_trend(self.history),
        }


@dataclass(frozen=True)
class TimesFMForecast:
    model_id: str
    horizon: int
    period: str
    series: tuple[ForecastSeries, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "horizon": self.horizon,
            "period": self.period,
            "series": [series.to_payload() for series in self.series],
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
        }


class TimesFMRunner(Protocol):
    model_id: str

    def forecast(
        self, *, series: list[tuple[str, list[float]]], horizon: int
    ) -> list[ForecastSeries]:
        raise NotImplementedError


class DeterministicTimesFMRunner:
    model_id = "deterministic-timesfm-test-runner"

    def forecast(
        self, *, series: list[tuple[str, list[float]]], horizon: int
    ) -> list[ForecastSeries]:
        forecasts: list[ForecastSeries] = []
        for series_id, values in series:
            trend = _recent_trend(tuple(values))
            current = values[-1]
            point = tuple(
                round(max(0.0, current + trend * step), 6) for step in range(1, horizon + 1)
            )
            q10 = tuple(round(value * 0.9, 6) for value in point)
            q90 = tuple(round(value * 1.1, 6) for value in point)
            forecasts.append(
                ForecastSeries(
                    series_id=series_id,
                    history=tuple(values),
                    horizon=horizon,
                    point_forecast=point,
                    q10=q10,
                    q50=point,
                    q90=q90,
                )
            )
        return forecasts


class LocalTimesFMRunner:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_TIMESFM_MODEL_ID,
        max_context: int = 1024,
        max_horizon: int = 256,
    ) -> None:
        self.model_id = model_id
        self.max_context = max_context
        self.max_horizon = max_horizon
        self._model: object | None = None

    def forecast(
        self, *, series: list[tuple[str, list[float]]], horizon: int
    ) -> list[ForecastSeries]:
        model = self._load_model(horizon)
        import numpy as np

        inputs = [
            np.asarray(values[-self.max_context :], dtype=np.float32) for _id, values in series
        ]
        point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=inputs)  # type: ignore[attr-defined]
        forecasts: list[ForecastSeries] = []
        for index, (series_id, values) in enumerate(series):
            point = _finite_tuple(point_forecast[index])
            q10 = _finite_tuple(quantile_forecast[index, :, 1])
            q50 = _finite_tuple(quantile_forecast[index, :, 5])
            q90 = _finite_tuple(quantile_forecast[index, :, 9])
            forecasts.append(
                ForecastSeries(
                    series_id=series_id,
                    history=tuple(values),
                    horizon=horizon,
                    point_forecast=point,
                    q10=q10,
                    q50=q50,
                    q90=q90,
                )
            )
        return forecasts

    def _load_model(self, horizon: int) -> object:
        if self._model is not None:
            return self._model
        import timesfm
        import torch

        torch.set_float32_matmul_precision("high")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=max(horizon, self.max_horizon),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        return model


class TimesFMSpecialist(ServiceSpecialist):
    def __init__(
        self,
        specialist_id: str = "timesfm",
        task_types: set[str] | tuple[str, ...] = ("demand_forecast", "tool_request"),
        *,
        version: str = "2.5-200m-pytorch",
        runner: TimesFMRunner | None = None,
        artifact_store: ContentAddressedStore | Path | str | None = None,
    ) -> None:
        super().__init__(specialist_id, task_types, version=f"timesfm-{version}")
        self.runner = runner or LocalTimesFMRunner(
            model_id=os.environ.get("TIMESFM_MODEL_ID", DEFAULT_TIMESFM_MODEL_ID)
        )
        if isinstance(artifact_store, ContentAddressedStore):
            self.artifact_store = artifact_store
        else:
            root = artifact_store or os.environ.get("VECL_ARTIFACT_STORE")
            self.artifact_store = ContentAddressedStore(
                root
                or environment_directory("VECL_ARTIFACT_STORE", prefix="vecl-timesfm-artifacts-")
            )

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def query(self, request: SpecialistRequest) -> SpecialistResponse:
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
                source_id=TIMESFM_SOURCE_ID,
                request=request,
                artifact_store=self.artifact_store,
            )
            if config_response is not None:
                return config_response
            series = _series_from_payload(request.input_payload)
            horizon = _positive_int(request.input_payload.get("horizon", 8), "horizon")
            period = str(request.input_payload.get("period") or "week")
            _validate_forecast_inputs(series, horizon)
            forecast = TimesFMForecast(
                model_id=self.runner.model_id,
                horizon=horizon,
                period=period,
                series=tuple(self.runner.forecast(series=series, horizon=horizon)),
                assumptions=("zero-shot forecast; no custom TimesFM fine-tuning",),
                limitations=(
                    "forecast is probabilistic and should not trigger side effects without review",
                    "inventory recommendation depends on caller-supplied stock policy",
                ),
            )
            records = self._write_artifacts(request, forecast)
            claim = self._claim_from_forecast(request, forecast, records)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
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
                "model_id": forecast.model_id,
                "horizon": forecast.horizon,
                "series_count": len(forecast.series),
            },
        )

    def _write_artifacts(
        self, request: SpecialistRequest, forecast: TimesFMForecast
    ) -> list[ArtifactRecord]:
        parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
        payload = forecast.to_payload()
        input_hash = stable_hash({"request": request.input_payload, "forecast": payload})
        return [
            self.artifact_store.write_text(
                json.dumps(payload, sort_keys=True, indent=2),
                producer_specialist_id=self.specialist_id,
                producer_version=self.version,
                input_hash=input_hash,
                output_format="json",
                parent_event_id=parent_event_id,
            ),
            self.artifact_store.write_text(
                _forecast_csv(forecast),
                producer_specialist_id=self.specialist_id,
                producer_version=self.version,
                input_hash=input_hash,
                output_format="csv",
                parent_event_id=parent_event_id,
            ),
        ]

    def _claim_from_forecast(
        self,
        request: SpecialistRequest,
        forecast: TimesFMForecast,
        records: list[ArtifactRecord],
    ) -> SpecialistClaim:
        payload = forecast.to_payload()
        artifact_ids = [record.artifact_id for record in records]
        claim_id = "timesfm-" + stable_hash({"payload": payload, "artifact_ids": artifact_ids})[:16]
        primary_series = forecast.series[0].to_payload()
        return SpecialistClaim(
            claim_id=claim_id,
            specialist_id=self.specialist_id,
            claim_text="timesfm_forecast="
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            claim_type="time_series_forecast",
            confidence=0.86,
            evidence_ids=[stable_hash(request.input_payload)],
            source_ids=[TIMESFM_SOURCE_ID],
            artifact_ids=artifact_ids,
            assumptions=list(forecast.assumptions),
            limitations=list(forecast.limitations)
            + [
                f"primary_series_forecast_sum={primary_series['forecast_sum']}",
                f"primary_series_forecast_max={primary_series['forecast_max']}",
            ],
        )


def create_timesfm_trust_anchor(
    model_id: str = DEFAULT_TIMESFM_MODEL_ID,
    version_name: str = "TimesFM 2.5 200M PyTorch",
    *,
    issued_at: datetime | None = None,
) -> TrustAnchor:
    issued = issued_at or datetime.now(UTC)
    credential_hash = stable_hash({"model_id": model_id, "version_name": version_name})
    return TrustAnchor(
        anchor_id=f"timesfm-v2.5-{credential_hash[:12]}",
        source_id=TIMESFM_SOURCE_ID,
        root_kind=TrustRootKind.VERIFIED_OPERATIONAL_RECORD,
        trust_value=0.9,
        issued_by="vecl-qb",
        issued_at=issued,
        expires_at=issued + timedelta(days=365),
        credential_hash=credential_hash,
    )


def _series_from_payload(payload: dict[str, Any]) -> list[tuple[str, list[float]]]:
    if "series" in payload and isinstance(payload["series"], list):
        parsed: list[tuple[str, list[float]]] = []
        for index, item in enumerate(payload["series"], start=1):
            if not isinstance(item, dict):
                raise ValueError("series entries must be objects")
            series_id = str(item.get("series_id") or item.get("id") or f"series-{index}")
            parsed.append((series_id, _numeric_values(item.get("values") or item.get("history"))))
        return parsed
    series_id = str(payload.get("series_id") or "demand")
    return [(series_id, _numeric_values(payload.get("values") or payload.get("history")))]


def _numeric_values(raw: Any) -> list[float]:
    if not isinstance(raw, list | tuple):
        raise ValueError("history or values must be a list of numbers")
    values = [float(value) for value in raw]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("time series values must be finite")
    return values


def _validate_forecast_inputs(series: list[tuple[str, list[float]]], horizon: int) -> None:
    if horizon > 256:
        raise ValueError("horizon must be <= 256 for Phase 7b")
    for series_id, values in series:
        if not series_id:
            raise ValueError("series_id must be non-empty")
        if len(values) < 8:
            raise ValueError("at least 8 historical observations are required")


def _positive_int(raw: Any, name: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _forecast_csv(forecast: TimesFMForecast) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["series_id", "step", "point_forecast", "q10", "q50", "q90", "period"])
    for series in forecast.series:
        for index, point in enumerate(series.point_forecast, start=1):
            writer.writerow(
                [
                    series.series_id,
                    index,
                    point,
                    series.q10[index - 1],
                    series.q50[index - 1],
                    series.q90[index - 1],
                    forecast.period,
                ]
            )
    return output.getvalue()


def _recent_trend(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    window = values[-min(6, len(values)) :]
    return (window[-1] - window[0]) / max(1, len(window) - 1)


def _finite_tuple(values: Any) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in parsed):
        raise ValueError("TimesFM forecast returned non-finite values")
    return parsed
