from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from vecl._env import load_dotenv_if_present
from vecl.qb._llm_inference import (
    DEFAULT_ROUTING_MODEL_ID,
    RoutingInferenceConfig,
    gemma_route_once,
)
from vecl.qb.prompted_router import (
    RoutingDecision,
    build_routing_prompt_from_cards,
    parse_routing_response,
)
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import SpecialistRequest


@dataclass(frozen=True)
class ModelDriverResult:
    provider: str
    model_id: str
    raw_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRouteResult:
    decision: RoutingDecision
    response: ModelDriverResult


@runtime_checkable
class ModelDriver(Protocol):
    provider: str
    model_id: str

    def route(
        self, request: SpecialistRequest, specialist_cards: Sequence[SpecialistCard]
    ) -> ModelRouteResult:
        raise NotImplementedError

    def synthesize(self, prompt: str) -> ModelDriverResult:
        raise NotImplementedError


class TextGenerationDriver:
    provider: str
    model_id: str

    def route(
        self, request: SpecialistRequest, specialist_cards: Sequence[SpecialistCard]
    ) -> ModelRouteResult:
        prompt = build_routing_prompt_from_cards(list(specialist_cards), request)
        response = self.synthesize(prompt)
        return ModelRouteResult(
            decision=parse_routing_response(response.raw_text),
            response=response,
        )

    def synthesize(self, prompt: str) -> ModelDriverResult:
        raise NotImplementedError


@dataclass(frozen=True)
class GemmaDriverConfig:
    model_id: str = DEFAULT_ROUTING_MODEL_ID
    max_new_tokens: int = 192
    dtype: str = "bfloat16"
    device_map: str = "auto"
    hf_token: str | None = None

    @classmethod
    def from_env(cls) -> GemmaDriverConfig:
        load_dotenv_if_present()
        return cls(
            model_id=os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID),
            max_new_tokens=int(os.environ.get("VECL_ROUTING_MAX_NEW_TOKENS", "192")),
            dtype=os.environ.get("VECL_ROUTING_DTYPE", "bfloat16"),
            device_map=os.environ.get("VECL_ROUTING_DEVICE_MAP", "auto"),
            hf_token=os.environ.get("HF_TOKEN"),
        )


class GemmaDriver(TextGenerationDriver):
    provider = "gemma"

    def __init__(
        self,
        config: GemmaDriverConfig | None = None,
        inference_fn: Callable[[str, RoutingInferenceConfig], str] | None = None,
    ) -> None:
        self.config = config or GemmaDriverConfig.from_env()
        self.model_id = self.config.model_id
        self._inference_fn = inference_fn or gemma_route_once

    def synthesize(self, prompt: str) -> ModelDriverResult:
        start = time.perf_counter()
        config = RoutingInferenceConfig(
            model_id=self.config.model_id,
            max_new_tokens=self.config.max_new_tokens,
            dtype=self.config.dtype,  # type: ignore[arg-type]
            device_map=self.config.device_map,
            hf_token=self.config.hf_token,
        )
        raw_text = self._inference_fn(prompt, config)
        return ModelDriverResult(
            provider=self.provider,
            model_id=self.model_id,
            raw_text=raw_text,
            latency_ms=_elapsed_ms(start),
        )


@dataclass(frozen=True)
class OpenAIResponsesConfig:
    model_id: str
    api_key: str
    max_output_tokens: int = 1024
    timeout_seconds: float = 60.0
    endpoint: str = "https://api.openai.com/v1/responses"

    @classmethod
    def from_env(cls) -> OpenAIResponsesConfig:
        load_dotenv_if_present()
        model_id = os.environ.get("VECL_OPENAI_MODEL", "").strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not model_id:
            raise ValueError("VECL_OPENAI_MODEL must be set for the OpenAI model driver")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for the OpenAI model driver")
        return cls(
            model_id=model_id,
            api_key=api_key,
            max_output_tokens=int(os.environ.get("VECL_OPENAI_MAX_OUTPUT_TOKENS", "1024")),
            timeout_seconds=float(os.environ.get("VECL_OPENAI_TIMEOUT_SECONDS", "60")),
        )


class OpenAIResponsesDriver(TextGenerationDriver):
    provider = "openai"

    def __init__(
        self,
        config: OpenAIResponsesConfig | None = None,
        transport: Callable[[urllib.request.Request, float], bytes] | None = None,
    ) -> None:
        self.config = config or OpenAIResponsesConfig.from_env()
        self.model_id = self.config.model_id
        self._transport = transport or _urlopen_bytes

    def synthesize(self, prompt: str) -> ModelDriverResult:
        start = time.perf_counter()
        payload = {
            "model": self.config.model_id,
            "input": prompt,
            "max_output_tokens": self.config.max_output_tokens,
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response = _read_json(self._transport(request, self.config.timeout_seconds))
        raw_text = _openai_output_text(response)
        return ModelDriverResult(
            provider=self.provider,
            model_id=str(response.get("model") or self.model_id),
            raw_text=raw_text,
            usage=dict(response.get("usage") or {}),
            latency_ms=_elapsed_ms(start),
            finish_reason=str(response.get("status") or "") or None,
            metadata={
                "response_id": response.get("id"),
                "incomplete_details": response.get("incomplete_details"),
                "estimated_cost_usd": estimate_model_cost_usd(
                    self.provider,
                    str(response.get("model") or self.model_id),
                    dict(response.get("usage") or {}),
                ),
            },
        )


@dataclass(frozen=True)
class AnthropicMessagesConfig:
    model_id: str
    api_key: str
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    endpoint: str = "https://api.anthropic.com/v1/messages"
    anthropic_version: str = "2023-06-01"

    @classmethod
    def from_env(cls) -> AnthropicMessagesConfig:
        load_dotenv_if_present()
        model_id = os.environ.get("VECL_ANTHROPIC_MODEL", "").strip()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not model_id:
            raise ValueError("VECL_ANTHROPIC_MODEL must be set for the Anthropic model driver")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set for the Anthropic model driver")
        return cls(
            model_id=model_id,
            api_key=api_key,
            max_tokens=int(os.environ.get("VECL_ANTHROPIC_MAX_TOKENS", "1024")),
            timeout_seconds=float(os.environ.get("VECL_ANTHROPIC_TIMEOUT_SECONDS", "60")),
        )


class AnthropicMessagesDriver(TextGenerationDriver):
    provider = "anthropic"

    def __init__(
        self,
        config: AnthropicMessagesConfig | None = None,
        transport: Callable[[urllib.request.Request, float], bytes] | None = None,
    ) -> None:
        self.config = config or AnthropicMessagesConfig.from_env()
        self.model_id = self.config.model_id
        self._transport = transport or _urlopen_bytes

    def synthesize(self, prompt: str) -> ModelDriverResult:
        start = time.perf_counter()
        payload = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": self.config.anthropic_version,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response = _read_json(self._transport(request, self.config.timeout_seconds))
        raw_text = _anthropic_output_text(response)
        return ModelDriverResult(
            provider=self.provider,
            model_id=str(response.get("model") or self.model_id),
            raw_text=raw_text,
            usage=dict(response.get("usage") or {}),
            latency_ms=_elapsed_ms(start),
            finish_reason=str(response.get("stop_reason") or "") or None,
            metadata={
                "response_id": response.get("id"),
                "estimated_cost_usd": estimate_model_cost_usd(
                    self.provider,
                    str(response.get("model") or self.model_id),
                    dict(response.get("usage") or {}),
                ),
            },
        )


def resolve_model_driver(name: str | None = None) -> ModelDriver:
    load_dotenv_if_present()
    selected = (name or os.environ.get("VECL_MODEL_DRIVER") or "gemma").strip().lower()
    if selected in {"", "gemma"}:
        return validate_model_driver(GemmaDriver())
    if selected == "openai":
        return validate_model_driver(OpenAIResponsesDriver())
    if selected == "anthropic":
        return validate_model_driver(AnthropicMessagesDriver())
    raise ValueError(f"unknown VECL model driver: {selected}")


def validate_model_driver(driver: object) -> ModelDriver:
    if not isinstance(driver, ModelDriver):
        raise TypeError("model driver must provide provider, model_id, route, and synthesize")
    if not str(driver.provider).strip():
        raise TypeError("model driver provider must be non-empty")
    if not str(driver.model_id).strip():
        raise TypeError("model driver model_id must be non-empty")
    return driver


def estimate_model_cost_usd(provider: str, model_id: str, usage: dict[str, Any]) -> float | None:
    del model_id
    pricing = _pricing_from_env(provider)
    if pricing is None:
        return None
    input_tokens = _token_count(usage, "input_tokens")
    output_tokens = _token_count(usage, "output_tokens")
    return round(
        (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000,
        8,
    )


def _pricing_from_env(provider: str) -> dict[str, float] | None:
    prefix = f"VECL_{provider.upper()}_"
    input_rate = _env_float(prefix + "INPUT_USD_PER_1M")
    output_rate = _env_float(prefix + "OUTPUT_USD_PER_1M")
    if input_rate is None or output_rate is None:
        return None
    return {"input": input_rate, "output": output_rate}


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _token_count(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _openai_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "\n".join(part for part in parts if part).strip()


def _anthropic_output_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in response.get("content") or []:
        if isinstance(content, dict) and content.get("type") == "text":
            parts.append(str(content.get("text") or ""))
    return "\n".join(part for part in parts if part).strip()


def _read_json(raw: bytes) -> dict[str, Any]:
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict):
        raise RuntimeError("model provider returned non-object JSON")
    return payload


def _urlopen_bytes(request: urllib.request.Request, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"model provider HTTP {exc.code}: {body}") from exc


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
