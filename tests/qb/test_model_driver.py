from __future__ import annotations

import json
import urllib.request

from vecl.qb.model_driver import (
    AnthropicMessagesConfig,
    AnthropicMessagesDriver,
    GemmaDriver,
    GemmaDriverConfig,
    ModelDriver,
    ModelDriverResult,
    OpenAIResponsesConfig,
    OpenAIResponsesDriver,
    estimate_model_cost_usd,
    resolve_model_driver,
    validate_model_driver,
)
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import SpecialistRequest


def _request() -> SpecialistRequest:
    return SpecialistRequest(
        "req-driver",
        "tenant",
        "chess_eval",
        {"query": "What is the best chess move?"},
        {},
        {"parent_event_id": "evt-request"},
    )


def _cards() -> list[SpecialistCard]:
    return [
        SpecialistCard(
            "stockfish",
            {"chess_eval"},
            {},
            cost_hint=1.0,
            latency_hint=1.0,
            version="stockfish-18",
            description="Analyze chess positions.",
        )
    ]


def test_gemma_driver_wraps_existing_inference_path() -> None:
    seen: dict[str, object] = {}

    def inference(prompt: str, config: object) -> str:
        seen["prompt"] = prompt
        seen["config"] = config
        return '{"tool":"stockfish","confidence":0.9,"reasoning":"chess"}'

    driver = GemmaDriver(
        GemmaDriverConfig(model_id="local-gemma", max_new_tokens=16, hf_token="token"),
        inference_fn=inference,
    )

    result = driver.route(_request(), _cards())

    assert result.decision.specialist_id == "stockfish"
    assert result.response.provider == "gemma"
    assert result.response.model_id == "local-gemma"
    assert "stockfish" in str(seen["prompt"])


def test_openai_responses_driver_posts_prompt_and_extracts_output_text(monkeypatch) -> None:
    monkeypatch.setenv("VECL_OPENAI_INPUT_USD_PER_1M", "1.25")
    monkeypatch.setenv("VECL_OPENAI_OUTPUT_USD_PER_1M", "10")
    captured: dict[str, object] = {}

    def transport(request: urllib.request.Request, timeout: float) -> bytes:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())  # type: ignore[union-attr]
        captured["auth_header"] = request.headers["Authorization"]
        return json.dumps(
            {
                "id": "resp-1",
                "model": "gpt-5.5-2026-04-23",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"tool":"stockfish","confidence":0.8,"reasoning":"chess"}',
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()

    driver = OpenAIResponsesDriver(
        OpenAIResponsesConfig(
            model_id="gpt-5.5-2026-04-23",
            api_key="key",
            max_output_tokens=64,
            endpoint="https://example.test/v1/responses",
        ),
        transport=transport,
    )

    result = driver.route(_request(), _cards())

    assert result.decision.specialist_id == "stockfish"
    assert result.response.usage["output_tokens"] == 5
    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["auth_header"] == "Bearer key"
    assert captured["body"]["model"] == "gpt-5.5-2026-04-23"  # type: ignore[index]
    assert captured["body"]["max_output_tokens"] == 64  # type: ignore[index]
    assert result.response.metadata["estimated_cost_usd"] == 0.0000625


def test_anthropic_messages_driver_posts_prompt_and_extracts_text_content(monkeypatch) -> None:
    monkeypatch.setenv("VECL_ANTHROPIC_INPUT_USD_PER_1M", "15")
    monkeypatch.setenv("VECL_ANTHROPIC_OUTPUT_USD_PER_1M", "75")
    captured: dict[str, object] = {}

    def transport(request: urllib.request.Request, timeout: float) -> bytes:
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())  # type: ignore[union-attr]
        captured["api_key_header"] = request.headers["X-api-key"]
        captured["version_header"] = request.headers["Anthropic-version"]
        return json.dumps(
            {
                "id": "msg-1",
                "model": "claude-opus-4-7",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": '{"tool":"stockfish","confidence":0.7,"reasoning":"chess"}',
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 6},
            }
        ).encode()

    driver = AnthropicMessagesDriver(
        AnthropicMessagesConfig(
            model_id="claude-opus-4-7",
            api_key="key",
            max_tokens=80,
            endpoint="https://example.test/v1/messages",
        ),
        transport=transport,
    )

    result = driver.route(_request(), _cards())

    assert result.decision.specialist_id == "stockfish"
    assert result.response.finish_reason == "end_turn"
    assert captured["api_key_header"] == "key"
    assert captured["version_header"] == "2023-06-01"
    assert captured["body"]["max_tokens"] == 80  # type: ignore[index]
    assert result.response.metadata["estimated_cost_usd"] == 0.000615


def test_resolver_defaults_to_gemma_when_driver_is_blank(monkeypatch) -> None:
    monkeypatch.setenv("VECL_MODEL_DRIVER", "")
    monkeypatch.setenv("VECL_ROUTING_MODEL_ID", "gemma-test")

    driver = resolve_model_driver()

    assert isinstance(driver, GemmaDriver)
    assert driver.model_id == "gemma-test"


def test_result_envelope_keeps_raw_output_visible() -> None:
    result = ModelDriverResult(
        provider="openai",
        model_id="model",
        raw_text='{"tool":null}',
        usage={"input_tokens": 1},
        latency_ms=12,
        finish_reason="completed",
    )

    assert result.raw_text == '{"tool":null}'
    assert result.usage["input_tokens"] == 1
    assert result.latency_ms == 12


def test_model_driver_protocol_is_runtime_checkable() -> None:
    driver = GemmaDriver(
        GemmaDriverConfig(model_id="local-gemma"),
        inference_fn=lambda _prompt, _config: '{"tool":null}',
    )

    assert isinstance(driver, ModelDriver)
    assert validate_model_driver(driver) is driver


def test_validate_model_driver_rejects_missing_contract() -> None:
    class MissingProvider:
        model_id = "model"

        def route(self, request, specialist_cards):  # noqa: ANN001
            raise NotImplementedError

        def synthesize(self, prompt):  # noqa: ANN001
            raise NotImplementedError

    try:
        validate_model_driver(MissingProvider())
    except TypeError as exc:
        assert "provider" in str(exc)
    else:
        raise AssertionError("validate_model_driver should reject missing provider")


def test_estimate_model_cost_returns_none_without_configured_rates(monkeypatch) -> None:
    monkeypatch.delenv("VECL_OPENAI_INPUT_USD_PER_1M", raising=False)
    monkeypatch.delenv("VECL_OPENAI_OUTPUT_USD_PER_1M", raising=False)

    assert estimate_model_cost_usd("openai", "unknown", {"input_tokens": 1}) is None
