import pytest
from pydantic import ValidationError

from scripts.smoke_gemma import DEFAULT_ALLOWED_CUDA_DEVICES, DEFAULT_MODEL_ID, SmokeGemmaConfig


def test_smoke_gemma_config_defaults() -> None:
    config = SmokeGemmaConfig.from_env({})
    assert config.model_id == DEFAULT_MODEL_ID
    assert config.max_new_tokens == 64
    assert config.dtype == "bfloat16"
    assert config.device_map == "auto"
    assert config.hf_token is None
    assert config.require_cuda
    assert config.allowed_cuda_devices == DEFAULT_ALLOWED_CUDA_DEVICES


def test_smoke_gemma_config_env_overrides_and_coercion() -> None:
    config = SmokeGemmaConfig.from_env(
        {
            "GEMMA_MODEL_ID": "google/gemma-4-E4B",
            "GEMMA_PROMPT": "Say hello.",
            "GEMMA_MAX_NEW_TOKENS": "12",
            "GEMMA_DTYPE": "float16",
            "GEMMA_DEVICE_MAP": "cuda:0",
            "HF_TOKEN": "hf_test",
            "GEMMA_REQUIRE_CUDA": "false",
            "GEMMA_ALLOWED_CUDA_DEVICES": "A100",
        }
    )
    assert config.model_id == "google/gemma-4-E4B"
    assert config.prompt == "Say hello."
    assert config.max_new_tokens == 12
    assert config.dtype == "float16"
    assert config.device_map == "cuda:0"
    assert config.hf_token == "hf_test"
    assert not config.require_cuda
    assert config.allowed_cuda_devices == "A100"


@pytest.mark.parametrize(
    "env",
    [
        {"GEMMA_MAX_NEW_TOKENS": "0"},
        {"GEMMA_MAX_NEW_TOKENS": "513"},
        {"GEMMA_DTYPE": "int8"},
        {"GEMMA_MODEL_ID": ""},
        {"GEMMA_ALLOWED_CUDA_DEVICES": ""},
    ],
)
def test_smoke_gemma_config_rejects_invalid_values(env: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        SmokeGemmaConfig.from_env(env)
