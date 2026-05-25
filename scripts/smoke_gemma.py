#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vecl._gemma import has_allowed_cuda_device, resolve_gemma_model_class, resolve_torch_dtype

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_PROMPT = "In one short paragraph, explain why provenance matters for learning systems."
DEFAULT_ALLOWED_CUDA_DEVICES = "A100,H100"


class SmokeGemmaConfig(BaseModel):
    """Validated runtime config for the manual Gemma smoke script."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(default=DEFAULT_MODEL_ID, min_length=1)
    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "bfloat16"
    device_map: str = Field(default="auto", min_length=1)
    hf_token: str | None = None
    require_cuda: bool = True
    allowed_cuda_devices: str = Field(default=DEFAULT_ALLOWED_CUDA_DEVICES, min_length=1)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SmokeGemmaConfig:
        env = env or os.environ
        values: dict[str, object] = {}
        env_map = {
            "model_id": "GEMMA_MODEL_ID",
            "prompt": "GEMMA_PROMPT",
            "max_new_tokens": "GEMMA_MAX_NEW_TOKENS",
            "dtype": "GEMMA_DTYPE",
            "device_map": "GEMMA_DEVICE_MAP",
            "hf_token": "HF_TOKEN",
            "require_cuda": "GEMMA_REQUIRE_CUDA",
            "allowed_cuda_devices": "GEMMA_ALLOWED_CUDA_DEVICES",
        }
        for field_name, env_name in env_map.items():
            if env_name in env:
                values[field_name] = env[env_name]
        return cls.model_validate(values)


def main() -> int:
    try:
        config = SmokeGemmaConfig.from_env()
    except ValidationError as exc:
        print(f"Invalid Gemma smoke configuration:\n{exc}", file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError:
        print(
            "Missing dependency: torch. Install project dependencies before running the Gemma smoke.",
            file=sys.stderr,
        )
        return 2

    if config.require_cuda and not has_allowed_cuda_device(torch, config.allowed_cuda_devices):
        print(
            "No approved CUDA device detected. Run this smoke on an approved A100/H100-class "
            "GPU, set GEMMA_ALLOWED_CUDA_DEVICES, or set GEMMA_REQUIRE_CUDA=false for local "
            "debugging.",
            file=sys.stderr,
        )
        return 2

    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        print(
            "Transformers could not import its processor stack. This usually means the GPU "
            "environment has incompatible torch/torchvision/transformers versions. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2

    model_cls = resolve_gemma_model_class()
    if model_cls is None:
        print(
            "Unsupported Transformers install: no Gemma 4 image/text generation auto-model class "
            "is available. Upgrade Transformers on the GPU environment.",
            file=sys.stderr,
        )
        return 2

    dtype = resolve_torch_dtype(torch, config.dtype)
    try:
        processor = AutoProcessor.from_pretrained(config.model_id, token=config.hf_token)
        model_kwargs = {"device_map": config.device_map, "token": config.hf_token}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = model_cls.from_pretrained(config.model_id, **model_kwargs)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": config.prompt}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
        input_len = inputs["input_ids"].shape[-1]
        text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            print(
                "Gemma smoke failed with CUDA out-of-memory. Use E4B, reduce context, or move "
                "to a larger GPU.",
                file=sys.stderr,
            )
            return 4
        print(f"Gemma smoke failed at runtime: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(
            "Gemma smoke could not access model files. Check HF_TOKEN, model access, and network "
            f"connectivity. Details: {exc}",
            file=sys.stderr,
        )
        return 3
    except Exception as exc:  # noqa: BLE001 - CLI smoke should preserve actionable failure text.
        print(f"Gemma smoke failed: {exc}", file=sys.stderr)
        return 3

    print(f"Model: {config.model_id}")
    print(f"Prompt: {config.prompt}")
    print("Output:")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
