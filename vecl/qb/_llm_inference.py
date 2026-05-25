from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from vecl._env import load_dotenv_if_present
from vecl._gemma import resolve_gemma_model_class, resolve_torch_dtype

DEFAULT_ROUTING_MODEL_ID = "google/gemma-4-31B-it"


@dataclass(frozen=True)
class RoutingInferenceConfig:
    model_id: str
    max_new_tokens: int
    dtype: Literal["auto", "bfloat16", "float16", "float32"]
    device_map: str
    hf_token: str | None

    @classmethod
    def from_env(cls) -> RoutingInferenceConfig:
        load_dotenv_if_present()
        return cls(
            model_id=os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID),
            max_new_tokens=int(os.environ.get("VECL_ROUTING_MAX_NEW_TOKENS", "192")),
            dtype=os.environ.get("VECL_ROUTING_DTYPE", "bfloat16"),  # type: ignore[arg-type]
            device_map=os.environ.get("VECL_ROUTING_DEVICE_MAP", "auto"),
            hf_token=os.environ.get("HF_TOKEN"),
        )


@dataclass
class _RoutingModelState:
    config: RoutingInferenceConfig | None = None
    processor: object | None = None
    model: object | None = None


_STATE = _RoutingModelState()


def gemma_route_once(prompt: str, config: RoutingInferenceConfig | None = None) -> str:
    config = config or RoutingInferenceConfig.from_env()
    processor, model = _load_model(config)
    import torch

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)  # type: ignore[attr-defined]
    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()  # type: ignore[attr-defined]


def _load_model(config: RoutingInferenceConfig) -> tuple[object, object]:
    if _STATE.config == config and _STATE.processor is not None and _STATE.model is not None:
        return _STATE.processor, _STATE.model

    import torch
    from transformers import AutoProcessor

    model_cls = resolve_gemma_model_class()
    if model_cls is None:
        raise RuntimeError("no Gemma-compatible Transformers auto-model class is available")
    dtype = resolve_torch_dtype(torch, config.dtype)
    model_kwargs: dict[str, object] = {
        "device_map": config.device_map,
        "token": config.hf_token,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    processor = AutoProcessor.from_pretrained(config.model_id, token=config.hf_token)
    model = model_cls.from_pretrained(config.model_id, **model_kwargs)
    _STATE.config = config
    _STATE.processor = processor
    _STATE.model = model
    return processor, model
