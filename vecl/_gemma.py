from __future__ import annotations

from typing import Any


def has_allowed_cuda_device(torch_module: Any, allowed_devices: str) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return False
    allowed = [name.strip().upper() for name in allowed_devices.split(",") if name.strip()]
    device_count = cuda.device_count()
    return any(
        any(allowed_name in cuda.get_device_name(index).upper() for allowed_name in allowed)
        for index in range(device_count)
    )


def resolve_torch_dtype(torch_module: Any, dtype: str) -> Any | None:
    if dtype == "auto":
        return None
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[dtype]


def resolve_gemma_model_class() -> Any | None:
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        pass
    try:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM
    except ImportError:
        return None
