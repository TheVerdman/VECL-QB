from __future__ import annotations

from vecl._gemma import has_allowed_cuda_device, resolve_torch_dtype


def test_resolve_torch_dtype_maps_supported_values() -> None:
    torch_module = _TorchDTypes()

    assert resolve_torch_dtype(torch_module, "auto") is None
    assert resolve_torch_dtype(torch_module, "bfloat16") == "bf16"
    assert resolve_torch_dtype(torch_module, "float16") == "fp16"
    assert resolve_torch_dtype(torch_module, "float32") == "fp32"


def test_has_allowed_cuda_device_matches_any_configured_name() -> None:
    torch_module = _TorchCuda(["NVIDIA A100-SXM4-80GB", "L4"])

    assert has_allowed_cuda_device(torch_module, "H100,A100")
    assert not has_allowed_cuda_device(torch_module, "H100")


def test_has_allowed_cuda_device_returns_false_without_cuda() -> None:
    assert not has_allowed_cuda_device(object(), "A100")


class _TorchDTypes:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"


class _TorchCuda:
    def __init__(self, names: list[str]) -> None:
        self.cuda = _Cuda(names)


class _Cuda:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.names)

    def get_device_name(self, index: int) -> str:
        return self.names[index]
