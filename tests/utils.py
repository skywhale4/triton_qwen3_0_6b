import os
import torch
import pytest


TRITON_DEVICE = os.getenv("TRITON_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_IS_CPU = TRITON_DEVICE == "cpu"


def triton_device() -> torch.device:
    return torch.device(TRITON_DEVICE)


def using_cuda() -> bool:
    return TRITON_DEVICE.startswith("cuda")


def require_triton_device() -> None:
    if using_cuda() and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def to_triton(tensor: torch.Tensor) -> torch.Tensor:
    if _IS_CPU:
        return tensor
    return tensor.to(triton_device())


def clone_to_triton(tensor: torch.Tensor) -> torch.Tensor:
    return to_triton(tensor.clone())


def from_triton(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.cpu()


__all__ = [
    "TRITON_DEVICE",
    "using_cuda",
    "require_triton_device",
    "to_triton",
    "clone_to_triton",
    "from_triton",
    "triton_device",
]

