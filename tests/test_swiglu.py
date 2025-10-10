import torch
import torch.nn.functional as F
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.swiglu import triton_swiglu_activation, triton_silu
from tests import require_triton_device, to_triton, from_triton

CPU_DEVICE = torch.device("cpu")


def pytorch_swiglu_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


@pytest.mark.parametrize("shape", [
    (1, 1, 3072),
    (1, 9, 3072),
    (1, 128, 3072),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swiglu_activation_correctness(shape, dtype):
    require_triton_device()

    torch.manual_seed(0)
    gate = to_triton(torch.randn(shape, device=CPU_DEVICE, dtype=dtype))
    up = to_triton(torch.randn(shape, device=CPU_DEVICE, dtype=dtype))
    torch_out = pytorch_swiglu_activation(gate.cpu(), up.cpu()).to(torch.float32)

    triton_out = from_triton(triton_swiglu_activation(gate, up)).to(torch.float32)

    tol = 2e-6 if dtype == torch.float32 else 8e-3
    assert torch.allclose(torch_out, triton_out, atol=tol)


@pytest.mark.parametrize("shape", [
    (1, 1, 3072),
    (1, 9, 3072),
    (2, 16, 3072),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_silu_correctness(shape, dtype):
    require_triton_device()

    torch.manual_seed(0)
    x = to_triton(torch.randn(shape, device=CPU_DEVICE, dtype=dtype))
    torch_out = F.silu(x.cpu().to(torch.float32)).to(torch.float32)

    triton_out = from_triton(triton_silu(x)).to(torch.float32)

    if dtype == torch.float32:
        assert torch.allclose(torch_out, triton_out, atol=1e-6)
        return

    diff = (torch_out - triton_out).abs()
    max_diff = diff.max().item()
    assert max_diff < 1e-2


def test_swiglu_edge_cases():
    require_triton_device()

    gate_low = to_triton(torch.full((1, 1, 100), -10.0, device=CPU_DEVICE))
    gate_high = to_triton(torch.full((1, 1, 100), 10.0, device=CPU_DEVICE))
    up = to_triton(torch.ones((1, 1, 100), device=CPU_DEVICE))

    torch_low = pytorch_swiglu_activation(gate_low.cpu(), up.cpu()).to(torch.float32)
    torch_high = pytorch_swiglu_activation(gate_high.cpu(), up.cpu()).to(torch.float32)

    low_acc = from_triton(triton_swiglu_activation(gate_low, up)).to(torch.float32)
    high_acc = from_triton(triton_swiglu_activation(gate_high, up)).to(torch.float32)

    assert torch.allclose(torch_low, low_acc, atol=1e-5)
    assert torch.allclose(torch_high, high_acc, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])

