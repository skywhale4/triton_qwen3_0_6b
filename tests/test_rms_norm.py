import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rms_norm import triton_rms_norm
from tests import require_triton_device, to_triton, from_triton


CPU_DEVICE = torch.device("cpu")


def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x_norm = (x32 * torch.rsqrt(var + eps))
    return (weight.to(torch.float32) * x_norm).to(x.dtype)


@pytest.mark.parametrize("shape", [
    (1, 1, 1024),
    (1, 9, 1024),
    (1, 128, 1024),
    (2, 16, 1024),
    (1, 1, 128),
])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rms_norm_correctness(shape, eps, dtype):

    hidden_size = shape[-1]

    require_triton_device()

    torch.manual_seed(0)
    x = to_triton(torch.randn(shape, device=CPU_DEVICE, dtype=dtype))
    weight = to_triton(torch.randn(hidden_size, device=CPU_DEVICE, dtype=dtype))

    torch_out = pytorch_rms_norm(x.cpu(), weight.cpu(), eps)

    triton_out = from_triton(triton_rms_norm(x, weight, eps)).to(torch.float32)

    tol = 1e-5 if dtype == torch.float32 else 6.5e-2
    assert torch.allclose(torch_out, triton_out, atol=tol, rtol=tol)


def test_rms_norm_stability():

    batch_size, seq_len, hidden_size = 4, 16, 1024

    require_triton_device()

    torch.manual_seed(0)
    x = to_triton(torch.randn(batch_size, seq_len, hidden_size, device=CPU_DEVICE, dtype=torch.float32) * 1e5)
    weight = to_triton(torch.ones(hidden_size, device=CPU_DEVICE, dtype=torch.float32))

    out = from_triton(triton_rms_norm(x, weight))
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__])

