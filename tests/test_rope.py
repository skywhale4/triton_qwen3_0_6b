import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rope import triton_rope
from kernels.rope_gather import triton_gather_rope
from models.qwen3_torch import rotate_half
from tests import require_triton_device, to_triton, from_triton

CPU_DEVICE = torch.device("cpu")


def pytorch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


@pytest.mark.parametrize("shape", [
    (1, 16, 9, 128),
    (1, 16, 1, 128),
    (1, 8, 32, 128),
])
def test_rope_correctness(shape):
    require_triton_device()

    B, H, S, D = shape

    torch.manual_seed(0)
    x = to_triton(torch.randn(B, H, S, D, device=CPU_DEVICE, dtype=torch.float32))
    positions = to_triton(torch.arange(S, device=CPU_DEVICE, dtype=torch.long))
    inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, D, 2, device=CPU_DEVICE).float() / D))
    freqs = torch.einsum("i,j->ij", positions.cpu().float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = to_triton(torch.cos(emb).unsqueeze(0).unsqueeze(0))
    sin = to_triton(torch.sin(emb).unsqueeze(0).unsqueeze(0))

    with torch.no_grad():
        torch_out = pytorch_rope(x.cpu(), cos.cpu(), sin.cpu()).to(torch.float32)

    triton_out = from_triton(triton_rope(x, cos, sin))

    assert torch.allclose(torch_out, triton_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("S", [1, 9, 32])
def test_rope_gather(S):
    require_triton_device()

    D = 128
    positions = to_triton(torch.arange(S, device=CPU_DEVICE, dtype=torch.int32))
    inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, D, 2, device=CPU_DEVICE).float() / D))
    base = torch.arange(0, 2048, device=CPU_DEVICE).float()
    freqs = torch.einsum("i,j->ij", base, inv_freq)
    emb = to_triton(torch.cat((freqs, freqs), dim=-1))

    gathered = from_triton(triton_gather_rope(emb, positions))
    reference = emb.cpu()[positions.cpu()]

    assert torch.allclose(gathered, reference)


if __name__ == "__main__":
    pytest.main([__file__])

