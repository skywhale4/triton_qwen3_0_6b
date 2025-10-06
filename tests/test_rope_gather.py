import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rope_gather import triton_gather_rope


@pytest.mark.parametrize("S,D", [
    (1, 128),
    (10, 256),
    (32, 512),
])
def test_rope_gather(S, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    table = torch.randn(2048, D, device='cuda', dtype=torch.float32)
    positions = torch.randint(0, 2048, (S,), device='cuda', dtype=torch.int32)

    torch_out = table[positions]
    triton_out = triton_gather_rope(table, positions)

    assert torch.allclose(torch_out, triton_out, atol=1e-6)


def test_rope_gather_sorted():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    table = torch.randn(1024, 64, device='cuda', dtype=torch.float32)
    positions = torch.arange(0, 32, device='cuda', dtype=torch.int32)

    torch_out = table[positions]
    triton_out = triton_gather_rope(table, positions)

    assert torch.allclose(torch_out, triton_out)


