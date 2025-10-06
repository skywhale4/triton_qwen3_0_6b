import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.cache import triton_kv_concat


def _make_tensors(B, H, S_old, S_new, D, dtype=torch.float32):
    past = torch.randn(B, H, S_old, D, device='cuda', dtype=dtype)
    curr = torch.randn(B, H, S_new, D, device='cuda', dtype=dtype)
    return past, curr


@pytest.mark.parametrize("B,H,S_old,S_new,D", [
    (1, 2, 5, 1, 16),
    (2, 4, 32, 8, 64),
    (1, 8, 1, 4, 128),
])
def test_kv_concat(B, H, S_old, S_new, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    past, curr = _make_tensors(B, H, S_old, S_new, D)

    torch_out = torch.cat([past, curr], dim=2)
    triton_out = triton_kv_concat(past, curr)

    assert torch.allclose(torch_out, triton_out, atol=1e-6)


if __name__ == "__main__":
    test_kv_concat(1, 2, 5, 1, 16)
