import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.repeat import triton_repeat_kv_heads


@pytest.mark.parametrize("B,Hkv,S,D,groups", [
    (1, 2, 5, 16, 2),
    (2, 4, 32, 64, 3),
    (1, 8, 1, 128, 4),
])
def test_repeat_heads(B, Hkv, S, D, groups):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(B, Hkv, S, D, device='cuda', dtype=torch.float32)
    torch_out = x.repeat_interleave(groups, dim=1)
    triton_out = triton_repeat_kv_heads(x, groups)

    assert torch.allclose(torch_out, triton_out, atol=1e-6)


def test_repeat_heads_single_value():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.arange(2*2*1*3, device='cuda', dtype=torch.float32).view(2, 2, 1, 3)
    out = triton_repeat_kv_heads(x, groups=2)
    ref = x.repeat_interleave(2, dim=1)

    assert torch.allclose(out, ref)


if __name__ == "__main__":
    test_repeat_heads(1, 2, 5, 16, 2)
    test_repeat_heads_single_value()
