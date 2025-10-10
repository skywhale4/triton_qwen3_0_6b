import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.cache import triton_kv_concat
from tests import require_triton_device, to_triton, from_triton


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_kv_concat(dtype):
    B, H, S_old, S_new, D = 2, 4, 16, 8, 64
    torch.manual_seed(0)
    past = torch.randn(B, H, S_old, D, device=torch.device("cpu"), dtype=dtype)
    curr = torch.randn(B, H, S_new, D, device=torch.device("cpu"), dtype=dtype)

    ref = torch.cat([past, curr], dim=2)

    require_triton_device()
    out = from_triton(triton_kv_concat(to_triton(past), to_triton(curr)))

    tol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(out, ref, atol=tol, rtol=tol)


if __name__ == "__main__":
    pytest.main([__file__])

