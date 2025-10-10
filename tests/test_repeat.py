import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.repeat import triton_repeat_kv_heads
from tests import require_triton_device, to_triton, from_triton


CPU_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("shape, groups", [
    ((1, 8, 9, 128), 2),
    ((1, 16, 9, 128), 1),
    ((2, 8, 32, 128), 4),
])
def test_repeat_kv_heads(shape, groups):
    require_triton_device()

    torch.manual_seed(0)
    x_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=torch.float32)
    out_torch = x_cpu.repeat_interleave(groups, dim=1).to(torch.float32)

    out_triton = from_triton(triton_repeat_kv_heads(to_triton(x_cpu), groups)).to(torch.float32)

    assert torch.allclose(out_triton, out_torch)


def test_repeat_kv_heads_small_case():
    require_triton_device()

    x_cpu = torch.arange(2 * 2 * 1 * 3, device=CPU_DEVICE, dtype=torch.float32).view(2, 2, 1, 3)
    out_torch = x_cpu.repeat_interleave(2, dim=1).to(torch.float32)

    out_triton = from_triton(triton_repeat_kv_heads(to_triton(x_cpu), 2)).to(torch.float32)

    assert torch.allclose(out_triton, out_torch)


if __name__ == "__main__":
    pytest.main([__file__])

