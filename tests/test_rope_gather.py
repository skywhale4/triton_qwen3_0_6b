import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rope_gather import triton_gather_rope
from tests import require_triton_device, to_triton, from_triton

CPU_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("S, D", [
    (9, 128),
    (32, 64),
])
def test_rope_gather_correctness(S, D):
    require_triton_device()

    torch.manual_seed(0)
    table = to_triton(torch.randn(2048, D, device=CPU_DEVICE, dtype=torch.float32))
    positions = to_triton(torch.randint(0, 2048, (S,), device=CPU_DEVICE, dtype=torch.int32))
    reference = table.cpu()[positions.cpu()]

    gathered = from_triton(triton_gather_rope(table, positions))
    assert torch.allclose(gathered, reference)


@pytest.mark.parametrize("S, D", [
    (9, 128),
    (32, 64),
])
def test_rope_gather_monotonic_positions(S, D):
    require_triton_device()

    torch.manual_seed(0)
    table = to_triton(torch.randn(1024, D, device=CPU_DEVICE, dtype=torch.float32))
    positions = to_triton(torch.arange(0, S * 4, 4, device=CPU_DEVICE, dtype=torch.int32))
    reference = table.cpu()[positions.cpu()]

    gathered = from_triton(triton_gather_rope(table, positions))
    assert torch.allclose(gathered, reference)


if __name__ == "__main__":
    pytest.main([__file__])

