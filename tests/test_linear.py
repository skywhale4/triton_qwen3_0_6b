import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.linear import triton_linear
from tests import require_triton_device, to_triton, from_triton


CPU_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("in_feat,out_feat,seq_len", [
    (1024, 2048, 9),
    (1024, 1024, 9),
    (1024, 3072, 9),
    (3072, 1024, 9),
    (1024, 151936, 1),
])
def test_linear_correctness(in_feat, out_feat, seq_len):
    require_triton_device()
    torch.manual_seed(0)
    x_cpu = torch.randn(1, seq_len, in_feat, device=CPU_DEVICE, dtype=torch.float32)
    linear_torch = nn.Linear(in_feat, out_feat, bias=False, device=CPU_DEVICE)

    with torch.no_grad():
        output_torch = linear_torch(x_cpu)

    x_acc = to_triton(x_cpu)
    weight_acc = to_triton(linear_torch.weight)

    output_triton = from_triton(triton_linear(x_acc, weight_acc))
    
    max_diff = (output_torch - output_triton).abs().max().item()
    rel_err = max_diff / output_torch.abs().max().item()
    
    assert max_diff < 0.1, f"Max diff too large: {max_diff}"


def test_linear_with_bias():
    require_triton_device()
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 9, 1024, device=CPU_DEVICE)
    linear_torch = nn.Linear(1024, 2048, bias=True, device=CPU_DEVICE)

    with torch.no_grad():
        output_torch = linear_torch(x_cpu)

    x_acc = to_triton(x_cpu)
    w_acc = to_triton(linear_torch.weight)
    b_acc = to_triton(linear_torch.bias)

    output_triton = from_triton(triton_linear(x_acc, w_acc, b_acc))
    
    max_diff = (output_torch - output_triton).abs().max().item()
    
    assert max_diff < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
