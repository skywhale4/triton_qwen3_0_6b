import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.matmul import triton_matmul
from tests import require_triton_device, to_triton, from_triton

CPU_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M,N,K", [
    (16, 16, 16),
    (64, 64, 64),
    (128, 128, 128),
    (9, 1024, 128),
    (9, 9, 128),
])
def test_matmul_correctness(M, N, K, dtype):
    require_triton_device()
    tol = 0.05 if dtype == torch.float32 else 0.1

    torch.manual_seed(0)
    a_cpu = torch.randn(M, K, device=CPU_DEVICE, dtype=dtype)
    b_cpu = torch.randn(K, N, device=CPU_DEVICE, dtype=dtype)

    with torch.no_grad():
        c_torch = torch.matmul(a_cpu, b_cpu)

    c_triton = from_triton(triton_matmul(to_triton(a_cpu), to_triton(b_cpu)))
    
    max_diff = (c_torch - c_triton).abs().max().item()
    mean_diff = (c_torch - c_triton).abs().mean().item()
    
    assert max_diff < tol, f"Max diff too large: {max_diff}"


def test_matmul_attention_qk():
    require_triton_device()

    B, H, S, S_kv, D = 1, 16, 9, 9, 128

    torch.manual_seed(0)
    q_cpu = torch.randn(B, H, S, D, device=CPU_DEVICE, dtype=torch.float32)
    k_cpu = torch.randn(B, H, S_kv, D, device=CPU_DEVICE, dtype=torch.float32)

    with torch.no_grad():
        scores_torch = torch.matmul(q_cpu, k_cpu.transpose(-2, -1))

    scores_triton = from_triton(triton_matmul(to_triton(q_cpu), to_triton(k_cpu).transpose(-2, -1)))
    
    max_diff = (scores_torch - scores_triton).abs().max().item()
    
    assert max_diff < 0.05


def test_matmul_attention_pv():
    require_triton_device()

    B, H, S, S_kv, D = 1, 16, 9, 9, 128

    torch.manual_seed(0)
    probs_cpu = torch.randn(B, H, S, S_kv, device=CPU_DEVICE, dtype=torch.float32)
    v_cpu = torch.randn(B, H, S_kv, D, device=CPU_DEVICE, dtype=torch.float32)

    with torch.no_grad():
        context_torch = torch.matmul(probs_cpu, v_cpu)

    context_triton = from_triton(triton_matmul(to_triton(probs_cpu), to_triton(v_cpu)))
    
    max_diff = (context_torch - context_triton).abs().max().item()
    
    assert max_diff < 0.05


def test_matmul_batched():
    require_triton_device()

    B, H, S, D = 1, 16, 9, 128

    torch.manual_seed(0)
    a_cpu = torch.randn(B, H, S, D, device=CPU_DEVICE, dtype=torch.float32)
    b_cpu = torch.randn(B, H, D, S, device=CPU_DEVICE, dtype=torch.float32)

    with torch.no_grad():
        c_torch = torch.matmul(a_cpu, b_cpu)

    c_triton = from_triton(triton_matmul(to_triton(a_cpu), to_triton(b_cpu)))
    
    max_diff = (c_torch - c_triton).abs().max().item()
    
    assert max_diff < 0.05


if __name__ == "__main__":
    pytest.main([__file__])
