import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.matmul import triton_matmul


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", [
    (16, 16, 16),
    (64, 64, 64),
    (128, 128, 128),
    (9, 1024, 128),
    (9, 9, 128),
])
def test_matmul_correctness(M, N, K, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tol = 0.05 if dtype == torch.float32 else 0.1
    
    a = torch.randn(M, K, device='cuda', dtype=dtype)
    b = torch.randn(K, N, device='cuda', dtype=dtype)
    
    c_torch = torch.matmul(a, b)
    c_triton = triton_matmul(a, b)
    
    max_diff = (c_torch - c_triton).abs().max().item()
    mean_diff = (c_torch - c_triton).abs().mean().item()
    
    print(f"[{M}x{K}] @ [{K}x{N}] -> [{M}x{N}] dtype={dtype}")
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    assert max_diff < tol, f"Max diff too large: {max_diff}"


def test_matmul_attention_qk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
                                         
    B, H, S, S_kv, D = 1, 16, 9, 9, 128
    
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, S_kv, D, device='cuda', dtype=torch.float32)
    
             
    scores_torch = torch.matmul(q, k.transpose(-2, -1))
    
                          
    k_t = k.transpose(-2, -1)
    scores_triton = triton_matmul(q, k_t)
    
    max_diff = (scores_torch - scores_triton).abs().max().item()
    
    print(f"Q@K.T: max_diff={max_diff:.6e}")
    
    assert max_diff < 0.05


def test_matmul_attention_pv():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B, H, S, S_kv, D = 1, 16, 9, 9, 128
    
    probs = torch.randn(B, H, S, S_kv, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, S_kv, D, device='cuda', dtype=torch.float32)
    
             
    context_torch = torch.matmul(probs, v)
    
            
    context_triton = triton_matmul(probs, v)
    
    max_diff = (context_torch - context_triton).abs().max().item()
    
    print(f"P@V: max_diff={max_diff:.6e}")
    
    assert max_diff < 0.05


def test_matmul_batched():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
                                        
    B, H, S, D = 1, 16, 9, 128
    
    a = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    b = torch.randn(B, H, D, S, device='cuda', dtype=torch.float32)
    
    c_torch = torch.matmul(a, b)
    c_triton = triton_matmul(a, b)
    
    max_diff = (c_torch - c_triton).abs().max().item()
    
    print(f"Batched [B,H,S,D]@[B,H,D,S]: max_diff={max_diff:.6e}")
    
    assert max_diff < 0.05


if __name__ == "__main__":
    print("Testing Matmul Triton kernel...")
    print("="*60)
    
    if torch.cuda.is_available():
        test_matmul_correctness(64, 64, 64)
        test_matmul_correctness(9, 1024, 128)
        test_matmul_attention_qk()
        test_matmul_attention_pv()
        test_matmul_batched()
        
        print("\n" + "="*60)
        print("All Matmul tests passed!")
    else:
        print("CUDA not available")
