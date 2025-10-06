import torch
import torch.nn.functional as F
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.swiglu import triton_swiglu_activation, triton_silu


def pytorch_swiglu_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


@pytest.mark.parametrize("shape", [
    (1, 1, 3072),
    (1, 9, 3072),
    (1, 128, 3072),
    (2, 16, 3072),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu_activation_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    gate = torch.randn(shape, device='cuda', dtype=dtype)
    up = torch.randn(shape, device='cuda', dtype=dtype)
    
                
    output_torch = pytorch_swiglu_activation(gate, up)
    
               
    output_triton = triton_swiglu_activation(gate, up)
    
        
    max_diff = (output_torch - output_triton).abs().max().item()
    mean_diff = (output_torch - output_triton).abs().mean().item()
    
    print(f"Shape: {shape}")
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    tol = 2e-6 if dtype == torch.float32 else (8e-3 if dtype == torch.float16 else 6.5e-2)
    assert max_diff < tol, f"Max diff too large: {max_diff}"


@pytest.mark.parametrize("shape", [
    (1, 1, 3072),
    (1, 9, 3072),
    (2, 16, 3072),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_silu_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    x = torch.randn(shape, device='cuda', dtype=dtype)
    
    output_torch = F.silu(x)
    output_triton = triton_silu(x)
    
    max_diff = (output_torch - output_triton).abs().max().item()
    
    print(f"SiLU shape: {shape}, max_diff: {max_diff:.6e}")
    
    tol = 1e-6 if dtype == torch.float32 else 1e-3
    assert max_diff < tol


def test_swiglu_qwen3_config():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
                                                
    B, S, intermediate = 1, 9, 3072
    
    gate = torch.randn(B, S, intermediate, device='cuda', dtype=torch.float32)
    up = torch.randn(B, S, intermediate, device='cuda', dtype=torch.float32)
    
    output_torch = pytorch_swiglu_activation(gate, up)
    output_triton = triton_swiglu_activation(gate, up)
    
    assert torch.allclose(output_torch, output_triton, rtol=1e-5, atol=1e-6)
    print("Qwen3 MLP config test passed")


def test_swiglu_edge_cases():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
         
    gate = torch.full((1, 1, 100), -10.0, device='cuda')
    up = torch.ones((1, 1, 100), device='cuda')
    
    output_torch = pytorch_swiglu_activation(gate, up)
    output_triton = triton_swiglu_activation(gate, up)
    
    assert torch.allclose(output_torch, output_triton, rtol=1e-5, atol=1e-6)
    
         
    gate = torch.full((1, 1, 100), 10.0, device='cuda')
    output_torch = pytorch_swiglu_activation(gate, up)
    output_triton = triton_swiglu_activation(gate, up)
    
    assert torch.allclose(output_torch, output_triton, rtol=1e-5, atol=1e-6)
    
    print("Edge cases test passed")


if __name__ == "__main__":
    print("Testing SwiGLU Triton kernel...")
    print("="*60)
    
    if torch.cuda.is_available():
        test_swiglu_activation_correctness((1, 9, 3072))
        test_silu_correctness((1, 9, 3072))
        test_swiglu_qwen3_config()
        test_swiglu_edge_cases()
        
        print("\n" + "="*60)
        print("All SwiGLU tests passed!")
    else:
        print("CUDA not available, skipping tests")
