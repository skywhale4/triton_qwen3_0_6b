import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rms_norm import triton_rms_norm


def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x = (x32 * torch.rsqrt(var + eps)).to(x.dtype)
    return weight * x


@pytest.mark.parametrize("shape", [
    (1, 1, 1024),               
    (1, 9, 1024),           
    (1, 128, 1024),          
    (2, 16, 1024),              
    (1, 1, 128),                            
    (1, 9, 16, 128),                 
])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_correctness(shape, eps, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    hidden_size = shape[-1]
    
            
    x = torch.randn(shape, device='cuda', dtype=dtype)
    weight = torch.randn(hidden_size, device='cuda', dtype=dtype)
    
                
    output_torch = pytorch_rms_norm(x, weight, eps)
    
               
    output_triton = triton_rms_norm(x, weight, eps)
    
        
    max_diff = (output_torch - output_triton).abs().max().item()
    mean_diff = (output_torch - output_triton).abs().mean().item()
    
    print(f"Shape: {shape}, eps: {eps}")
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
                    
    tol = 1e-5 if dtype == torch.float32 else 6.5e-2
    assert max_diff < tol, f"Max diff too large: {max_diff}"


def test_rms_norm_backward_compat():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
                       
    batch_size = 1
    seq_len = 9
    hidden_size = 1024
    eps = 1e-6
    
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
    weight = torch.ones(hidden_size, device='cuda', dtype=torch.float32)
    
    output_torch = pytorch_rms_norm(x, weight, eps)
    output_triton = triton_rms_norm(x, weight, eps)
    
    assert torch.allclose(output_torch, output_triton, rtol=1e-5, atol=1e-6)
    print("Backward compatibility test passed")



if __name__ == "__main__":
              
    print("Testing RMSNorm Triton kernel...")
    print("="*60)
    
    if torch.cuda.is_available():
        test_rms_norm_correctness((1, 9, 1024), 1e-6)
        test_rms_norm_correctness((1, 1, 128), 1e-6)
        test_rms_norm_backward_compat()
        test_rms_norm_dtype(torch.float32)
        
        print("\n" + "="*60)
        print("All tests passed!")
    else:
        print("CUDA not available, skipping tests")
