import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.linear import triton_linear


@pytest.mark.parametrize("in_feat,out_feat,seq_len", [
    (1024, 2048, 9),                  
    (1024, 1024, 9),                    
    (1024, 3072, 9),                     
    (3072, 1024, 9),                     
    (1024, 151936, 1),           
])
def test_linear_correctness(in_feat, out_feat, seq_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    x = torch.randn(1, seq_len, in_feat, device='cuda', dtype=torch.float32)
    
                    
    linear_torch = nn.Linear(in_feat, out_feat, bias=False).cuda()
    
    output_torch = linear_torch(x)
    output_triton = triton_linear(x, linear_torch.weight)
    
    max_diff = (output_torch - output_triton).abs().max().item()
    rel_err = max_diff / output_torch.abs().max().item()
    
    print(f"[{seq_len}, {in_feat}] -> [{seq_len}, {out_feat}]")
    print(f"  Max diff: {max_diff:.6e}, Rel err: {rel_err:.6e}")
    
    assert max_diff < 0.1, f"Max diff too large: {max_diff}"


def test_linear_with_bias():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    x = torch.randn(1, 9, 1024, device='cuda')
    linear_torch = nn.Linear(1024, 2048, bias=True).cuda()
    
    output_torch = linear_torch(x)
    output_triton = triton_linear(x, linear_torch.weight, linear_torch.bias)
    
    max_diff = (output_torch - output_triton).abs().max().item()
    
    print(f"Linear with bias: max_diff={max_diff:.6e}")
    
    assert max_diff < 0.1


if __name__ == "__main__":
    print("Testing Linear Triton implementation...")
    print("="*60)
    
    if torch.cuda.is_available():
        test_linear_correctness(1024, 2048, 9)
        test_linear_correctness(1024, 151936, 1)
        test_linear_with_bias()
        
        print("\n" + "="*60)
        print("All Linear tests passed!")
    else:
        print("CUDA not available")
