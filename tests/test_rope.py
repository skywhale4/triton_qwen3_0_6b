import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rope import triton_rope


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def pytorch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


@pytest.mark.parametrize("shape", [
    (1, 1, 1, 128),                 
    (1, 16, 1, 128),                        
    (1, 16, 9, 128),            
    (1, 8, 64, 128),                   
    (2, 16, 32, 128),             
])
def test_rope_correctness(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B, H, S, D = shape
    
            
    x = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    
                            
    positions = torch.arange(S, device='cuda')
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, D, 2, device='cuda').float() / D))
    freqs = torch.einsum('i,j->ij', positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)                
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    
                
    output_torch = pytorch_rope(x, cos, sin)
    
               
    output_triton = triton_rope(x, cos, sin)
    
        
    max_diff = (output_torch - output_triton).abs().max().item()
    mean_diff = (output_torch - output_triton).abs().mean().item()
    
    print(f"Shape: {shape}")
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    assert max_diff < 1e-5, f"Max diff too large: {max_diff}"
    assert mean_diff < 1e-6, f"Mean diff too large: {mean_diff}"


def test_rope_with_qwen3_config():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
                   
    B, H, S, D = 1, 16, 9, 128
    theta = 1_000_000.0
    
    x = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    
                          
    positions = torch.arange(S, device='cuda')
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device='cuda').float() / D))
    freqs = torch.einsum('i,j->ij', positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    
    output_torch = pytorch_rope(x, cos, sin)
    output_triton = triton_rope(x, cos, sin)
    
    assert torch.allclose(output_torch, output_triton, rtol=1e-5, atol=1e-6)
    print("Qwen3 config test passed")


def test_rope_decode_phase():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B, H, S, D = 1, 16, 1, 128               
    
    x = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    
                             
    position = 1000
    positions = torch.tensor([position], device='cuda')
    
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, D, 2, device='cuda').float() / D))
    freqs = torch.einsum('i,j->ij', positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    
    output_torch = pytorch_rope(x, cos, sin)
    output_triton = triton_rope(x, cos, sin)
    
    max_diff = (output_torch - output_triton).abs().max().item()
    assert max_diff < 1e-5, f"Decode phase diff: {max_diff}"
    print(f"Decode phase (position={position}), diff={max_diff:.6e}")


if __name__ == "__main__":
    print("Testing RoPE Triton kernel...")
    print("="*60)
    
    if torch.cuda.is_available():
        test_rope_correctness((1, 16, 9, 128))
        test_rope_correctness((1, 8, 64, 128))
        test_rope_with_qwen3_config()
        test_rope_decode_phase()
        
        print("\n" + "="*60)
        print("All RoPE tests passed!")
    else:
        print("CUDA not available, skipping tests")
