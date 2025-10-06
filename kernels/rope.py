import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    X,                
    Cos,                
    Sin,                
    Out,                
    n_elements,
    half_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
                    
    x = tl.load(X + offsets, mask=mask, other=0.0)
    cos = tl.load(Cos + offsets, mask=mask, other=1.0)
    sin = tl.load(Sin + offsets, mask=mask, other=0.0)
    
                           
    local_idx = offsets % (half_dim * 2)
    
                             
                                                      
                                                       
    in_first_half = local_idx < half_dim
    
                 
    pair_local_idx = tl.where(in_first_half, local_idx + half_dim, local_idx - half_dim)
    base_offset = offsets - local_idx                 
    pair_offsets = base_offset + pair_local_idx
    pair_mask = pair_offsets < n_elements
    
            
    x_pair = tl.load(X + pair_offsets, mask=pair_mask, other=0.0)
    
                                    
    rotated = tl.where(in_first_half, -x_pair, x_pair)
    
          
    output = x * cos + rotated * sin
    
    tl.store(Out + offsets, output, mask=mask)


def triton_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:

    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    
    B, H, S, D = x.shape
    
            
    x_flat = x.reshape(-1)
    
                             
    cos_flat = cos.expand(B, H, S, D).reshape(-1)
    sin_flat = sin.expand(B, H, S, D).reshape(-1)
    
    output_flat = torch.empty_like(x_flat)
    
    n_elements = x_flat.numel()
    half_dim = D // 2
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    rope_kernel[grid](
        x_flat, cos_flat, sin_flat, output_flat,
        n_elements,
        half_dim=half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_flat.reshape(B, H, S, D)
