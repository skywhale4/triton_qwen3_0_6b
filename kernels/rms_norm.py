import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    
              
    row_start = row_idx * n_cols
    
              
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
                                      
            
    square_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
        square_sum += tl.sum(x_vals * x_vals)
    
            
    mean = square_sum / n_cols
    rrms = tl.math.rsqrt(mean + eps)
    
              
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols
        
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
        w_vals = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0)
        
        output_vals = x_vals.to(tl.float32) * rrms.to(tl.float32) * w_vals.to(tl.float32)
        tl.store(output_ptr + row_start + col_offsets, output_vals.to(x_vals.dtype), mask=col_mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.shape[-1] == weight.shape[0], f"Shape mismatch"
    
                   
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    
    output = torch.empty_like(x_2d)
    
                               
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)         
    
                        
    grid = (n_rows,)
    
    rms_norm_kernel[grid](
        x_2d, weight, output,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.reshape(original_shape)
