import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr, out_ptr, n_rows, n_cols,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = x_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
    mask = col_offsets < n_cols
    
    x = tl.load(x_ptrs, mask=mask, other=float('-inf'))
    
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x_shifted = tl.where(mask, x - x_max, float('-inf'))
    
    exp_x = tl.where(mask, tl.exp(x_shifted), 0.0)
    sum_exp = tl.sum(exp_x, axis=0)
    
    softmax_out = exp_x / sum_exp
    
    out_ptrs = out_ptr + row_idx * stride_out_row + col_offsets * stride_out_col
    tl.store(out_ptrs, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax along the last dimension
    Args:
        x: [M, N] tensor
    Returns:
        [M, N] softmax output
    """
    assert x.is_cuda and x.dtype == torch.float32, f"Only float32 supported, got {x.dtype}"
    
    M, N = x.shape
    device = x.device
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    _softmax_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out