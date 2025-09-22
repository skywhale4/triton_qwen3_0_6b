import triton
import triton.language as tl
import torch


@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr, n, eps,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = x_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
    mask = col_offsets < n
    
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    
    # Compute RMS
    x_sq = x_f32 * x_f32
    mean_sq = tl.sum(x_sq, axis=0) / n
    inv_rms = tl.rsqrt(mean_sq + eps)
    
    # Normalize and apply weight
    normed = x_f32 * inv_rms
    weight_ptrs = weight_ptr + col_offsets
    weight = tl.load(weight_ptrs, mask=mask, other=0.0)
    result = normed * weight.to(tl.float32)
    
    out_ptrs = out_ptr + row_idx * stride_out_row + col_offsets * stride_out_col
    tl.store(out_ptrs, result.to(tl.float32), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, \
        f"Only float32 supported, got x.dtype={x.dtype}, weight.dtype={weight.dtype}"
    assert x.shape[-1] == weight.shape[0]
    
    *batch_dims, D = x.shape
    x_flat = x.view(-1, D)
    B = x_flat.shape[0]
    
    out = torch.empty_like(x_flat)
    
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (B,)
    
    _rmsnorm_kernel[grid](
        x_flat, weight, out, D, eps,
        x_flat.stride(0), x_flat.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.view(*batch_dims, D)


def rmsnorm_2d(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm for 2D tensors"""
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32
    assert x.shape[-1] == weight.shape[0]
    
    N, D = x.shape
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (N,)
    
    _rmsnorm_kernel[grid](
        x, weight, out, D, eps,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out