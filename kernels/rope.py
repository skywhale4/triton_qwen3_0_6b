import triton
import triton.language as tl
import torch


def build_rope_cache(head_dim: int, max_pos: int, theta: float = 1000000.0, device: str = 'cuda', dtype: torch.dtype = torch.float32):
    """
    Build RoPE cache with specified dtype
    
    Args:
        head_dim: dimension of each attention head
        max_pos: maximum sequence length
        theta: base frequency (default 1000000.0 for Qwen3)
        device: device to store cache
        dtype: data type for cache (float32 for consistency)
    """
    half = head_dim // 2
    inv = torch.pow(theta, -torch.arange(0, half, device=device, dtype=torch.float32) / half)
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = t[:, None] * inv[None, :]
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos.contiguous(), sin.contiguous()


@triton.jit
def _apply_rope(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_b, stride_t, stride_h, stride_d,
    stride_cos, stride_sin,
    B, T, H, D,
    BLOCK_HALF: tl.constexpr,
    X_DTYPE: tl.constexpr,  # 新增：输入数据类型
):
    pid = tl.program_id(axis=0)
    bh = pid // H
    h = pid % H
    b = bh // T
    t = bh % T
    base = b * stride_b + t * stride_t + h * stride_h
    half = D // 2
    offs_half = tl.arange(0, BLOCK_HALF)
    mask_half = offs_half < half
    idx_even = 2 * offs_half
    idx_odd = idx_even + 1
    
    # Load values
    x_even = tl.load(x_ptr + base + idx_even, mask=mask_half, other=0.0)
    x_odd = tl.load(x_ptr + base + idx_odd, mask=mask_half, other=0.0)
    cos = tl.load(cos_ptr + t * stride_cos + offs_half, mask=mask_half, other=0.0)
    sin = tl.load(sin_ptr + t * stride_sin + offs_half, mask=mask_half, other=0.0)
    
    # Apply RoPE rotation
    y_even = x_even * cos - x_odd * sin
    y_odd = x_odd * cos + x_even * sin
    
    # Store results
    tl.store(out_ptr + base + idx_even, y_even, mask=mask_half)
    tl.store(out_ptr + base + idx_odd, y_odd, mask=mask_half)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embedding
    
    Args:
        x: [B, T, H, D] tensor
        cos: [T, D//2] cosine cache
        sin: [T, D//2] sine cache
        
    Returns:
        [B, T, H, D] tensor with RoPE applied
    """
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    assert x.dtype == cos.dtype == sin.dtype, f"dtypes must match: x={x.dtype}, cos={cos.dtype}, sin={sin.dtype}"
    assert x.dtype == torch.float32, f"Only float32 supported, got {x.dtype}"
    
    B, T, H, D = x.shape
    assert D % 2 == 0, f"D must be even, got {D}"
    assert cos.shape == sin.shape == (T, D // 2), f"Shape mismatch: cos={cos.shape}, sin={sin.shape}"
    
    y = torch.empty_like(x)
    grid = (B * T * H,)
    
    # 根据输入类型设置 Triton dtype
    x_dtype = tl.float32 if x.dtype == torch.float32 else tl.bfloat16
    
    _apply_rope[grid](
        x, cos, sin, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), sin.stride(0),
        B, T, H, D,
        BLOCK_HALF=triton.next_power_of_2(D // 2),
        X_DTYPE=x_dtype,  # 传递数据类型
    )
    return y

