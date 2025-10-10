import torch
import triton
import triton.language as tl


@triton.jit
def _repeat_heads_kernel(
    src_ptr,
    dst_ptr,
    stride_src_b,
    stride_src_h,
    stride_src_s,
    stride_src_d,
    stride_dst_b,
    stride_dst_h,
    stride_dst_s,
    stride_dst_d,
    S,
    D,
    REP,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h_out = tl.program_id(1)

    h_in = h_out // REP

    src_base = src_ptr + b * stride_src_b + h_in * stride_src_h
    dst_base = dst_ptr + b * stride_dst_b + h_out * stride_dst_h

    for s_block in range(0, S, BLOCK_S):
        offs_s = s_block + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        for d_block in range(0, D, BLOCK_D):
            offs_d = d_block + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            mask = mask_s[:, None] & mask_d[None, :]

            src_ptrs = src_base + offs_s[:, None] * stride_src_s + offs_d[None, :] * stride_src_d
            dst_ptrs = dst_base + offs_s[:, None] * stride_dst_s + offs_d[None, :] * stride_dst_d

            vals = tl.load(src_ptrs, mask=mask, other=0.0)
            tl.store(dst_ptrs, vals, mask=mask)


def triton_repeat_kv_heads(x: torch.Tensor, groups: int) -> torch.Tensor:
    assert x.dim() == 4                  
    B, H_src, S, D = x.shape
    H_dst = H_src * groups

    out = torch.empty((B, H_dst, S, D), device=x.device, dtype=x.dtype)

    BLOCK_S = 32
    BLOCK_D = 64

    grid = (B, H_dst)

    _repeat_heads_kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        S,
        D,
        groups,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    return out
