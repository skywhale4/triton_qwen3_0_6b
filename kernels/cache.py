import torch
import triton
import triton.language as tl


@triton.jit
def _kv_concat_kernel(
    past_ptr,
    curr_ptr,
    out_ptr,
    stride_past_b,
    stride_past_s,
    stride_past_d,
    stride_curr_b,
    stride_curr_s,
    stride_curr_d,
    stride_out_b,
    stride_out_s,
    stride_out_d,
    S_old,
    S_new,
    S_total,
    D,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    s_block = tl.program_id(1)

    offs_s = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_seq = offs_s < S_total

    past_base = past_ptr + bh * stride_past_b
    curr_base = curr_ptr + bh * stride_curr_b
    out_base = out_ptr + bh * stride_out_b

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask = mask_seq[:, None] & mask_d[None, :]

        out_ptrs = out_base + offs_s[:, None] * stride_out_s + offs_d[None, :] * stride_out_d

                          
        mask_past = mask & (offs_s[:, None] < S_old)
        past_ptrs = past_base + offs_s[:, None] * stride_past_s + offs_d[None, :] * stride_past_d
        past_vals = tl.load(past_ptrs, mask=mask_past, other=0.0)
        tl.store(out_ptrs, past_vals, mask=mask_past)

                             
        mask_curr = mask & (offs_s[:, None] >= S_old)
        curr_idx = tl.maximum(offs_s - S_old, 0)
        curr_ptrs = curr_base + curr_idx[:, None] * stride_curr_s + offs_d[None, :] * stride_curr_d
        curr_vals = tl.load(curr_ptrs, mask=mask_curr, other=0.0)
        tl.store(out_ptrs, curr_vals, mask=mask_curr)


def triton_kv_concat(past: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Concat along sequence dimension using Triton (for KV cache)."""
    assert past.is_cuda and current.is_cuda
    assert past.dtype == current.dtype
    assert past.shape[0] == current.shape[0]
    assert past.shape[1] == current.shape[1]
    assert past.shape[3] == current.shape[3]

    bh = past.shape[0] * past.shape[1]
    S_old = past.shape[2]
    S_new = current.shape[2]
    S_total = S_old + S_new
    D = past.shape[3]

    past_3d = past.contiguous().view(bh, S_old, D)
    curr_3d = current.contiguous().view(bh, S_new, D)
    out = torch.empty((bh, S_total, D), device=past.device, dtype=past.dtype)

    BLOCK_S = 32
    BLOCK_D = 64

    grid = (bh, triton.cdiv(S_total, BLOCK_S))

    _kv_concat_kernel[grid](
        past_3d,
        curr_3d,
        out,
        past_3d.stride(0),
        past_3d.stride(1),
        past_3d.stride(2),
        curr_3d.stride(0),
        curr_3d.stride(1),
        curr_3d.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        S_old,
        S_new,
        S_total,
        D,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    return out.view(past.shape[0], past.shape[1], S_total, D)
