import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kernel(
    table_ptr,
    pos_ptr,
    out_ptr,
    stride_table_s,
    stride_table_d,
    stride_out_s,
    stride_out_d,
    stride_pos,
    S,
    D,
    BLOCK_D: tl.constexpr,
):
    s_block = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    pos = tl.load(pos_ptr + s_block * stride_pos)

    table_row_ptr = table_ptr + pos * stride_table_s
    out_row_ptr = out_ptr + s_block * stride_out_s

    for d in range(0, D, BLOCK_D):
        idx_d = d + offs_d
        mask = idx_d < D

        table_ptrs = table_row_ptr + idx_d * stride_table_d
        out_ptrs = out_row_ptr + idx_d * stride_out_d

        vals = tl.load(table_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, vals, mask=mask)


def triton_gather_rope(table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    S = positions.numel()
    D = table.shape[1]

    positions = positions.contiguous()

    out = torch.empty((S, D), device=table.device, dtype=table.dtype)

    BLOCK_D = 128
    grid = (S,)

    _gather_kernel[grid](
        table,
        positions,
        out,
        table.stride(0),
        table.stride(1),
        out.stride(0),
        out.stride(1),
        positions.stride(0),
        S,
        D,
        BLOCK_D=BLOCK_D,
    )

    return out
