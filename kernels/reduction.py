import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_blocks_kernel(
    x_ptr,
    idx_ptr,
    val_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    data = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    block_idx = tl.argmax(data, axis=0)
    block_val = tl.max(data, axis=0)
    block_abs_idx = block_start + block_idx

    tl.store(idx_ptr + pid, block_abs_idx)
    tl.store(val_ptr + pid, block_val)


def triton_argmax(x: torch.Tensor) -> int:

    flat = x.reshape(-1)
    n = flat.numel()

    BLOCK_SIZE = 1024
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    idx_buffer = torch.empty(n_blocks, device=x.device, dtype=torch.int64)
    val_buffer = torch.empty(n_blocks, device=x.device, dtype=torch.float32)

    _argmax_blocks_kernel[(n_blocks,)](flat, idx_buffer, val_buffer, n, BLOCK_SIZE=BLOCK_SIZE)

    max_val, max_idx = val_buffer[0], idx_buffer[0]
    for i in range(1, n_blocks):
        cand_val = val_buffer[i]
        if cand_val > max_val:
            max_val = cand_val
            max_idx = idx_buffer[i]

    return int(max_idx.item())
