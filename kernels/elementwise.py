import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
              
    max_val = float('-inf')
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols
        vals = tl.load(input_ptr + row_start + col_offsets, mask=col_mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(vals))
    
                            
    sum_exp = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols
        vals = tl.load(input_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(col_mask, exp_vals, 0.0))
    
                  
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols
        vals = tl.load(input_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
        softmax_vals = tl.exp(vals - max_val) / sum_exp
        tl.store(output_ptr + row_start + col_offsets, softmax_vals, mask=col_mask)


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    output = a.to(tl.float32) + b.to(tl.float32)
    tl.store(output_ptr + offsets, output.to(a.dtype), mask=mask)


@triton.jit
def multiply_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    output = a.to(tl.float32) * b.to(tl.float32)
    tl.store(output_ptr + offsets, output.to(a.dtype), mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:

    assert x.is_cuda
    
    if dim != -1 and dim != x.dim() - 1:
        x = x.transpose(dim, -1)
    
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    
    output = torch.empty_like(x_2d)
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 2048)
    grid = (n_rows,)
    
    softmax_kernel[grid](
        x_2d, output,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    result = output.reshape(original_shape)
    
                                      
    if dim != -1 and dim != len(original_shape) - 1:
        result = result.transpose(dim, -1)
    
    return result


def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    n_elements = a_flat.numel()
    
    output_flat = torch.empty_like(a_flat)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    add_kernel[grid](
        a_flat, b_flat, output_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_flat.reshape(a.shape)


@triton.jit
def cos_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.cos(x.to(tl.float32)).to(x.dtype)

    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def sin_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.sin(x.to(tl.float32)).to(x.dtype)

    tl.store(output_ptr + offsets, y, mask=mask)


def triton_cos(x: torch.Tensor) -> torch.Tensor:

    assert x.is_cuda

    x_flat = x.reshape(-1)
    out_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x_flat.numel(), BLOCK_SIZE),)

    cos_kernel[grid](
        x_flat,
        out_flat,
        x_flat.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_flat.reshape(x.shape)


def triton_sin(x: torch.Tensor) -> torch.Tensor:

    assert x.is_cuda

    x_flat = x.reshape(-1)
    out_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x_flat.numel(), BLOCK_SIZE),)

    sin_kernel[grid](
        x_flat,
        out_flat,
        x_flat.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_flat.reshape(x.shape)


@triton.jit
def argmax_kernel(
    x_ptr,
    idx_ptr,
    val_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    offs = tl.arange(0, BLOCK_SIZE)
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    max_idx = tl.full((), -1, dtype=tl.int32)

    for start in range(0, n_elements, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_elements

        vals = tl.load(x_ptr + idx, mask=mask, other=-float('inf')).to(tl.float32)
        idx_i32 = idx.to(tl.int32)

        chunk_max = tl.max(vals, axis=0)
        chunk_idx = tl.argmax(vals, axis=0)

        better = chunk_max > max_val
        max_val = tl.where(better, chunk_max, max_val)
        best_idx = tl.sum(idx_i32 * tl.where(tl.arange(0, BLOCK_SIZE)[:, None] == chunk_idx[None, :], 1, 0), axis=0)
        max_idx = tl.where(better, best_idx, max_idx)

    tl.store(idx_ptr, max_idx)
    tl.store(val_ptr, max_val)


def triton_argmax(x: torch.Tensor) -> int:

    assert x.is_cuda

    flat = x.reshape(-1)
    idx_out = torch.empty(1, device=x.device, dtype=torch.int32)
    val_out = torch.empty(1, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = (1,)

    argmax_kernel[grid](
        flat,
        idx_out,
        val_out,
        flat.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return int(idx_out.item())


def triton_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    n_elements = a_flat.numel()
    
    output_flat = torch.empty_like(a_flat)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    multiply_kernel[grid](
        a_flat, b_flat, output_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_flat.reshape(a.shape)
