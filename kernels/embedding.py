import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    n_tokens,
    hidden_size,
    stride_weight_vocab,
    stride_weight_hidden,
    stride_output_token,
    stride_output_hidden,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= n_tokens:
        return

    token_id = tl.load(input_ids_ptr + token_idx)

    offs = tl.arange(0, BLOCK_H)
    for h in range(0, hidden_size, BLOCK_H):
        cur = h + offs
        mask = cur < hidden_size
        weight_ptrs = weight_ptr + token_id * stride_weight_vocab + cur * stride_weight_hidden
        values = tl.load(weight_ptrs, mask=mask, other=0.0).to(tl.float32)
        out_ptrs = output_ptr + token_idx * stride_output_token + cur * stride_output_hidden
        tl.store(out_ptrs, values, mask=mask)


def triton_embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert input_ids.dtype in (torch.int32, torch.int64)

    flat_ids = input_ids.contiguous().view(-1)
    n_tokens = flat_ids.shape[0]
    hidden_size = weight.shape[1]

    output = torch.empty((n_tokens, hidden_size), device=weight.device, dtype=weight.dtype)

    grid = (triton.cdiv(n_tokens, 1),)

    _embedding_kernel[grid](
        flat_ids,
        weight,
        output,
        n_tokens,
        hidden_size,
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_H=128,
    )

    return output.view(*input_ids.shape, hidden_size)
