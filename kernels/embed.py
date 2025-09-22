import triton
import triton.language as tl
import torch


@triton.jit
def _gather_embed_kernel(ids_ptr, emb_ptr, out_ptr,
                         V, D, num_ids,
                         BLOCK_D: tl.constexpr):
    token_id = tl.program_id(axis=0)
    if token_id >= num_ids:
        return
        
    # Load token ID
    idx = tl.load(ids_ptr + token_id)
    
    # Boundary check for token ID
    if idx < 0 or idx >= V:
        # Handle out-of-bounds (set to zero or raise error)
        offs_d = tl.arange(0, BLOCK_D)
        tl.store(out_ptr + token_id * D + offs_d, 
                tl.zeros([BLOCK_D], dtype=tl.float32), 
                mask=offs_d < D)
        return
    
    # Load embedding vector
    offs_d = tl.arange(0, BLOCK_D)
    row_ptr = emb_ptr + idx * D
    vals = tl.load(row_ptr + offs_d, mask=offs_d < D, other=0.0)
    tl.store(out_ptr + token_id * D + offs_d, vals, mask=offs_d < D)


def embed(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup using Triton
    
    Args:
        ids: [B, N] or [B] tensor of token IDs
        table: [V, D] embedding table
        
    Returns:
        [B, N, D] or [B, D] tensor of embeddings
    """
    assert ids.is_cuda and table.is_cuda
    assert table.dim() == 2, "Embedding table must be 2D [V, D]"
    
    # Handle both [B] and [B, N] input shapes
    orig_shape = ids.shape
    ids_flat = ids.view(-1)  # [B*N] or [B]
    num_ids = ids_flat.shape[0]
    D = table.shape[1]
    
    # Create output tensor
    out = torch.empty((num_ids, D), device=ids.device, dtype=table.dtype)
    
    # Launch kernel
    block_d = triton.next_power_of_2(D)
    grid = (num_ids,)
    _gather_embed_kernel[grid](
        ids_flat, table, out,
        table.shape[0], D, num_ids,  # V, D, num_ids
        BLOCK_D=block_d,
    )
    
    # Reshape back to original shape
    return out.view(*orig_shape, D)


# 测试代码
def test_embed():
    # Test 1D input
    ids_1d = torch.tensor([1, 2, 3], device='cuda')
    table = torch.randn(100, 128, device='cuda')
    out_1d = embed(ids_1d, table)
    print(f"1D test: {out_1d.shape}")  # Should be [3, 128]
    
    # Test 2D input
    ids_2d = torch.tensor([[1, 2], [3, 4]], device='cuda')
    out_2d = embed(ids_2d, table)
    print(f"2D test: {out_2d.shape}")  # Should be [2, 2, 128]
    
    # Verify correctness
    expected = table[ids_2d]
    print(f"Correctness: {torch.allclose(out_2d, expected)}")


if __name__ == "__main__":
    test_embed()