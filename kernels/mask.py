import triton
import triton.language as tl
import torch


@triton.jit
def _causal_mask_4d_kernel(
    scores_ptr, B, H, T, 
    stride_b, stride_h, stride_q, stride_k, 
    neg_value,
    BLOCK_T: tl.constexpr
):
    # Get current query position (row)
    pid_row = tl.program_id(axis=0)  # ranges from 0 to B*H*T-1
    pid_block = tl.program_id(axis=1)  # ranges from 0 to ceil(T/BLOCK_T)-1
    
    # Decode pid_row to get (b, h, i) where i is query position
    b = pid_row // (H * T)
    temp = pid_row % (H * T)
    h = temp // T
    i = temp % T  # current query position
    
    # Get key positions for this block
    j_start = pid_block * BLOCK_T
    j_offsets = j_start + tl.arange(0, BLOCK_T)
    mask_cols = j_offsets < T  # valid key positions
    
    # Calculate row pointer in the attention matrix
    row_ptr = (scores_ptr + 
              b * stride_b + 
              h * stride_h + 
              i * stride_q)
    
    # Load the row
    scores_row = tl.load(row_ptr + j_offsets * stride_k, 
                        mask=mask_cols, other=0.0)
    
    # Create causal mask: mask out future positions (j > i)
    causal_mask = j_offsets > i
    
    # Apply mask: set future positions to neg_value
    masked_scores = tl.where(causal_mask & mask_cols, neg_value, scores_row)
    
    # Store back
    tl.store(row_ptr + j_offsets * stride_k, 
            masked_scores, mask=mask_cols)


def apply_causal_mask(scores: torch.Tensor, neg_value: float = -1e4):
    """
    Apply causal mask to attention scores
    
    Args:
        scores: [B, H, T, T] attention scores
        neg_value: value to use for masked positions (default -1e4)
        
    Returns:
        [B, H, T, T] masked scores (in-place)
    """
    assert scores.is_cuda
    assert scores.dim() == 4, f"Expected 4D tensor, got {scores.shape}"
    
    B, H, T, T2 = scores.shape
    assert T == T2, f"Expected square attention matrix, got {T}x{T2}"
    
    # Grid: one thread block per query position per head per batch
    # axis=0: B * H * T (all query positions)
    # axis=1: ceil(T / BLOCK_T) (blocks along key dimension)
    grid = (B * H * T, triton.cdiv(T, 128))
    
    _causal_mask_4d_kernel[grid](
        scores, B, H, T,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        neg_value,
        BLOCK_T=128
    )
    
    return scores
