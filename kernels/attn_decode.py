import torch
from .matmul import matmul
from .softmax import softmax


def attention_decode(q, k_cache, v_cache):
    """
    Decode-phase attention with KV cache
    
    Args:
        q: [B, H, Dh] - query for current token
        k_cache: [B, T, H, Dh] - cached keys
        v_cache: [B, T, H, Dh] - cached values
        
    Returns:
        out: [B, H, Dh] - attention output
    """
    B, H, Dh = q.shape
    T = k_cache.shape[1]
    device = q.device
    scale = Dh ** -0.5
    
    # 使用 float32 而不是 bfloat16
    dtype = torch.float32
    
    # Reshape for processing
    q_flat = q.view(B * H, Dh)
    k_flat = k_cache.permute(0, 2, 1, 3).contiguous().view(B * H, T, Dh)
    v_flat = v_cache.permute(0, 2, 1, 3).contiguous().view(B * H, T, Dh)
    
    # Pre-allocate reusable tensors with float32
    scores = torch.empty((1, T), device=device, dtype=dtype)
    tmp_out = torch.empty((1, Dh), device=device, dtype=dtype)
    final_out = torch.empty((B * H, Dh), device=device, dtype=dtype)
    
    for bh in range(B * H):
        q_bh = q_flat[bh:bh+1]  # [1, Dh]
        k_bh = k_flat[bh]       # [T, Dh]  
        v_bh = v_flat[bh]       # [T, Dh]
        
        # Compute attention scores: Q @ K.T
        matmul(q_bh, k_bh.T, scores)
        scores *= scale
        
        # Softmax and weighted sum
        probs = softmax(scores)
        matmul(probs, v_bh, tmp_out)
        
        final_out[bh] = tmp_out[0]
    
    return final_out.view(B, H, Dh)