import torch
from .matmul import matmul
from .softmax import softmax


def attention_prefill(q, k, v):
    """
    Prefill-phase attention (causal)

    Args:
        q: [B, T, H, Dh]
        k: [B, T, H, Dh]
        v: [B, T, H, Dh]
    Returns:
        out: [B, T, H, Dh]
    """
    B, T, H, Dh = q.shape
    device = q.device
    scale = Dh ** -0.5
    dtype = torch.float32

    # 直接按 (b, h) 处理，避免错误展平与来回 permute
    out = torch.empty((B, T, H, Dh), device=device, dtype=dtype)

    for b in range(B):
        for h in range(H):
            q_bh = q[b, :, h, :]  # [T, Dh]
            k_bh = k[b, :, h, :]  # [T, Dh]
            v_bh = v[b, :, h, :]  # [T, Dh]

            # scores: [T, T] = q @ k.T
            scores = torch.empty((T, T), device=device, dtype=dtype)
            matmul(q_bh, k_bh.T, scores)
            scores *= scale

            # causal mask（上三角置 -inf，不含对角线）
            mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))

            # probs @ v -> out[b, :, h, :]
            probs = softmax(scores)                    # [T, T]
            matmul(probs, v_bh, out[b, :, h, :])      # [T, Dh]

    return out