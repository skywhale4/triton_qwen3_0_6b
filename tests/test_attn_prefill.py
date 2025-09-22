import pytest
import torch

from kernels.attn_prefill import attention_prefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")

# 参考实现走“纯 FP32”
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

RTOL = 1e-3
ATOL = 1e-3


def _rand(shape, low=-1.0, high=1.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


def _ref_attention_prefill(q, k, v):
    # q,k,v: [B,T,H,Dh]
    B, T, H, Dh = q.shape
    scale = Dh ** -0.5

    # 计算注意力分数
    # scores[b,h] = q[b,:,h] @ k[b,:,h].T -> [T,T]
    q_ = q.permute(0, 2, 1, 3).contiguous()  # [B,H,T,Dh]
    k_ = k.permute(0, 2, 1, 3).contiguous()
    v_ = v.permute(0, 2, 1, 3).contiguous()

    out = torch.empty((B, H, T, Dh), device=q.device, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            scores = torch.matmul(q_[b, h], k_[b, h].transpose(-1, -2)) * scale  # [T,T]
            # 因果遮罩：上三角置 -inf（不含对角线）
            mask = torch.triu(torch.ones((T, T), device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = torch.matmul(probs, v_[b, h])  # [T,Dh]
    return out.permute(0, 2, 1, 3).contiguous()  # [B,T,H,Dh]


@pytest.mark.parametrize("B,T,H,Dh", [
    (1, 1, 1, 16),
    (1, 8, 2, 32),
    (2, 16, 3, 64),
    (2, 33, 2, 32),   # 非 2 的幂长度
])
def test_attention_prefill_matches_torch_fp32(B, T, H, Dh):
    q = _rand((B, T, H, Dh))
    k = _rand((B, T, H, Dh))
    v = _rand((B, T, H, Dh))

    out = attention_prefill(q, k, v)
    ref = _ref_attention_prefill(q, k, v)

    assert out.shape == (B, T, H, Dh) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_attention_prefill_causal_property():
    # 验证因果性：第 t 个位置的输出只依赖 <= t 的 K/V
    B, T, H, Dh = 1, 12, 2, 32
    q = _rand((B, T, H, Dh))
    k = _rand((B, T, H, Dh))
    v = _rand((B, T, H, Dh))

    out = attention_prefill(q, k, v)

    # 构造一个只到 t 的截断参考并对比每个 t
    for t in [1, 3, 7, 11]:
        out_t = attention_prefill(q[:, :t+1], k[:, :t+1], v[:, :t+1])
        # 只比较最后一个位置 t 的输出
        torch.testing.assert_close(out[:, t:t+1], out_t[:, -1:], rtol=RTOL, atol=ATOL)