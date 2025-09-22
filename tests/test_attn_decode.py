import pytest
import torch

from kernels.attn_decode import attention_decode

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


def _ref_attention_decode(q, k_cache, v_cache):
    # q: [B, H, Dh], k_cache/v_cache: [B, T, H, Dh]
    B, H, Dh = q.shape
    scale = Dh ** -0.5
    # scores: [B, H, T]
    scores = torch.einsum("bhd,bthd->bht", q, k_cache) * scale
    probs = torch.softmax(scores, dim=-1)
    # out: [B, H, Dh]
    out = torch.einsum("bht,bthd->bhd", probs, v_cache)
    return out


@pytest.mark.parametrize("B,H,T,Dh", [
    (1, 1, 1, 16),
    (2, 2, 17, 32),
    (1, 4, 128, 64),
    (3, 2, 257, 32),
    (2, 3, 513, 16),
])
def test_attention_decode_matches_torch_fp32(B, H, T, Dh):
    q = _rand((B, H, Dh))
    k_cache = _rand((B, T, H, Dh))
    v_cache = _rand((B, T, H, Dh))

    out = attention_decode(q, k_cache, v_cache)
    ref = _ref_attention_decode(q, k_cache, v_cache)

    assert out.shape == (B, H, Dh)
    assert out.dtype == torch.float32

    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_attention_decode_uniform_scores_fp32():
    # 当所有 scores 相等时，softmax 为均匀分布；输出为 V 在 T 维的均值
    B, H, T, Dh = 2, 3, 129, 32
    q = torch.zeros((B, H, Dh), device="cuda", dtype=torch.float32)
    k_cache = torch.zeros((B, T, H, Dh), device="cuda", dtype=torch.float32)
    v_cache = _rand((B, T, H, Dh))

    out = attention_decode(q, k_cache, v_cache)
    ref = v_cache.mean(dim=1)  # 均匀权重下等于 V 的时间维均值

    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_attention_decode_shapes_small_cases_fp32():
    # 额外覆盖一些小尺寸与边界
    for B, H, T, Dh in [(1, 1, 5, 8), (1, 2, 3, 16), (2, 1, 7, 8)]:
        q = _rand((B, H, Dh))
        k_cache = _rand((B, T, H, Dh))
        v_cache = _rand((B, T, H, Dh))
        out = attention_decode(q, k_cache, v_cache)
        assert out.shape == (B, H, Dh)
        assert out.dtype == torch.float32