import pytest
import torch

from kernels.mask import apply_causal_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


def _rand_scores(B, H, T, low=-3.0, high=3.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand((B, H, T, T), device="cuda", generator=g, dtype=torch.float32) + low


@pytest.mark.parametrize("B,H,T", [
    (1, 1, 1),
    (1, 2, 8),
    (2, 3, 17),   # 非 2 的幂
    (2, 1, 33),
])
@pytest.mark.parametrize("neg", [-1e4, -1e9])
def test_causal_mask_value_and_structure(B, H, T, neg):
    scores = _rand_scores(B, H, T)
    out = apply_causal_mask(scores.clone(), neg_value=neg)

    # 结构检查：对角线及以下保持原值；上三角应为 neg
    tril_mask = torch.tril(torch.ones((T, T), device="cuda", dtype=torch.bool), diagonal=0)
    for b in range(B):
        for h in range(H):
            # 下三角保持不变
            torch.testing.assert_close(out[b, h][tril_mask], scores[b, h][tril_mask], rtol=0, atol=0)
            # 上三角为 neg
            assert torch.all(out[b, h][~tril_mask] == neg)


def test_causal_mask_idempotent():
    B, H, T = 2, 2, 16
    scores = _rand_scores(B, H, T)
    neg = -1e6
    out1 = apply_causal_mask(scores.clone(), neg_value=neg)
    out2 = apply_causal_mask(out1.clone(), neg_value=neg)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_causal_mask_large_T_block_coverage():
    # 覆盖多个 BLOCK_T 的情况（BLOCK_T=128）
    B, H, T = 1, 1, 257
    scores = _rand_scores(B, H, T)
    neg = -1e4
    out = apply_causal_mask(scores.clone(), neg_value=neg)

    tril_mask = torch.tril(torch.ones((T, T), device="cuda", dtype=torch.bool), diagonal=0)
    torch.testing.assert_close(out[0, 0][tril_mask], scores[0, 0][tril_mask], rtol=0, atol=0)
    assert torch.all(out[0, 0][~tril_mask] == neg)