import pytest
import torch

from kernels.rope import apply_rope, build_rope_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


def _rand(shape, low=-1.0, high=1.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


def rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, D], cos/sin: [T, D//2]
    B, T, H, D = x.shape
    half = D // 2
    x_even = x[..., 0::2]   # [B, T, H, half]
    x_odd  = x[..., 1::2]   # [B, T, H, half]
    cos_broadcast = cos[None, :, None, :]   # [1, T, 1, half]
    sin_broadcast = sin[None, :, None, :]   # [1, T, 1, half]
    y_even = x_even * cos_broadcast - x_odd * sin_broadcast
    y_odd  = x_odd  * cos_broadcast + x_even * sin_broadcast
    y = torch.empty_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd
    return y


@pytest.mark.parametrize("B,T,H,D", [
    (1, 1, 1, 8),
    (2, 3, 2, 16),
    (1, 8, 4, 64),
    (2, 17, 3, 32),
    (3, 33, 2, 128),
])
def test_rope_matches_torch_fp32(B, T, H, D):
    x = _rand((B, T, H, D))
    cos, sin = build_rope_cache(D, T, dtype=torch.float32, device="cuda")
    out = apply_rope(x, cos, sin)
    ref = rope_ref(x, cos, sin)

    assert out.shape == (B, T, H, D)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_rope_identity_when_zero_angle():
    # cos=1, sin=0 -> 应为恒等变换
    B, T, H, D = 2, 5, 3, 32
    x = _rand((B, T, H, D))
    cos = torch.ones((T, D // 2), device="cuda", dtype=torch.float32)
    sin = torch.zeros((T, D // 2), device="cuda", dtype=torch.float32)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(out, x, rtol=0, atol=0)


@pytest.mark.parametrize("B,T,H,D", [
    (1, 7, 1, 8),
    (2, 9, 2, 64),
])
def test_rope_against_manual_batching(B, T, H, D):
    # 拆分 batch/head 逐元素验证
    x = _rand((B, T, H, D))
    cos, sin = build_rope_cache(D, T, dtype=torch.float32, device="cuda")
    out = apply_rope(x, cos, sin)

    out_manual = torch.empty_like(x)
    for b in range(B):
        for h in range(H):
            out_manual[b, :, h] = rope_ref(x[b:b+1, :, h:h+1, :], cos, sin)[0, :, 0, :]

    torch.testing.assert_close(out, out_manual, rtol=1e-6, atol=1e-6)


def test_rope_raises_on_odd_D():
    B, T, H, D = 1, 2, 1, 7  # 奇数维度应触发断言
    x = _rand((B, T, H, D))
    cos = torch.ones((T, D // 2), device="cuda", dtype=torch.float32)
    sin = torch.zeros((T, D // 2), device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError):
        apply_rope(x, cos, sin)