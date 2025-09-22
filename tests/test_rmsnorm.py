import pytest
import torch

from kernels.rmsnorm import rmsnorm, rmsnorm_2d

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


def _rand(shape, low=-1.0, high=1.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [..., D], weight: [D]
    x2 = x.float() * x.float()
    mean_sq = x2.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    y = (x.float() * inv_rms) * weight.float()
    return y.to(torch.float32)


@pytest.mark.parametrize("shape", [
    (2, 8),
    (3, 33),
    (4, 128),
    (1, 7, 5, 32),
    (2, 3, 4, 64),
])
@pytest.mark.parametrize("eps", [0.0, 1e-6, 1e-4])
def test_rmsnorm_matches_torch_fp32(shape, eps):
    x = _rand(shape)
    D = x.shape[-1]
    weight = _rand((D,))
    out = rmsnorm(x, weight, eps=eps)
    ref = rmsnorm_ref(x, weight, eps=eps)
    assert out.shape == x.shape and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("N,D", [
    (1, 8),
    (5, 33),
    (7, 128),
])
@pytest.mark.parametrize("eps", [0.0, 1e-6, 1e-4])
def test_rmsnorm_2d_matches_torch_fp32(N, D, eps):
    x = _rand((N, D))
    weight = _rand((D,))
    out = rmsnorm_2d(x, weight, eps=eps)
    ref = rmsnorm_ref(x, weight, eps=eps)
    assert out.shape == x.shape and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("N,D", [
    (3, 65),
    (4, 256),
])
def test_rmsnorm_and_rmsnorm_2d_consistency(N, D):
    x = _rand((N, D))
    weight = _rand((D,))
    out_a = rmsnorm(x, weight)
    out_b = rmsnorm_2d(x, weight)
    torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-6)


def test_rmsnorm_noncontiguous_last_dim():
    N, D = 4, 128
    x_full = _rand((N, D * 2))
    x = x_full[:, ::2]  # 形状 (N, D)，沿列非连续
    assert not x.is_contiguous()
    weight = _rand((D,))
    out = rmsnorm(x, weight)
    ref = rmsnorm_ref(x, weight)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_rmsnorm_broadcast_over_batch_dims():
    B, T, H, D = 2, 5, 3, 64
    x = _rand((B, T, H, D))
    weight = _rand((D,))
    out = rmsnorm(x, weight)
    ref = rmsnorm_ref(x, weight)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_rmsnorm_raises_on_mismatched_weight():
    x = _rand((2, 33))
    wrong_weight = _rand((34,))
    with pytest.raises(AssertionError):
        rmsnorm(x, wrong_weight)


def test_rmsnorm_requires_fp32():
    x = _rand((2, 16)).to(torch.float16)
    weight = _rand((16,))
    with pytest.raises(AssertionError):
        rmsnorm(x, weight)
    x = _rand((2, 16))
    weight_fp16 = _rand((16,)).to(torch.float16)
    with pytest.raises(AssertionError):
        rmsnorm(x, weight_fp16)