import pytest
import torch

from kernels.softmax import softmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


def _rand(shape, low=-5.0, high=5.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


@pytest.mark.parametrize("M,N", [
    (1, 1),
    (2, 7),
    (3, 33),
    (4, 128),
    (2, 257),
    (5, 513),
])
def test_softmax_matches_torch_fp32_various_shapes(M, N):
    x = _rand((M, N))
    out = softmax(x)
    ref = torch.softmax(x, dim=-1)

    assert out.shape == (M, N)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("M,N", [
    (1, 5),
    (3, 33),
    (2, 257),
])
def test_softmax_row_sums_to_one(M, N):
    x = _rand((M, N))
    out = softmax(x)
    row_sums = out.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(M, device="cuda", dtype=torch.float32), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("M,N,scale", [
    (2, 33, 20.0),
    (3, 127, 50.0),
    (4, 257, 80.0),
])
def test_softmax_numerical_stability_large_values(M, N, scale):
    x = _rand((M, N)) * scale
    out = softmax(x)
    ref = torch.softmax(x, dim=-1)

    # 仍应匹配参考实现，且每行和为 1
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(M, device="cuda", dtype=torch.float32), rtol=1e-6, atol=1e-6)


def test_softmax_noncontiguous_last_dim():
    M, N = 4, 256
    x_full = _rand((M, N))
    x = x_full[:, ::2]  # 形状 (M, N/2)，沿列非连续
    out = softmax(x)
    ref = torch.softmax(x, dim=-1)
    assert not x.is_contiguous()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("M,N", [
    (2, 17),
    (3, 64),
])
def test_softmax_uniform_input(M, N):
    x = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    out = softmax(x)
    expected = torch.full((M, N), 1.0 / N, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_softmax_single_large_max():
    M, N = 3, 33
    x = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    max_indices = [5, 0, 17]
    for i, k in enumerate(max_indices):
        x[i, k] = 50.0  # 远大于其它值

    out = softmax(x)
    # 应非常接近 one-hot，且最大位置一致
    topk = out.argmax(dim=-1)
    torch.testing.assert_close(topk.to(torch.int64), torch.tensor(max_indices, device="cuda", dtype=torch.int64))
    # 在最大位置的概率接近 1
    vals = out[torch.arange(M, device="cuda"), torch.tensor(max_indices, device="cuda")]
    torch.testing.assert_close(vals, torch.ones(M, device="cuda"), rtol=1e-6, atol=1e-6)