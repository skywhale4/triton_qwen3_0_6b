import pytest
import torch

from kernels.argmax import argmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


def _rand(shape, low=-3.0, high=3.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


@pytest.mark.parametrize("M,N", [
    (1, 1),
    (1, 257),
    (2, 1024),
    (3, 4096),
    (4, 4096 + 7),
    (5, 10000),
])
def test_argmax_matches_torch_fp32_various_shapes(M, N):
    x = _rand((M, N))

    idx, val = argmax(x)  # 默认 block_size=4096
    ref_val, ref_idx = torch.max(x, dim=1)

    assert idx.shape == (M,)
    assert val.shape == (M,)
    assert idx.dtype == torch.int32
    assert val.dtype == torch.float32

    # 索引数值一致（类型不同，转换比较）
    torch.testing.assert_close(idx.to(torch.int64), ref_idx)
    torch.testing.assert_close(val, ref_val, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("block_size,N", [
    (64, 1000),       # 多块且非 2 的幂
    (128, 5000),      # 多块
    (1024, 5000),     # 多块
    (4096, 12345),    # 多块，覆盖 stage2 的 next_power_of_2
])
def test_argmax_custom_block_size_fp32(block_size, N):
    M = 3
    x = _rand((M, N))

    idx, val = argmax(x, block_size=block_size)
    ref_val, ref_idx = torch.max(x, dim=1)

    assert idx.dtype == torch.int32 and val.dtype == torch.float32
    torch.testing.assert_close(idx.to(torch.int64), ref_idx)
    torch.testing.assert_close(val, ref_val, rtol=1e-5, atol=1e-6)


def test_argmax_noncontiguous_stride_fp32():
    M, N = 4, 2560  # 经过切片后有效 N 为 1280
    x_full = _rand((M, N))
    x = x_full[:, ::2]  # 沿 N 维步长为 2 的非连续张量

    idx, val = argmax(x)
    ref_val, ref_idx = torch.max(x, dim=1)

    assert not x.is_contiguous()
    assert idx.dtype == torch.int32 and val.dtype == torch.float32
    torch.testing.assert_close(idx.to(torch.int64), ref_idx)
    torch.testing.assert_close(val, ref_val, rtol=1e-5, atol=1e-6)


def test_argmax_ties_returns_first_index_fp32():
    M, N = 3, 32
    x = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    # 为每一行构造并列最大值（相同值）在不同位置
    tie_positions = [(3, 7), (0, 1), (10, 10)]  # 第三行只有一个最大值（退化为无并列）
    for i, (i1, i2) in enumerate(tie_positions):
        x[i, i1] = 1.0
        x[i, i2] = 1.0

    idx, val = argmax(x)
    ref_val, ref_idx = torch.max(x, dim=1)

    # 期望与 torch 行为一致：返回第一个最大值的索引
    torch.testing.assert_close(idx.to(torch.int64), ref_idx)
    torch.testing.assert_close(val, ref_val, rtol=0, atol=0)