import pytest
import torch

from kernels.matmul import matmul

RTOL = 1e-4
ATOL = 1e-4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")

# 让参考实现尽量使用纯 FP32，避免 TF32 干扰对比
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")


def _rand(shape, low=-1.0, high=1.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


@pytest.mark.parametrize("M,K,N", [
    (1, 1, 1),
    (2, 3, 4),
    (17, 19, 23),      # 非整块尺寸
    (64, 32, 48),
    (128, 64, 80),
])
def test_matmul_matches_torch_fp32_various_shapes(M, K, N):
    a = _rand((M, K))
    b = _rand((K, N))

    out = matmul(a, b)
    ref = torch.matmul(a, b)

    assert out.shape == (M, N) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_matmul_with_out_buffer_autotune():
    # 选择非整块尺寸以覆盖边界
    M, K, N = 127, 95, 131
    a = _rand((M, K))
    b = _rand((K, N))
    out_buf = torch.empty((M, N), device="cuda", dtype=torch.float32)

    out = matmul(a, b, out=out_buf)  # 不手动指定 block 参数
    ref = torch.matmul(a, b)

    assert out.data_ptr() == out_buf.data_ptr()
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_matmul_noncontiguous_inputs_slice():
    # 通过步长切片构造非连续视图
    M, K, N = 65, 97, 129
    a_full = _rand((M, K * 2))
    b_full = _rand((K * 2, N))
    a = a_full[:, ::2]   # (M, K)，非连续
    b = b_full[::2, :]   # (K, N)，非连续
    assert not a.is_contiguous() and not b.is_contiguous()

    out = matmul(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_matmul_noncontiguous_inputs_transpose():
    # 通过转置构造非连续视图
    M, K, N = 33, 48, 55
    a = _rand((K, M)).t()        # (M, K)，非连续
    b = _rand((N, K)).t()        # (K, N)，非连续
    assert not a.is_contiguous() and not b.is_contiguous()

    out = matmul(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_matmul_raises_on_shape_mismatch():
    a = _rand((8, 16))
    b = _rand((15, 8))  # K 不匹配
    with pytest.raises(AssertionError):
        matmul(a, b)


def test_matmul_raises_on_dtype_or_device():
    a = _rand((4, 4)).cpu()  # 非 CUDA
    b = _rand((4, 4))
    with pytest.raises(AssertionError):
        matmul(a, b)

    a = _rand((4, 4)).to(torch.float16)
    b = _rand((4, 4))
    with pytest.raises(AssertionError):
        matmul(a, b)


def test_matmul_large_square_1024():
    M = K = N = 1024
    a = _rand((M, K))
    b = _rand((K, N))
    out = matmul(a, b)  # 不手动指定 block 参数，走 autotune/默认配置
    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)