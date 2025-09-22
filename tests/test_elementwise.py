import pytest
import torch

from kernels.elementwise import add, scale, silu_mul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


def _rand(shape, low=-1.0, high=1.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


@pytest.mark.parametrize("shape", [(1,), (257,), (1024,), (4096 + 17,), (32, 33)])
def test_add_matches_torch_fp32(shape):
    a = _rand(shape).to(torch.float32)
    b = _rand(shape).to(torch.float32)

    # 无 out
    out = add(a, b)
    ref = a + b
    assert out.dtype == torch.float32 and out.shape == a.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

    # 带 out
    out_buf = torch.empty_like(a, dtype=torch.float32)
    out2 = add(a, b, out=out_buf)
    torch.testing.assert_close(out2, ref, rtol=1e-5, atol=1e-6)
    assert out2.data_ptr() == out_buf.data_ptr()


@pytest.mark.parametrize("shape", [(1,), (257,), (2048 + 13,), (16, 17)])
@pytest.mark.parametrize("alpha", [-1.5, 0.5, 2.0])
def test_scale_matches_torch_fp32(shape, alpha):
    a = _rand(shape).to(torch.float32)

    out = scale(a, alpha)
    ref = a * alpha

    assert out.dtype == torch.float32 and out.shape == a.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

    # 带 out
    out_buf = torch.empty_like(a, dtype=torch.float32)
    out2 = scale(a, alpha, out=out_buf)
    torch.testing.assert_close(out2, ref, rtol=1e-5, atol=1e-6)
    assert out2.data_ptr() == out_buf.data_ptr()


@pytest.mark.parametrize("shape", [(1,), (257,), (4096 + 3,), (8, 9, 10)])
def test_silu_mul_matches_torch_fp32(shape):
    up = _rand(shape).to(torch.float32)
    gate = _rand(shape, low=-8.0, high=8.0).to(torch.float32)

    silu = gate * torch.sigmoid(gate)
    ref = silu * up

    # 注意：这里交换参数顺序为 (gate, up)
    out = silu_mul(gate, up)
    assert out.dtype == torch.float32 and out.shape == up.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

    # 带 out，同样按 (gate, up) 顺序
    out_buf = torch.empty_like(up, dtype=torch.float32)
    out2 = silu_mul(gate, up, out=out_buf)
    torch.testing.assert_close(out2, ref, rtol=1e-5, atol=1e-6)
    assert out2.data_ptr() == out_buf.data_ptr()