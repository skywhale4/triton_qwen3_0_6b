import pytest
import torch

from kernels.mlp import mlp_swiglu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")

# 参考实现走“纯 FP32”
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

RTOL = 3e-3
ATOL = 1e-2  # 接近 0 的位置用绝对误差主导，给稍大容差以覆盖大矩阵相消


def _rand(shape, low=-1.0, high=1.0):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (high - low) * torch.rand(shape, device="cuda", generator=g, dtype=torch.float32) + low


def swiglu_ref(x, gate_w, up_w, down_w):
    # x: [B, T, D]
    B, T, D = x.shape
    inter = gate_w.shape[0]  # [intermediate, D]
    x2 = x.reshape(B * T, D)
    gate = x2 @ gate_w.t()            # [B*T, inter]
    up = x2 @ up_w.t()                # [B*T, inter]
    silu = gate * torch.sigmoid(gate) # [B*T, inter]
    out = (silu * up) @ down_w.t()           # [B*T, D]
    return out.reshape(B, T, D)


@pytest.mark.parametrize("B,T,D,inter", [
    (1, 1, 16, 32),
    (2, 3, 64, 128),
    (1, 8, 128, 256),
    (2, 4, 256, 512),
])
def test_mlp_swiglu_matches_torch_fp32(B, T, D, inter):
    x = _rand((B, T, D))
    # 权重按 [out_features, in_features]
    gate_w = _rand((inter, D))
    up_w = _rand((inter, D))
    down_w = _rand((D, inter))

    out = mlp_swiglu(x, gate_w, up_w, down_w)
    ref = swiglu_ref(x, gate_w, up_w, down_w)

    assert out.shape == (B, T, D) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_mlp_swiglu_large_hidden():
    # 更贴近实际大模型尺寸的用例（适度放大）
    B, T, D, inter = 1, 4, 512, 1536
    x = _rand((B, T, D))
    gate_w = _rand((inter, D))
    up_w = _rand((inter, D))
    down_w = _rand((D, inter))

    out = mlp_swiglu(x, gate_w, up_w, down_w)
    ref = swiglu_ref(x, gate_w, up_w, down_w)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_mlp_swiglu_noncontiguous_x():
    B, T, D, inter = 2, 3, 64, 128
    x_full = _rand((B, T, D * 2))
    x = x_full[..., ::2]  # 非连续：保留每隔一列
    assert not x.is_contiguous()

    gate_w = _rand((inter, D))
    up_w = _rand((inter, D))
    down_w = _rand((D, inter))

    out = mlp_swiglu(x, gate_w, up_w, down_w)
    ref = swiglu_ref(x, gate_w, up_w, down_w)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)