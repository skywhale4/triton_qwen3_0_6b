import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.elementwise import (
    triton_softmax,
    triton_add,
    triton_multiply,
    triton_cos,
    triton_sin,
)
from kernels.reduction import triton_argmax
from tests import require_triton_device, to_triton, from_triton

CPU_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_correctness(shape, dtype):
    require_triton_device()
    torch.manual_seed(0)
    a_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=dtype)
    b_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=dtype)
    torch_out = (a_cpu + b_cpu).to(torch.float32)

    triton_out = from_triton(triton_add(to_triton(a_cpu), to_triton(b_cpu))).to(torch.float32)

    tol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(torch_out, triton_out, atol=tol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_multiply_correctness(shape, dtype):
    require_triton_device()
    torch.manual_seed(0)
    a_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=dtype)
    b_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=dtype)
    torch_out = (a_cpu * b_cpu).to(torch.float32)

    triton_out = from_triton(triton_multiply(to_triton(a_cpu), to_triton(b_cpu))).to(torch.float32)

    tol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(torch_out, triton_out, atol=tol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cos_correctness(shape, dtype):
    require_triton_device()
    torch.manual_seed(0)
    x_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=dtype)
    torch_out = torch.cos(x_cpu.to(torch.float32))

    triton_out = from_triton(triton_cos(to_triton(x_cpu))).to(torch.float32)
    if dtype == torch.float32:
        atol = rtol = 1e-5
    else:
        atol = rtol = 5e-3
    assert torch.allclose(torch_out, triton_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sin_correctness(shape, dtype):
    require_triton_device()
    torch.manual_seed(0)
    x_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=dtype)
    torch_out = torch.sin(x_cpu.to(torch.float32))

    triton_out = from_triton(triton_sin(to_triton(x_cpu))).to(torch.float32)
    if dtype == torch.float32:
        atol = rtol = 1e-5
    else:
        atol = rtol = 5e-3
    assert torch.allclose(torch_out, triton_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
])
def test_argmax(shape):
    require_triton_device()
    torch.manual_seed(0)
    x_cpu = torch.randn(shape, device=CPU_DEVICE, dtype=torch.float32)
    torch_idx = torch.argmax(x_cpu).item()

    triton_idx = triton_argmax(to_triton(x_cpu))
    assert torch_idx == triton_idx


