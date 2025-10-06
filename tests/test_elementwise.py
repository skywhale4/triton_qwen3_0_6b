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


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_add_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(shape, device='cuda', dtype=dtype)
    b = torch.randn(shape, device='cuda', dtype=dtype)
    torch_out = a + b
    triton_out = triton_add(a, b)
    tol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(torch_out, triton_out, atol=tol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_multiply_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.randn(shape, device='cuda', dtype=dtype)
    b = torch.randn(shape, device='cuda', dtype=dtype)
    torch_out = a * b
    triton_out = triton_multiply(a, b)
    tol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(torch_out, triton_out, atol=tol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cos_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(shape, device='cuda', dtype=dtype)
    torch_out = torch.cos(x)
    triton_out = triton_cos(x)
    if dtype == torch.float32:
        atol = rtol = 1e-5
    elif dtype == torch.float16:
        atol = rtol = 5e-3
    else:
        atol = rtol = 6.5e-2
    assert torch.allclose(torch_out, triton_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
    (2, 4, 8),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sin_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(shape, device='cuda', dtype=dtype)
    torch_out = torch.sin(x)
    triton_out = triton_sin(x)
    if dtype == torch.float32:
        atol = rtol = 1e-5
    elif dtype == torch.float16:
        atol = rtol = 5e-3
    else:
        atol = rtol = 6.5e-2
    assert torch.allclose(torch_out, triton_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [
    (32,),
    (8, 16),
])
def test_argmax(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    torch_idx = torch.argmax(x).item()
    triton_idx = triton_argmax(x)
    assert torch_idx == triton_idx


