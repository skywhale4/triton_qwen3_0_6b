import triton
import triton.language as tl
import torch


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n):
    pid = tl.program_id(axis=0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32, \
        f"Only float32 supported, got a.dtype={a.dtype}, b.dtype={b.dtype}"
    assert a.shape == b.shape
    n = a.numel()
    
    if out is None:
        out = torch.empty_like(a, dtype=torch.float32)
    else:
        assert out.dtype == torch.float32, f"Output must be float32, got {out.dtype}"
    
    grid = (triton.cdiv(n, 1024),)
    _add_kernel[grid](a, b, out, n)
    return out


@triton.jit
def _scale_kernel(a_ptr, out_ptr, n, scale):
    pid = tl.program_id(axis=0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * scale, mask=mask)


def scale(a: torch.Tensor, scale: float, out: torch.Tensor | None = None) -> torch.Tensor:
    assert a.is_cuda
    n = a.numel()
    if out is None:
        out = torch.empty_like(a)
    _scale_kernel[(triton.cdiv(n, 1024),)](a, out, n, scale)
    return out


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n, input_dtype: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < n
    up = tl.load(up_ptr + offs, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0)
    
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    gate_f32 = gate.to(tl.float32)
    silu = gate_f32 * tl.sigmoid(gate_f32)  # 使用更稳定的 sigmoid
    result_f32 = silu * up.to(tl.float32)
    
    # 根据输入类型转换回原类型
    if input_dtype == tl.float16:
        result = result_f32.to(tl.float16)
    elif input_dtype == tl.bfloat16:
        result = result_f32.to(tl.bfloat16)
    else:
        result = result_f32  # 保持 float32
    
    tl.store(out_ptr + offs, result, mask=mask)


def silu_mul(gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    assert up.is_cuda and gate.is_cuda
    assert up.shape == gate.shape
    n = up.numel()
    if out is None:
        out = torch.empty_like(up)
    
    # 根据输入类型设置 dtype
    if up.dtype == torch.float16:
        input_dtype = tl.float16
    elif up.dtype == torch.bfloat16:
        input_dtype = tl.bfloat16
    else:
        input_dtype = tl.float32
    
    _silu_mul_kernel[(triton.cdiv(n, 1024),)](gate, up, out, n, input_dtype)
    return out

