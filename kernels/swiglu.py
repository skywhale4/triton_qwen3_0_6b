import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_activation_kernel(
    gate_ptr,                                  
    up_ptr,                                  
    output_ptr,                     
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
                  
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)
 
                                       
                                    
    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate
 
           
    output = silu_gate * up_f32
    tl.store(output_ptr + offsets, output.to(gate.dtype), mask=mask)


@triton.jit  
def silu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    output = x_f32 * tl.sigmoid(x_f32)
    tl.store(output_ptr + offsets, output.to(x.dtype), mask=mask)


def triton_swiglu_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:

    assert gate.is_cuda and up.is_cuda
    assert gate.shape == up.shape
    
        
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    output_flat = torch.empty_like(gate_flat)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    swiglu_activation_kernel[grid](
        gate_flat, up_flat, output_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_flat.reshape(gate.shape)


def triton_silu(x: torch.Tensor) -> torch.Tensor:

    assert x.is_cuda
    
    x_flat = x.reshape(-1)
    n_elements = x_flat.numel()

    output_flat = torch.empty_like(x_flat)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    silu_kernel[grid](
        x_flat, output_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_flat.reshape(x.shape)
