import torch
from .matmul import matmul
from .elementwise import silu_mul

def mlp_swiglu(x, gate_w, up_w, down_w):
    """
    SwiGLU MLP: (silu(x @ gate_w) * (x @ up_w)) @ down_w
    Weight shapes: gate_w, up_w, down_w are stored as [out_features, in_features]
    """
    assert x.is_cuda and gate_w.is_cuda and up_w.is_cuda and down_w.is_cuda
    assert x.dtype == torch.float32, f"Only float32 supported, got {x.dtype}"
    
    B, T, D = x.shape
    # gate_w shape: [intermediate_size, D] -> we want [D, intermediate_size] for matmul
    intermediate_size = gate_w.shape[0]  # e.g., 3072
    
    x_flat = x.reshape(B * T, D)
    
    # Pre-allocate tensors
    gate_out = torch.empty((B * T, intermediate_size), device=x.device, dtype=torch.float32)
    up_out = torch.empty((B * T, intermediate_size), device=x.device, dtype=torch.float32)
    silu_out = torch.empty((B * T, intermediate_size), device=x.device, dtype=torch.float32)
    output = torch.empty((B * T, D), device=x.device, dtype=torch.float32)
    
    # Compute: x @ W_gate.T and x @ W_up.T
    # gate_w.T: [D, intermediate_size]
    # matmul(x_flat, gate_w.t(), gate_out)  # [B*T, D] @ [D, 3072] -> [B*T, 3072]
    gate_out = torch.matmul(x_flat, gate_w.t())  # [B*T, D] @ [D, 3072] -> [B*T, 3072]
    # matmul(x_flat, up_w.t(), up_out)      # [B*T, D] @ [D, 3072] -> [B*T, 3072]
    up_out = torch.matmul(x_flat, up_w.t())  # [B*T, D] @ [D, 3072] -> [B*T, 3072]
    
    # Apply SwiGLU: silu(gate_out) * up_out
    silu_mul(gate_out, up_out, silu_out)  # [B*T, 3072]
    
    # Final projection: result @ W_down.T
    # down_w.T: [3072, D]
    # matmul(silu_out, down_w.t(), output)  # [B*T, 3072] @ [3072, 1024] -> [B*T, 1024]
    output = torch.matmul(silu_out, down_w.t())  # [B*T, 3072] @ [3072, 1024] -> [B*T, 1024]
    
    return output.reshape(B, T, D)