import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
                  
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
                 
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
                                
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        
            
        acc += tl.dot(a, b)
    
                    
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_dtype = C.dtype.element_ty
    tl.store(c_ptrs, acc.to(out_dtype), mask=c_mask)


def _promote_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:

    if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16:
        return torch.bfloat16
    if a.dtype == torch.float16 or b.dtype == torch.float16:
        return torch.float16
    return torch.float32


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.is_cuda and b.is_cuda
    assert a.shape[-1] == b.shape[-2], f"Shape mismatch: {a.shape} @ {b.shape}"
    
            
    orig_shape_a = a.shape
    orig_shape_b = b.shape
    
                 
    if a.dim() > 2 or b.dim() > 2:
                 
        a_2d = a.reshape(-1, a.shape[-1])
        b_2d = b.reshape(-1, b.shape[-1]) if b.dim() > 2 else b
        
                                                   
                      
        if a.dim() == 4 and b.dim() == 4:
                                         
            a_2d = a.reshape(a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
            b_2d = b.reshape(b.shape[0] * b.shape[1], b.shape[2], b.shape[3])
            
                                           
                             
            outputs = []
            for i in range(a_2d.shape[0]):
                out_i = _matmul_2d(a_2d[i], b_2d[i])
                outputs.append(out_i)
            
            result = torch.stack(outputs, dim=0)
                                         
            return result.reshape(orig_shape_a[0], orig_shape_a[1], result.shape[1], result.shape[2])
    
                   
    return _matmul_2d(a, b)


def _matmul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.dim() == 2 and b.dim() == 2
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    out_dtype = _promote_dtype(a, b)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    
                 
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return c


def _matmul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.dim() == 2 and b.dim() == 2
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return c
