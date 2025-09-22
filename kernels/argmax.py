import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_stage1(
    x_ptr, temp_max_ptr, temp_idx_ptr,
    M, N, stride_x_m, stride_x_n,
    BLOCK_N: tl.constexpr
):
    """Stage 1: Compute local max and argmax for each block"""
    pid_m = tl.program_id(axis=0)  # batch dimension
    pid_n = tl.program_id(axis=1)  # block dimension
    
    # Calculate block range
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    
    # Load data
    x_ptrs = x_ptr + pid_m * stride_x_m + n_offsets * stride_x_n
    x = tl.load(x_ptrs, mask=n_mask, other=-float('inf'))
    
    # Convert to float32 for precise computation
    x_f32 = x.to(tl.float32)
    
    # Compute local max and argmax
    local_max = tl.max(x_f32, axis=0)
    local_argmax = tl.argmax(x_f32, axis=0)
    
    # Convert to global index
    global_argmax = local_argmax + n_start
    
    # Store results
    num_blocks = tl.cdiv(N, BLOCK_N)
    temp_offset = pid_m * num_blocks + pid_n
    tl.store(temp_max_ptr + temp_offset, local_max)
    tl.store(temp_idx_ptr + temp_offset, global_argmax)


@triton.jit
def _argmax_stage2(
    temp_max_ptr, temp_idx_ptr, out_max_ptr, out_idx_ptr,
    M, num_blocks, BLOCK_SIZE: tl.constexpr
):
    """Stage 2: Find global max from local results"""
    pid_m = tl.program_id(axis=0)
    
    # Load all block results for this batch
    block_offsets = tl.arange(0, BLOCK_SIZE)
    block_mask = block_offsets < num_blocks
    
    base_offset = pid_m * num_blocks
    max_values = tl.load(temp_max_ptr + base_offset + block_offsets, 
                        mask=block_mask, other=-float('inf'))
    idx_values = tl.load(temp_idx_ptr + base_offset + block_offsets, 
                        mask=block_mask, other=0)
    
    # Find global maximum
    global_max = tl.max(max_values, axis=0)
    max_block_idx = tl.argmax(max_values, axis=0)
    
    # Get corresponding global index
    final_idx = tl.load(temp_idx_ptr + base_offset + max_block_idx)
    
    # Store final results
    tl.store(out_max_ptr + pid_m, global_max)
    tl.store(out_idx_ptr + pid_m, final_idx)


def argmax(x: torch.Tensor, block_size: int = 4096):
    """
    Multi-stage argmax implementation for large vocabularies
    
    Args:
        x: Input tensor [M, N] where N is vocabulary size
        block_size: Block size for stage 1 (default: 4096)
        
    Returns:
        idx: argmax indices [M]
        val: maximum values [M]
    """
    M, N = x.shape
    device = x.device
    dtype = x.dtype
    
    # Ensure block size doesn't exceed Triton limits
    if block_size > 8192:
        block_size = 8192
    
    num_blocks = triton.cdiv(N, block_size)
    
    # Temporary storage for stage 1 results
    temp_max = torch.empty((M, num_blocks), device=device, dtype=torch.float32)
    temp_idx = torch.empty((M, num_blocks), device=device, dtype=torch.int32)
    
    # Final output tensors
    out_max = torch.empty(M, device=device, dtype=dtype)
    out_idx = torch.empty(M, device=device, dtype=torch.int32)
    
    # Stage 1: Parallel local argmax
    grid1 = (M, num_blocks)
    _argmax_stage1[grid1](
        x, temp_max, temp_idx,
        M, N, x.stride(0), x.stride(1),
        BLOCK_N=block_size
    )
    
    # Stage 2: Global reduction
    stage2_block_size = triton.next_power_of_2(min(num_blocks, 1024))
    grid2 = (M,)
    _argmax_stage2[grid2](
        temp_max, temp_idx, out_max, out_idx,
        M, num_blocks, BLOCK_SIZE=stage2_block_size
    )
    
    return out_idx, out_max