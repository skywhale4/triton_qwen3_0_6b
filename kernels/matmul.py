import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        # 大形状稳健配置（更大的 BLOCK_K，减少累加轮数）
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'SPLIT_K': 1}, num_warps=8, num_stages=3),
        # split-K：将 K 维拆分并用原子加合并，降低长 K 的线性累加误差
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'SPLIT_K': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'SPLIT_K': 4}, num_warps=8, num_stages=3),
        # 中等形状备选
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'SPLIT_K': 1}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel_fp32_tuned(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # 2D + split-K 网格
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 本程序实例负责的 K 片段
    k_per_split = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 显式 K 偏移寻址，减少边界掩码误差；大 BLOCK_K 降低累加轮数
    k0 = k_start
    while k0 < k_end:
        k_vec = k0 + offs_k
        k_mask = k_vec < k_end

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_vec[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_vec[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)

        k0 += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # split-K 合并
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


@triton.jit
def _matmul_kernel_fp32_fixed(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_per_split = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = k_start
    while k0 < k_end:
        k_vec = k0 + offs_k
        k_mask = k_vec < k_end

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_vec[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_vec[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)

        k0 += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


@triton.jit
def _matmul_kernel_fp32_kahan(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D 网格：每个程序实例计算 [BLOCK_M, BLOCK_N] 的子块
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器 + Kahan 补偿器（降低长 K 维的舍入误差）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 显式 K 偏移寻址，避免指针滚动的边界差异；较大的 BLOCK_K 降低累加轮数
    for k0 in range(0, K, BLOCK_K):
        k_vec = k0 + offs_k
        k_mask = k_vec < K

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_vec[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_vec[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        partial = tl.dot(a, b, allow_tf32=False)  # FP32 FMA

        # Kahan 补偿求和：acc += partial（带补偿）
        y = partial - comp
        t = acc + y
        comp = (t - acc) - y
        acc = t

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None,
    block_m: int | None = None, block_n: int | None = None, block_k: int | None = None,
    split_k: int | None = None, num_warps: int | None = None, num_stages: int | None = None,
):
    """
    FP32 matmul (A @ B)，2D 网格 + 显式 K 偏移 + 可选 split-K（原子加合并）。
    - 常规/推理：不传块大小 → autotune 路径，自动选择 BLOCK_* 与 SPLIT_K
    - 测试/调参：传入 block_* / split_k → 固定块大小路径
    """
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    else:
        assert out.shape == (M, N) and out.is_cuda and out.dtype == torch.float32

    # 自动调优路径
    if block_m is None or block_n is None or block_k is None:
        # 使用可调用 grid，让 Triton 按选中的配置动态设置第三维 SPLIT_K
        def grid(meta):
            BM = meta['BLOCK_M']
            BN = meta['BLOCK_N']
            SK = meta['SPLIT_K']
            return (triton.cdiv(M, BM), triton.cdiv(N, BN), SK)

        # 若可能走 split-K，先清零 out，以保证 atomic_add 合并从 0 开始
        out.zero_()

        _matmul_kernel_fp32_tuned[grid](
            a, b, out,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
        )
        return out

    # 固定块大小路径
    assert block_m > 0 and block_n > 0 and block_k > 0
    if split_k is None or split_k < 1:
        split_k = 1

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), split_k)

    # split-K 需要先将 out 清零
    if split_k > 1:
        out.zero_()

    launch_kw = {}
    if num_warps is not None:
        launch_kw['num_warps'] = num_warps
    if num_stages is not None:
        launch_kw['num_stages'] = num_stages

    _matmul_kernel_fp32_fixed[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        **launch_kw,
    )
    return out