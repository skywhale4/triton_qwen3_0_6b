import pytest
import torch

from kernels.embed import embed

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")


def _rand_ids(shape, V):
    g = torch.Generator(device="cuda").manual_seed(0)
    return torch.randint(0, V, shape, generator=g, device="cuda", dtype=torch.long)


def _rand_table(V, D):
    g = torch.Generator(device="cuda").manual_seed(1)
    return torch.randn(V, D, generator=g, device="cuda", dtype=torch.float32)


@pytest.mark.parametrize("B,N,V,D", [
    (1, 1, 17, 8),
    (2, 3, 31, 16),
    (4, 5, 97, 64),
])
def test_embed_matches_torch_gather_2d(B, N, V, D):
    ids = _rand_ids((B, N), V)
    table = _rand_table(V, D)

    out = embed(ids, table)          # [B,N,D]
    ref = table[ids]                 # PyTorch 索引

    assert out.shape == (B, N, D) and out.dtype == table.dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("B,V,D", [
    (3, 50, 32),
    (5, 127, 48),
])
def test_embed_matches_torch_gather_1d(B, V, D):
    ids = _rand_ids((B,), V)
    table = _rand_table(V, D)

    out = embed(ids, table)          # [B,D]
    ref = table[ids]

    assert out.shape == (B, D) and out.dtype == table.dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_embed_out_of_bounds_is_zero():
    V, D = 16, 32
    table = _rand_table(V, D)
    ids = torch.tensor([[0, 5, -1, 20]], device="cuda", dtype=torch.long)  # -1 和 20 越界
    out = embed(ids, table)  # [1,4,D]

    ref = torch.empty_like(out)
    ref[0, 0] = table[0]
    ref[0, 1] = table[5]
    ref[0, 2] = torch.zeros(D, device="cuda", dtype=table.dtype)
    ref[0, 3] = torch.zeros(D, device="cuda", dtype=table.dtype)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)