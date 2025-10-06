import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.embedding import triton_embedding


@pytest.mark.parametrize("batch, seq_len, vocab, hidden", [
    (1, 9, 100, 1024),
    (2, 16, 151936, 1024),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_embedding_correctness(batch, seq_len, vocab, hidden, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ids = torch.randint(0, vocab, (batch, seq_len), device='cuda', dtype=torch.long)
    emb = nn.Embedding(vocab, hidden).cuda().to(dtype)

    torch_out = emb(ids)
    triton_out = triton_embedding(ids, emb.weight)

    max_diff = (torch_out - triton_out).abs().max().item()
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    assert max_diff < tol


def test_embedding_single_token():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ids = torch.tensor([[0]], device='cuda', dtype=torch.long)
    embed = nn.Embedding(10, 4).cuda()
    embed.weight.data.copy_(torch.arange(40, device='cuda').view(10, 4))

    torch_out = embed(ids)
    triton_out = triton_embedding(ids, embed.weight)

    assert torch.allclose(torch_out, triton_out)


if __name__ == "__main__":
    test_embedding_correctness(1, 9, 100, 1024)
    test_embedding_single_token()
