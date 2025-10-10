import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.embedding import triton_embedding
from tests import require_triton_device, to_triton, from_triton


CPU_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("batch, seq_len, vocab, hidden", [
    (1, 9, 151936, 1024),
    (2, 16, 5000, 256),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_embedding_correctness(batch, seq_len, vocab, hidden, dtype):
    torch.manual_seed(0)
    ids = to_triton(torch.randint(0, vocab, (batch, seq_len), device=CPU_DEVICE, dtype=torch.long))
    emb_cpu = nn.Embedding(vocab, hidden, device=CPU_DEVICE, dtype=dtype)
    emb_triton = nn.Embedding(vocab, hidden, device=ids.device, dtype=dtype)
    emb_triton.weight.data.copy_(emb_cpu.weight.data.to(ids.device))

    torch_out = emb_cpu(ids.cpu()).to(torch.float32)

    triton_out = from_triton(triton_embedding(ids, emb_triton.weight)).to(torch.float32)

    tol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(torch_out, triton_out, atol=tol)


def test_embedding_single_token():
    torch.manual_seed(0)
    ids = to_triton(torch.tensor([[0]], device=CPU_DEVICE, dtype=torch.long))
    embed_cpu = nn.Embedding(10, 4, device=CPU_DEVICE)
    embed_cpu.weight.data.copy_(torch.arange(40, device=CPU_DEVICE).view(10, 4))
    embed_triton = nn.Embedding(10, 4, device=ids.device)
    embed_triton.weight.data.copy_(embed_cpu.weight.data.to(ids.device))

    torch_out = embed_cpu(ids.cpu()).to(torch.float32)

    triton_out = from_triton(triton_embedding(ids, embed_triton.weight)).to(torch.float32)

    assert torch.allclose(torch_out, triton_out)


if __name__ == "__main__":
    pytest.main([__file__])

