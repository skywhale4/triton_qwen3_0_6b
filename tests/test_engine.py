import os
import pytest
import torch

from model import Qwen3Triton
from kernels.argmax import argmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# 参考实现使用纯 FP32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

def _fake_ids(B, T, V, seed=123):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, V, (B, T), generator=g, device="cuda")


def test_engine_greedy_loop_small():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(ROOT, 'Qwen3-0.6B')
    m = Qwen3Triton(model_dir, dtype=torch.float32)

    B, T = 1, 4
    ids = _fake_ids(B, T, m.cfg.vocab_size)

    with torch.no_grad():
        # 预填充
        x = m.prefill(ids)
        pos = T

        # 第一步
        logits = m.logits(x[:, -1, :])
        nxt, _ = argmax(logits)
        seq = [int(nxt[0].item())]

        # 连续 decode 若干步（贪心，确定性）
        steps = 6
        for _ in range(steps):
            logits = m.decode(nxt[0], start_pos=pos)
            pos += 1
            nxt, _ = argmax(logits)
            seq.append(int(nxt[0].item()))

    # 基本断言：长度与词表范围
    assert len(seq) == steps + 1
    assert all(0 <= t < m.cfg.vocab_size for t in seq)

    # KV cache 递增性（第0层）
    assert m.k_caches[0] is not None and m.v_caches[0] is not None
    assert m.k_caches[0].shape[1] >= T + steps
    assert m.v_caches[0].shape[1] >= T + steps