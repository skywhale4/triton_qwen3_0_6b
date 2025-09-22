import pytest
import torch

from kernels.sampling import sample_with_temperature

# 采样是随机的，但我们通过保存/恢复 RNG 状态，实现与“参考实现”逐位一致的可重复性
def _save_rng(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device), "cuda"
    return torch.random.get_rng_state(), "cpu"


def _restore_rng(state, device: torch.device, kind: str):
    if kind == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.random.set_rng_state(state)


def _softmax_with_temperature_and_topp(logits: torch.Tensor, temperature: float, top_p: float):
    scaled_logits = logits / temperature
    if top_p < 1.0:
        probs = torch.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs <= top_p
        mask[:, 0] = True  # 至少保留一个
        filtered_logits = scaled_logits.clone()
        sorted_mask = torch.zeros_like(mask, dtype=torch.bool)
        sorted_mask.scatter_(1, sorted_indices, mask)
        filtered_logits.masked_fill_(~sorted_mask, float("-inf"))
        probs = torch.softmax(filtered_logits, dim=-1)
        return probs
    else:
        return torch.softmax(scaled_logits, dim=-1)


def _rand_logits(shape, low=-5.0, high=5.0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    return (high - low) * torch.rand(shape, device=device, generator=g, dtype=torch.float32) + low


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Prefer CUDA for consistency with other tests")
@pytest.mark.parametrize("M,N", [(1, 5), (2, 33), (3, 257)])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_sampling_temperature_only_matches_reference(M, N, temperature):
    device = torch.device("cuda")
    logits = _rand_logits((M, N), device="cuda")
    top_p = 1.0

    # 设定全局随机种子并保存当前 RNG 状态
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    state, kind = _save_rng(device)

    # 参考实现：用同一 RNG 状态进行 multinomial
    probs_ref = _softmax_with_temperature_and_topp(logits, temperature, top_p)
    ref = torch.multinomial(probs_ref, num_samples=1).squeeze(-1)

    # 恢复 RNG 状态后，调用被测函数，应得到一致的采样结果
    _restore_rng(state, device, kind)
    out = sample_with_temperature(logits, temperature=temperature, top_p=top_p)

    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Prefer CUDA for consistency with other tests")
@pytest.mark.parametrize("M,N", [(1, 10), (2, 50), (4, 500)])
@pytest.mark.parametrize("top_p", [0.9, 0.8, 0.5])
def test_sampling_with_top_p_matches_reference(M, N, top_p):
    device = torch.device("cuda")
    logits = _rand_logits((M, N), device="cuda")
    temperature = 0.7

    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    state, kind = _save_rng(device)

    probs_ref = _softmax_with_temperature_and_topp(logits, temperature, top_p)
    ref = torch.multinomial(probs_ref, num_samples=1).squeeze(-1)

    _restore_rng(state, device, kind)
    out = sample_with_temperature(logits, temperature=temperature, top_p=top_p)

    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Prefer CUDA for consistency with other tests")
@pytest.mark.parametrize("M,N,top_p", [(3, 100, 0.2), (2, 1000, 0.05)])
def test_sampling_result_is_within_top_p_set(M, N, top_p):
    device = torch.device("cuda")
    logits = _rand_logits((M, N), device="cuda")

    out = sample_with_temperature(logits, temperature=1.0, top_p=top_p)

    # 计算每个样本的 top-p 索引集合，并验证采样索引一定落在集合内
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum_probs <= top_p
    mask[:, 0] = True
    keep = torch.zeros_like(mask, dtype=torch.bool)
    keep.scatter_(1, sorted_indices, mask)

    for i in range(M):
        assert keep[i, out[i]].item() is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Prefer CUDA for consistency with other tests")
@pytest.mark.parametrize("M,N", [(1, 17), (2, 101)])
def test_top_p_very_small_reduces_to_argmax(M, N):
    # 当 top_p 极小（保留第一个最高概率 token），采样结果应等于 argmax（确定性）
    device = torch.device("cuda")
    logits = _rand_logits((M, N), device="cuda")

    out = sample_with_temperature(logits, temperature=1.0, top_p=1e-8)
    ref = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

    torch.testing.assert_close(out, ref)