import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen3_triton import build_triton_model, load_triton_from_hf
from kernels.reduction import triton_argmax


def pick_argmax(logits: torch.Tensor) -> int:
    logits = logits.squeeze(0)
    if logits.device.type == "cuda":
        return triton_argmax(logits)
    return int(torch.argmax(logits).item())


def format_tokens(tokenizer, token_ids: List[int]) -> str:
    if not token_ids:
        return "<empty>"
    parts = []
    for tid in token_ids:
        text = tokenizer.decode([tid], skip_special_tokens=False)
        display = text.replace("\n", "\\n").replace("\t", "\\t")
        if display == "":
            display = "<EMPTY>"
        parts.append(f"{tid}:{display}")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton Qwen3-0.6B inference")
    parser.add_argument("--model-path", type=str, default="qwen3-0-6B")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override number of tokens to generate")
    parser.add_argument("--plain-prompt", action="store_true", help="Use prompt as-is without chat template")
    parser.add_argument("--single-layer", action="store_true", help="Limit decoder to one layer and generate single token")
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--parity", action="store_true")
    return parser.parse_args()


def locate_model(path_str: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    mp = Path(path_str)
    model_dir = mp if mp.is_absolute() else (root / mp)
    if model_dir.exists():
        return model_dir
    for cand in (root / "qwen3-0-6B", root / "Qwen3-0-6B"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Model dir not found: {model_dir}")


def to_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_triton(model_dir: Path, use_triton: bool, dtype: torch.dtype, single_layer: bool):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    if single_layer:
        hf_model.config.num_hidden_layers = 1
        hf_model.model.layers = hf_model.model.layers[:1]
    model = build_triton_model(str(model_dir / "config.json"), use_triton=use_triton, dtype=dtype)
    if single_layer:
        model.config.num_hidden_layers = 1
        model.model.layers = torch.nn.ModuleList(model.model.layers[:1])
    model.to(dtype=dtype, device="cpu")
    load_triton_from_hf(hf_model, model)
    return tokenizer, hf_model, model


def parity_check(tokenizer, hf_model, triton_model, model_dir: Path, prompt: str, device: str, dtype: torch.dtype):
    from models.qwen3_torch import build_model_from_config, load_from_hf

    torch_model = build_model_from_config(str(model_dir / "config.json"))
    torch_model.to(dtype=torch.float32, device=device)
    load_from_hf(hf_model, torch_model)
    triton_model.to(device=device, dtype=dtype)

    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    inp = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        torch_logits, _, _ = torch_model(inp["input_ids"], past_kvs=None, use_cache=False)
        triton_logits, _, _ = triton_model(inp["input_ids"], past_kvs=None, use_cache=False)

    diff = (torch_logits[:, -1, :].float() - triton_logits[:, -1, :].float()).abs()
    print(f"max diff {diff.max().item():.3e}, mean diff {diff.mean().item():.3e}")


def run_once(tokenizer, model, device: str, prompt: str, max_length: int, max_new_tokens: Optional[int], use_chat: bool, dtype: torch.dtype, single_layer: bool) -> Tuple[str, float, int, Optional[torch.Tensor]]:
    if use_chat:
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()

    print("Input tokens :")
    print(f"  {format_tokens(tokenizer, input_ids)}")
    if single_layer:
        max_new = 1
    elif max_new_tokens is not None:
        max_new = max_new_tokens
    else:
        max_new = max(1, max_length - len(input_ids))

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        logits, past_kvs, hidden_states = model(
            torch.tensor([input_ids], device=device, dtype=torch.long),
            past_kvs=None,
            use_cache=not single_layer,
            output_hidden_states=single_layer,
        )

    generated: List[int] = []
    hidden_state: Optional[torch.Tensor] = None
    if single_layer:
        hidden_state = hidden_states[-1][0, -1, :].float().cpu()
        nxt = pick_argmax(logits[:, -1, :])
        generated.append(nxt)
    else:
        eos_token = tokenizer.eos_token_id
        for _ in range(max_new):
            next_id = pick_argmax(logits[:, -1, :])
            generated.append(next_id)

            if isinstance(eos_token, int):
                if next_id == eos_token:
                    break
            elif isinstance(eos_token, list):
                if next_id in eos_token:
                    break

            with torch.no_grad():
                logits, past_kvs, _ = model(
                    torch.tensor([[next_id]], device=device, dtype=torch.long),
                    past_kvs=past_kvs,
                    use_cache=True,
                )

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    text_out = tokenizer.decode(input_ids + generated, skip_special_tokens=True)
    print("Generated tokens:")
    print(f"  {format_tokens(tokenizer, generated)}")
    return text_out, t1 - t0, len(generated), hidden_state


def main():
    args = parse_args()
    model_dir = locate_model(args.model_path)
    dtype = to_dtype(args.dtype)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"model: {model_dir}")
    print(f"device: {args.device}")
    print(f"backend: {'triton' if not args.no_triton else 'torch'}")

    if not args.no_triton:
        print("kernels: embedding, linear, rmsnorm, rope, attention, swiglu, kv-cache, argmax")

    tokenizer, hf_model, triton_model = load_triton(model_dir, use_triton=not args.no_triton, dtype=dtype, single_layer=args.single_layer)

    generate_tokens = args.max_new_tokens if args.max_new_tokens is not None else None
    use_chat_template = not args.plain_prompt

    if args.parity:
        parity_check(tokenizer, hf_model, triton_model, model_dir, args.prompt, args.device, dtype)
        return

    del hf_model
    triton_model.to(args.device, dtype=dtype)
    triton_model.eval()

    if args.benchmark:
        latencies, tokens = [], []
        for _ in range(args.num_iterations):
            _, lat, tok, _ = run_once(tokenizer, triton_model, args.device, args.prompt, args.max_length, generate_tokens, use_chat_template, dtype, args.single_layer)
            latencies.append(lat)
            tokens.append(tok)
        total_time = sum(latencies)
        print(f"avg latency {total_time/len(latencies):.3f}s")
        print(f"avg tokens  {sum(tokens)/len(tokens):.1f}")
        print(f"throughput  {sum(tokens)/max(total_time, 1e-6):.2f} tok/s")
        return

    out, lat, tok, _ = run_once(tokenizer, triton_model, args.device, args.prompt, args.max_length, generate_tokens, use_chat_template, dtype, args.single_layer)
    print(out)
    print("=" * 60)
    print(f"Generation time: {lat:.3f}s")
    print(f"Tokens generated: {tok}")
    print(f"Throughput: {tok/max(lat, 1e-6):.2f} tokens/s")


if __name__ == "__main__":
    main()
