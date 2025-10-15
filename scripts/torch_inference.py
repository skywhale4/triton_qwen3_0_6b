import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen3_torch import build_model_from_config, load_from_hf


def sample_next_argmax(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits.squeeze(0)).item())


def decode_text(tokenizer, gen_ids: List[int]) -> str:
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Torch Inference (manual forward)")
    parser.add_argument("--model-path", type=str, default="qwen3-0-6B", help="相对项目根目录的模型目录")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?", help="输入提示")
    parser.add_argument("--max-length", type=int, default=1000, help="最大总长度（与 hf_inference 对齐）")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="显式控制续写长度")
    parser.add_argument("--plain-prompt", action="store_true", help="不使用 chat 模板，直接将 prompt 输入模型")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda/cpu")
    parser.add_argument("--benchmark", action="store_true", help="基准测试模式（多次迭代）")
    parser.add_argument("--num-iterations", type=int, default=10, help="基准测试迭代次数")
    parser.add_argument("--parity", action="store_true", help="与HF logits对齐检测后退出")
    parser.add_argument("--parity-layers", action="store_true", help="逐层对齐：打印各层隐藏状态/最终logits差异")
    parser.add_argument("--debug-single-layer", action="store_true", help="调试模式：只运行第一层并打印所有中间张量")
    parser.add_argument("--single-layer", action="store_true", help="仅保留单层并直接根据隐藏状态生成")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    mp = Path(args.model_path)
    model_dir = (mp if mp.is_absolute() else (root / mp)).resolve()
    if not model_dir.exists() and not mp.is_absolute():
                   
        candidates = []
        if mp.name.lower() == "qwen3-0-6b":
            candidates = [root / "qwen3-0-6B", root / "Qwen3-0-6B"]
        else:
            candidates = [root / "qwen3-0-6B", root / "Qwen3-0-6B"]
        for cand in candidates:
            if cand.exists():
                model_dir = cand.resolve()
                break
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

                               
    dtype = torch.float32
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading model from: {model_dir}")
    print(f"Device: {args.device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
                                              
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )

    torch_model = build_model_from_config(str(model_dir / "config.json"))
    torch_model.to(dtype=torch.float32, device="cpu")
    if args.single_layer:
        torch_model.config.num_hidden_layers = 1
        torch_model.model.layers = torch.nn.ModuleList(torch_model.model.layers[:1])
    load_from_hf(hf_model, torch_model)

                                       
    if args.parity or args.parity_layers:
        tok = tokenizer
        if args.plain_prompt:
            text = args.prompt
        else:
            messages = [{"role": "user", "content": args.prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok([text], return_tensors="pt")
        with torch.no_grad():
                                           
            hf_out = hf_model(**inp, output_hidden_states=True, use_cache=False)
            hf_logits = hf_out.logits[:, -1, :].float()
            hf_hiddens = [h.float() for h in hf_out.hidden_states[1:]]                         

            my_logits, _, my_hiddens = torch_model(
                inp["input_ids"], past_kvs=None, use_cache=False, output_hidden_states=True
            )
            my_logits = my_logits[:, -1, :].float()

                     
        diff = (hf_logits - my_logits).abs()
        print(f"parity max_abs_diff={diff.max().item():.6f} mean_abs_diff={diff.mean().item():.6f}")

        if args.parity_layers:
                                                                                 
            num_layers = min(len(hf_hiddens), len(my_hiddens))
            for i in range(num_layers):
                            
                a = hf_hiddens[i]
                b = my_hiddens[i]
                                     
                da = a[:, -1, :]
                db = b[:, -1, :]
                d = (da - db).abs()
                print(f"layer[{i}] last_token max={d.max().item():.6f} mean={d.mean().item():.6f}")
        return
    del hf_model
                              
    torch_model.to(args.device, dtype=torch.float32)
    torch_model.eval()
    if args.single_layer:
        torch_model.model.debug_single_layer = True
    
            
    if args.debug_single_layer:
        print("\n" + "="*80)
        print("调试模式：只运行第一层，打印所有中间张量")
        print("="*80 + "\n")
        torch_model.set_debug_mode(debug=True, single_layer=True)

    def run_once(prompt: str, debug_print: bool = False) -> Tuple[str, float, int]:
        if args.plain_prompt:
            text = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if debug_print and not args.plain_prompt:
            print("\nProcessed input (after prompt formatting):")
            print("=" * 60)
            print(text)
            print("=" * 60)

        inputs = tokenizer([text], return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()

        print("Input tokens :")
        print(f"  {format_tokens(tokenizer, input_ids)}")

                                                                   
        if args.max_new_tokens is not None:
            max_new_tokens = max(1, args.max_new_tokens)
        else:
            max_new_tokens = max(1, args.max_length - len(input_ids))

        if args.single_layer:
            max_new_tokens = 1

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            logits, past_kvs, _ = torch_model(
                torch.tensor([input_ids], device=args.device, dtype=torch.long),
                past_kvs=None,
                use_cache=True,
            )

                             
        if args.debug_single_layer:
            print("\n" + "="*80)
            print("调试模式完成：Prefill 阶段已结束")
            print("="*80)
            return "", 0.0, 0

        generated: List[int] = []
        hidden_state = None
        if args.single_layer:
            inputs_tensor = torch.tensor([input_ids], device=args.device, dtype=torch.long)
            hidden_out, _, _ = torch_model.model(
                inputs_tensor,
                past_kvs=None,
                use_cache=False,
                output_hidden_states=False,
            )
            hidden_state = hidden_out[0, -1, :].float().cpu()
            lm_logits = torch_model.lm_head(hidden_out[:, -1:, :])
            next_id = sample_next_argmax(lm_logits[:, 0, :])
            generated.append(next_id)
        else:
            eos_token = tokenizer.eos_token_id
            for _ in range(max_new_tokens):
                next_id = sample_next_argmax(logits[:, -1, :])
                generated.append(next_id)

                if isinstance(eos_token, int):
                    if next_id == eos_token:
                        break
                elif isinstance(eos_token, list):
                    if next_id in eos_token:
                        break

                with torch.no_grad():
                    logits, past_kvs, _ = torch_model(
                        torch.tensor([[next_id]], device=args.device, dtype=torch.long),
                        past_kvs=past_kvs,
                        use_cache=True,
                    )

        print("Generated tokens:")
        print(f"  {format_tokens(tokenizer, generated)}")

        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

                                              
        text_out = tokenizer.decode(input_ids + generated, skip_special_tokens=True)
        latency = t1 - t0
        new_tok = len(generated)
        return text_out, latency, new_tok

    if args.debug_single_layer:
                            
        run_once(args.prompt, debug_print=True)
        return
    
    if args.benchmark:
        print("\n" + "=" * 60)
        print(f"Running benchmark mode ({args.num_iterations} iterations)")
        print("=" * 60)

        latencies, tokens_generated = [], []
        first_out = None

        for i in range(args.num_iterations):
            out, lat, ntok = run_once(args.prompt, debug_print=(i == 0))
            latencies.append(lat)
            tokens_generated.append(ntok)
            if i == 0:
                first_out = out
                print("\nIteration 1 output:")
                print(f"Prompt: {args.prompt}")
                print(f"Generated: {first_out}")
                print("\n" + "=" * 60)
            print(f"Iteration {i+1}/{args.num_iterations}: {lat:.3f}s, {ntok} tokens, {ntok/max(lat,1e-6):.2f} tokens/s")

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        avg_throughput = avg_tokens / max(avg_latency, 1e-6)

        print("\n" + "=" * 60)
        print("Benchmark Results:")
        print("=" * 60)
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"Average tokens generated: {avg_tokens:.1f}")
        print(f"Average throughput: {avg_throughput:.2f} tokens/s")
        print(f"Min latency: {min(latencies):.3f}s")
        print(f"Max latency: {max(latencies):.3f}s")
    else:
        if args.plain_prompt:
            text = args.prompt
        else:
            messages = [{"role": "user", "content": args.prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _ = tokenizer([text], return_tensors="pt")

        out, lat, ntok = run_once(args.prompt, debug_print=False)
        print(f"Generated text:\n{out}\n")
        print("=" * 60)
        print(f"Generation time: {lat:.3f}s")
        print(f"Tokens generated: {ntok}")
        print(f"Throughput: {ntok/max(lat,1e-6):.2f} tokens/s")


if __name__ == "__main__":
    main()