import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import List


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sample_next_argmax(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits.squeeze(0)).item())


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B HF Inference (显式前向)")
    parser.add_argument("--model-path", type=str, default="./qwen3-0-6B", help="模型目录")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?", help="输入提示")
    parser.add_argument("--max-length", type=int, default=1000, help="最大总长度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda/cpu")
    parser.add_argument("--benchmark", action="store_true", help="基准测试模式（多次迭代）")
    parser.add_argument("--num-iterations", type=int, default=10, help="基准测试迭代次数")
    parser.add_argument("--debug-single-layer", action="store_true", help="调试模式：只运行第一层并打印所有中间张量")
    
    args = parser.parse_args()
    
                                         
    dtype = torch.float32
    print(f"Loading model from: {args.model_path}")
    print(f"Device: {args.device}, dtype: {dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    
    print(f"\nModel loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters\n")
    
                                
    if args.debug_single_layer:
        print("\n" + "="*80)
        print("调试模式：只运行第一层，打印所有中间张量（HF标准）")
        print("="*80 + "\n")
        
                
        captured = {}
        
        def make_capture_hook(name):
            def hook(module, input, output):
                            
                if isinstance(output, tuple):
                    captured[name] = output[0].detach()
                else:
                    captured[name] = output.detach()
            return hook
        
                  
        layer_0 = model.model.layers[0]
        layer_0.input_layernorm.register_forward_hook(make_capture_hook('input_norm'))
        layer_0.self_attn.q_proj.register_forward_hook(make_capture_hook('q_proj'))
        layer_0.self_attn.k_proj.register_forward_hook(make_capture_hook('k_proj'))
        layer_0.self_attn.v_proj.register_forward_hook(make_capture_hook('v_proj'))
        layer_0.self_attn.q_norm.register_forward_hook(make_capture_hook('q_norm'))
        layer_0.self_attn.k_norm.register_forward_hook(make_capture_hook('k_norm'))
        
                                
        def capture_o_proj_input(module, input, output):
            captured['o_proj_input'] = input[0].detach()                 
            captured['o_proj'] = output.detach()
        
        layer_0.self_attn.o_proj.register_forward_hook(capture_o_proj_input)
        layer_0.post_attention_layernorm.register_forward_hook(make_capture_hook('post_attn_norm'))
        layer_0.mlp.gate_proj.register_forward_hook(make_capture_hook('gate_proj'))
        layer_0.mlp.up_proj.register_forward_hook(make_capture_hook('up_proj'))
        layer_0.mlp.down_proj.register_forward_hook(make_capture_hook('down_proj'))
    
    def run_once_hf(prompt: str, debug_print: bool = False):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if debug_print:
            print("Processed input (after chat template formatting):")
            print("=" * 60)
            print(text)
            print("=" * 60 + "\n")

        inputs = tokenizer([text], return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"][0].tolist()

        max_new_tokens = max(1, args.max_length - len(input_ids))

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

                   
        with torch.no_grad():
                                  
            if args.debug_single_layer:
                       
                original_layers = model.model.layers
                          
                model.model.layers = original_layers[:1]
                
                      
                embed_out = model.model.embed_tokens(inputs["input_ids"])
                print("\n[HF] Embedding output [0,-1,:50]:")
                print(f"     {embed_out[0, -1, :50].cpu().tolist()}\n")
                print("="*80)
                print("Processing Layer 0 (HF) - Using model forward with hooks")
                print("="*80 + "\n")
                
                outputs = model(input_ids=inputs["input_ids"], use_cache=True, output_hidden_states=True)
                
                       
                model.model.layers = original_layers
                
                           
                print(f"[HF] input_layernorm output [0,-1,:50]:")
                print(f"     {captured['input_norm'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] q_proj output [0,-1,:50]:")
                print(f"     {captured['q_proj'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] k_proj output [0,-1,:50]:")
                print(f"     {captured['k_proj'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] v_proj output [0,-1,:50]:")
                print(f"     {captured['v_proj'][0, -1, :50].cpu().tolist()}")
                
                                                  
                print(f"[HF] q_norm output [0,-1,0,:50] (head 0):")
                print(f"     {captured['q_norm'][0, -1, 0, :50].cpu().tolist()}")
                print(f"[HF] k_norm output [0,-1,0,:50] (head 0):")
                print(f"     {captured['k_norm'][0, -1, 0, :50].cpu().tolist()}")
                
                print(f"[HF] Context (before o_proj) [0,-1,:50]:")
                print(f"     {captured['o_proj_input'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] o_proj output [0,-1,:50]:")
                print(f"     {captured['o_proj'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] post_attention_layernorm output [0,-1,:50]:")
                print(f"     {captured['post_attn_norm'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] gate_proj output [0,-1,:50]:")
                print(f"     {captured['gate_proj'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] up_proj output [0,-1,:50]:")
                print(f"     {captured['up_proj'][0, -1, :50].cpu().tolist()}")
                print(f"[HF] down_proj output [0,-1,:50]:")
                print(f"     {captured['down_proj'][0, -1, :50].cpu().tolist()}")
                
                hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()
                print(f"\n[HF] After Layer 0 + final norm [0,-1,:50]:")
                print(f"     {hidden[:50].tolist()}\n")
                print("="*80)
                print("调试模式完成：Prefill 阶段已结束（HF标准）")
                print("="*80)
                return "", 0.0, 0
            
            outputs = model(input_ids=inputs["input_ids"], use_cache=True, output_hidden_states=True)
            past_kvs = outputs.past_key_values
            logits = outputs.logits

                                           
        hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()
        print("[HF] Prefill hidden_state (last token, first 50):")
        print(hidden[:50].tolist())
        print()

        generated: List[int] = []
        for _ in range(max_new_tokens):
            next_id = sample_next_argmax(logits[:, -1, :])
            generated.append(next_id)
            eos_ids = tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                if next_id == eos_ids:
                    break
            elif isinstance(eos_ids, list) and next_id in eos_ids:
                break
                         
            inp_next = torch.tensor([[next_id]], device=args.device, dtype=torch.long)
            with torch.no_grad():
                outputs = model(input_ids=inp_next, past_key_values=past_kvs, use_cache=True)
                past_kvs = outputs.past_key_values
                logits = outputs.logits

        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        text_out = tokenizer.decode(input_ids + generated, skip_special_tokens=True)
        latency = t1 - t0
        ntok = len(generated)
        return text_out, latency, ntok

    if args.debug_single_layer:
                            
        run_once_hf(args.prompt, debug_print=True)
        return
    
    if args.benchmark:
        print("=" * 60)
        print(f"Running benchmark mode ({args.num_iterations} iterations)")
        print("=" * 60)
        
        latencies, tokens_generated = [], []

        for i in range(args.num_iterations):
            out, lat, ntok = run_once_hf(args.prompt, debug_print=(i == 0))
            latencies.append(lat)
            tokens_generated.append(ntok)
            if i == 0:
                print(f"\nIteration 1 output:")
                print(f"Prompt: {args.prompt}")
                print(f"Generated: {out}")
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
        print(f"Prompt: {args.prompt}")
        print(f"Generating (max_length={args.max_length})...\n")
        out, lat, ntok = run_once_hf(args.prompt, debug_print=True)
        print(f"\nGenerated text:\n{out}\n")
        print("=" * 60)
        print(f"Generation time: {lat:.3f}s")
        print(f"Tokens generated: {ntok}")
        print(f"Throughput: {ntok/max(lat,1e-6):.2f} tokens/s")


if __name__ == "__main__":
    main()
