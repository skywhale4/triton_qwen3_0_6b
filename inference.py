import os
import torch
import argparse
from transformers import AutoTokenizer
from model import Qwen3Triton
from kernels.sampling import sample_with_temperature


def generate_text(model, tokenizer, prompt, max_new_tokens=64, temperature=0.7, top_p=0.9):
    """使用 Triton 内核进行文本生成"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
    
    # prefill builds KV cache and returns hidden states
    x = model.prefill(input_ids)
    pos = input_ids.shape[1]
    
    # first token from prefill last state
    logits = model.logits(x[:, -1, :])
    
    # 使用温度采样而非 argmax
    next_token = sample_with_temperature(logits, temperature=temperature, top_p=top_p)
    generated = [int(next_token[0].item())]
    
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151645
    
    for _ in range(max_new_tokens - 1):
        # decode one step using last token, append to cache inside
        logits = model.decode(next_token[0], start_pos=pos)
        pos += 1
        
        # 使用温度采样
        next_token = sample_with_temperature(logits, temperature=temperature, top_p=top_p)
        tid = int(next_token[0].item())
        
        if tid == eos_id:
            break
        generated.append(tid)
    
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Triton 推理脚本")
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="输入提示文本"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=64,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="温度参数"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="top-p 采样参数"
    )
    
    args = parser.parse_args()
    
    # 初始化模型和tokenizer
    model_dir = os.path.join(os.getcwd(), 'Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3Triton(model_dir, dtype=torch.float32)
    
    if args.prompt:
        # 单次推理模式
        print(f"input prompt: {args.prompt}")
        print("inference result: ")
        
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(result)
    else:
        # 默认示例
        examples = [
            "hello",
            "Please introduce yourself",
            "What is artificial intelligence?"
        ]
        
        for prompt in examples:
            print(f"input prompt: {prompt}")
            print("inference result: ")
            
            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            print(result)
            print()


if __name__ == '__main__':
    main()