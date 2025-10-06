# Triton-Qwen3-0.6B

Qwen3-0.6B 推理实验，三条路径：

- `scripts/hf_inference.py`：直接用 HuggingFace
- `scripts/torch_inference.py`：纯 PyTorch 手写前向
- `scripts/triton_inference.py`：换成 Triton kernel

## 模型

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./qwen3-0-6B
```

## 运行

```bash
python scripts/hf_inference.py --prompt "Hello"
python scripts/torch_inference.py --prompt "Hello"
python scripts/triton_inference.py --prompt "Hello"
python scripts/triton_inference.py --prompt "Hello" --dtype fp16
```

`torch` 与 `triton` 脚本默认做 argmax 采样，方便逐行对比。

`triton_inference.py` 默认以 `fp32` 运行，可用 `--dtype {fp32,fp16,bf16}` 切换精度。

## 其他

- 逐层对齐：`--debug-single-layer`
- 测速度：`--benchmark`


