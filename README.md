# Triton-Qwen3-0.6B


## 环境准备

```bash
uv pip install -r requirements.txt
```

可选环境变量：

- `TRITON_DEVICE`：指定 Triton kernel 的目标设备，默认自动选择 `cuda`（若可用），否则使用 `cpu`。

## 运行单测

```bash
pytest tests
```

单测会：

- 在 CPU 上构造 PyTorch 参考结果；
- 将张量通过 `tests.utils.to_triton()` 传入 Triton kernel；
- 比较 Triton 与 PyTorch 输出一致性。

自定义后端环境可通过设置 `TRITON_DEVICE=cpu` 来保持数据驻留 CPU，并替换底层 Triton runtime。

## 推理脚本

推理脚本都支持 `--plain-prompt`（跳过 chat 模板）以及 `--max-new-tokens`（显式限制生成长度）。

- `scripts/hf_inference.py`
- `scripts/torch_inference.py`
- `scripts/triton_inference.py`

示例：

```bash
python scripts/triton_inference.py --prompt "hello" --plain-prompt --max-new-tokens 1
python scripts/triton_inference.py --prompt "hello" --plain-prompt --max-new-tokens 1 --dtype fp16
python scripts/torch_inference.py --prompt "hello" --plain-prompt --max-new-tokens 1
python scripts/hf_inference.py --prompt "hello" --plain-prompt --max-new-tokens 1
```

```bash
python scripts/triton_inference.py --prompt "hello" --plain-prompt --single-layer
python scripts/triton_inference.py --prompt "hello" --plain-prompt --single-layer --dtype fp16
python scripts/torch_inference.py --prompt "hello" --plain-prompt --single-layer
python scripts/hf_inference.py --prompt "hello" --plain-prompt --single-layer
```

- 默认使用 CPU 张量与 Triton kernel；
- `--device` 缺省取 `TRITON_DEVICE`



