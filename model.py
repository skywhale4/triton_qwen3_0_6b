import torch
from safetensors.torch import load_file

from kernels.embed import embed
from kernels.rmsnorm import rmsnorm, rmsnorm_2d
from kernels.matmul import matmul
from kernels.attn_prefill import attention_prefill
from kernels.mlp import mlp_swiglu
from kernels.rope import build_rope_cache, apply_rope
from kernels.elementwise import add
from kernels.attn_decode import attention_decode


class Qwen3Config:
    def __init__(self, cfg: dict):
        self.hidden_size = cfg['hidden_size']
        self.intermediate_size = cfg['intermediate_size']
        self.num_hidden_layers = cfg['num_hidden_layers']
        self.num_attention_heads = cfg['num_attention_heads']
        self.num_key_value_heads = cfg['num_key_value_heads']
        self.head_dim = cfg['head_dim']
        self.vocab_size = cfg['vocab_size']
        self.rms_norm_eps = cfg.get('rms_norm_eps', 1e-6)
        self.rope_theta = cfg.get('rope_theta', 1000000.0)


class Qwen3Triton:
    def __init__(self, model_dir: str, device: str = 'cuda', dtype: torch.dtype = torch.float32):
        import json, os
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            cfg = json.load(f)
        self.cfg = Qwen3Config(cfg)
        self.device = device
        self.dtype = dtype
        self.weights = load_file(os.path.join(model_dir, 'model.safetensors'))
        self._cast_to_device()
        self.cos, self.sin = build_rope_cache(self.cfg.head_dim, 40960, self.cfg.rope_theta, device=device)
        self.k_caches = [None] * self.cfg.num_hidden_layers
        self.v_caches = [None] * self.cfg.num_hidden_layers

    def _cast_to_device(self):
        for k in list(self.weights.keys()):
            self.weights[k] = self.weights[k].to(self.device).to(self.dtype)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embedding 权重也需要转换为正确 dtype
        embed_weight = self.weights['model.embed_tokens.weight'].to(self.dtype)
        return embed(input_ids, embed_weight)

    def layer_forward_prefill(self, x: torch.Tensor, layer_idx: int, pos: torch.Tensor):
        cfg = self.cfg
        hs = cfg.hidden_size
        num_h = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        dh = cfg.head_dim

        x = rmsnorm(x, self.weights[f'model.layers.{layer_idx}.input_layernorm.weight'], cfg.rms_norm_eps)
        B, T, _ = x.shape
        x2 = x.reshape(B * T, hs)
        
        # 使用 self.dtype 而不是硬编码 bfloat16
        q = torch.empty((B * T, num_h * dh), device=x.device, dtype=self.dtype)
        k = torch.empty((B * T, num_kv * dh), device=x.device, dtype=self.dtype)
        v = torch.empty((B * T, num_kv * dh), device=x.device, dtype=self.dtype)
        
        matmul(x2, self.weights[f'model.layers.{layer_idx}.self_attn.q_proj.weight'].t(), q)
        matmul(x2, self.weights[f'model.layers.{layer_idx}.self_attn.k_proj.weight'].t(), k)
        matmul(x2, self.weights[f'model.layers.{layer_idx}.self_attn.v_proj.weight'].t(), v)

        q = q.reshape(B, T, num_h, dh)
        k = k.reshape(B, T, num_kv, dh)
        v = v.reshape(B, T, num_kv, dh)
        q_w = self.weights[f'model.layers.{layer_idx}.self_attn.q_norm.weight']
        k_w = self.weights[f'model.layers.{layer_idx}.self_attn.k_norm.weight']
        q = rmsnorm_2d(q.reshape(-1, dh), q_w, 1e-6).reshape(B, T, num_h, dh)
        k = rmsnorm_2d(k.reshape(-1, dh), k_w, 1e-6).reshape(B, T, num_kv, dh)

        if num_h != num_kv:
            repeat = num_h // num_kv
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        idx = torch.clamp(pos.to(torch.long), 0, self.cos.shape[0]-1)
        cos_t = self.cos.index_select(0, idx).contiguous()
        sin_t = self.sin.index_select(0, idx).contiguous()
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)

        ctx = attention_prefill(q, k, v)
        ctx2 = ctx.reshape(B * T, num_h * dh)
        out = torch.empty((B * T, hs), device=x.device, dtype=self.dtype)
        matmul(ctx2, self.weights[f'model.layers.{layer_idx}.self_attn.o_proj.weight'].t(), out)
        x = add(x, out.reshape(B, T, hs))

        y = rmsnorm(x, self.weights[f'model.layers.{layer_idx}.post_attention_layernorm.weight'], cfg.rms_norm_eps)
        y2 = mlp_swiglu(y,
                        self.weights[f'model.layers.{layer_idx}.mlp.gate_proj.weight'],
                        self.weights[f'model.layers.{layer_idx}.mlp.up_proj.weight'],
                        self.weights[f'model.layers.{layer_idx}.mlp.down_proj.weight'])
        x = add(x, y2)
        self.k_caches[layer_idx] = k
        self.v_caches[layer_idx] = v
        return x

    def prefill(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        B, T = input_ids.shape
        pos = torch.arange(T, device=self.device)
        for i in range(self.cfg.num_hidden_layers):
            x = self.layer_forward_prefill(x, i, pos)
        x = rmsnorm(x, self.weights['model.norm.weight'], self.cfg.rms_norm_eps)
        return x

    def logits(self, x_last: torch.Tensor) -> torch.Tensor:
        # 使用 self.dtype
        out = torch.empty((x_last.shape[0], self.cfg.vocab_size), device=self.device, dtype=self.dtype)
        matmul(x_last, self.weights['lm_head.weight'].t(), out)
        return out

    def layer_forward_decode(self, x: torch.Tensor, layer_idx: int, pos: int):
        """
        执行单层的 decode 前向传播
        
        Args:
            x: 输入张量 [B, 1, hs] (decode 时序列长度为1)
            layer_idx: 层索引
            pos: 当前位置索引
        
        Returns:
            x: 输出张量 [B, 1, hs]
        """
        cfg = self.cfg
        hs = cfg.hidden_size
        num_h = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        dh = cfg.head_dim
        
        # Input LayerNorm
        x = rmsnorm(x, self.weights[f'model.layers.{layer_idx}.input_layernorm.weight'], cfg.rms_norm_eps)
        
        B, T, _ = x.shape  # T应该是1
        x2 = x.reshape(B * T, hs)
        
        # QKV Projections
        q = torch.empty((B * T, num_h * dh), device=x.device, dtype=self.dtype)
        k = torch.empty((B * T, num_kv * dh), device=x.device, dtype=self.dtype)
        v = torch.empty((B * T, num_kv * dh), device=x.device, dtype=self.dtype)
        
        matmul(x2, self.weights[f'model.layers.{layer_idx}.self_attn.q_proj.weight'].t(), q)
        matmul(x2, self.weights[f'model.layers.{layer_idx}.self_attn.k_proj.weight'].t(), k)
        matmul(x2, self.weights[f'model.layers.{layer_idx}.self_attn.v_proj.weight'].t(), v)
        
        # Reshape for attention
        q = q.reshape(B, T, num_h, dh)
        k = k.reshape(B, T, num_kv, dh)
        v = v.reshape(B, T, num_kv, dh)
        
        # Q/K LayerNorm
        q_w = self.weights[f'model.layers.{layer_idx}.self_attn.q_norm.weight']
        k_w = self.weights[f'model.layers.{layer_idx}.self_attn.k_norm.weight']
        q = rmsnorm_2d(q.reshape(-1, dh), q_w, 1e-6).reshape(B, T, num_h, dh)
        k = rmsnorm_2d(k.reshape(-1, dh), k_w, 1e-6).reshape(B, T, num_kv, dh)
        
        # GQA: Repeat K/V if needed
        if num_h != num_kv:
            repeat = num_h // num_kv
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        
        # RoPE
        idx = torch.clamp(torch.tensor([pos], device=self.device, dtype=torch.long), 0, self.cos.shape[0]-1)
        cos_t = self.cos.index_select(0, idx).contiguous()
        sin_t = self.sin.index_select(0, idx).contiguous()
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)
        
        # Update KV Cache
        if self.k_caches[layer_idx] is None:
            self.k_caches[layer_idx] = k
            self.v_caches[layer_idx] = v
        else:
            self.k_caches[layer_idx] = torch.cat([self.k_caches[layer_idx], k], dim=1)
            self.v_caches[layer_idx] = torch.cat([self.v_caches[layer_idx], v], dim=1)
        
        # Attention Decode
        ctx = attention_decode(q[:, 0], self.k_caches[layer_idx], self.v_caches[layer_idx])
        
        # Output Projection
        ctx2 = ctx.reshape(B, num_h * dh)
        out = torch.empty((B, hs), device=x.device, dtype=self.dtype)
        matmul(ctx2, self.weights[f'model.layers.{layer_idx}.self_attn.o_proj.weight'].t(), out)
        
        # Residual Connection
        x = add(x.reshape(B, -1, hs), out.reshape(B, 1, hs))
        
        # Post-Attention LayerNorm
        y = rmsnorm(x, self.weights[f'model.layers.{layer_idx}.post_attention_layernorm.weight'], cfg.rms_norm_eps)
        
        # MLP
        y2 = mlp_swiglu(y,
                        self.weights[f'model.layers.{layer_idx}.mlp.gate_proj.weight'],
                        self.weights[f'model.layers.{layer_idx}.mlp.up_proj.weight'],
                        self.weights[f'model.layers.{layer_idx}.mlp.down_proj.weight'])
        
        # Residual Connection
        x = add(x, y2)
        
        return x


    # 同时可以简化 decode 方法，复用 layer_forward_decode
    def decode(self, input_id: torch.Tensor, start_pos: int):
        """使用 layer_forward_decode 的简化版本"""
        x = self.embed(input_id.reshape(1, -1))
        
        for i in range(self.cfg.num_hidden_layers):
            x = self.layer_forward_decode(x, i, start_pos)
        
        x = rmsnorm(x, self.weights['model.norm.weight'], self.cfg.rms_norm_eps)
        logits = self.logits(x[:, -1, :])
        return logits

