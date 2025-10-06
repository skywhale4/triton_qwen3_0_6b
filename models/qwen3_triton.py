import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.qwen3_torch import (
    Qwen3Config,
    load_qwen3_config,
    rotate_half,
)

                
try:
    from kernels.rms_norm import triton_rms_norm
    from kernels.rope import triton_rope
    from kernels.swiglu import triton_swiglu_activation
    from kernels.elementwise import triton_softmax, triton_add
    from kernels.rope_gather import triton_gather_rope
    from kernels.matmul import triton_matmul
    from kernels.linear import triton_linear
    from kernels.embedding import triton_embedding
    from kernels.cache import triton_kv_concat
    from kernels.repeat import triton_repeat_kv_heads
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton kernels not available, falling back to PyTorch")



class TritonRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_triton = use_triton
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and TRITON_AVAILABLE:
            return triton_rms_norm(x, self.weight, self.eps)
        
                    
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class TritonRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int, theta: float = 1_000_000.0, use_triton: bool = False):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.use_triton = use_triton
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.use_triton and TRITON_AVAILABLE:
            cos = triton_gather_rope(self.cos, positions).unsqueeze(0).unsqueeze(0)
            sin = triton_gather_rope(self.sin, positions).unsqueeze(0).unsqueeze(0)
            return triton_rope(x, cos, sin)
        else:
            cos = self.cos[positions].unsqueeze(0).unsqueeze(0)
            sin = self.sin[positions].unsqueeze(0).unsqueeze(0)
            return (x * cos) + (rotate_half(x) * sin)


class TritonQwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, use_triton: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.use_triton = use_triton
        self.compute_dtype = dtype

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = TritonRMSNorm(self.head_dim, config.rms_norm_eps, use_triton)
        self.k_norm = TritonRMSNorm(self.head_dim, config.rms_norm_eps, use_triton)
        self.rotary = TritonRotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta, use_triton)
        
        self.attn_dropout_p = config.attention_dropout
        self.debug_mode = False
    
    def _shape(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, n_heads, self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape

        if self.use_triton and TRITON_AVAILABLE:
            q = self._shape(triton_linear(x, self.q_proj.weight), self.n_heads)
            k = self._shape(triton_linear(x, self.k_proj.weight), self.n_kv_heads)
            v = self._shape(triton_linear(x, self.v_proj.weight), self.n_kv_heads)
            q = q.to(self.compute_dtype)
            k = k.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        else:
            q = self._shape(self.q_proj(x), self.n_heads)
            k = self._shape(self.k_proj(x), self.n_kv_heads)
            v = self._shape(self.v_proj(x), self.n_kv_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rotary(q, positions)
        k = self.rotary(k, positions)

        if past_kv is not None:
            pk, pv = past_kv
            if self.use_triton and TRITON_AVAILABLE:
                k = triton_kv_concat(pk, k)
                v = triton_kv_concat(pv, v)
            else:
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)

        groups = self.n_heads // self.n_kv_heads
        if self.use_triton and TRITON_AVAILABLE:
            k_rep = triton_repeat_kv_heads(k, groups)
            v_rep = triton_repeat_kv_heads(v, groups)
        else:
            k_rep = k.repeat_interleave(groups, dim=1)
            v_rep = v.repeat_interleave(groups, dim=1)

                      
        if self.use_triton and TRITON_AVAILABLE:
                               
                     
            k_t = k_rep.transpose(-2, -1)
            attn_scores = triton_matmul(q, k_t) * self.scale
            
                      
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask
            
                     
            attn_probs = triton_softmax(attn_scores, dim=-1)
            
                   
            context = triton_matmul(attn_probs, v_rep)
        else:
                        
            attn_scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask
            
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.attn_dropout_p > 0 and self.training:
                attn_probs = F.dropout(attn_probs, p=self.attn_dropout_p)
            
            context = torch.matmul(attn_probs, v_rep)
        context = context.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)
        
        if self.use_triton and TRITON_AVAILABLE:
            out = triton_linear(context, self.o_proj.weight)
        else:
            out = self.o_proj(context)
        return out, (k, v)


class TritonQwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, use_swiglu: bool = True, use_triton: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.use_triton = use_triton
        self.compute_dtype = dtype
        hs = config.hidden_size
        ims = config.intermediate_size
        
        if use_swiglu:
            self.gate_proj = nn.Linear(hs, ims, bias=False)
            self.up_proj = nn.Linear(hs, ims, bias=False)
            self.down_proj = nn.Linear(ims, hs, bias=False)
        else:
            self.fc1 = nn.Linear(hs, ims, bias=False)
            self.fc2 = nn.Linear(ims, hs, bias=False)
        self.debug_mode = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            if self.use_triton and TRITON_AVAILABLE:
                                  
                gate_out = triton_linear(x, self.gate_proj.weight)
                up_out = triton_linear(x, self.up_proj.weight)
                hidden = triton_swiglu_activation(gate_out, up_out)
                return triton_linear(hidden, self.down_proj.weight)
            else:
                            
                gate_out = self.gate_proj(x)
                up_out = self.up_proj(x)
                hidden = F.silu(gate_out) * up_out
                return self.down_proj(hidden)
        else:
            return self.fc2(F.silu(self.fc1(x)))


class TritonQwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, use_triton: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.attn_norm = TritonRMSNorm(config.hidden_size, config.rms_norm_eps, use_triton)
        self.self_attn = TritonQwen3Attention(config, use_triton, dtype=dtype)
        self.mlp_norm = TritonRMSNorm(config.hidden_size, config.rms_norm_eps, use_triton)
        self.mlp = TritonQwen3MLP(config, use_swiglu=True, use_triton=use_triton)
        self.debug_mode = False
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv = self.self_attn(self.attn_norm(x), positions, past_kv=past_kv, attn_mask=attn_mask)
        if self.self_attn.use_triton and TRITON_AVAILABLE:
            h = triton_add(x, attn_out)
            o = triton_add(h, self.mlp(self.mlp_norm(h)))
        else:
            h = x + attn_out
            o = h + self.mlp(self.mlp_norm(h))
        return o, kv


class TritonQwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config, use_triton: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.use_triton = use_triton
        self.compute_dtype = dtype
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TritonQwen3DecoderLayer(config, use_triton, dtype=dtype) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = TritonRMSNorm(config.hidden_size, config.rms_norm_eps, use_triton)
        self.debug_mode = False
        self.debug_single_layer = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
        B, S = input_ids.shape
        device = input_ids.device
        if self.use_triton and TRITON_AVAILABLE:
            x = triton_embedding(input_ids, self.tok_embeddings.weight)
        else:
            x = self.tok_embeddings(input_ids)
        if self.compute_dtype != torch.float32:
            x = x.to(self.compute_dtype)

        past_len = 0 if not past_kvs else past_kvs[0][0].shape[2]
        positions = torch.arange(past_len, past_len + S, device=device, dtype=torch.long)

        attn_mask = None
        if S > 1 or past_len == 0:
            tot = past_len + S
            mask = torch.ones((S, tot), device=device).triu(diagonal=1 + past_len)
            attn_mask = mask.masked_fill(mask == 1, float("-inf")).unsqueeze(0).unsqueeze(0)

        new_kvs = [] if use_cache else None
        all_hidden = [] if output_hidden_states else None
        
        num_layers = 1 if self.debug_single_layer else len(self.layers)
        
        for i, layer in enumerate(self.layers[:num_layers]):
            pkv = past_kvs[i] if past_kvs is not None else None
            x, kv = layer(x, positions, past_kv=pkv, attn_mask=attn_mask)
            if use_cache:
                new_kvs.append(kv)
            if output_hidden_states:
                all_hidden.append(x)

        x = self.norm(x)
        return x, new_kvs, all_hidden


class TritonQwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config, use_triton: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.use_triton = use_triton
        self.model = TritonQwen3Model(config, use_triton, dtype=dtype)
        self.compute_dtype = dtype
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.weight
    
    def set_triton_mode(self, use_triton: bool):
        self.use_triton = use_triton
        self.model.use_triton = use_triton
        for layer in self.model.layers:
            layer.attn_norm.use_triton = use_triton
            layer.mlp_norm.use_triton = use_triton
            layer.self_attn.use_triton = use_triton
            layer.self_attn.q_norm.use_triton = use_triton
            layer.self_attn.k_norm.use_triton = use_triton
            layer.self_attn.rotary.use_triton = use_triton
            layer.mlp.use_triton = use_triton
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
        h, new_kvs, all_hidden = self.model(
            input_ids, past_kvs=past_kvs, use_cache=use_cache, output_hidden_states=output_hidden_states
        )
        if self.use_triton and TRITON_AVAILABLE:
            logits = triton_linear(h[:, -1:, :], self.lm_head.weight)
        else:
            logits = self.lm_head(h[:, -1:, :])
        if input_ids.shape[1] > 1:
            backend = 'Triton' if self.use_triton else 'PyTorch'
            print(f"[{backend}] Prefill hidden_state (last token, first 50):")
            print(h[0, -1, :50].cpu().tolist())
            print()
        return logits, new_kvs, all_hidden


def build_triton_model(config_json_path: str, use_triton: bool = False, dtype: torch.dtype = torch.float32) -> TritonQwen3ForCausalLM:
    cfg = load_qwen3_config(config_json_path)
    model = TritonQwen3ForCausalLM(cfg, use_triton=use_triton, dtype=dtype)
    return model


@torch.no_grad()
def load_triton_from_hf(hf_model, triton_model: TritonQwen3ForCausalLM) -> None:
    sd = hf_model.state_dict()

    triton_model.model.tok_embeddings.weight.copy_(sd["model.embed_tokens.weight"])
    if not triton_model.config.tie_word_embeddings and "lm_head.weight" in sd:
        triton_model.lm_head.weight.copy_(sd["lm_head.weight"])

    for i, layer in enumerate(triton_model.model.layers):
        layer.attn_norm.weight.copy_(sd[f"model.layers.{i}.input_layernorm.weight"])
        layer.mlp_norm.weight.copy_(sd[f"model.layers.{i}.post_attention_layernorm.weight"])

        layer.self_attn.q_proj.weight.copy_(sd[f"model.layers.{i}.self_attn.q_proj.weight"])
        layer.self_attn.k_proj.weight.copy_(sd[f"model.layers.{i}.self_attn.k_proj.weight"])
        layer.self_attn.v_proj.weight.copy_(sd[f"model.layers.{i}.self_attn.v_proj.weight"])
        layer.self_attn.o_proj.weight.copy_(sd[f"model.layers.{i}.self_attn.o_proj.weight"])
        
        layer.self_attn.q_norm.weight.copy_(sd[f"model.layers.{i}.self_attn.q_norm.weight"])
        layer.self_attn.k_norm.weight.copy_(sd[f"model.layers.{i}.self_attn.k_norm.weight"])

        has_gate = f"model.layers.{i}.mlp.gate_proj.weight" in sd
        if has_gate:
            layer.mlp.gate_proj.weight.copy_(sd[f"model.layers.{i}.mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight.copy_(sd[f"model.layers.{i}.mlp.up_proj.weight"])
            layer.mlp.down_proj.weight.copy_(sd[f"model.layers.{i}.mlp.down_proj.weight"])

    triton_model.model.norm.weight.copy_(sd["model.norm.weight"])
