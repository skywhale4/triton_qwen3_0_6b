import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen3Config:
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0


def load_qwen3_config(path: str) -> Qwen3Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Qwen3Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rope_theta=cfg.get("rope_theta", 1_000_000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        attention_dropout=cfg.get("attention_dropout", 0.0),
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int, theta: float = 1_000_000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)                         
                                                          
        emb = torch.cat((freqs, freqs), dim=-1)                       
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                                           
        cos = self.cos[positions]          
        sin = self.sin[positions]
        cos = cos.unsqueeze(0).unsqueeze(0)                
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (x * cos) + (rotate_half(x) * sin)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

                                                         
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
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

        if self.debug_mode:
            print(f"  [Attn] Input x shape: {x.shape}, first 50 elements:")
            print(f"         {x[0, -1, :50].cpu().tolist()}")

        q = self._shape(self.q_proj(x), self.n_heads)                      
        k = self._shape(self.k_proj(x), self.n_kv_heads)                   
        v = self._shape(self.v_proj(x), self.n_kv_heads)                   

        if self.debug_mode:
            print(f"  [Attn] After Q/K/V proj (before norm), Q[0,-1,0,:50]:")
            print(f"         {q[0, -1, 0, :50].cpu().tolist()}")
            print(f"  [Attn] After Q/K/V proj (before norm), K[0,-1,0,:50]:")
            print(f"         {k[0, -1, 0, :50].cpu().tolist()}")
            print(f"  [Attn] After Q/K/V proj (before norm), V[0,-1,0,:50]:")
            print(f"         {v[0, -1, 0, :50].cpu().tolist()}")

                                                                                       
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.debug_mode:
            print(f"  [Attn] After Q/K norm, Q[0,-1,0,:50]:")
            print(f"         {q[0, -1, 0, :50].cpu().tolist()}")
            print(f"  [Attn] After Q/K norm, K[0,-1,0,:50]:")
            print(f"         {k[0, -1, 0, :50].cpu().tolist()}")

                                                                    
        q = q.transpose(1, 2)                                      
        k = k.transpose(1, 2)                                      
        v = v.transpose(1, 2)                                      

                                          
        q = self.rotary(q, positions)
        k = self.rotary(k, positions)

        if self.debug_mode:
            print(f"  [Attn] After RoPE, Q[0,0,-1,:50] (head 0, last token):")
            print(f"         {q[0, 0, -1, :50].cpu().tolist()}")
            print(f"  [Attn] After RoPE, K[0,0,-1,:50] (head 0, last token):")
            print(f"         {k[0, 0, -1, :50].cpu().tolist()}")

        if past_kv is not None:
            pk, pv = past_kv                     
            if self.debug_mode:
                print(f"  [Attn] past_kv shape: pk={pk.shape}, pv={pv.shape}")
                print(f"  [Attn] current k shape (before cat): {k.shape}")
            k = torch.cat([pk, k], dim=2)                         
            v = torch.cat([pv, v], dim=2)
            if self.debug_mode:
                print(f"  [Attn] k shape after cat: {k.shape}")

        groups = self.n_heads // self.n_kv_heads
        k_rep = k.repeat_interleave(groups, dim=1)                                
        v_rep = v.repeat_interleave(groups, dim=1)

                                                  
        attn_scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale                    
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if self.debug_mode:
            print(f"  [Attn] Attention scores shape: {attn_scores.shape}, [0,0,-1,:10] (last query, first 10 keys):")
            print(f"         {attn_scores[0, 0, -1, :10].cpu().tolist()}")

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.attn_dropout_p > 0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.attn_dropout_p)

        if self.debug_mode:
            print(f"  [Attn] Attention probs [0,0,-1,:10]:")
            print(f"         {attn_probs[0, 0, -1, :10].cpu().tolist()}")

        context = torch.matmul(attn_probs, v_rep)                
        context = context.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)
        
        if self.debug_mode:
            print(f"  [Attn] Context (before o_proj) shape: {context.shape}, [0,-1,:50]:")
            print(f"         {context[0, -1, :50].cpu().tolist()}")

        out = self.o_proj(context)
        
        if self.debug_mode:
            print(f"  [Attn] Output (after o_proj) [0,-1,:50]:")
            print(f"         {out[0, -1, :50].cpu().tolist()}")

        return out, (k, v)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, use_swiglu: bool = True):
        super().__init__()
        self.use_swiglu = use_swiglu
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
        if self.debug_mode:
            print(f"  [MLP] Input x [0,-1,:50]:")
            print(f"        {x[0, -1, :50].cpu().tolist()}")

        if self.use_swiglu:
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)
            if self.debug_mode:
                print(f"  [MLP] After gate_proj [0,-1,:50]:")
                print(f"        {gate_out[0, -1, :50].cpu().tolist()}")
                print(f"  [MLP] After up_proj [0,-1,:50]:")
                print(f"        {up_out[0, -1, :50].cpu().tolist()}")
            
            silu_out = F.silu(gate_out)
            if self.debug_mode:
                print(f"  [MLP] After SiLU(gate) [0,-1,:50]:")
                print(f"        {silu_out[0, -1, :50].cpu().tolist()}")
            
            intermediate = silu_out * up_out
            if self.debug_mode:
                print(f"  [MLP] After gate*up [0,-1,:50]:")
                print(f"        {intermediate[0, -1, :50].cpu().tolist()}")
            
            result = self.down_proj(intermediate)
            if self.debug_mode:
                print(f"  [MLP] Output (after down_proj) [0,-1,:50]:")
                print(f"        {result[0, -1, :50].cpu().tolist()}")
            return result
        else:
            return self.fc2(F.silu(self.fc1(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config, use_swiglu=True)
        self.debug_mode = False

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.debug_mode:
            print(f"[Layer] Input x [0,-1,:50]:")
            print(f"        {x[0, -1, :50].cpu().tolist()}")
        
        attn_normed = self.attn_norm(x)
        if self.debug_mode:
            print(f"[Layer] After attn_norm [0,-1,:50]:")
            print(f"        {attn_normed[0, -1, :50].cpu().tolist()}")
        
        attn_out, kv = self.self_attn(attn_normed, positions, past_kv=past_kv, attn_mask=attn_mask)
        
        if self.debug_mode:
            print(f"[Layer] After self_attn (residual add before) [0,-1,:50]:")
            print(f"        {attn_out[0, -1, :50].cpu().tolist()}")
        
        h = x + attn_out
        
        if self.debug_mode:
            print(f"[Layer] After attn residual add [0,-1,:50]:")
            print(f"        {h[0, -1, :50].cpu().tolist()}")
        
        mlp_normed = self.mlp_norm(h)
        if self.debug_mode:
            print(f"[Layer] After mlp_norm [0,-1,:50]:")
            print(f"        {mlp_normed[0, -1, :50].cpu().tolist()}")
        
        mlp_out = self.mlp(mlp_normed)
        
        if self.debug_mode:
            print(f"[Layer] After MLP (residual add before) [0,-1,:50]:")
            print(f"        {mlp_out[0, -1, :50].cpu().tolist()}")
        
        o = h + mlp_out
        
        if self.debug_mode:
            print(f"[Layer] Final output (after mlp residual add) [0,-1,:50]:")
            print(f"        {o[0, -1, :50].cpu().tolist()}")
        
        return o, kv


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
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
        x = self.tok_embeddings(input_ids)             

        if self.debug_mode:
            print(f"\n[Model] Embedding output [0,-1,:50]:")
            print(f"        {x[0, -1, :50].cpu().tolist()}\n")

                                                            
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
            if self.debug_mode:
                print(f"\n{'='*80}")
                print(f"Processing Layer {i}")
                print(f"{'='*80}\n")
            
            pkv = past_kvs[i] if past_kvs is not None else None
            x, kv = layer(x, positions, past_kv=pkv, attn_mask=attn_mask)
            if use_cache:
                new_kvs.append(kv)
            if output_hidden_states:
                all_hidden.append(x)

        x = self.norm(x)
        
        if self.debug_mode:
            print(f"\n[Model] After final norm [0,-1,:50]:")
            print(f"        {x[0, -1, :50].cpu().tolist()}\n")
        
        return x, new_kvs, all_hidden


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.weight
    
    def set_debug_mode(self, debug: bool, single_layer: bool = False):
        self.model.debug_mode = debug
        self.model.debug_single_layer = single_layer
        for layer in self.model.layers:
            layer.debug_mode = debug
            layer.self_attn.debug_mode = debug
            layer.mlp.debug_mode = debug

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
        logits = self.lm_head(h[:, -1:, :])           
        if input_ids.shape[1] > 1:
            print("[Torch] Prefill hidden_state (last token, first 50):")
            print(h[0, -1, :50].cpu().tolist())
            print()
        return logits, new_kvs, all_hidden


@torch.no_grad()
def load_from_hf(hf_model, torch_model: Qwen3ForCausalLM) -> None:
    sd = hf_model.state_dict()

    torch_model.model.tok_embeddings.weight.copy_(sd["model.embed_tokens.weight"])
    if not torch_model.config.tie_word_embeddings and "lm_head.weight" in sd:
        torch_model.lm_head.weight.copy_(sd["lm_head.weight"])

    for i, layer in enumerate(torch_model.model.layers):
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
        else:
            layer.mlp.fc1.weight.copy_(sd[f"model.layers.{i}.mlp.up_proj.weight"])
            layer.mlp.fc2.weight.copy_(sd[f"model.layers.{i}.mlp.down_proj.weight"])

    torch_model.model.norm.weight.copy_(sd["model.norm.weight"])


def build_model_from_config(config_json_path: str) -> Qwen3ForCausalLM:
    cfg = load_qwen3_config(config_json_path)
    return Qwen3ForCausalLM(cfg)