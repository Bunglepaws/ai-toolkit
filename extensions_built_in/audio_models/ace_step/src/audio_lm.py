"""
Qwen3 LM for AceStep 1.5 audio code generation.
Ported from ComfyUI comfy/text_encoders/llama.py — no ComfyUI dependencies.

Supports Qwen3 2B and 4B ACE15 variants (vocab_size=217204, q/k norm, RoPE).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMS Norm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm) * self.weight.to(x.device, x.dtype)


# ── RoPE (ComfyUI-style half-rotate variant) ──────────────────────────────────

def _precompute_freqs(head_dim, position_ids, theta=1000000.0, device=None):
    theta_nums = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_nums / head_dim))
    inv_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    pos_expanded = position_ids[:, None, :].float()
    freqs = (inv_expanded @ pos_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)   # [B, 1, T, D]
    sin = emb.sin().unsqueeze(1)   # [B, 1, T, D]
    half = sin.shape[-1] // 2
    return cos, sin[..., :half], -sin[..., half:]  # cos, sin_first, neg_sin_second


def _apply_rope(xq, xk, freqs_cis):
    cos, sin, nsin = freqs_cis
    cos = cos.to(xq.dtype)
    sin = sin.to(xq.dtype)
    nsin = nsin.to(xq.dtype)

    q = xq.clone()
    half = q.shape[-1] // 2
    q[..., :half] = xq[..., :half] * cos[..., :half] + xq[..., half:] * nsin
    q[..., half:] = xq[..., half:] * cos[..., half:] + xq[..., :half] * sin

    k = xk.clone()
    k[..., :half] = xk[..., :half] * cos[..., :half] + xk[..., half:] * nsin
    k[..., half:] = xk[..., half:] * cos[..., half:] + xk[..., :half] * sin
    return q, k


# ── Attention with Q/K RMSNorm (Qwen3-style) ─────────────────────────────────

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.inner_size = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, self.inner_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(self, x, freqs_cis, past_key_value=None):
        B, T, _ = x.shape
        xq = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq, xk = _apply_rope(xq, xk, freqs_cis)

        present_key_value = None
        if past_key_value is not None:
            if len(past_key_value) == 3:
                past_key, past_value, index = past_key_value
                if past_key.shape[2] >= (index + T):
                    past_key[:, :, index:index + T] = xk
                    past_value[:, :, index:index + T] = xv
                    xk = past_key[:, :, :index + T]
                    xv = past_value[:, :, :index + T]
                    present_key_value = (past_key, past_value, index + T)
                else:
                    xk = torch.cat([past_key[:, :, :index], xk], dim=2)
                    xv = torch.cat([past_value[:, :, :index], xv], dim=2)
                    present_key_value = (xk, xv, index + T)
            else:
                present_key_value = (xk, xv, T)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)

        is_causal = (T > 1)
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, T, self.inner_size)
        return self.o_proj(out), present_key_value


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps=1e-6):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x, freqs_cis, past_key_value=None):
        residual = x
        x, kv = self.self_attn(self.input_layernorm(x), freqs_cis, past_key_value)
        x = residual + x
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        return residual + x, kv


# ── Qwen3 ACE15 LM configs ────────────────────────────────────────────────────

_CONFIGS = {
    "4b": dict(
        vocab_size=217204, hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0,
    ),
    "2b": dict(
        vocab_size=217204, hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0,
    ),
}


# ── Qwen3 LM model ────────────────────────────────────────────────────────────

class Qwen3ACE15LM(nn.Module):
    """Qwen3 LM for audio code generation. Tied embeddings (no separate lm_head)."""

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([
            TransformerBlock(
                config['hidden_size'], config['num_attention_heads'],
                config['num_key_value_heads'], config['head_dim'],
                config['intermediate_size'], config['rms_norm_eps'],
            )
            for _ in range(config['num_hidden_layers'])
        ])
        self.norm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def _freqs(self, T, past_len, device, dtype):
        pos = torch.arange(past_len, past_len + T, device=device).unsqueeze(0)
        cos, sin, nsin = _precompute_freqs(self.cfg['head_dim'], pos, self.cfg['rope_theta'], device)
        return cos.to(dtype), sin.to(dtype), nsin.to(dtype)

    def forward(self, input_ids=None, embeds=None, past_key_values=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(input_ids)
        B, T, D = x.shape

        past_len = 0
        if past_key_values and len(past_key_values) > 0 and len(past_key_values[0]) == 3:
            past_len = past_key_values[0][2]

        freqs_cis = self._freqs(T, past_len, x.device, x.dtype)

        next_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if (past_key_values and len(past_key_values) > i) else []
            x, kv = layer(x, freqs_cis, past_kv)
            if kv is not None:
                next_kvs.append(kv)

        x = self.norm(x)
        return x, next_kvs

    def logits(self, x):
        return F.linear(x, self.embed_tokens.weight.to(x.device, x.dtype))


# ── Loading ───────────────────────────────────────────────────────────────────

def _infer_config(sd):
    """Infer model size from state dict shapes."""
    hidden = sd['embed_tokens.weight'].shape[1]
    for size, cfg in _CONFIGS.items():
        if cfg['hidden_size'] == hidden:
            return cfg
    # fallback: derive from shapes
    inter = sd['layers.0.mlp.gate_proj.weight'].shape[0]
    n_layers = max(int(k.split('.')[1]) for k in sd if k.startswith('layers.')) + 1
    n_heads = sd['layers.0.self_attn.q_proj.weight'].shape[0] // 128
    n_kv = sd['layers.0.self_attn.k_proj.weight'].shape[0] // 128
    vocab = sd['embed_tokens.weight'].shape[0]
    return dict(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=n_heads, num_key_value_heads=n_kv,
        head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0,
    )


def load_qwen3_lm(path, device='cpu', dtype=torch.bfloat16):
    """Load a Qwen3 ACE15 LM from a safetensors file."""
    from safetensors.torch import load_file
    sd = load_file(path)

    # Strip 'model.' prefix if present (HuggingFace format)
    if any(k.startswith('model.') for k in sd):
        sd = {k[len('model.'):]: v for k, v in sd.items() if not k.startswith('lm_head.')}

    config = _infer_config(sd)
    model = Qwen3ACE15LM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Qwen3 LM missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  Qwen3 LM unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    return model.to(device).to(dtype).eval()


# ── Prompt building ───────────────────────────────────────────────────────────

def build_lm_prompts(caption, lyrics, bpm, duration, keyscale, timesignature):
    """Build positive and negative LM prompts for audio code generation."""
    import math as _math
    import yaml

    dur = int(_math.ceil(float(duration))) if duration else 30
    ts = str(timesignature or "4/4")
    if ts.endswith("/4"):
        ts = ts[:-2]

    meta = {}
    if bpm:
        try:
            meta["bpm"] = int(bpm)
        except (ValueError, TypeError):
            meta["bpm"] = bpm
    meta["duration"] = dur
    if keyscale:
        meta["keyscale"] = keyscale
    if ts:
        try:
            meta["timesignature"] = int(ts)
        except (ValueError, TypeError):
            meta["timesignature"] = ts

    meta_yaml = yaml.dump(meta, allow_unicode=True, sort_keys=True).strip() if meta else ""
    cot_pos = f"<think>\n{meta_yaml}\n</think>"
    cot_neg = "<think>\n\n</think>"

    lm_tmpl = (
        "<|im_start|>system\n# Instruction\n"
        "Generate audio semantic tokens based on the given conditions:\n\n"
        "<|im_end|>\n<|im_start|>user\n# Caption\n{cap}\n\n# Lyric\n{lyr}\n"
        "<|im_end|>\n<|im_start|>assistant\n{cot}\n\n<|im_end|>\n"
    )
    pos = lm_tmpl.format(cap=caption, lyr=(lyrics or "").strip(), cot=cot_pos)
    neg = lm_tmpl.format(cap=caption, lyr=(lyrics or "").strip(), cot=cot_neg)
    return pos, neg


# ── Audio code generation ─────────────────────────────────────────────────────

AUDIO_START_ID = 151669
AUDIO_END_ID   = 215669
EOS_TOKEN_ID   = 151645
PAD_TOKEN_ID   = 151643


@torch.no_grad()
def generate_audio_codes(
    lm: Qwen3ACE15LM,
    tokenizer,
    pos_prompt: str,
    neg_prompt: str,
    duration: float,
    seed: int = 0,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    device=None,
    dtype=None,
):
    """
    Autoregressively generate audio semantic codes from the LM.
    Returns list of audio code indices (offset from AUDIO_START_ID).
    """
    if device is None:
        device = next(lm.parameters()).device
    if dtype is None:
        dtype = next(lm.parameters()).dtype

    max_tokens = int(math.ceil(float(duration))) * 5
    min_tokens = max_tokens

    # Tokenize
    pos_ids = tokenizer(pos_prompt, return_tensors="pt").input_ids[0].tolist()
    neg_ids = tokenizer(neg_prompt, return_tensors="pt").input_ids[0].tolist()

    # Pad shorter to same length
    if cfg_scale != 1.0 and len(neg_ids) < len(pos_ids):
        neg_ids = [PAD_TOKEN_ID] * (len(pos_ids) - len(neg_ids)) + neg_ids
    elif cfg_scale != 1.0 and len(pos_ids) < len(neg_ids):
        pos_ids = [PAD_TOKEN_ID] * (len(neg_ids) - len(pos_ids)) + pos_ids

    ids = [pos_ids, neg_ids] if cfg_scale != 1.0 else [pos_ids]

    # Initial forward pass (process full prompt, build KV cache)
    ids_tensor = torch.tensor(ids, device=device, dtype=torch.long)
    embeds = lm.embed_tokens(ids_tensor).to(dtype)

    batch = embeds.shape[0]
    cfg_hidden = lm.cfg
    kv_alloc_len = embeds.shape[1] + min_tokens
    past_key_values = []
    for _ in range(cfg_hidden['num_hidden_layers']):
        kv_shape = [batch, cfg_hidden['num_key_value_heads'], kv_alloc_len, cfg_hidden['head_dim']]
        past_key_values.append((
            torch.empty(kv_shape, device=device, dtype=dtype),
            torch.empty(kv_shape, device=device, dtype=dtype),
            0,
        ))

    hidden, past_key_values = lm(embeds=embeds, past_key_values=past_key_values)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    remove_val = torch.finfo(dtype).min

    audio_codes = []

    for step in range(max_tokens):
        logits = lm.logits(hidden[:, -1:])  # [batch, 1, vocab]
        next_logits = logits[:, 0, :]       # [batch, vocab]

        # CFG
        if cfg_scale != 1.0:
            cond_l = next_logits[0:1]
            uncond_l = next_logits[1:2]
            cfg_l = uncond_l + cfg_scale * (cond_l - uncond_l)
        else:
            cfg_l = next_logits[0:1]

        # EOS handling (only after min_tokens)
        use_eos = (EOS_TOKEN_ID is not None and EOS_TOKEN_ID < AUDIO_START_ID and step >= min_tokens)
        if use_eos:
            eos_score = cfg_l[:, EOS_TOKEN_ID].clone()

        # Restrict to audio token range
        cfg_l[:, :AUDIO_START_ID] = remove_val
        cfg_l[:, AUDIO_END_ID:] = remove_val

        if use_eos:
            cfg_l[:, EOS_TOKEN_ID] = eos_score

        # top-k
        if top_k and top_k > 0:
            top_k_vals, _ = torch.topk(cfg_l, top_k)
            cfg_l[cfg_l < top_k_vals[..., -1:]] = remove_val

        # min-p
        if min_p and min_p > 0:
            probs = torch.softmax(cfg_l, dim=-1)
            p_max = probs.max(dim=-1, keepdim=True).values
            cfg_l[probs < min_p * p_max] = remove_val

        # top-p
        if top_p and top_p < 1.0:
            sorted_l, sorted_idx = torch.sort(cfg_l, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
            sorted_remove = cum_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_mask = sorted_remove.scatter(1, sorted_idx, sorted_remove)
            cfg_l[remove_mask] = remove_val

        # Sample
        if temperature > 0:
            cfg_l = cfg_l / temperature
            token = torch.multinomial(torch.softmax(cfg_l, dim=-1), 1, generator=rng).item()
        else:
            token = cfg_l.argmax(dim=-1).item()

        if use_eos and token == EOS_TOKEN_ID:
            break

        audio_codes.append(token - AUDIO_START_ID)

        # Embed next token and forward
        next_embed = lm.embed_tokens(
            torch.tensor([[token]], device=device, dtype=torch.long)
        ).to(dtype).expand(batch, -1, -1)
        hidden, past_key_values = lm(embeds=next_embed, past_key_values=past_key_values)

    return audio_codes
