"""
FSQ quantizer + AudioTokenDetokenizer for AceStep 1.5.
Ported from ComfyUI comfy/ldm/ace/ace_step15.py — no ComfyUI dependencies.

Key structure matches ComfyUI's AceStepEncoderLayer / AudioTokenDetokenizer exactly,
so weights load from the AIO safetensors file without remapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMS Norm (compatible with all PyTorch 2.x versions) ──────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight.to(device=x.device, dtype=x.dtype)


# ── RoPE ─────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def _rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ── Attention — keys match AceStepAttention exactly ──────────────────────────

class AceStepAttention(nn.Module):
    """Self-attention matching ComfyUI AceStepAttention key layout."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary(q, k, cos, sin)
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


# ── MLP — keys match ComfyUI MLP exactly (gate_proj / up_proj / down_proj) ───

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Encoder Layer — matches AceStepEncoderLayer key layout exactly ────────────

class AceStepEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps=1e-6):
        super().__init__()
        self.self_attn = AceStepAttention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, position_embeddings=None):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, position_embeddings)
        hidden_states = residual + x
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(x)
        return hidden_states


# ── FSQ (codebook lookup only — no encoding needed for text-to-music) ─────────

class FSQ(nn.Module):
    """Finite Scalar Quantizer — codebook lookup path only (get_output_from_indices)."""

    def __init__(self, levels, dim=None):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer('_levels', _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0)
        self.register_buffer('_basis', _basis, persistent=False)

        self.codebook_dim = len(levels)
        self.dim = dim if dim is not None else self.codebook_dim
        requires_projection = self.dim != self.codebook_dim
        if requires_projection:
            self.project_in = nn.Linear(self.dim, self.codebook_dim)
            self.project_out = nn.Linear(self.codebook_dim, self.dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        codebook_size = int(_levels.prod().item())
        indices = torch.arange(codebook_size)
        codebook = self._build_codebook(indices)
        self.register_buffer('implicit_codebook', codebook, persistent=False)

    def _build_codebook(self, indices):
        indices = indices.unsqueeze(-1)
        codes = (indices // self._basis) % self._levels
        return codes.float() * (2.0 / (self._levels.float() - 1)) - 1.0


class ResidualFSQ(nn.Module):
    """Residual FSQ — only get_output_from_indices needed for inference."""

    # Fixed for all AceStep 1.5 models
    FSQ_LEVELS = [8, 8, 8, 5, 5, 5]

    def __init__(self, dim, num_quantizers=1):
        super().__init__()
        codebook_dim = len(self.FSQ_LEVELS)
        requires_projection = codebook_dim != dim
        if requires_projection:
            self.project_in = nn.Linear(dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.layers = nn.ModuleList([
            FSQ(self.FSQ_LEVELS, dim=codebook_dim)  # dim=codebook_dim so FSQ has Identity projections
            for _ in range(num_quantizers)
        ])

        levels_t = torch.tensor(self.FSQ_LEVELS, dtype=torch.float32)
        scales = torch.stack([levels_t.float() ** -i for i in range(num_quantizers)])
        self.register_buffer('scales', scales, persistent=False)

    def get_output_from_indices(self, indices, dtype=torch.float32):
        """
        Convert audio code indices to continuous embeddings.
        indices: [B, T] or [B, T, num_quantizers]
        Returns: [B, T, dim]
        """
        if indices.dim() == 2:
            indices = indices.unsqueeze(-1)  # [B, T, 1]
        all_codes = []
        for i, layer in enumerate(self.layers):
            idx = indices[..., i].long()
            codebook = layer.implicit_codebook.to(device=idx.device, dtype=dtype)
            codes = F.embedding(idx, codebook)  # [B, T, codebook_dim]
            scale = self.scales[i].to(device=idx.device, dtype=dtype)
            all_codes.append(codes * scale)
        summed = torch.stack(all_codes).sum(dim=0)  # [B, T, codebook_dim]
        return self.project_out(summed)             # [B, T, dim]


# ── AudioTokenDetokenizer — matches ComfyUI exactly ──────────────────────────

class AudioTokenDetokenizer(nn.Module):
    """Upsamples 5Hz FSQ codes to 25Hz acoustic latents.

    Key layout matches ComfyUI AudioTokenDetokenizer exactly:
      embed_tokens, special_tokens, rotary_emb, layers.X.*, norm, proj_out
    """

    def __init__(self, hidden_size, pool_window_size, audio_acoustic_hidden_dim, num_layers=2, head_dim=128):
        super().__init__()
        self.pool_window_size = pool_window_size
        self.embed_tokens = nn.Linear(hidden_size, hidden_size)
        self.special_tokens = nn.Parameter(torch.empty(1, pool_window_size, hidden_size))
        self.rotary_emb = RotaryEmbedding(head_dim)
        self.layers = nn.ModuleList([
            AceStepEncoderLayer(hidden_size, 16, 8, head_dim, hidden_size * 3)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, audio_acoustic_hidden_dim)

    def forward(self, x):
        """x: [B, T_5Hz, hidden_size] → [B, T_25Hz, audio_acoustic_hidden_dim]"""
        B, T, D = x.shape
        x = self.embed_tokens(x)
        x = x.unsqueeze(2).repeat(1, 1, self.pool_window_size, 1)  # [B, T, P, D]
        spec = self.special_tokens.to(device=x.device, dtype=x.dtype).expand(B, T, -1, -1)
        x = x + spec
        x = x.view(B * T, self.pool_window_size, D)
        cos, sin = self.rotary_emb(x, seq_len=self.pool_window_size)
        for layer in self.layers:
            x = layer(x, (cos, sin))
        x = self.norm(x)
        x = self.proj_out(x)
        return x.view(B, T * self.pool_window_size, -1)


# ── Loading helpers ───────────────────────────────────────────────────────────

_FSQ_LEVELS = [8, 8, 8, 5, 5, 5]


def load_audio_tokenizer_from_aio(aio_sd: dict, device='cpu', dtype=torch.bfloat16):
    """
    Extract and instantiate ResidualFSQ quantizer + AudioTokenDetokenizer from
    the AIO model state dict (keys stripped of 'model.diffusion_model.' prefix).
    """
    det_keys = {k: v for k, v in aio_sd.items() if k.startswith('detokenizer.')}
    tok_keys = {k: v for k, v in aio_sd.items() if k.startswith('tokenizer.')}

    if not det_keys:
        raise ValueError(
            "No detokenizer.* keys in AIO state dict. "
            "Make sure you're pointing at the XL base model (not turbo)."
        )

    # Infer detokenizer parameters from weight shapes
    det_embed_w = det_keys['detokenizer.embed_tokens.weight']
    hidden_size = det_embed_w.shape[0]
    det_special = det_keys['detokenizer.special_tokens']
    pool_window_size = det_special.shape[1]
    det_proj_out_w = det_keys['detokenizer.proj_out.weight']
    audio_acoustic_dim = det_proj_out_w.shape[0]
    num_layers = max(
        int(k.split('.')[2]) for k in det_keys if k.startswith('detokenizer.layers.')
    ) + 1

    # Infer quantizer fsq_dim from project_out weight
    quant_proj_out = tok_keys.get('tokenizer.quantizer.project_out.weight')
    fsq_dim = quant_proj_out.shape[0] if quant_proj_out is not None else hidden_size

    print(f"  Audio tokenizer: fsq_dim={fsq_dim}, levels={_FSQ_LEVELS}")
    print(f"  Audio detokenizer: hidden={hidden_size}, pool={pool_window_size}, "
          f"out={audio_acoustic_dim}, layers={num_layers}")

    # Instantiate and load quantizer
    quantizer = ResidualFSQ(dim=fsq_dim, num_quantizers=1)
    quant_sd = {
        k.removeprefix('tokenizer.quantizer.'): v
        for k, v in tok_keys.items()
        if k.startswith('tokenizer.quantizer.')
    }
    miss, unex = quantizer.load_state_dict(quant_sd, strict=False)
    if miss:
        print(f"    Quantizer missing: {miss}")
    quantizer = quantizer.to(device).to(dtype).eval()

    # Instantiate and load detokenizer
    detokenizer = AudioTokenDetokenizer(
        hidden_size=hidden_size,
        pool_window_size=pool_window_size,
        audio_acoustic_hidden_dim=audio_acoustic_dim,
        num_layers=num_layers,
    )
    det_sd = {k.removeprefix('detokenizer.'): v for k, v in det_keys.items()}
    miss, unex = detokenizer.load_state_dict(det_sd, strict=False)
    if miss:
        print(f"    Detokenizer missing: {miss}")
    detokenizer = detokenizer.to(device).to(dtype).eval()

    return quantizer, detokenizer


@torch.no_grad()
def codes_to_context_latents(
    audio_codes: list,
    quantizer: ResidualFSQ,
    detokenizer: AudioTokenDetokenizer,
    latent_len: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    audio_codes: list of ints from generate_audio_codes (offset from AUDIO_START_ID)
    Returns: context_latents [1, latent_len, 2*audio_acoustic_dim] (e.g. [1, T, 128])
    """
    pool = detokenizer.pool_window_size
    expected_5hz = (latent_len + pool - 1) // pool

    codes = list(audio_codes)
    if len(codes) < expected_5hz:
        pad_val = codes[-1] if codes else 0
        codes = codes + [pad_val] * (expected_5hz - len(codes))
    codes = codes[:expected_5hz]

    idx = torch.tensor(codes, device=device, dtype=torch.long).unsqueeze(0)  # [1, T_5Hz]
    lm_hints_5hz = quantizer.get_output_from_indices(idx, dtype=dtype)       # [1, T_5Hz, fsq_dim]
    lm_hints = detokenizer(lm_hints_5hz)                                     # [1, T_25Hz, 64]
    lm_hints = lm_hints[:, :latent_len, :]                                   # trim to latent_len

    chunk_masks = torch.ones_like(lm_hints)
    return torch.cat([lm_hints, chunk_masks], dim=-1)                        # [1, latent_len, 128]
