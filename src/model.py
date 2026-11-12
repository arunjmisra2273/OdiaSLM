#!/usr/bin/env python3
"""
Step 4.1: Odia SLM Decoder-Only Transformer Architecture (Llama-3 / SmolLM Style).
Supports both the 60.06M and 97.54M parameter configurations with:
- Grouped-Query Attention (GQA) via PyTorch F.scaled_dot_product_attention
- Half-Split Rotary Position Embeddings (RoPE, 100% Hugging Face Llama compatible)
- FP16-Safe RMSNorm (float32 variance accumulation)
- SwiGLU Feed-Forward Network
- Tied Input/Output Token Embeddings
- Depth-Scaled Residual Initialization (0.02 / sqrt(2 * n_layers))
- Float32 Cross-Entropy + Z-Loss Regularization.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OdiaSLMConfig:
    model_variant: str = "60m"
    vocab_size: int = 16384
    n_layers: int = 12
    d_model: int = 640
    n_heads: int = 10
    n_kv_heads: int = 2
    d_ff: int = 1728
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    z_loss_coeff: float = 1e-4
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    @property
    def kv_group_size(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)


def get_model_config(variant: str = "60m") -> OdiaSLMConfig:
    """Returns the exact verified architectural specification for '60m' or '100m'."""
    v = variant.lower().strip()
    if v == "60m":
        return OdiaSLMConfig(
            model_variant="60m",
            vocab_size=16384,
            n_layers=12,
            d_model=640,
            n_heads=10,
            n_kv_heads=2,
            d_ff=1728,
            max_seq_len=1024,
        )
    elif v == "100m":
        return OdiaSLMConfig(
            model_variant="100m",
            vocab_size=16384,
            n_layers=14,
            d_model=768,
            n_heads=12,
            n_kv_heads=4,
            d_ff=2048,
            max_seq_len=1024,
        )
    else:
        raise ValueError(f"Unknown model variant '{variant}'. Expected '60m' or '100m'.")


class RMSNorm(nn.Module):
    """
    FP16-safe Root Mean Square Layer Normalization.
    Computes variance strictly in float32 to prevent FP16 overflow on T4 GPUs.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return x_normed.to(input_dtype) * self.weight


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes float32 Cosine and Sine tables of shape (1, 1, max_seq_len, head_dim)
    using the Hugging Face Llama half-split convention.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, half_dim)
    emb = torch.cat([freqs, freqs], dim=-1)   # (max_seq_len, head_dim)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_len, head_dim)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_len, head_dim)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits last dimension into halves [-x2, x1] matching HF LlamaForCausalLM."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embeddings to q and k of shape (B, H, T, head_dim).
    Performs rotation in float32 and casts back to q.dtype.
    """
    seq_len = q.shape[2]
    cos_t = cos[:, :, :seq_len, :]
    sin_t = sin[:, :, :seq_len, :]

    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)

    q_rot = (q_fp32 * cos_t) + (rotate_half(q_fp32) * sin_t)
    k_rot = (k_fp32 * cos_t) + (rotate_half(k_fp32) * sin_t)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Causal Self-Attention (GQA) with bias-free projections."""

    def __init__(self, config: OdiaSLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_group_size = config.kv_group_size
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)
        self.o_proj._is_residual_proj = True

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Expand Key and Value heads if GQA group size > 1
        if self.kv_group_size > 1:
            k = (
                k[:, :, None, :, :]
                .expand(bsz, self.n_kv_heads, self.kv_group_size, seq_len, self.head_dim)
                .reshape(bsz, self.n_heads, seq_len, self.head_dim)
            )
            v = (
                v[:, :, None, :, :]
                .expand(bsz, self.n_kv_heads, self.kv_group_size, seq_len, self.head_dim)
                .reshape(bsz, self.n_heads, seq_len, self.head_dim)
            )

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)


class SwiGLUFFN(nn.Module):
    """Llama-style SwiGLU Feed-Forward Network (gate_proj, up_proj, down_proj)."""

    def __init__(self, config: OdiaSLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.down_proj._is_residual_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Decoder Layer."""

    def __init__(self, config: OdiaSLMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class OdiaSLM(nn.Module):
    """Complete Odia Small Language Model with tied input/output embeddings and Z-Loss."""

    def __init__(self, config: OdiaSLMConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute non-persistent RoPE cos/sin buffers up to max_seq_len
        cos, sin = precompute_rope_cache(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights before tying
        self.apply(self._init_weights)

        # Tie input and output embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_is_residual_proj", False):
                std = 0.02 / math.sqrt(2.0 * self.config.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) int64 tensor of token IDs
            targets: Optional (B, T) int64 tensor of target token IDs
        Returns dict with keys:
            'logits': (B, T, V) tensor
            'loss_total': scalar total loss (CE + Z-loss) if targets provided
            'loss_ce': scalar pure Cross-Entropy loss (detached, for perplexity/BPB)
            'z_loss': scalar Z-loss penalty (detached)
        """
        bsz, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
        )

        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        h = self.norm(h)
        logits = self.lm_head(h)

        if targets is None:
            return {
                "logits": logits,
                "loss_total": None,
                "loss_ce": None,
                "z_loss": None,
            }

        # Upcast logits to float32 for numerical stability in FP16 training
        logits_fp32 = logits.to(torch.float32).view(-1, self.config.vocab_size)
        targets_flat = targets.reshape(-1)

        loss_ce = F.cross_entropy(logits_fp32, targets_flat)

        # Auxiliary Z-Loss: 1e-4 * mean(logsumexp(logits)^2)
        log_z = torch.logsumexp(logits_fp32, dim=-1)
        z_loss = self.config.z_loss_coeff * torch.mean(log_z.pow(2))
        loss_total = loss_ce + z_loss

        return {
            "logits": logits,
            "loss_total": loss_total,
            "loss_ce": loss_ce.detach(),
            "z_loss": z_loss.detach(),
        }

    def count_parameters(self) -> Dict[str, int]:
        """Counts exact unique trainable parameters (accounting for tied embeddings)."""
        unique_params = {id(p): p for p in self.parameters() if p.requires_grad}
        total_unique = sum(p.numel() for p in unique_params.values())
        emb_params = self.embed_tokens.weight.numel()
        non_emb_params = total_unique - emb_params
        return {
            "total_trainable_params": total_unique,
            "embedding_params": emb_params,
            "non_embedding_params": non_emb_params,
        }

    def configure_optimizers(
        self,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> torch.optim.AdamW:
        """
        Separates unique parameters into:
        - decay group (weight_decay=0.1): all 2D matrices (Linear projections & Embedding table)
        - no_decay group (weight_decay=0.0): all 1D tensors (RMSNorm gamma weights)
        """
        seen_ids = set()
        decay_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad or id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        use_fused = torch.cuda.is_available()
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            fused=use_fused,
        )
