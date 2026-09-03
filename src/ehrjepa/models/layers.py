"""Shared building blocks: Fourier scalar features, RoPE, and a pre-LN block.

Nothing here knows about EHR. The two pieces worth naming are:

**Fourier scalar features.** Age, the log-gap since the previous event and the
per-code z-score are continuous, and a single linear projection of a raw float
gives a network almost nothing to work with at multiple scales. Each scalar is
lifted to ``[sin(2*pi*f*x), cos(2*pi*f*x)]`` over ``n_freq`` log-spaced
frequencies before an MLP sees it, which is the standard NeRF/Transformer
positional trick applied to a value rather than to an index.

**RoPE.** Rotary position embeddings on the *sequence index*, applied to queries
and keys inside attention. Rotary is relative, so a constant offset in the index
(the encoder prepends a CLS token and the predictor does not) cancels within a
module; only the gaps between events matter, and the irregular wall-clock spacing
is carried separately by the ``log_delta`` feature rather than by the position.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "FourierFeatures",
    "RotaryEmbedding",
    "ScalarEncoder",
    "TransformerBlock",
    "apply_rope",
]


class FourierFeatures(nn.Module):
    """Lift a scalar to ``2 * n_freq`` sinusoids at log-spaced frequencies.

    The frequency grid is ``logspace(log10(min_freq), log10(max_freq), n_freq)``
    in cycles per unit of the input. The defaults span periods of 100 down to 0.1
    input units, which covers a lifetime in years, a decade of log-hours and a
    clipped z-score with the same module.
    """

    def __init__(self, n_freq: int = 16, min_freq: float = 1e-2, max_freq: float = 1e1) -> None:
        super().__init__()
        if n_freq < 1:
            raise ValueError("n_freq must be positive")
        if not 0 < min_freq < max_freq:
            raise ValueError("need 0 < min_freq < max_freq")
        freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), n_freq)
        self.register_buffer("freqs", freqs * 2.0 * math.pi, persistent=True)
        self.n_freq = n_freq

    @property
    def out_features(self) -> int:
        return 2 * self.n_freq

    def forward(self, x: Tensor) -> Tensor:
        """``x`` of shape ``(...)`` becomes ``(..., 2 * n_freq)``."""
        scaled = x.unsqueeze(-1) * self.freqs.to(x.dtype)
        return torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=-1)


class ScalarEncoder(nn.Module):
    """``scalar -> Fourier features -> 2-layer MLP -> dim``."""

    def __init__(
        self,
        dim: int,
        n_freq: int = 16,
        hidden: int | None = None,
        min_freq: float = 1e-2,
        max_freq: float = 1e1,
    ) -> None:
        super().__init__()
        self.fourier = FourierFeatures(n_freq, min_freq=min_freq, max_freq=max_freq)
        hidden = dim if hidden is None else hidden
        self.mlp = nn.Sequential(
            nn.Linear(self.fourier.out_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.fourier(x))


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings over the head dimension, cached per length."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE needs an even head dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._cache: tuple[Tensor, Tensor] | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        cache = self._cache
        if cache is None or seq_len > self._cached_len or cache[0].device != device:
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            angles = torch.outer(pos, self.inv_freq.to(device))
            emb = torch.cat((angles, angles), dim=-1)
            self._cache = (emb.cos(), emb.sin())
            self._cached_len = seq_len
        cos, sin = self._cache  # type: ignore[misc]
        return cos[:seq_len].to(dtype), sin[:seq_len].to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` of shape ``(batch, heads, seq, head_dim)`` in place of adding a PE."""
    return x * cos + _rotate_half(x) * sin


class _SelfAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE, via ``scaled_dot_product_attention``."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, attn_mask: Tensor | None) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class _SwiGLU(nn.Module):
    """``(W1 x) * silu(W2 x) -> W3``; the hidden width is scaled by 2/3 to match GELU params."""

    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(round(dim * ratio * 2 / 3 / 8)) * 8 or dim
        self.gate = nn.Linear(dim, 2 * hidden)
        self.out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.gate(x).chunk(2, dim=-1)
        return self.drop(self.out(value * torch.nn.functional.silu(gate)))


class _GELUMLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(round(dim * ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.net(x))


class TransformerBlock(nn.Module):
    """Pre-LN block: ``x + attn(LN(x))`` then ``x + mlp(LN(x))``."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp: str = "swiglu",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, heads, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        if mlp == "swiglu":
            self.mlp: nn.Module = _SwiGLU(dim, mlp_ratio, dropout)
        elif mlp == "gelu":
            self.mlp = _GELUMLP(dim, mlp_ratio, dropout)
        else:
            raise ValueError(f"mlp must be 'swiglu' or 'gelu', got {mlp!r}")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, attn_mask: Tensor | None) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, attn_mask))
        return x + self.mlp(self.norm2(x))
