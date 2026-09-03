"""The event encoder: a pre-LN, bidirectional, RoPE transformer with a CLS token.

The encoder sees already-embedded tokens (see :mod:`~ehrjepa.models.embedding`)
and a boolean validity mask, prepends a learned CLS token whose output is the
subject embedding, and returns both that and the per-event outputs.

Attention is bidirectional by default -- this is a joint-embedding model, not a
language model, and the context window is a *set* of events the predictor is
allowed to look at from both sides. Masking is key-side only: an invalid
(padding, or context-dropped) position still produces an output row, it just
never gets attended to. That keeps the tensor rectangular and, because CLS is
always a valid key, guarantees no attention row is fully masked, which is where
``scaled_dot_product_attention`` would otherwise return NaN.

``causal=True`` switches the same module into the autoregressive regime used by
the next-code baseline: the key-padding mask is intersected with a lower
triangle, so position ``i`` sees positions ``0..i`` and nothing later. CLS keeps
its place at index 0 and is therefore a visible *key* for every position. Its own
output row is then a function of the CLS parameter alone -- a causal model cannot
have a prefix token that reads the sequence without leaking the future back
through it at the next layer -- which is why :mod:`ehrjepa.eval.probe` offers
``last`` (the final valid token) as a pooling option for causal checkpoints.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ehrjepa.models.layers import RotaryEmbedding, TransformerBlock

__all__ = ["Encoder", "EncoderOutput"]


class EncoderOutput(dict):
    """``{"cls": (B, D), "tokens": (B, L, D)}`` -- a dict so it survives checkpointing.

    With ``return_penultimate=True`` it also carries ``cls_penultimate`` and
    ``tokens_penultimate``: the residual stream after the second-to-last block,
    *before* the final LayerNorm.
    """

    @property
    def cls(self) -> Tensor:
        return self["cls"]

    @property
    def tokens(self) -> Tensor:
        return self["tokens"]

    @property
    def cls_penultimate(self) -> Tensor:
        return self["cls_penultimate"]

    @property
    def tokens_penultimate(self) -> Tensor:
        return self["tokens_penultimate"]


class Encoder(nn.Module):
    """Pre-LN transformer over event tokens.

    Parameters
    ----------
    dim, depth, heads:
        Width, number of blocks, attention heads.
    mlp:
        ``"swiglu"`` or ``"gelu"``.
    mlp_ratio:
        Hidden width multiple. SwiGLU scales it by ``2/3`` internally so the two
        MLP choices have comparable parameter counts.
    dropout, attn_dropout:
        Residual/MLP dropout and attention dropout.
    causal:
        Restrict attention to the past (plus the CLS key at index 0).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp: str = "swiglu",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.causal = causal
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.rope = RotaryEmbedding(dim // heads)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, mlp, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_linear)

    def forward(
        self, tokens: Tensor, valid_mask: Tensor, return_penultimate: bool = False
    ) -> EncoderOutput:
        """``tokens`` is ``(B, L, D)``; ``valid_mask`` is ``(B, L)`` and truthy on
        positions the encoder is allowed to attend to."""
        b, n, _ = tokens.shape
        x = torch.cat((self.cls_token.expand(b, 1, -1).to(tokens.dtype), tokens), dim=1)
        keep = torch.cat(
            (torch.ones(b, 1, dtype=torch.bool, device=tokens.device), valid_mask.bool()), dim=1
        )
        # (B, 1, 1, L+1): key-side padding, broadcast over heads and queries.
        attn_mask = keep[:, None, None, :]
        if self.causal:
            # (L+1, L+1) lower triangle, intersected with the key padding. Column
            # 0 is CLS, which is a valid key and is at or before every query, so
            # no row goes fully masked -- including a padding row, whose own
            # diagonal entry the key mask removes.
            tri = torch.ones(n + 1, n + 1, dtype=torch.bool, device=tokens.device).tril()
            attn_mask = attn_mask & tri[None, None]
        cos, sin = self.rope(n + 1, tokens.device, x.dtype)
        cos, sin = cos[None, None], sin[None, None]
        penultimate: Tensor | None = None
        last = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            if return_penultimate and i == last:
                penultimate = x
            x = block(x, cos, sin, attn_mask)
        x = self.norm(x)
        out = EncoderOutput(cls=x[:, 0], tokens=x[:, 1:])
        if return_penultimate:
            # For depth 1 there is no "block before the last", and the residual
            # stream entering the only block -- the embedding -- is what falls out.
            assert penultimate is not None
            out["cls_penultimate"] = penultimate[:, 0]
            out["tokens_penultimate"] = penultimate[:, 1:]
        return out


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
