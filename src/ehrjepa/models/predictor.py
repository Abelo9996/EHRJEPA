"""The latent predictor: context representations plus target *time* -> target latents.

I-JEPA/V-JEPA 2 shape: a deliberately narrower transformer than the encoder, so
representation quality cannot be offloaded into it. It runs over one sequence of
the encoder's original length, where each position is one of three things:

* a **context** position -- the encoder's output there, linearly projected to the
  predictor width;
* a **target** position -- a learned ``MASK`` embedding plus the target's time
  features (``age`` and ``log_delta`` through Fourier MLPs) and nothing else;
* **dropped** -- neither, and excluded from attention.

Keeping the original layout is what makes "RoPE at the target's original sequence
index" free: position ``i`` in the predictor is event ``i`` of the window, so the
relative offsets the rotary embedding encodes are the true ones.

**What the predictor must not see.** No ``code_id``, ``value_bin`` or ``value_z``
of any target ever enters this module -- not through the mask tokens, which are
built from time alone, and not through the context representations, because the
context encoder pass never attended to the target positions. Asking the predictor
to recover a latent it could have copied would make the objective vacuous;
``tests/test_models.py`` asserts the property numerically by perturbing target
codes and values and checking the predictions are bit-identical.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ehrjepa.models.encoder import _init_linear
from ehrjepa.models.layers import RotaryEmbedding, ScalarEncoder, TransformerBlock

__all__ = ["Predictor"]


class Predictor(nn.Module):
    """Narrow transformer mapping context latents + target times to target latents.

    Parameters
    ----------
    encoder_dim:
        Width of the representations coming in and going out.
    dim, depth, heads:
        The predictor's own (narrower) width, depth and head count.
    time_encoders:
        Optional ``(age_enc, delta_enc)`` to share with the event embedding rather
        than allocating fresh ones. Shared modules are projected into the
        predictor width by a small linear map, since the embedding's encoders emit
        ``encoder_dim``.
    mask_token_time:
        When ``False`` a mask token is the bare learned ``MASK`` embedding, with
        RoPE at the target's index its only remaining positional information, and
        the whole module becomes independent of ``age`` and ``log_delta``. The
        time encoders are still allocated -- unused parameters are skipped by
        AdamW and keeping them leaves the checkpoint layout comparable -- they
        simply never run.
    """

    def __init__(
        self,
        encoder_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp: str = "swiglu",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        n_freq: int = 16,
        time_encoders: tuple[nn.Module, nn.Module] | None = None,
        mask_token_time: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.dim = dim
        self.mask_token_time = mask_token_time
        self.in_proj = nn.Linear(encoder_dim, dim)
        self.out_proj = nn.Linear(dim, encoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.shared_time = time_encoders is not None
        if time_encoders is None:
            self.age_enc: nn.Module = ScalarEncoder(dim, n_freq)
            self.delta_enc: nn.Module = ScalarEncoder(dim, n_freq)
            self.time_proj: nn.Module = nn.Identity()
        else:
            self.age_enc, self.delta_enc = time_encoders
            self.time_proj = nn.Linear(encoder_dim, dim, bias=False)
        self.rope = RotaryEmbedding(dim // heads)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, mlp, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_linear)

    def target_tokens(self, age: Tensor, log_delta: Tensor) -> Tensor:
        """Mask tokens for every position: ``MASK + f(age) + f(log_delta)``.

        With ``mask_token_time=False`` this is the bare ``MASK`` embedding, the
        same vector at every position; what distinguishes one target from
        another is then RoPE alone.
        """
        if not self.mask_token_time:
            return self.mask_token.expand(age.shape[0], age.shape[1], self.dim)
        time = self.age_enc(age) + self.delta_enc(log_delta)
        return self.mask_token + self.time_proj(time)

    def forward(
        self,
        context: Tensor,
        age: Tensor,
        log_delta: Tensor,
        context_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """Predict encoder-width latents at every position.

        ``context`` is the encoder's ``(B, L, encoder_dim)`` output. Only rows
        selected by ``target_mask`` are meaningful in the result; the caller
        gathers those. Positions in neither mask are dropped from attention and
        zeroed on input so nothing downstream can read them.
        """
        context_mask = context_mask.bool()
        target_mask = target_mask.bool()
        attend = context_mask | target_mask
        x = torch.where(
            target_mask.unsqueeze(-1),
            self.target_tokens(age, log_delta).to(context.dtype),
            self.in_proj(context),
        )
        x = x * attend.unsqueeze(-1).to(x.dtype)
        attn_mask = attend[:, None, None, :]
        blank = ~attend.any(dim=-1)  # (B,) sequences with nothing visible at all
        if bool(blank.any()):
            # Softmax over zero keys is NaN; let such a sequence's rows see
            # themselves. They are masked out of the loss anyway.
            eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
            attn_mask = attn_mask | (blank[:, None, None, None] & eye)
        cos, sin = self.rope(x.shape[1], x.device, x.dtype)
        cos, sin = cos[None, None], sin[None, None]
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        return self.out_proj(self.norm(x))
