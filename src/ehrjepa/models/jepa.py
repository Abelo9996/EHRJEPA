"""``EHRJEPA``: embedding + encoder + predictor, and the two ways to make targets.

The forward pass is three encoder-shaped things:

1. **Context pass.** The encoder runs over the whole window but attends only to
   ``context_mask`` positions. This is the pass that carries gradients, and the
   pass whose outputs the anti-collapse regularizer looks at.
2. **Target pass.** The encoder (or an EMA copy of it) runs over the *full*
   window under ``no_grad``. Target latents are read at ``target_mask``
   positions.
3. **Prediction.** The predictor sees the context pass's outputs and, for each
   target, only its position and its time features.

``target_mode``
    ``"shared"`` (default, LeJEPA): targets come from the same weights under
    stop-gradient. There is no second copy of the model, nothing to schedule, and
    collapse is prevented by SIGReg rather than by an asymmetry between two
    networks.
    ``"ema"`` (V-JEPA 2): a frozen copy of embedding+encoder updated as
    ``p_ema <- m * p_ema + (1 - m) * p`` with ``m`` on a schedule from
    ``0.996`` to ``1.0`` over training, so the target network stops moving as the
    run ends.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
from torch import Tensor, nn

from ehrjepa.models.embedding import EventEmbedding
from ehrjepa.models.encoder import Encoder
from ehrjepa.models.predictor import Predictor

__all__ = ["EHRJEPA", "EHRJEPAConfig", "ema_momentum"]

TARGET_MODES = ("shared", "ema")


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Linear momentum schedule, ``start`` at step 0 and ``end`` at ``total_steps``."""
    if total_steps <= 0:
        return end
    frac = min(max(step / total_steps, 0.0), 1.0)
    return start + (end - start) * frac


@dataclass
class EHRJEPAConfig:
    """Everything that determines the shape of the three networks."""

    vocab_size: int
    dim: int = 256
    depth: int = 6
    heads: int = 4
    mlp: str = "swiglu"
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    n_freq: int = 16

    #: Causal (autoregressive) attention. Required by ``objective.kind: ar``;
    #: a JEPA encoder is bidirectional and leaves this ``False``.
    causal: bool = False
    #: Tie the next-code output projection to the code embedding table. Only the
    #: AR model reads this; :class:`EHRJEPA` has no output vocabulary.
    tie_embeddings: bool = True

    pred_dim: int = 128
    pred_depth: int = 4
    pred_heads: int = 4
    pred_mlp_ratio: float = 4.0

    target_mode: str = "shared"
    ema_start: float = 0.996
    ema_end: float = 1.0

    #: Reuse the embedding's age/log_delta encoders inside the predictor.
    share_time_encoders: bool = False

    def __post_init__(self) -> None:
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {self.target_mode!r}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> EHRJEPAConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**values)  # type: ignore[arg-type]


@dataclass
class JEPAOutput:
    """One forward pass, with everything the loss and the diagnostics need."""

    predictions: Tensor  # (n_targets, dim)
    targets: Tensor  # (n_targets, dim), stop-gradient
    target_index: tuple[Tensor, Tensor]  # (batch_idx, position_idx)
    context_tokens: Tensor  # (B, L, dim), the gradient-carrying encoder pass
    context_mask: Tensor  # (B, L) bool
    cls: Tensor  # (B, dim)
    extras: dict[str, Tensor] = field(default_factory=dict)


class EHRJEPA(nn.Module):
    """Embedding + encoder + predictor, wired for joint-embedding prediction."""

    def __init__(self, config: EHRJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = EventEmbedding(
            config.vocab_size, config.dim, n_freq=config.n_freq, dropout=config.dropout
        )
        self.encoder = Encoder(
            config.dim,
            config.depth,
            config.heads,
            mlp=config.mlp,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            causal=config.causal,
        )
        shared = (self.embed.age_enc, self.embed.delta_enc) if config.share_time_encoders else None
        self.predictor = Predictor(
            config.dim,
            config.pred_dim,
            config.pred_depth,
            config.pred_heads,
            mlp=config.mlp,
            mlp_ratio=config.pred_mlp_ratio,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            n_freq=config.n_freq,
            time_encoders=shared,
        )
        if config.target_mode == "ema":
            self.target_embed = copy.deepcopy(self.embed).requires_grad_(False)
            self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        else:
            self.target_embed = None
            self.target_encoder = None

    # ------------------------------------------------------------------ #

    @property
    def uses_ema(self) -> bool:
        return self.config.target_mode == "ema"

    def n_parameters(self) -> dict[str, int]:
        """Trainable parameter counts per component, plus the total."""

        def count(module: nn.Module | None) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        counts = {
            "embedding": count(self.embed),
            "encoder": count(self.encoder),
            "predictor": count(self.predictor),
        }
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        """``p_ema <- m * p_ema + (1 - m) * p`` over parameters and buffers."""
        if not self.uses_ema:
            return
        pairs = ((self.embed, self.target_embed), (self.encoder, self.target_encoder))
        for online, target in pairs:
            assert target is not None
            for p_online, p_target in zip(online.parameters(), target.parameters(), strict=True):
                p_target.mul_(momentum).add_(p_online.detach(), alpha=1.0 - momentum)
            for b_online, b_target in zip(online.buffers(), target.buffers(), strict=True):
                b_target.copy_(b_online)

    # ------------------------------------------------------------------ #

    def embed_batch(self, batch: Mapping[str, Tensor], target_side: bool = False) -> Tensor:
        module = self.target_embed if (target_side and self.uses_ema) else self.embed
        assert module is not None
        return module(
            batch["code_id"],
            batch["value_bin"],
            batch["value_z"],
            batch["age"],
            batch["log_delta"],
        )

    def encode(self, batch: Mapping[str, Tensor], valid_mask: Tensor | None = None) -> Tensor:
        """The subject (CLS) embedding for a batch, attending to ``valid_mask``.

        This is the inference entry point: no masking, no predictor, no targets.
        """
        mask = batch["attention_mask"] if valid_mask is None else valid_mask
        return self.encoder(self.embed_batch(batch), mask).cls

    def forward(
        self,
        batch: Mapping[str, Tensor],
        context_mask: Tensor,
        target_mask: Tensor,
    ) -> JEPAOutput:
        context_mask = context_mask.bool()
        target_mask = target_mask.bool()
        tokens = self.embed_batch(batch)
        context = self.encoder(tokens, context_mask)

        with torch.no_grad():
            if self.uses_ema:
                assert self.target_encoder is not None
                target_tokens = self.embed_batch(batch, target_side=True)
                target_out = self.target_encoder(target_tokens, batch["attention_mask"])
            else:
                target_out = self.encoder(tokens.detach(), batch["attention_mask"])
            target_repr = target_out.tokens.detach()

        predicted = self.predictor(
            context.tokens, batch["age"], batch["log_delta"], context_mask, target_mask
        )
        index = target_mask.nonzero(as_tuple=True)
        return JEPAOutput(
            predictions=predicted[index],
            targets=target_repr[index],
            target_index=index,
            context_tokens=context.tokens,
            context_mask=context_mask,
            cls=context.cls,
        )
