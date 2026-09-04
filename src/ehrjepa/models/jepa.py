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

**What the target is allowed to be.** Three config flags, all defaulting to the
behaviour above, exist because diagnostics on the first pilot grid found the task
soluble without the encoder:

``mask_token_time``
    Off, the predictor's mask tokens carry no ``age``/``log_delta``, so a
    time-conditional prior cannot stand in for a prediction.
``target_time_features``
    Off, the target encoder's input is content only (code + value), so the
    quantity being predicted is not itself mostly a clock.
``target_span_only``
    On, the target encoder sees the target span alone rather than the full
    window, so a target latent cannot absorb the context the predictor was given
    and be recovered by copying.

``time_feature_dropout`` drops both time terms per token on the *online* pass, so
that under shared or EMA weights the encoder has seen inputs shaped like the
content-only ones the target pass produces.

``objective.lambda_pred``
    Weight on the prediction loss. At ``0`` the target pass is skippable --
    see ``forward``'s ``compute_targets`` -- and, with ``objective.lambda_recon``
    positive, the predictor is trained purely to name each target's code.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
from torch import Tensor, nn

from ehrjepa.data.tokenize import N_VALUE_BINS
from ehrjepa.models.embedding import EventEmbedding
from ehrjepa.models.encoder import Encoder
from ehrjepa.models.predictor import Predictor

__all__ = ["EHRJEPA", "EHRJEPAConfig", "ema_momentum"]

TARGET_MODES = ("shared", "ema")

#: The per-event tensors an :class:`~ehrjepa.models.embedding.EventEmbedding` reads.
EVENT_FIELDS = ("code_id", "value_bin", "value_z", "age", "log_delta")


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

    #: Mask tokens carry the target's ``age``/``log_delta`` (``predictor.mask_token_time``).
    mask_token_time: bool = True
    #: The target encoder's input carries the time terms (``target.time_features``).
    target_time_features: bool = True
    #: The target encoder runs on the target span alone (``target.span_only``).
    target_span_only: bool = False
    #: Per-token probability of dropping both time terms on the online pass
    #: (``train.time_feature_dropout``).
    time_feature_dropout: float = 0.0
    #: Build the auxiliary code-reconstruction head (``objective.lambda_recon``).
    recon_head: bool = False
    #: Build the auxiliary ``value_bin`` head (``objective.recon_value``).
    recon_value_head: bool = False

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
            config.vocab_size,
            config.dim,
            n_freq=config.n_freq,
            dropout=config.dropout,
            time_dropout=config.time_feature_dropout,
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
            mask_token_time=config.mask_token_time,
        )
        if config.target_mode == "ema":
            self.target_embed = copy.deepcopy(self.embed).requires_grad_(False)
            self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
            # Time-feature dropout is an augmentation of the *online* input; the
            # target must see the clean (or cleanly time-free) distribution.
            self.target_embed.time_dropout = 0.0
        else:
            self.target_embed = None
            self.target_encoder = None
        # The auxiliary heads read the predictor's encoder-width output, so they
        # are the AR next-code head applied one step off its usual place. Built
        # only when asked for: an unconditional head would add rows to every
        # checkpoint's ``state_dict`` and break every existing one.
        self.recon_head: nn.Module | None = None
        self.recon_value_head: nn.Module | None = None
        if config.recon_head:
            # Imported here, not at module scope: ``ehrjepa.objectives`` imports
            # the loss, which imports this module, so a top-level import of the
            # AR head would close the cycle.
            from ehrjepa.objectives.ar import NextCodeHead

            self.recon_head = NextCodeHead(
                config.dim, config.vocab_size, tied_weight=self.embed.code_emb.weight
            )
        if config.recon_value_head:
            self.recon_value_head = nn.Linear(config.dim, N_VALUE_BINS + 1)

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
            use_time=self.config.target_time_features if target_side else True,
        )

    def encode(self, batch: Mapping[str, Tensor], valid_mask: Tensor | None = None) -> Tensor:
        """The subject (CLS) embedding for a batch, attending to ``valid_mask``.

        This is the inference entry point: no masking, no predictor, no targets.
        """
        mask = batch["attention_mask"] if valid_mask is None else valid_mask
        return self.encoder(self.embed_batch(batch), mask).cls

    @property
    def _target_stack(self) -> tuple[nn.Module, nn.Module]:
        """The (embedding, encoder) pair the target pass runs through."""
        if self.uses_ema:
            assert self.target_embed is not None and self.target_encoder is not None
            return self.target_embed, self.target_encoder
        return self.embed, self.encoder

    def _span_targets(self, batch: Mapping[str, Tensor], target_mask: Tensor) -> Tensor:
        """Encode each sequence's target positions on their own, then scatter back.

        The span is compacted left, keeping the original order, and run through
        the target encoder with its own attention mask and its own CLS. RoPE
        therefore sees the *within-span* offsets: contiguous targets keep their
        true relative positions, and multi-block targets lose the gaps between
        blocks. That is the price of the property this buys -- a target latent
        computed from the span alone cannot contain the context.
        """
        b, length = target_mask.shape
        dim = self.config.dim
        counts = target_mask.sum(dim=1)
        width = max(1, int(counts.max()))
        device = target_mask.device

        # Column ``rank`` of row ``r`` gets the original index of that row's
        # ``rank``-th target; non-targets are parked in a scratch column.
        rank = (target_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        park = torch.where(target_mask, rank, torch.full_like(rank, width))
        source = torch.arange(length, device=device).expand(b, length)
        index = torch.zeros(b, width + 1, dtype=torch.long, device=device)
        index.scatter_(1, park, source)
        index = index[:, :width]
        span_mask = torch.arange(width, device=device)[None, :] < counts[:, None]

        span = {key: batch[key].gather(1, index) for key in EVENT_FIELDS}
        embed, encoder = self._target_stack
        tokens = embed(
            span["code_id"],
            span["value_bin"],
            span["value_z"],
            span["age"],
            span["log_delta"],
            use_time=self.config.target_time_features,
        )
        encoded = encoder(tokens, span_mask).tokens

        # Scatter into an (L + 1)-wide buffer so the padding columns, whose index
        # is a duplicate 0, land in a scratch slot instead of clobbering a real
        # target at position 0.
        scatter_to = torch.where(span_mask, index, torch.full_like(index, length))
        full = torch.zeros(b, length + 1, dim, device=encoded.device, dtype=encoded.dtype)
        full.scatter_(1, scatter_to[:, :, None].expand(-1, -1, dim), encoded)
        return full[:, :length]

    def forward(
        self,
        batch: Mapping[str, Tensor],
        context_mask: Tensor,
        target_mask: Tensor,
        compute_targets: bool = True,
    ) -> JEPAOutput:
        """``compute_targets=False`` skips the target pass entirely.

        Set by the caller when ``objective.lambda_pred`` is 0: with no
        prediction-loss term to feed, the target encoder's forward pass (an EMA
        copy's full encoder call, or a re-embedding under shared weights) is pure
        waste, so it is never run and ``targets`` comes back as a zero
        placeholder of the right shape instead. The predictor still runs -- a
        ``lambda_pred: 0`` / ``lambda_recon > 0`` config predicts codes through
        the same predictor, just with nothing pulling on the latent itself.
        """
        context_mask = context_mask.bool()
        target_mask = target_mask.bool()
        tokens = self.embed_batch(batch)
        context = self.encoder(tokens, context_mask)
        index = target_mask.nonzero(as_tuple=True)

        target_repr: Tensor | None = None
        if compute_targets:
            with torch.no_grad():
                if self.config.target_span_only:
                    target_repr = self._span_targets(batch, target_mask).detach()
                elif self.uses_ema or not self.config.target_time_features:
                    # The shared-weight path re-embeds only when it has to: with
                    # the time terms on, the online tokens are the same tensor.
                    assert self.target_encoder is not None or not self.uses_ema
                    target_tokens = self.embed_batch(batch, target_side=True)
                    _, encoder = self._target_stack
                    target_repr = encoder(target_tokens, batch["attention_mask"]).tokens.detach()
                else:
                    target_repr = self.encoder(
                        tokens.detach(), batch["attention_mask"]
                    ).tokens.detach()

        predicted = self.predictor(
            context.tokens, batch["age"], batch["log_delta"], context_mask, target_mask
        )
        predictions = predicted[index]
        targets = target_repr[index] if target_repr is not None else torch.zeros_like(predictions)
        extras: dict[str, Tensor] = {}
        if self.recon_head is not None:
            extras["recon_code_id"] = batch["code_id"][index]
        if self.recon_value_head is not None:
            extras["recon_value_bin"] = batch["value_bin"][index]
        return JEPAOutput(
            predictions=predictions,
            targets=targets,
            target_index=index,
            context_tokens=context.tokens,
            context_mask=context_mask,
            cls=context.cls,
            extras=extras,
        )
