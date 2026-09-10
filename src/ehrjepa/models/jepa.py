"""``EHRJEPA``: embedding + encoder + predictor, and the three ways to make targets.

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
    ``"frozen"``: the same second copy, loaded once from a finished run's
    checkpoint (``model.target_init``) and never updated at all. Where ``ema``
    asks "can a slow-moving copy of yourself be a target", this asks "can a
    *pretrained* teacher replace the schedule entirely" -- the target is a fixed
    function from step zero, so there is no momentum to tune and no chance of the
    two networks collapsing together. ``model.init_from`` is the orthogonal knob:
    it starts the *online* stack from a checkpoint, so "student initialized from
    the teacher" and "student initialized from scratch" are separate rows.

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
from dataclasses import dataclass, field, fields, replace

import torch
from torch import Tensor, nn

from ehrjepa.data.tokenize import N_VALUE_BINS
from ehrjepa.models.embedding import CODE_INITS, EventEmbedding
from ehrjepa.models.encoder import Encoder
from ehrjepa.models.predictor import Predictor
from ehrjepa.models.pretrained import load_encoder_weights

__all__ = ["EHRJEPA", "EHRJEPAConfig", "build_event_stack", "effective_valid", "ema_momentum"]

TARGET_MODES = ("shared", "ema", "frozen")

#: ``model.encoder``: the from-scratch stack, or the pretrained language model.
ENCODER_KINDS = ("scratch", "lm")

#: The target modes that allocate a second embedding+encoder pair.
TARGET_COPY_MODES = ("ema", "frozen")

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
    #: ``target_mode: frozen`` only: the ``final.pt`` whose ``embed``/``encoder``
    #: weights become the fixed teacher.
    target_init: str | None = None
    #: Optional: initialize the *online* ``embed``/``encoder`` from this
    #: checkpoint instead of from scratch. Independent of ``target_init``.
    init_from: str | None = None

    #: ``random`` (the default, and the only behaviour before this existed) or
    #: ``text``, which overwrites the code embedding table with the projected
    #: sentence embeddings of each code's description -- see
    #: :mod:`ehrjepa.data.code_text`.
    code_init: str = "random"
    #: Where the ``code_init: text`` table lives. Filled in from the cache
    #: directory and ``dim`` by :meth:`PretrainConfig.model_config` when unset.
    code_init_path: str | None = None
    #: Freeze the code embedding table. At this repository's vocabulary
    #: (30,000 x 256) that is 7.68M parameters, 58% of a base-size hybrid's
    #: 13.21M trainable ones and 61% of the AR model's 12.67M.
    freeze_code_embeddings: bool = False

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
    #: Build the continuous ``value_z`` regression head (``objective.lambda_value``).
    #: ``dim -> 1``; the loss masks it to targets whose ``value_bin`` is non-zero.
    value_head: bool = False

    #: Which event stack runs: ``scratch`` is this repository's
    #: :class:`~ehrjepa.models.embedding.EventEmbedding` + RoPE
    #: :class:`~ehrjepa.models.encoder.Encoder`; ``lm`` serialises each event to a
    #: short text span and reads the event representation off a frozen pretrained
    #: causal language model with LoRA adapters
    #: (:mod:`ehrjepa.models.lm`).
    encoder: str = "scratch"
    #: ``encoder: lm`` only: the Hugging Face model id. Its tokenizer is used
    #: unless ``lm_tokenizer`` names another, and both are recorded in the
    #: checkpoint's model config so ``load_encoder`` rebuilds the same stack.
    lm_name: str = "Qwen/Qwen2.5-0.5B"
    #: Tokenizer id, when it differs from ``lm_name``.
    lm_tokenizer: str | None = None
    #: Hard cap on tokens per window. A window over the cap is truncated from the
    #: *left* (the oldest events are dropped), which is why ``last`` pooling and
    #: the causal next-latent targets are unaffected by it.
    lm_max_tokens: int = 2048
    lm_lora_r: int = 8
    lm_lora_alpha: int = 16
    #: Comma-separated LoRA target module names. The default is Qwen2's attention
    #: projections; a GPT-2 shaped model wants ``c_attn``.
    lm_lora_targets: str = "q_proj,k_proj,v_proj,o_proj"
    lm_grad_checkpointing: bool = True
    #: The cache directory whose ``vocab.parquet`` supplies each code's words and
    #: whose ``quantizer.parquet`` supplies the per-code mean/std that turn
    #: ``value_z`` back into a printable number. Filled in from ``data.cache_dir``
    #: by :meth:`PretrainConfig.model_config`.
    lm_cache_dir: str | None = None

    #: Build the transformer :class:`~ehrjepa.models.predictor.Predictor`. The
    #: causal latent objectives in :mod:`ehrjepa.models.latent` replace it with
    #: their own MLP heads and set this ``False``; a masked-span JEPA leaves it
    #: alone. Guarding the construction rather than deleting it afterwards keeps
    #: the module-initialisation order -- and therefore the RNG stream of every
    #: existing ``jepa``/``ar`` run -- exactly as it was.
    build_predictor: bool = True
    #: ``objective.horizons``: the step offsets ``nextlatent`` predicts, one MLP
    #: head each. Recorded here so a checkpoint rebuilds the same heads.
    horizons: list[int] = field(default_factory=lambda: [1])
    #: ``objective.window_horizons``: the horizons in days ``window`` pools over,
    #: one learned horizon embedding row each.
    window_horizons: list[float] = field(default_factory=lambda: [30.0, 365.0])

    def __post_init__(self) -> None:
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {self.target_mode!r}")
        if self.code_init not in CODE_INITS:
            raise ValueError(f"code_init must be one of {CODE_INITS}, got {self.code_init!r}")
        # "frozen without a target_init" is *not* an error here: it is what
        # :meth:`for_reload` produces, and a frozen-but-untrained teacher is the
        # right thing for the ``random_init`` control arm. A training config that
        # forgets the path is caught by ``PretrainConfig.model_config``.
        if self.target_init and self.target_mode != "frozen":
            raise ValueError(
                f"model.target_init is only read by target_mode='frozen', "
                f"but target_mode is {self.target_mode!r}"
            )
        if self.code_init == "text" and not self.code_init_path:
            raise ValueError("code_init='text' needs model.code_init_path to name a .npy table")
        if self.encoder not in ENCODER_KINDS:
            raise ValueError(f"model.encoder must be one of {ENCODER_KINDS}, got {self.encoder!r}")
        if self.encoder == "lm":
            self._check_lm()

    def _check_lm(self) -> None:
        """What ``encoder: lm`` needs, and the three knobs it cannot honour.

        Each rejection is a thing the LM stack genuinely has no counterpart for,
        and each is better a construction-time error than a silently ignored
        setting:

        (``objective.kind: jepa`` is rejected one level up, by
        :meth:`~ehrjepa.train.config.PretrainConfig.model_config`, which is the
        only place that sees both the objective and the model section.)

        ``target_mode`` other than ``shared``
            An EMA or frozen teacher is a *second copy* of the encoder. For a
            0.5B base that is another gigabyte of weights resident and another
            gigabyte written into every checkpoint, for a teacher whose frozen
            trunk is bit-identical to the student's. The shared-weight target --
            the same forward pass under stop-gradient -- is what this repository
            calls ``shared`` and is the only target mode offered here.
        ``objective.kind: jepa``
            Masked-span JEPA drops context positions with a key-side attention
            mask. A serialised text window has no such mask: dropping an event
            means re-tokenising the window. The causal objectives (``ar``,
            ``nextlatent``) need no context mask at all.
        ``target.span_only`` / ``model.share_time_encoders`` / ``code_init: text``
            All three reach into :class:`~ehrjepa.models.embedding.EventEmbedding`
            internals -- a span re-embedding, the Fourier time encoders, the code
            table as the model's view of a code -- that the LM stack does not have.
        """
        if not self.lm_cache_dir:
            raise ValueError(
                "model.encoder='lm' needs model.lm_cache_dir to name the cache whose "
                "vocab.parquet and quantizer.parquet describe each code in words"
            )
        if self.target_mode != "shared":
            raise ValueError(
                f"model.encoder='lm' supports target_mode='shared' only, got "
                f"{self.target_mode!r}: a second copy of the base model does not fit"
            )
        if self.target_span_only:
            raise ValueError("model.encoder='lm' cannot honour target.span_only")
        if self.share_time_encoders:
            raise ValueError("model.encoder='lm' has no Fourier time encoders to share")
        if self.code_init != "random":
            raise ValueError(
                f"model.encoder='lm' reads code descriptions itself; "
                f"code_init={self.code_init!r} has nothing to initialize"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> EHRJEPAConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**values)  # type: ignore[arg-type]

    def for_reload(self) -> EHRJEPAConfig:
        """The same architecture, with every *initialization source* cleared.

        A checkpoint records the paths it was initialized from, and rebuilding
        that model to load its ``state_dict`` -- which is what
        :func:`ehrjepa.eval.probe.load_encoder` does -- must not re-read them:
        the weights are all about to be overwritten, and on an eval machine the
        source checkpoint and the text-init ``.npy`` may not exist at all. The
        *shape* fields, ``target_mode`` included, are kept, because the target
        stack has to be there for a strict load to succeed.
        """
        return replace(self, target_init=None, init_from=None, code_init="random")


def build_event_stack(
    config: EHRJEPAConfig, *, time_dropout: float = 0.0, causal: bool = False
) -> tuple[nn.Module, nn.Module]:
    """``(embedding, encoder)`` for ``config.encoder``, in that construction order.

    The order and the argument lists of the ``scratch`` branch are exactly what
    :class:`EHRJEPA` and :class:`~ehrjepa.models.ar.EHRAR` used before this
    function existed, so the number and sequence of RNG draws -- and therefore
    every recorded checksum -- is unchanged. ``time_dropout`` and ``causal`` are
    parameters rather than reads of ``config`` because the two callers differ on
    both: the AR model has never applied time-feature dropout and is always
    causal.

    The ``lm`` branch returns a
    :class:`~ehrjepa.models.lm.LMEventTokenizer` and a
    :class:`~ehrjepa.models.lm.LMEncoder`, which honour the same two-call
    protocol (``embed(...)`` then ``encoder(tokens, valid_mask)``) so the models,
    the trainer and :mod:`ehrjepa.eval.probe` need no branch of their own.
    """
    if config.encoder == "lm":
        # Imported here so that `import ehrjepa.models` does not require
        # transformers/peft, which are the optional `lm` extra.
        from ehrjepa.models.lm import LMEncoder, LMEventTokenizer

        return LMEventTokenizer(config), LMEncoder(config)
    embed = EventEmbedding(
        config.vocab_size,
        config.dim,
        n_freq=config.n_freq,
        dropout=config.dropout,
        time_dropout=time_dropout,
        code_init=config.code_init,
        code_init_path=config.code_init_path,
        freeze_code_embeddings=config.freeze_code_embeddings,
    )
    encoder = Encoder(
        config.dim,
        config.depth,
        config.heads,
        mlp=config.mlp,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        causal=causal,
    )
    return embed, encoder


def effective_valid(tokens: object, valid: Tensor) -> Tensor:
    """``valid``, intersected with the events the encoder's input actually kept.

    The from-scratch stack keeps everything -- one event in, one token out -- and
    this is the identity. The LM stack caps a window at ``model.lm_max_tokens``
    and drops the oldest events that do not fit, reporting them in
    ``LMTokens.kept``; a dropped event has no hidden state, so it must not be
    scored, attended to as a target, or pooled over.
    """
    kept = getattr(tokens, "kept", None)
    # Always ``bool``: callers use the result both as an attention mask (where an
    # integer tensor is harmless) and as a boolean index (where it is not -- an
    # integer tensor there selects rows by position instead of by predicate).
    return valid.bool() if kept is None else valid.bool() & kept.bool()


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
        self.embed, self.encoder = build_event_stack(
            config, time_dropout=config.time_feature_dropout, causal=config.causal
        )
        self.predictor: Predictor | None = None
        if config.build_predictor:
            shared = (
                (self.embed.age_enc, self.embed.delta_enc) if config.share_time_encoders else None
            )
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
        if config.init_from:
            # Before the target copy is taken, so that "frozen teacher, student
            # initialized from the same checkpoint" starts the two identical.
            for note in load_encoder_weights(
                self.embed, self.encoder, config.init_from, config, "init_from"
            ):
                print(f"[note] model.init_from differs in {note}", flush=True)
        if config.target_mode in TARGET_COPY_MODES:
            self.target_embed = copy.deepcopy(self.embed).requires_grad_(False)
            self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
            # Time-feature dropout is an augmentation of the *online* input; the
            # target must see the clean (or cleanly time-free) distribution.
            self.target_embed.time_dropout = 0.0
            if config.target_init:
                stack = (self.target_embed, self.target_encoder)
                for note in load_encoder_weights(*stack, config.target_init, config, "target_init"):
                    print(f"[note] model.target_init differs in {note}", flush=True)
                # ``load_state_dict`` writes into the existing Parameters, so the
                # flag survives -- but say so, because "the teacher is frozen" is
                # the whole content of this target mode.
                self.target_embed.requires_grad_(False)
                self.target_encoder.requires_grad_(False)
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
        # Last, and only when asked for: an unconditional head would add a row to
        # every checkpoint's ``state_dict`` and take a draw from the RNG stream
        # that every recorded loss checksum depends on.
        self.value_head: nn.Module | None = None
        if config.value_head:
            self.value_head = nn.Linear(config.dim, 1)

    # ------------------------------------------------------------------ #

    @property
    def uses_ema(self) -> bool:
        """Whether the trainer should move the target copy after each step."""
        return self.config.target_mode == "ema"

    @property
    def uses_target_copy(self) -> bool:
        """Whether there *is* a second embedding+encoder pair -- ``ema`` or ``frozen``."""
        return self.config.target_mode in TARGET_COPY_MODES

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
        module = self.target_embed if (target_side and self.uses_target_copy) else self.embed
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
        if self.uses_target_copy:
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

    @torch.no_grad()
    def window_targets(
        self, batch: Mapping[str, Tensor], tokens: Tensor, hidden: Tensor | None = None
    ) -> Tensor:
        """Target latents for **every** position of the window, under ``no_grad``.

        Which stack runs is the same decision :meth:`forward` makes for the
        masked-span objective, minus ``target_span_only`` (which is defined by a
        target *span*, and the causal objectives in :mod:`ehrjepa.models.latent`
        have none): the EMA copy when there is one, a content-only re-embedding
        when ``target.time_features`` is off, and otherwise the online tokens run
        through the online encoder under stop-gradient.

        ``hidden`` is the online pass's output, offered by the caller as a
        shortcut for the one case where the second forward is provably redundant:
        the LM stack under shared targets with the time terms on. There the
        target pass would re-run the *same* frozen base on the *same* token ids
        with no dropout anywhere, so it returns the same numbers as the online
        pass at the price of a second 0.5B forward per step. The from-scratch
        stack does not take this path -- its encoder has residual dropout, so the
        two passes are genuinely different samples and the existing behaviour is
        preserved exactly.
        """
        valid = effective_valid(tokens, batch["attention_mask"])
        if self.uses_target_copy or not self.config.target_time_features:
            target_tokens = self.embed_batch(batch, target_side=True)
            _, encoder = self._target_stack
            return encoder(target_tokens, effective_valid(target_tokens, valid)).tokens.detach()
        if hidden is not None and self.config.encoder == "lm":
            return hidden.detach()
        return self.encoder(tokens.detach(), valid).tokens.detach()

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
            if self.config.target_span_only:
                with torch.no_grad():
                    target_repr = self._span_targets(batch, target_mask).detach()
            else:
                # The shared-weight path re-embeds only when it has to: with the
                # time terms on, the online tokens are the same tensor.
                target_repr = self.window_targets(batch, tokens)

        assert self.predictor is not None, "EHRJEPA.forward needs model.build_predictor"
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
        if self.value_head is not None:
            extras["value_target_z"] = batch["value_z"][index]
            extras["value_target_bin"] = batch["value_bin"][index]
        return JEPAOutput(
            predictions=predictions,
            targets=targets,
            target_index=index,
            context_tokens=context.tokens,
            context_mask=context_mask,
            cls=context.cls,
            extras=extras,
        )
