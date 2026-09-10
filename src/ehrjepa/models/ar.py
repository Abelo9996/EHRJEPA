"""``EHRAR``: the same embedding and encoder as :class:`~ehrjepa.models.jepa.EHRJEPA`,
with a next-code head instead of a predictor.

This is deliberately not a subclass and deliberately not a flag on ``EHRJEPA``:
the two models share their *parts*, not their forward pass, and a shared
``forward`` that branched on the objective would make the one thing the
comparison rests on -- that only the loss differs -- harder to read, not easier.
What they do share is :class:`~ehrjepa.models.jepa.EHRJEPAConfig`, so a config
file, a checkpoint payload and the probe can treat the two interchangeably; the
``pred_*`` fields are simply unused here, and ``causal``/``tie_embeddings`` are
unused by ``EHRJEPA``.

Only the positions that actually have a next event reach the vocabulary at all,
and this module stops one matmul short of them: ``forward`` returns the gathered
hidden rows, and :func:`~ehrjepa.objectives.ar.ar_loss_chunked` projects those in
slices. At batch 64 x 512 with a 30,000-code vocabulary a dense ``(B, L, V)``
logit tensor is 3.9 GB in float32, which does not belong on a 16 GB laptop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ehrjepa.data.tokenize import N_VALUE_BINS, PAD_ID
from ehrjepa.models.jepa import EHRJEPAConfig, build_event_stack, effective_valid
from ehrjepa.models.pretrained import load_encoder_weights
from ehrjepa.objectives.ar import NextCodeHead, next_code_targets

__all__ = ["AROutput", "EHRAR"]


@dataclass
class AROutput:
    """One forward pass: the scored hidden rows and their targets, plus the encoder's.

    ``hidden`` stops one matmul short of logits on purpose. The loss projects it
    in slices (see :func:`~ehrjepa.objectives.ar.ar_loss_chunked`), and a
    ``(n_targets, 30000)`` tensor returned from here would be exactly the
    allocation that trick exists to avoid.
    """

    hidden: Tensor  # (n_targets, dim)
    targets: Tensor  # (n_targets,)
    tokens: Tensor  # (B, L, dim)
    cls: Tensor  # (B, dim)
    valid_mask: Tensor  # (B, L) bool
    #: The *next* event's ``value_z`` and ``value_bin`` at the scored positions,
    #: for ``objective.lambda_value``. Gathered unconditionally -- two index
    #: operations on tensors already in memory -- so a run that does not ask for
    #: the term pays nothing measurable and computes nothing different.
    value_z: Tensor | None = None  # (n_targets,)
    value_bin: Tensor | None = None  # (n_targets,)


class EHRAR(nn.Module):
    """Causal transformer language model over EHR event codes."""

    def __init__(self, config: EHRJEPAConfig) -> None:
        super().__init__()
        if not config.causal:
            raise ValueError("the autoregressive model needs model.causal=true")
        self.config = config
        self.embed, self.encoder = build_event_stack(config, causal=True)
        self.head = NextCodeHead(
            config.dim,
            config.vocab_size,
            tied_weight=self.embed.code_emb.weight if config.tie_embeddings else None,
        )
        # After the head, and only when the objective asks for them, so the RNG
        # stream of every existing AR run is untouched.
        self.recon_value_head: nn.Module | None = None
        if config.recon_value_head:
            self.recon_value_head = nn.Linear(config.dim, N_VALUE_BINS + 1)
        self.value_head: nn.Module | None = None
        if config.value_head:
            self.value_head = nn.Linear(config.dim, 1)
        if config.init_from:
            for note in load_encoder_weights(
                self.embed, self.encoder, config.init_from, config, "init_from"
            ):
                print(f"[note] model.init_from differs in {note}", flush=True)

    # ------------------------------------------------------------------ #

    @property
    def uses_ema(self) -> bool:
        """Never; the trainer asks every model this."""
        return False

    def n_parameters(self) -> dict[str, int]:
        """Trainable parameter counts per component, plus the total.

        ``head`` counts only what the head *owns*: with tied embeddings that is
        the LayerNorm and the output bias, because the ``vocab_size x dim``
        matrix is already counted under ``embedding``.
        """

        def count(module: nn.Module | None) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        counts = {
            "embedding": count(self.embed),
            "encoder": count(self.encoder),
            "predictor": 0,
            "head": count(self.head),
        }
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:  # pragma: no cover - trivial
        """No target network to move."""

    # ------------------------------------------------------------------ #

    def embed_batch(self, batch: Mapping[str, Tensor], target_side: bool = False) -> Tensor:
        return self.embed(
            batch["code_id"],
            batch["value_bin"],
            batch["value_z"],
            batch["age"],
            batch["log_delta"],
        )

    def encode(self, batch: Mapping[str, Tensor], valid_mask: Tensor | None = None) -> Tensor:
        """The CLS row, for interface parity with ``EHRJEPA``.

        Under causal attention this row is a function of the CLS parameter alone;
        :mod:`ehrjepa.eval.probe` is where the pooling that matters is chosen.
        """
        mask = batch["attention_mask"] if valid_mask is None else valid_mask
        return self.encoder(self.embed_batch(batch), mask).cls

    def forward(self, batch: Mapping[str, Tensor]) -> AROutput:
        tokens = self.embed_batch(batch)
        valid = effective_valid(tokens, batch["attention_mask"])
        encoded = self.encoder(tokens, valid)
        targets = next_code_targets(batch["code_id"], valid)
        keep = targets != PAD_ID
        # The value of the event being predicted, at the same positions -- so the
        # regression target is the *next* event's number, exactly as the code
        # target is the next event's code.
        shifted_z = torch.zeros_like(batch["value_z"])
        shifted_z[:, :-1] = batch["value_z"][:, 1:]
        shifted_bin = torch.zeros_like(batch["value_bin"])
        shifted_bin[:, :-1] = batch["value_bin"][:, 1:]
        return AROutput(
            hidden=encoded.tokens[keep],
            targets=targets[keep],
            tokens=encoded.tokens,
            cls=encoded.cls,
            valid_mask=valid,
            value_z=shifted_z[keep],
            value_bin=shifted_bin[keep],
        )
