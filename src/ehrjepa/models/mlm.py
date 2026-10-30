"""``EHRMLM``: bidirectional masked-code prediction (``objective.kind: mlm``).

The Imperial-style baseline (arXiv 2610.29528 compares JEPA against a
bidirectional MLM on MIMIC-IV): the same embedding and encoder as every other
objective here, with attention bidirectional, a fraction ``objective.mlm_prob``
of each window's events selected, and the code of each selected event predicted
from the encoder's own output at that position through the tied code head.

Selection follows BERT: of the selected events 80% have their code replaced by
``[MASK]``, 10% by a random non-special code, 10% left unchanged. A replaced
event also loses its value channels (``value_bin`` and ``value_z`` set to "no
number"), because a decile and z-score are per-code statistics and would name
the masked code almost as well as the code itself. Time channels are kept: they
are where the event is, not what it is. Every row with a valid event gets at
least one selection, so no window contributes an empty term.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ehrjepa.data.tokenize import MASK_ID, PAD_ID
from ehrjepa.models.ar import EHRAR, AROutput
from ehrjepa.models.jepa import EHRJEPAConfig, build_event_stack, effective_valid
from ehrjepa.models.pretrained import load_encoder_weights
from ehrjepa.objectives.ar import NextCodeHead

__all__ = ["N_SPECIAL", "EHRMLM", "mlm_inputs"]

#: ``[PAD]``, ``[UNK]``, ``[CLS]``, ``[MASK]`` -- never drawn as a random replacement.
N_SPECIAL = 4


def mlm_inputs(
    code_id: Tensor,
    valid: Tensor,
    prob: float,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(input_code, selected, blank_value)``, all ``(B, L)``, drawn on ``code_id``'s device.

    ``selected`` marks the scored positions; ``blank_value`` the ones whose code
    was replaced (``[MASK]`` or random) and whose value channels must go.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError(f"mlm_prob must be in (0, 1), got {prob}")
    valid = valid.bool()
    cpu = code_id.cpu()
    shape = cpu.shape
    draw = torch.rand(shape, generator=generator)
    selected = (draw < prob) & valid.cpu()
    # Rows with valid events but no selection get their argmax-draw event.
    empty = ~selected.any(dim=1) & valid.cpu().any(dim=1)
    if bool(empty.any()):
        scores = torch.where(valid.cpu(), draw, torch.full_like(draw, -1.0))
        pick = scores.argmax(dim=1)
        rows = empty.nonzero(as_tuple=True)[0]
        selected[rows, pick[rows]] = True
    kind = torch.rand(shape, generator=generator)
    to_mask = selected & (kind < 0.8)
    to_random = selected & (kind >= 0.8) & (kind < 0.9)
    random_codes = torch.randint(
        N_SPECIAL, max(N_SPECIAL + 1, vocab_size), shape, generator=generator
    )
    out = torch.where(to_mask, torch.full_like(cpu, MASK_ID), cpu)
    out = torch.where(to_random, random_codes, out)
    device = code_id.device
    return out.to(device), selected.to(device), (to_mask | to_random).to(device)


class EHRMLM(EHRAR):
    """:class:`EHRAR`'s parts, bidirectional, scored at masked positions.

    Subclassed for the head, the parameter accounting and the probe interface;
    the forward pass is its own.
    """

    def __init__(self, config: EHRJEPAConfig) -> None:
        if config.causal:
            raise ValueError("the MLM model needs model.causal=false")
        nn.Module.__init__(self)
        self.config = config
        self.embed, self.encoder = build_event_stack(config, causal=False)
        self.head = NextCodeHead(
            config.dim,
            config.vocab_size,
            tied_weight=self.embed.code_emb.weight if config.tie_embeddings else None,
        )
        self.recon_value_head: nn.Module | None = None
        self.value_head: nn.Module | None = None
        if config.recon_value_head or config.value_head:
            raise ValueError("objective.kind=mlm does not build value heads")
        if config.init_from:
            for note in load_encoder_weights(
                self.embed, self.encoder, config.init_from, config, "init_from"
            ):
                print(f"[note] model.init_from differs in {note}", flush=True)

    def forward(  # type: ignore[override]
        self,
        batch: Mapping[str, Tensor],
        input_code: Tensor | None = None,
        selected: Tensor | None = None,
        blank_value: Tensor | None = None,
    ) -> AROutput:
        valid = batch["attention_mask"].bool()
        if input_code is None or selected is None or blank_value is None:
            raise ValueError("EHRMLM.forward needs the mlm_inputs() triple")
        value_bin = torch.where(
            blank_value, torch.zeros_like(batch["value_bin"]), batch["value_bin"]
        )
        value_z = torch.where(blank_value, torch.zeros_like(batch["value_z"]), batch["value_z"])
        tokens = self.embed(input_code, value_bin, value_z, batch["age"], batch["log_delta"])
        valid = effective_valid(tokens, valid)
        encoded = self.encoder(tokens, valid)
        keep = selected.bool() & valid & (batch["code_id"] != PAD_ID)
        return AROutput(
            hidden=encoded.tokens[keep],
            targets=batch["code_id"][keep],
            tokens=encoded.tokens,
            cls=encoded.cls,
            valid_mask=valid,
            value_z=batch["value_z"][keep],
            value_bin=batch["value_bin"][keep],
        )
