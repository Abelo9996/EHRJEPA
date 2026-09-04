"""The loss for the two causal latent objectives.

Same three terms as :class:`~ehrjepa.objectives.loss.JEPAObjective` -- a smooth
L1 latent prediction loss, SIGReg on the encoder's token outputs and CLS row, and
an optional auxiliary supervised term -- with two differences that follow from
what the models in :mod:`ehrjepa.models.latent` produce.

**The prediction loss is a mean over horizons, not over rows.** ``horizons: [1,
4, 16]`` scores ``L - 1``, ``L - 4`` and ``L - 16`` positions per window, so a
single ``smooth_l1`` over the concatenation would silently weight horizon 1 most
and horizon 16 least. Each horizon's loss is computed on its own slice and the
slices are averaged, so the reported number is the mean of three comparable
quantities and adding a longer horizon does not change what the shorter ones
contribute.

**The auxiliary term is a different term per objective.**

``nextlatent``
    ``lambda_recon`` is the *AR loss*: next-code cross-entropy from the encoder's
    own outputs through the tied code head, chunked exactly as
    :mod:`ehrjepa.objectives.ar` chunks it. So ``lambda_pred: 0`` with
    ``lambda_recon: 1`` and ``lambda_sigreg: 0`` reduces this objective to
    next-code AR, which is the thing the pilot grids say works, and any value in
    between is a hybrid. ``ce``/``top1``/``top10`` are logged from it, into the
    same columns the ``ar`` runs use.

``window``
    ``lambda_recon`` is a multi-label BCE (TransformEHR-style): from the
    *predicted pooled latent*, through the same tied code head, to the multi-hot
    set of codes occurring inside that anchor's horizon. Reduction is the
    PyTorch default -- the mean over rows *and* classes -- so the term starts at
    ``log 2`` and is dominated by the 30,000 absent codes rather than by the few
    dozen present ones. That is the standard formulation and the one
    ``lambda_recon: 0.1`` was chosen against; a ``pos_weight`` would be a
    different experiment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ehrjepa.data.tokenize import PAD_ID
from ehrjepa.models.jepa import JEPAOutput
from ehrjepa.objectives.ar import ar_loss_chunked
from ehrjepa.objectives.loss import ObjectiveConfig, jepa_loss
from ehrjepa.objectives.sigreg import SIGReg

__all__ = ["BCE_CHUNK", "LatentObjective", "multilabel_bce_chunked"]

#: Rows per slice of the multi-label head. The dense ``(chunk, vocab)`` target is
#: materialised alongside the logits and their gradient, so this is a third of
#: the AR chunk: 512 x 30,000 is 61 MB per tensor in float32.
BCE_CHUNK = 512


def multilabel_bce_chunked(
    head: nn.Module,
    hidden: Tensor,
    codes: Tensor,
    chunk: int = BCE_CHUNK,
) -> Tensor:
    """Mean binary cross-entropy over the multi-hot code set of each row.

    ``hidden`` is ``(N, dim)`` and ``codes`` is ``(N, L)``: the code ids inside
    that row's horizon, with :data:`~ehrjepa.data.tokenize.PAD_ID` everywhere
    else. The multi-hot target is scattered per slice rather than built once, so
    peak memory is ``chunk x vocab`` instead of ``N x vocab``; duplicates within a
    row scatter to the same column and a repeated code is one positive.
    """
    n = int(hidden.shape[0])
    if n == 0:
        return hidden.new_zeros(())
    vocab = int(head.vocab_size)
    total = hidden.new_zeros(())
    for start in range(0, n, max(1, chunk)):
        rows = hidden[start : start + chunk]
        picks = codes[start : start + chunk]
        if torch.is_grad_enabled() and rows.requires_grad:
            logits = checkpoint(head, rows, use_reentrant=False).float()
        else:
            logits = head(rows).float()
        target = torch.zeros_like(logits)
        target.scatter_(1, picks, 1.0)
        target[:, PAD_ID] = 0.0
        total = total + F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
    return total / (n * vocab)


class LatentObjective(nn.Module):
    """Prediction loss + SIGReg + the per-objective auxiliary term.

    ``recon_head`` is the model's own module, held in a list for the same reason
    :class:`~ehrjepa.objectives.loss.JEPAObjective` holds its heads that way: a
    registered submodule would put one tensor in two ``state_dict``\\ s and twice
    in the optimizer's weight-decay bookkeeping.
    """

    def __init__(self, config: ObjectiveConfig, recon_head: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self._heads = [recon_head]
        self.sigreg = SIGReg(
            n_directions=config.sigreg_directions,
            max_rows=config.sigreg_max_rows,
            n_grid=config.sigreg_grid,
            t_max=config.sigreg_t_max,
            sigma=config.sigreg_sigma,
            scale_by_n=config.sigreg_scale_by_n,
        )

    # ------------------------------------------------------------------ #

    def prediction_loss(self, output: JEPAOutput) -> Tensor:
        """Mean over horizons of each horizon's smooth L1 against its normed targets."""
        sizes = [int(n) for n in output.extras["horizon_sizes"].tolist()]
        losses = []
        start = 0
        for size in sizes:
            stop = start + size
            if size > 0:
                losses.append(
                    jepa_loss(
                        output.predictions[start:stop],
                        output.targets[start:stop],
                        beta=self.config.smooth_l1_beta,
                    )
                )
            start = stop
        if not losses:
            return output.predictions.new_zeros(())
        return torch.stack(losses).mean()

    def reconstruction(self, output: JEPAOutput) -> tuple[Tensor, dict[str, Tensor]]:
        """``(loss, extra scalars to log)`` for whichever auxiliary term applies."""
        zero = output.predictions.new_zeros(())
        head = self._heads[0]
        if self.config.lambda_recon == 0.0 or head is None:
            return zero, {}
        if self.config.kind == "nextlatent":
            hidden = output.extras.get("recon_hidden")
            codes = output.extras.get("recon_code_id")
            if hidden is None or codes is None:
                return zero, {}
            stats = ar_loss_chunked(head, hidden, codes, chunk=self.config.ar_chunk)
            return stats["loss"], {
                "ce": stats["ce"],
                "top1": stats["top1"],
                "top10": stats["top10"],
            }
        codes = output.extras.get("window_codes")
        if codes is None:
            return zero, {}
        return multilabel_bce_chunked(head, output.predictions, codes), {}

    def forward(self, output: JEPAOutput, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return ``loss`` plus every component and diagnostic, all scalar tensors."""
        cfg = self.config
        zero = output.predictions.new_zeros(())
        if cfg.lambda_pred == 0.0:
            pred_loss = output.predictions.new_full((), float("nan"))
            pred_term = zero
        else:
            pred_loss = self.prediction_loss(output)
            pred_term = cfg.lambda_pred * pred_loss

        mask = output.context_mask if valid_mask is None else valid_mask.bool()
        if cfg.lambda_sigreg == 0.0:
            sig_tokens, sig_cls = zero, zero
        else:
            sig_tokens = self.sigreg(output.context_tokens[mask].float())
            sig_cls = self.sigreg(output.cls.float())

        recon, extra = self.reconstruction(output)
        total = pred_term + cfg.lambda_sigreg * (sig_tokens + sig_cls)
        if cfg.lambda_recon != 0.0:
            total = total + cfg.lambda_recon * recon
        losses = {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "sigreg_tokens": sig_tokens.detach(),
            "sigreg_cls": sig_cls.detach(),
            "recon_loss": recon.detach(),
            "recon_value_loss": zero,
            **{name: value.detach() for name, value in extra.items()},
        }
        losses.update(_anchor_diagnostics(output))
        return losses


def _anchor_diagnostics(output: JEPAOutput) -> dict[str, Tensor]:
    """``skipped_frac`` and ``positives_per_anchor``, when the model reported them.

    ``skipped_frac`` is over anchor/horizon *pairs offered* -- an anchor counts
    once per horizon, because a 30-day pool can be observed inside the window
    when the 365-day one is not.
    """
    out: dict[str, Tensor] = {}
    offered = output.extras.get("anchors_offered")
    if offered is not None:
        skipped = output.extras["anchors_unobserved"] + output.extras["anchors_empty"]
        out["skipped_frac"] = skipped / offered.clamp(min=1.0)
    positives = output.extras.get("positives_per_anchor")
    if positives is not None:
        out["positives_per_anchor"] = positives
    return out
