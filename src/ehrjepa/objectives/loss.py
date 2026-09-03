"""The pretraining objective: latent prediction plus SIGReg.

.. math::

    \\mathcal{L} = \\mathcal{L}_{\\text{pred}}
      + \\lambda \\left( \\mathrm{SIGReg}(\\text{tokens})
                       + \\mathrm{SIGReg}(\\text{CLS}) \\right)

``L_pred`` is a smooth L1 between the predictor's outputs and the target
encoder's representations at the target positions, layer-normalised. The
LayerNorm is on the *target* side only and is what stops the trivial solution of
shrinking every target latent toward zero: the loss cannot be reduced by scaling
the targets down, only by predicting their direction and shape.

SIGReg is applied twice, at both granularities the model produces: to the valid
token outputs of the (gradient-carrying) context encoder pass, subsampled to at
most ``max_rows`` rows, and to the CLS embeddings. Both are needed -- an encoder
can produce perfectly isotropic token outputs while every subject embedding lands
in the same place.

``lambda_sigreg`` defaults to 0.05, the value LeJEPA reports as a robust default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ehrjepa.models.jepa import JEPAOutput
from ehrjepa.objectives.sigreg import DEFAULT_N_DIRECTIONS, SIGReg

__all__ = ["JEPAObjective", "ObjectiveConfig", "collapse_diagnostics", "jepa_loss"]


def jepa_loss(predictions: Tensor, targets: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth L1 between predictions and layer-normed targets, averaged over targets.

    Both tensors are ``(n_targets, dim)``. ``targets`` is expected to be detached
    already; it is layer-normalised here (no learned affine) so the scale of the
    target representation cannot be gamed.
    """
    if predictions.shape[0] == 0:
        return predictions.new_zeros(())
    normed = F.layer_norm(targets.float(), (targets.shape[-1],))
    return F.smooth_l1_loss(predictions.float(), normed, beta=beta, reduction="mean")


OBJECTIVE_KINDS = ("jepa", "ar")


@dataclass
class ObjectiveConfig:
    """Loss hyper-parameters.

    ``kind`` selects the pretraining objective: ``"jepa"`` (latent prediction plus
    SIGReg, everything below) or ``"ar"`` (next-code cross-entropy, see
    :mod:`ehrjepa.objectives.ar`, which uses none of the other fields).
    """

    kind: str = "jepa"
    #: AR only: positions per slice of the vocabulary projection. See
    #: :func:`ehrjepa.objectives.ar.ar_loss_chunked`.
    ar_chunk: int = 2048
    lambda_sigreg: float = 0.05
    smooth_l1_beta: float = 1.0
    sigreg_directions: int = DEFAULT_N_DIRECTIONS
    sigreg_max_rows: int = 8192
    sigreg_grid: int = 17
    sigreg_t_max: float = 5.0
    sigreg_sigma: float = 1.0
    sigreg_scale_by_n: bool = False

    def __post_init__(self) -> None:
        if self.kind not in OBJECTIVE_KINDS:
            raise ValueError(f"objective.kind must be one of {OBJECTIVE_KINDS}, got {self.kind!r}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ObjectiveConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown objective config keys: {sorted(unknown)}")
        return cls(**values)  # type: ignore[arg-type]


class JEPAObjective(nn.Module):
    """Combine the prediction loss and the two SIGReg terms."""

    def __init__(self, config: ObjectiveConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectiveConfig()
        self.sigreg = SIGReg(
            n_directions=self.config.sigreg_directions,
            max_rows=self.config.sigreg_max_rows,
            n_grid=self.config.sigreg_grid,
            t_max=self.config.sigreg_t_max,
            sigma=self.config.sigreg_sigma,
            scale_by_n=self.config.sigreg_scale_by_n,
        )

    def forward(self, output: JEPAOutput, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return ``loss`` plus its three components, all scalar tensors.

        ``valid_mask`` selects the token rows SIGReg sees; it defaults to the
        context mask, i.e. exactly the positions the encoder attended to.
        """
        cfg = self.config
        pred_loss = jepa_loss(output.predictions, output.targets, beta=cfg.smooth_l1_beta)

        mask = output.context_mask if valid_mask is None else valid_mask.bool()
        rows = output.context_tokens[mask]
        zero = pred_loss.new_zeros(())
        if cfg.lambda_sigreg == 0.0:
            sig_tokens, sig_cls = zero, zero
        else:
            sig_tokens = self.sigreg(rows.float())
            sig_cls = self.sigreg(output.cls.float())
        total = pred_loss + cfg.lambda_sigreg * (sig_tokens + sig_cls)
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "sigreg_tokens": sig_tokens.detach(),
            "sigreg_cls": sig_cls.detach(),
        }


@torch.no_grad()
def collapse_diagnostics(
    tokens: Tensor,
    mask: Tensor,
    predictions: Tensor,
    targets: Tensor,
    max_rows: int = 2048,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Cheap per-step evidence about whether the representation is collapsing.

    ``mean_std``
        Mean over dimensions of the per-dimension standard deviation of the valid
        token outputs. Goes to zero under total collapse.
    ``effective_rank``
        ``exp`` of the entropy of the normalised singular values of a
        ``max_rows``-row subsample -- the "how many directions are actually in
        use" number. 1.0 means rank-one.
    ``cos_real`` / ``cos_shuffled`` / ``cos_gap``
        Mean cosine between each prediction and its own target, versus the mean
        cosine between predictions and a rolled (mismatched) target. The *gap* is
        the signal: a predictor that has learned nothing but the batch mean scores
        the same on both.
    """
    out: dict[str, float] = {}
    rows = tokens[mask.bool()]
    if rows.shape[0] == 0:
        return {
            "mean_std": 0.0,
            "effective_rank": 0.0,
            "cos_real": 0.0,
            "cos_shuffled": 0.0,
            "cos_gap": 0.0,
        }
    rows = rows.float()
    out["mean_std"] = float(rows.std(dim=0).mean())

    # The subsample is drawn on the CPU from an optional dedicated generator, so
    # logging never perturbs the RNG stream the training step itself depends on.
    pick = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
    sample = rows[pick.to(rows.device)]
    sample = sample - sample.mean(dim=0, keepdim=True)
    try:
        sv = torch.linalg.svdvals(sample.cpu())
    except RuntimeError:  # pragma: no cover - LAPACK convergence
        sv = torch.zeros(1)
    total = sv.sum()
    if float(total) <= 0:
        out["effective_rank"] = 0.0
    else:
        p = sv / total
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
        out["effective_rank"] = float(torch.exp(entropy))

    if predictions.shape[0] >= 2:
        p = F.normalize(predictions.float(), dim=-1)
        t = F.normalize(F.layer_norm(targets.float(), (targets.shape[-1],)), dim=-1)
        out["cos_real"] = float((p * t).sum(-1).mean())
        out["cos_shuffled"] = float((p * t.roll(1, dims=0)).sum(-1).mean())
    else:
        out["cos_real"] = 0.0
        out["cos_shuffled"] = 0.0
    out["cos_gap"] = out["cos_real"] - out["cos_shuffled"]
    return out
