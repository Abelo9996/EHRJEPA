"""The pretraining objective: latent prediction plus SIGReg.

.. math::

    \\mathcal{L} = \\lambda_{\\text{pred}} \\mathcal{L}_{\\text{pred}}
      + \\lambda \\left( \\mathrm{SIGReg}(\\text{tokens})
                       + \\mathrm{SIGReg}(\\text{CLS}) \\right)

``L_pred`` is a smooth L1 between the predictor's outputs and the target
encoder's representations at the target positions, layer-normalised. The
LayerNorm is on the *target* side only and is what stops the trivial solution of
shrinking every target latent toward zero: the loss cannot be reduced by scaling
the targets down, only by predicting their direction and shape.

``lambda_pred`` defaults to ``1.0``. At ``0.0`` the term drops out of the total
entirely (not multiplied in -- the caller upstream never computed a real target
to multiply against, see :meth:`~ehrjepa.models.jepa.EHRJEPA.forward`'s
``compute_targets``), so ``lambda_pred: 0`` with ``lambda_recon`` positive gives
a pure masked-code-prediction objective through the same predictor. The
``pred_loss`` diagnostic is logged as ``NaN`` in that case rather than a number
computed against a placeholder target.

SIGReg is applied twice, at both granularities the model produces: to the valid
token outputs of the (gradient-carrying) context encoder pass, subsampled to at
most ``max_rows`` rows, and to the CLS embeddings. Both are needed -- an encoder
can produce perfectly isotropic token outputs while every subject embedding lands
in the same place.

``lambda_sigreg`` defaults to 0.05, the value LeJEPA reports as a robust default.

``lambda_recon`` adds a third term, off by default: cross-entropy from the
predictor's output at each target position to that event's ``code_id``, through
the tied code-embedding table (the AR next-code head, reused verbatim, applied
one step off its usual place). It is an auxiliary that forces the predicted
latent to carry code identity, which the first pilot grid's probes said the JEPA
encoders were discarding. ``recon_value`` adds the same for the 11-way
``value_bin``. Both are logged separately from the prediction loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ehrjepa.data.masking import DEFAULT_N_ANCHORS
from ehrjepa.models.jepa import JEPAOutput
from ehrjepa.objectives.ar import ar_loss_chunked
from ehrjepa.objectives.sigreg import DEFAULT_N_DIRECTIONS, SIGReg

__all__ = [
    "LATENT_KINDS",
    "OBJECTIVE_KINDS",
    "JEPAObjective",
    "ObjectiveConfig",
    "collapse_diagnostics",
    "jepa_loss",
]


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


OBJECTIVE_KINDS = ("jepa", "ar", "nextlatent", "window")

#: The two causal latent objectives, whose loss lives in
#: :mod:`ehrjepa.objectives.latent` and whose models live in
#: :mod:`ehrjepa.models.latent`. They share every field below except that
#: ``lambda_recon`` names a different auxiliary term for each.
LATENT_KINDS = ("nextlatent", "window")


@dataclass
class ObjectiveConfig:
    """Loss hyper-parameters.

    ``kind`` selects the pretraining objective: ``"jepa"`` (masked-span latent
    prediction plus SIGReg, everything below), ``"ar"`` (next-code cross-entropy,
    see :mod:`ehrjepa.objectives.ar`, which uses none of the other fields),
    ``"nextlatent"`` (dense causal next-latent prediction) or ``"window"``
    (pooled future-window latent prediction). The last two are described in
    :mod:`ehrjepa.models.latent`; they read the ``lambda_*`` and ``sigreg_*``
    fields plus their own ``horizons``/``window_horizons``/``window_anchors``,
    and ignore the masking section entirely.
    """

    kind: str = "jepa"
    #: AR only: positions per slice of the vocabulary projection. See
    #: :func:`ehrjepa.objectives.ar.ar_loss_chunked`.
    ar_chunk: int = 2048
    #: Weight on the latent smooth-L1 prediction term. ``0.0`` drops the term
    #: (not a zero-weighted multiply -- the model skips computing a real target
    #: to multiply against) and logs ``pred_loss`` as ``NaN``.
    lambda_pred: float = 1.0
    lambda_sigreg: float = 0.05
    smooth_l1_beta: float = 1.0
    sigreg_directions: int = DEFAULT_N_DIRECTIONS
    sigreg_max_rows: int = 8192
    sigreg_grid: int = 17
    sigreg_t_max: float = 5.0
    sigreg_sigma: float = 1.0
    sigreg_scale_by_n: bool = False
    #: Weight on the auxiliary code-reconstruction cross-entropy. ``0.0`` builds
    #: no head at all, so a run that leaves it alone is byte-identical.
    lambda_recon: float = 0.0
    #: Add an 11-way ``value_bin`` head alongside it, at the same weight.
    recon_value: bool = False

    #: ``nextlatent``: the step offsets predicted, one MLP head each. ``[1]`` is
    #: "the next event"; ``[1, 4, 16]`` adds two coarser look-aheads whose losses
    #: are averaged with it.
    horizons: list[int] = field(default_factory=lambda: [1])
    #: ``window``: the horizons in **days** whose events are pooled into one
    #: target latent per anchor.
    window_horizons: list[float] = field(default_factory=lambda: [30.0, 365.0])
    #: ``window``: anchors drawn per window (``K``).
    window_anchors: int = DEFAULT_N_ANCHORS

    def __post_init__(self) -> None:
        if self.kind not in OBJECTIVE_KINDS:
            raise ValueError(f"objective.kind must be one of {OBJECTIVE_KINDS}, got {self.kind!r}")
        # YAML and `--override objective.horizons=[1,4,16]` both hand these over
        # as lists already; a tuple from a hand-built config would survive to
        # `yaml.safe_dump` in the checkpoint and fail there instead of here.
        self.horizons = [int(k) for k in self.horizons]
        self.window_horizons = [float(h) for h in self.window_horizons]

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ObjectiveConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown objective config keys: {sorted(unknown)}")
        return cls(**values)  # type: ignore[arg-type]


class JEPAObjective(nn.Module):
    """Combine the prediction loss, the two SIGReg terms and the auxiliary heads.

    ``recon_head``/``recon_value_head`` are the model's own modules. They are kept
    in a plain list rather than assigned as submodules on purpose: registering
    them here would put the same parameters in two ``state_dict``\\ s and twice in
    the optimizer's weight-decay bookkeeping.
    """

    def __init__(
        self,
        config: ObjectiveConfig | None = None,
        recon_head: nn.Module | None = None,
        recon_value_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ObjectiveConfig()
        self._heads = [recon_head, recon_value_head]
        self.sigreg = SIGReg(
            n_directions=self.config.sigreg_directions,
            max_rows=self.config.sigreg_max_rows,
            n_grid=self.config.sigreg_grid,
            t_max=self.config.sigreg_t_max,
            sigma=self.config.sigreg_sigma,
            scale_by_n=self.config.sigreg_scale_by_n,
        )

    def forward(self, output: JEPAOutput, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return ``loss`` plus its components, all scalar tensors.

        ``valid_mask`` selects the token rows SIGReg sees; it defaults to the
        context mask, i.e. exactly the positions the encoder attended to.

        When ``lambda_pred`` is ``0``, ``output.targets`` is the zero placeholder
        :meth:`EHRJEPA.forward <ehrjepa.models.jepa.EHRJEPA.forward>` returns for
        ``compute_targets=False`` -- a real ``smooth_l1`` against it would be
        cheap to compute but meaningless, so it is skipped and ``pred_loss`` is
        reported as ``NaN`` instead.
        """
        cfg = self.config
        zero = output.predictions.new_zeros(())
        if cfg.lambda_pred == 0.0:
            pred_loss = output.predictions.new_full((), float("nan"))
            pred_term = zero
        else:
            pred_loss = jepa_loss(output.predictions, output.targets, beta=cfg.smooth_l1_beta)
            pred_term = cfg.lambda_pred * pred_loss

        mask = output.context_mask if valid_mask is None else valid_mask.bool()
        rows = output.context_tokens[mask]
        if cfg.lambda_sigreg == 0.0:
            sig_tokens, sig_cls = zero, zero
        else:
            sig_tokens = self.sigreg(rows.float())
            sig_cls = self.sigreg(output.cls.float())
        recon, recon_value = self.reconstruction(output)
        total = pred_term + cfg.lambda_sigreg * (sig_tokens + sig_cls)
        if cfg.lambda_recon != 0.0:
            total = total + cfg.lambda_recon * (recon + recon_value)
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "sigreg_tokens": sig_tokens.detach(),
            "sigreg_cls": sig_cls.detach(),
            "recon_loss": recon.detach(),
            "recon_value_loss": recon_value.detach(),
        }

    def reconstruction(self, output: JEPAOutput) -> tuple[Tensor, Tensor]:
        """``(code CE, value_bin CE)`` from the predicted latents, or two zeros.

        The code head is applied through :func:`~ehrjepa.objectives.ar.ar_loss_chunked`
        for the same reason the AR run is: a ``(n_targets, 30000)`` logit tensor
        and its gradient do not belong on a 16 GB laptop.
        """
        zero = output.predictions.new_zeros(())
        head, value_head = self._heads
        if self.config.lambda_recon == 0.0 or output.predictions.shape[0] == 0:
            return zero, zero
        recon, recon_value = zero, zero
        codes = output.extras.get("recon_code_id")
        if head is not None and codes is not None:
            recon = ar_loss_chunked(
                head, output.predictions, codes, chunk=self.config.ar_chunk, top_k=(1,)
            )["loss"]
        bins = output.extras.get("recon_value_bin")
        if value_head is not None and bins is not None:
            recon_value = F.cross_entropy(value_head(output.predictions).float(), bins)
        return recon, recon_value


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
