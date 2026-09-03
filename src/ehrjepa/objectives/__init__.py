"""Training objectives: latent prediction loss and anti-collapse regularization.

This subpackage will hold the loss terms. The predictive term scores predicted
target-span latents against the EMA target encoder's outputs over the masked
events, reduced only over valid (non-padding) target positions. The
regularization term is SIGReg-style, following LeJEPA (arXiv:2511.08544): rather
than penalizing feature covariance directly, it pushes the embedding
distribution toward an isotropic Gaussian by testing random one-dimensional
projections of the batch against the corresponding univariate normal, which
gives a collapse guarantee with a single trade-off coefficient and no need for
stop-gradient tricks, predictor tuning, or large-batch negatives. Planned
contents: the projection sampler, the per-projection goodness-of-fit statistic,
the combined loss module exposing its components for logging, and diagnostic
utilities (embedding rank, per-dimension variance) used to detect collapse
during training.
"""

from ehrjepa.data.masking import future_span_mask, multi_block_mask, sample_masks
from ehrjepa.objectives.ar import (
    ARObjective,
    ARStats,
    NextCodeHead,
    ar_loss,
    ar_loss_chunked,
    next_code_targets,
)
from ehrjepa.objectives.loss import (
    OBJECTIVE_KINDS,
    JEPAObjective,
    ObjectiveConfig,
    collapse_diagnostics,
    jepa_loss,
)
from ehrjepa.objectives.sigreg import SIGReg, epps_pulley, random_directions, sigreg

__all__ = [
    "OBJECTIVE_KINDS",
    "ARObjective",
    "ARStats",
    "JEPAObjective",
    "ObjectiveConfig",
    "NextCodeHead",
    "SIGReg",
    "ar_loss",
    "ar_loss_chunked",
    "collapse_diagnostics",
    "epps_pulley",
    "future_span_mask",
    "jepa_loss",
    "multi_block_mask",
    "next_code_targets",
    "random_directions",
    "sample_masks",
    "sigreg",
]
