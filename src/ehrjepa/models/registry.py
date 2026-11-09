"""``objective.kind`` -> model class, for the trainer and :func:`ehrjepa.eval.probe.load_encoder`.

One table instead of one per caller, so a new objective is loadable by the probe
and the fine-tune harness the moment the trainer can build it. ``jepa`` (and
checkpoints that predate the ``kind`` field) fall through to
:class:`~ehrjepa.models.jepa.EHRJEPA`.
"""

from __future__ import annotations

from torch import nn

from ehrjepa.models.ar import EHRAR
from ehrjepa.models.jepa import EHRJEPA
from ehrjepa.models.latent import LATENT_MODELS
from ehrjepa.models.mlm import EHRMLM
from ehrjepa.models.state import EHRStateModel

__all__ = ["MODEL_CLASSES", "model_class"]

MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "ar": EHRAR,
    "mlm": EHRMLM,
    **LATENT_MODELS,
    "state": EHRStateModel,
}


def model_class(kind: str) -> type[nn.Module]:
    return MODEL_CLASSES.get(kind, EHRJEPA)
