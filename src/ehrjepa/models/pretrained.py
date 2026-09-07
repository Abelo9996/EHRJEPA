"""Copy an *existing* run's embedding and encoder into a new model.

Two config keys use this, and they differ only in which stack they write into:

``model.target_init``
    With ``model.target_mode: frozen``, the target encoder is a copy of a
    finished run's embedding+encoder that is never updated -- a fixed teacher
    rather than a shared or EMA-tracked one. Loaded once at construction,
    ``requires_grad_(False)`` for the life of the run.
``model.init_from``
    The *online* stack starts from a finished run's weights instead of from
    scratch. Orthogonal to the target side: a run may do either, both (from the
    same checkpoint, which starts student and teacher identical), or neither.

Only ``embed.*`` and ``encoder.*`` are read. The heads -- a predictor, a tied
next-code head, one MLP per horizon -- belong to the objective that trained
them, and an ``ar`` checkpoint feeding a ``nextlatent`` run has none the new
model wants. Everything else in the payload (optimizer state, RNG, step) is
ignored: this is initialization, not a resume.

**Architecture is checked before anything is copied.** ``load_state_dict`` would
catch a width mismatch too, but it reports it as a wall of shape errors on
individual parameters; :func:`check_architecture` reports it as "dim 384 does
not match this model's 256", naming the file. The fields compared are exactly
those that determine the parameter shapes; ``causal`` is *not* one of them (it
is an attention-mask flag, and a bidirectional model taking a causal
checkpoint's weights as a frozen teacher is a legitimate, if unusual, thing to
ask for), so it is reported as a note rather than an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, fine for typing
    from ehrjepa.models.jepa import EHRJEPAConfig

__all__ = ["ARCHITECTURE_FIELDS", "check_architecture", "load_encoder_weights", "read_model_config"]

#: The config fields that determine the shape of ``embed.*`` + ``encoder.*``.
#: ``mlp``/``mlp_ratio``/``n_freq`` are in here because they size a matrix;
#: ``causal``, ``dropout`` and everything predictor-side are not.
ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "vocab_size",
    "dim",
    "depth",
    "heads",
    "mlp",
    "mlp_ratio",
    "n_freq",
)


def read_model_config(path: str | Path) -> dict[str, Any]:
    """The ``model_config`` mapping a training checkpoint records, or an error."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no checkpoint at {file}")
    payload = torch.load(file, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_config" not in payload:
        raise ValueError(f"{file} is not an ehrjepa training checkpoint (no 'model_config')")
    return dict(payload["model_config"])


def check_architecture(
    config: EHRJEPAConfig, other: Mapping[str, Any], path: str | Path, key: str
) -> list[str]:
    """Raise unless ``other`` names the same encoder shape as ``config``.

    Returns the list of *non-fatal* differences (currently just ``causal``) so
    the caller can print them; raises :class:`ValueError` naming ``key`` and
    ``path`` on the first fatal one.
    """
    bad = [
        f"{field}={other.get(field)!r} (this model: {getattr(config, field)!r})"
        for field in ARCHITECTURE_FIELDS
        if field in other and other[field] != getattr(config, field)
    ]
    if bad:
        raise ValueError(
            f"model.{key}={path} was trained with a different architecture: "
            + ", ".join(bad)
            + ". The embedding and encoder weights cannot be copied into this model."
        )
    missing = [f for f in ARCHITECTURE_FIELDS if f not in other]
    if missing:
        raise ValueError(
            f"model.{key}={path} records no {sorted(missing)} in its model_config, "
            "so its architecture cannot be checked against this one"
        )
    notes = []
    if "causal" in other and bool(other["causal"]) != bool(config.causal):
        notes.append(f"causal={other['causal']} (this model: {config.causal})")
    return notes


def load_encoder_weights(
    embed: nn.Module,
    encoder: nn.Module,
    path: str | Path,
    config: EHRJEPAConfig,
    key: str,
) -> list[str]:
    """Copy ``embed.*`` and ``encoder.*`` out of the checkpoint at ``path``.

    ``key`` is the config field being honoured (``target_init`` or
    ``init_from``) and appears in every error message, because "which of the two
    checkpoints is the wrong shape" is the only question a failure here raises.
    Returns the non-fatal architecture notes from :func:`check_architecture`.
    """
    file = Path(path)
    notes = check_architecture(config, read_model_config(file), file, key)
    state = torch.load(file, map_location="cpu", weights_only=False)["model"]
    for name, module in (("embed", embed), ("encoder", encoder)):
        prefix = f"{name}."
        part = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not part:
            raise ValueError(f"model.{key}={file} has no {prefix}* weights to copy")
        module.load_state_dict(part)
    return notes
