"""Typed configuration for pretraining: plain YAML in, dataclasses out.

No hydra. A config file is a mapping of sections -- ``data``, ``model``,
``predictor``, ``target``, ``objective``, ``masking``, ``optim``, ``train``,
``run`` -- each of which maps onto one dataclass, and every key is checked
against the dataclass fields, so a typo is an error at load time rather than a
silently ignored setting.

``predictor``, ``target`` and ``train`` are three small sections that name the
*objective's* shape rather than the network's, and each is folded into
:class:`~ehrjepa.models.jepa.EHRJEPAConfig` by :meth:`PretrainConfig.model_config`
so that a checkpoint records them and the probe rebuilds the same model. They are
where they are because "does the mask token carry time" is a question about the
prediction task, not about how wide the predictor is.

``--override key=value`` uses dotted paths (``optim.lr=1e-4``,
``model.depth=8``); values are parsed as YAML scalars, so ``true``, ``3``,
``1e-4`` and ``null`` all mean what they look like.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from ehrjepa.models.jepa import EHRJEPAConfig
from ehrjepa.objectives.loss import LATENT_KINDS, ObjectiveConfig

__all__ = [
    "DataConfig",
    "MaskingConfig",
    "OptimConfig",
    "PredictorConfig",
    "PretrainConfig",
    "RunConfig",
    "TargetConfig",
    "TrainConfig",
    "apply_overrides",
    "load_config",
]


@dataclass
class DataConfig:
    cache_dir: str = "data/cache/desynpuf-s1"
    split: str = "train"
    max_len: int = 512
    min_len: int = 16
    sampling: str = "random_window"
    num_workers: int = 0
    prefetch_factor: int | None = None


@dataclass
class MaskingConfig:
    p_future: float = 0.6
    future_min_span: int = 8
    future_max_span: int = 64
    future_cut_low: float = 0.3
    future_cut_high: float = 0.9
    block_min_blocks: int = 2
    block_max_blocks: int = 4
    block_block_low: float = 0.05
    block_block_high: float = 0.15
    block_context_drop: float = 0.3

    def kwargs(self) -> dict[str, Any]:
        """The ``future_*``/``block_*`` keywords ``sample_masks`` takes."""
        out = asdict(self)
        out.pop("p_future")
        return out


@dataclass
class PredictorConfig:
    """What the predictor's mask tokens are made of."""

    #: Mask tokens carry the target's ``age`` and ``log_delta``. With this off a
    #: mask token is the learned ``MASK`` embedding plus RoPE at the target index
    #: and nothing else, so no time-conditional prior can stand in for the
    #: prediction.
    mask_token_time: bool = True


@dataclass
class TargetConfig:
    """What the target encoder is shown."""

    #: The target encoder's input carries the time terms. Off, targets are
    #: content -- code and value only.
    time_features: bool = True
    #: Run the target encoder on the target span alone rather than on the full
    #: window, so a target latent cannot absorb the context.
    span_only: bool = False


@dataclass
class TrainConfig:
    """Input augmentation applied on the online (gradient-carrying) pass."""

    #: Per-token probability that both time terms are dropped from the online
    #: encoder's input, so the shared or EMA-copied weights have seen inputs
    #: shaped like the content-only ones ``target.time_features: false`` makes.
    time_feature_dropout: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 100
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    accum_steps: int = 1


@dataclass
class RunConfig:
    steps: int = 2000
    batch_size: int = 32
    seed: int = 20240917
    device: str = "auto"
    precision: str = "auto"
    out_dir: str = "runs/pretrain"
    log_every: int = 10
    ckpt_every: int = 500
    diagnostics_every: int = 0  # 0 means "whenever we log"
    tensorboard: bool = True
    max_seconds: float = 0.0  # 0 means no wall-clock limit
    resume: str | None = None
    compile: bool = False


@dataclass
class PretrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: dict[str, Any] = field(default_factory=dict)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def model_config(self, vocab_size: int) -> EHRJEPAConfig:
        """Resolve the model section, defaulting ``vocab_size`` to the cache's.

        The ``predictor``/``target``/``train`` sections and the two
        reconstruction knobs are folded in here and win over anything of the same
        name written under ``model``: there is one place to set each of them, and
        it is the section named after the thing it changes.
        """
        values = coerce_section(EHRJEPAConfig, self.model, "model")
        values.setdefault("vocab_size", vocab_size)
        values["mask_token_time"] = self.predictor.mask_token_time
        values["target_time_features"] = self.target.time_features
        values["target_span_only"] = self.target.span_only
        values["time_feature_dropout"] = self.train.time_feature_dropout
        values["recon_head"] = self.objective.lambda_recon != 0.0
        values["recon_value_head"] = values["recon_head"] and self.objective.recon_value
        # The causal latent objectives replace the transformer predictor with
        # their own MLP heads, and carry the horizon lists the heads are shaped
        # from -- into the *model* config, so a checkpoint rebuilds them without
        # the probe having to read the objective section.
        latent = self.objective.kind in LATENT_KINDS
        values["build_predictor"] = not latent
        values["horizons"] = list(self.objective.horizons)
        values["window_horizons"] = list(self.objective.window_horizons)
        return EHRJEPAConfig.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": asdict(self.data),
            "model": dict(self.model),
            "predictor": asdict(self.predictor),
            "target": asdict(self.target),
            "objective": asdict(self.objective),
            "masking": asdict(self.masking),
            "optim": asdict(self.optim),
            "train": asdict(self.train),
            "run": asdict(self.run),
        }


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "predictor": PredictorConfig,
    "target": TargetConfig,
    "objective": ObjectiveConfig,
    "masking": MaskingConfig,
    "optim": OptimConfig,
    "train": TrainConfig,
    "run": RunConfig,
}


_SCALARS: tuple[type, ...] = (bool, int, float, str)


def coerce_section(cls: type, values: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Check keys against ``cls``'s fields and coerce scalars to their annotated type.

    Worth the trouble because the common way to get this wrong is a shell quoting
    slip -- ``--override "run.steps=900 run.log_every=25"`` as one argument makes
    ``steps`` the *string* ``"900 run.log_every=25"``, which a plain dataclass
    accepts happily and which then fails a thousand lines later inside the loop.
    Here it fails at load time, naming the key.
    """
    annotations = {f.name: f.type for f in fields(cls)}
    unknown = set(values) - set(annotations)
    if unknown:
        raise ValueError(f"unknown keys in config section {where!r}: {sorted(unknown)}")
    out: dict[str, Any] = {}
    for key, value in values.items():
        expected = annotations[key]
        target = {"int": int, "float": float, "bool": bool, "str": str}.get(
            expected if isinstance(expected, str) else getattr(expected, "__name__", "")
        )
        if target is None or isinstance(value, target) or value is None:
            out[key] = value
            continue
        if not isinstance(value, _SCALARS):
            raise ValueError(f"{where}.{key} expects {target.__name__}, got {type(value).__name__}")
        try:
            out[key] = target(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.{key} expects {target.__name__}, got {value!r}") from exc
    return out


def _build_section(name: str, values: Mapping[str, Any]) -> Any:
    cls = _SECTIONS[name]
    return cls(**coerce_section(cls, values, name))


def from_mapping(raw: Mapping[str, Any]) -> PretrainConfig:
    """Build a :class:`PretrainConfig` from a plain nested mapping."""
    unknown = set(raw) - set(_SECTIONS) - {"model"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sections = {name: _build_section(name, raw.get(name) or {}) for name in _SECTIONS}
    config = PretrainConfig(model=dict(raw.get("model") or {}), **sections)
    if config.optim.betas is not None:
        config.optim.betas = tuple(config.optim.betas)  # type: ignore[assignment]
    return config


def parse_scalar(text: str) -> Any:
    """Parse an override's right-hand side.

    YAML 1.1 -- which PyYAML implements -- does not recognise ``1e-4`` as a float
    (it wants ``1.0e-4``), and typing ``optim.lr=1.0e-4`` on a command line is a
    trap nobody remembers. So: YAML first, then, if that produced a bare string,
    try ``int`` and ``float`` before giving up and keeping the string.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
    return value


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` strings to a nested mapping, in place, parsing YAML scalars."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        path, _, text = item.partition("=")
        keys = path.strip().split(".")
        node = raw
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[keys[-1]] = parse_scalar(text)
    return raw


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> PretrainConfig:
    """Read a YAML config, apply ``--override`` strings, and type-check every key."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return from_mapping(apply_overrides(raw, list(overrides)))


def config_to_yaml(config: PretrainConfig) -> str:
    payload = config.to_dict()
    for section in payload.values():
        if is_dataclass(section):  # pragma: no cover - defensive
            raise TypeError("to_dict must return plain mappings")
    return yaml.safe_dump(payload, sort_keys=False)
