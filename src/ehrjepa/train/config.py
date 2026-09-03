"""Typed configuration for pretraining: plain YAML in, dataclasses out.

No hydra. A config file is a mapping of five sections -- ``data``, ``model``,
``objective``, ``masking``, ``optim``, ``run`` -- each of which maps onto one
dataclass, and every key is checked against the dataclass fields, so a typo is an
error at load time rather than a silently ignored setting.

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
from ehrjepa.objectives.loss import ObjectiveConfig

__all__ = [
    "DataConfig",
    "MaskingConfig",
    "OptimConfig",
    "PretrainConfig",
    "RunConfig",
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
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def model_config(self, vocab_size: int) -> EHRJEPAConfig:
        """Resolve the model section, defaulting ``vocab_size`` to the cache's."""
        values = dict(self.model)
        values.setdefault("vocab_size", vocab_size)
        return EHRJEPAConfig.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": asdict(self.data),
            "model": dict(self.model),
            "objective": asdict(self.objective),
            "masking": asdict(self.masking),
            "optim": asdict(self.optim),
            "run": asdict(self.run),
        }


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "objective": ObjectiveConfig,
    "masking": MaskingConfig,
    "optim": OptimConfig,
    "run": RunConfig,
}


def _build_section(name: str, values: Mapping[str, Any]) -> Any:
    cls = _SECTIONS[name]
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in config section {name!r}: {sorted(unknown)}")
    return cls(**values)


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
