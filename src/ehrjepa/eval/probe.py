"""Frozen-encoder embeddings and the linear probes over them.

The encoder is run exactly the way inference should run it: ``eval()`` mode, no
masking, float32, ``no_grad``, over the last ``max_len`` events **strictly
before** the anchor. Nothing is fine-tuned -- a probe is a logistic regression on
frozen features, so a difference between two checkpoints is a difference in the
representation and not in how much supervised capacity was bolted on.

Each anchor yields ``concat(cls, mean of token outputs)``: the CLS row is what
the pretraining objective regularises, the masked mean is the cheap pooled
alternative, and concatenating costs one extra ``dim`` of probe parameters
rather than a decision about which one to trust.

``random_init`` builds the same architecture from the checkpoint's own model
config and leaves the weights at their initialisation. It is the control: a
pretrained checkpoint that does not beat its own untrained twin has not learned
anything a probe can use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ehrjepa.data.dataset import collate_events
from ehrjepa.eval.baselines import LOGISTIC_GRID, FittedModel, fit_logistic
from ehrjepa.eval.history import HistoryReader, anchor_minutes
from ehrjepa.models.jepa import EHRJEPA, EHRJEPAConfig

__all__ = ["embed", "embed_cached", "embedding_path", "few_shot", "fit_probe", "load_encoder"]

log = logging.getLogger(__name__)

#: Few-shot sizes: ``k`` positives and ``k`` negatives, or the whole train split.
FEW_SHOT_K: tuple[int | None, ...] = (32, 128, 512, None)

#: Seeds for the few-shot subsample.
FEW_SHOT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def load_encoder(
    checkpoint: str | Path | None,
    *,
    vocab_size: int | None = None,
    random_init: bool = False,
    seed: int = 0,
) -> tuple[EHRJEPA, int]:
    """``(model, max_len)`` from a training checkpoint.

    ``random_init=True`` keeps the checkpoint's architecture and discards its
    weights, which is the control arm. ``checkpoint=None`` requires
    ``vocab_size`` and builds a default-shaped model, which only the tests use.
    """
    if checkpoint is None:
        if vocab_size is None:
            raise ValueError("vocab_size is required when no checkpoint is given")
        config, max_len = EHRJEPAConfig(vocab_size=vocab_size), 512
        state = None
    else:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        config = EHRJEPAConfig.from_mapping(payload["model_config"])
        max_len = int(payload["config"]["data"]["max_len"])
        state = payload["model"]
    torch.manual_seed(seed)
    model = EHRJEPA(config)
    if state is not None and not random_init:
        model.load_state_dict(state)
    return model.eval().float(), max_len


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this machine
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def embed(
    checkpoint: str | Path | None,
    cache_dir: str | Path,
    anchors: pl.DataFrame,
    *,
    random_init: bool = False,
    batch_size: int = 64,
    device: str | None = None,
    seed: int = 0,
    vocab_size: int | None = None,
) -> np.ndarray:
    """``(n_anchors, 2 * dim)`` embeddings, in ``anchors`` row order.

    ``anchors`` needs ``subject_id``, ``anchor_time`` and ``split``.
    """
    model, max_len = load_encoder(
        checkpoint, vocab_size=vocab_size, random_init=random_init, seed=seed
    )
    dev = _device(device)
    model = model.to(dev)
    reader = HistoryReader(cache_dir, max_len=max_len)

    minutes = anchor_minutes(anchors["anchor_time"])
    subjects = anchors["subject_id"].to_numpy()
    splits = anchors["split"].to_list()
    out = np.zeros((anchors.height, 2 * model.config.dim), dtype=np.float32)

    for start in range(0, anchors.height, batch_size):
        stop = min(start + batch_size, anchors.height)
        items = []
        for i in range(start, stop):
            window = reader.dataset(splits[i]).windows_at(int(subjects[i]), int(minutes[i]))
            items.append(window)
        batch = collate_events(items)
        batch = {k: v.to(dev) for k, v in batch.items()}
        tokens = model.embed_batch(batch)
        encoded = model.encoder(tokens, batch["attention_mask"])
        mask = batch["attention_mask"].to(encoded.tokens.dtype).unsqueeze(-1)
        pooled = (encoded.tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        rows = torch.cat([encoded.cls, pooled], dim=-1).float().cpu().numpy()
        out[start:stop] = rows
        if start and start % (batch_size * 50) == 0:
            log.info("embedded %d/%d", start, anchors.height)
    return out


def embedding_path(cache_root: Path | str, source: str, model_name: str) -> Path:
    """Where embeddings are memoised: one parquet per (source, model).

    Keyed on ``(subject_id, anchor_time)`` rather than on a task, because the
    default anchor rule gives every task except ``readmission_30d`` the same
    anchors -- the ``new_dx`` tasks are subsets of them -- so a per-task cache
    would re-run the encoder over the same windows five times.
    """
    safe = model_name.replace("/", "__").replace(":", "-")
    return Path(cache_root) / source / f"emb__{safe}.parquet"


_KEY = ("subject_id", "anchor_time")


def embed_cached(
    checkpoint: str | Path | None,
    cache_dir: str | Path,
    anchors: pl.DataFrame,
    path: Path | str | None,
    **kwargs,
) -> np.ndarray:
    """:func:`embed`, but computing only the ``(subject_id, anchor_time)`` pairs missing."""
    key = anchors.select(*_KEY, "split")
    stored = pl.read_parquet(path) if path and Path(path).exists() else None
    if stored is None:
        stored = key.clear().with_columns(embedding=pl.Series([], dtype=pl.List(pl.Float32)))
    missing = key.join(stored.select(*_KEY), on=list(_KEY), how="anti")
    if missing.height:
        log.info("embedding %d new anchors (%d cached)", missing.height, stored.height)
        matrix = embed(checkpoint, cache_dir, missing, **kwargs)
        fresh = missing.with_columns(
            embedding=pl.Series(matrix.tolist(), dtype=pl.List(pl.Float32))
        )
        stored = pl.concat([stored, fresh], how="vertical_relaxed")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            stored.write_parquet(path)
    joined = key.join(
        stored.select(*_KEY, "embedding"), on=list(_KEY), how="left", maintain_order="left"
    )
    if joined.height != anchors.height or joined["embedding"].null_count():
        raise RuntimeError("embedding cache lookup did not cover every anchor")
    return np.asarray(joined["embedding"].to_list(), dtype=np.float32)


# --------------------------------------------------------------------------- #


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    grid: Sequence[float] = LOGISTIC_GRID,
    seed: int = 0,
    name: str = "probe",
) -> FittedModel:
    """Logistic regression on frozen embeddings, ``C`` tuned on ``tuning``."""
    return fit_logistic(
        x_train, y_train, x_tune, y_tune, grid=grid, seed=seed, name=name, scale=True
    )


def subsample(y: np.ndarray, k: int | None, seed: int) -> np.ndarray:
    """Row indices for ``k`` positives and ``k`` negatives, or everything.

    When a class has fewer than ``k`` rows, all of them are taken -- the draw is
    balanced where the data allows and the returned size says when it did not.
    """
    if k is None:
        return np.arange(y.size)
    rng = np.random.default_rng(seed)
    parts = []
    for label in (0, 1):
        rows = np.flatnonzero(y == label)
        take = min(k, rows.size)
        parts.append(rng.choice(rows, size=take, replace=False))
    return np.sort(np.concatenate(parts))


def few_shot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_tune: np.ndarray,
    y_tune: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    sizes: Sequence[int | None] = FEW_SHOT_K,
    seeds: Sequence[int] = FEW_SHOT_SEEDS,
    metric_fn=None,
    grid: Sequence[float] = LOGISTIC_GRID,
    fit_fn=fit_probe,
) -> list[dict]:
    """Mean and standard deviation of held-out AUROC/AUPRC per training-set size."""
    from ehrjepa.eval.metrics import auprc, auroc

    metric_fn = metric_fn or {"auroc": auroc, "auprc": auprc}
    rows = []
    for k in sizes:
        scores: dict[str, list[float]] = {name: [] for name in metric_fn}
        n_train = []
        for seed in seeds:
            index = subsample(y_train, k, seed)
            ys = y_train[index]
            n_train.append(int(index.size))
            if ys.min() == ys.max():
                continue
            model = fit_fn(x_train[index], ys, x_tune, y_tune, grid=grid, seed=seed)
            p = model.predict_proba(x_eval)
            for name, fn in metric_fn.items():
                scores[name].append(fn(y_eval, p))
            if k is None:
                break  # the full split has no sampling variance
        row = {"k": k, "n_train": int(np.mean(n_train)) if n_train else 0, "n_seeds": 0}
        for name, values in scores.items():
            clean = [v for v in values if np.isfinite(v)]
            row["n_seeds"] = len(clean)
            row[f"{name}_mean"] = float(np.mean(clean)) if clean else float("nan")
            row[f"{name}_std"] = float(np.std(clean, ddof=0)) if clean else float("nan")
        rows.append(row)
    return rows
