"""Count features over pre-anchor history, and the two tabular baselines.

The featurisation is a deliberately plain reimplementation of what
`MEDS-Tabular-AutoML <https://github.com/mmcdermott/MEDS_Tabular_AutoML>`_
(McDermott et al.) does: per-code occurrence counts in a few time windows before
the prediction time, plus a summary of each code's numeric values, fed to a
linear model and to gradient boosting. It is the reference point a pretrained
encoder has to beat, and on this kind of data it is a hard one.

Five blocks of ``vocab_size`` columns each, concatenated:

============== =========================================================
``count_30d``   ``log1p`` of the code's occurrences in ``(t - 30d, t)``
``count_365d``  ``log1p`` of the code's occurrences in ``(t - 365d, t)``
``count_all``   ``log1p`` of the code's occurrences in all history
``last_value``  the code's most recent ``value_z`` (0 if never numeric)
``value_count`` ``log1p`` of the code's numeric observations
============== =========================================================

Every window is open at ``t``: history comes from
:class:`ehrjepa.eval.history.HistoryReader`, which cuts strictly before the
anchor. Columns that are nonzero for fewer than ``min_df`` **training** rows are
dropped, and that support is fit on train alone.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ehrjepa.eval.history import HistoryReader, anchor_minutes
from ehrjepa.eval.metrics import auroc

__all__ = [
    "BLOCKS",
    "FittedModel",
    "count_features",
    "count_matrix",
    "fit_gbm",
    "fit_logistic",
    "prune_columns",
]

log = logging.getLogger(__name__)

#: Feature blocks, in column order.
BLOCKS = ("count_30d", "count_365d", "count_all", "last_value", "value_count")

_MINUTES_PER_DAY = 1440
_WINDOW_MINUTES = (30 * _MINUTES_PER_DAY, 365 * _MINUTES_PER_DAY)

#: Logistic regression grid, tuned on the ``tuning`` split.
LOGISTIC_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

#: Three gradient-boosting settings, tuned on the ``tuning`` split.
GBM_GRID = (
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 300},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 600},
)


def count_features(
    history: Mapping[str, np.ndarray], anchor_min: int, vocab_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse ``(column_index, value)`` pairs for one subject's history.

    ``history`` is what :meth:`HistoryReader.history` returns: ``code_id``,
    ``value_bin``, ``value_z`` and ``time_min`` arrays in time order, all of them
    strictly before ``anchor_min``.
    """
    codes = np.asarray(history["code_id"], dtype=np.int64)
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    times = np.asarray(history["time_min"], dtype=np.int64)
    bins = np.asarray(history["value_bin"], dtype=np.int64)
    values = np.asarray(history["value_z"], dtype=np.float64)

    indices: list[np.ndarray] = []
    data: list[np.ndarray] = []

    def emit(block: int, cols: np.ndarray, vals: np.ndarray) -> None:
        keep = vals != 0.0
        indices.append(cols[keep] + block * vocab_size)
        data.append(vals[keep])

    for block, span in enumerate(_WINDOW_MINUTES):
        window = codes[times > anchor_min - span]
        if window.size:
            counts = np.bincount(window, minlength=vocab_size)
            nz = np.flatnonzero(counts)
            emit(block, nz, np.log1p(counts[nz]))
        else:
            emit(block, np.zeros(0, dtype=np.int64), np.zeros(0))

    counts = np.bincount(codes, minlength=vocab_size)
    nz = np.flatnonzero(counts)
    emit(2, nz, np.log1p(counts[nz]))

    numeric = np.flatnonzero(bins > 0)
    if numeric.size:
        num_codes = codes[numeric]
        # Reverse, then take each code's first appearance: the last observation
        # in time order. Fancy-index assignment with duplicate indices is not
        # defined to keep the last write, so it cannot be used here.
        rev_codes = num_codes[::-1]
        rev_values = values[numeric][::-1]
        unique, first = np.unique(rev_codes, return_index=True)
        emit(3, unique, rev_values[first])
        num_counts = np.bincount(num_codes, minlength=vocab_size)
        nz_num = np.flatnonzero(num_counts)
        emit(4, nz_num, np.log1p(num_counts[nz_num]))

    return np.concatenate(indices), np.concatenate(data)


def count_matrix(
    reader: HistoryReader, anchors: pl.DataFrame, vocab_size: int | None = None
) -> sparse.csr_matrix:
    """A CSR matrix with one row per anchor, in ``anchors`` row order."""
    vocab_size = reader.vocab_size if vocab_size is None else vocab_size
    minutes = anchor_minutes(anchors["anchor_time"])
    subjects = anchors["subject_id"].to_numpy()
    splits = anchors["split"].to_list()

    indptr = np.zeros(anchors.height + 1, dtype=np.int64)
    all_indices: list[np.ndarray] = []
    all_data: list[np.ndarray] = []
    for i in range(anchors.height):
        history = reader.history(int(subjects[i]), splits[i], int(minutes[i]))
        cols, vals = count_features(history, int(minutes[i]), vocab_size)
        all_indices.append(cols)
        all_data.append(vals)
        indptr[i + 1] = indptr[i] + cols.size
    matrix = sparse.csr_matrix(
        (
            np.concatenate(all_data) if all_data else np.zeros(0),
            np.concatenate(all_indices) if all_indices else np.zeros(0, dtype=np.int64),
            indptr,
        ),
        shape=(anchors.height, len(BLOCKS) * vocab_size),
    )
    matrix.sum_duplicates()
    return matrix


def prune_columns(train: sparse.csr_matrix, min_df: int = 10) -> np.ndarray:
    """Column indices that are nonzero in at least ``min_df`` training rows."""
    support = np.diff(train.tocsc().indptr)
    return np.flatnonzero(support >= min_df)


# --------------------------------------------------------------------------- #


@dataclass
class FittedModel:
    """A tuned model plus the setting that won on ``tuning``."""

    name: str
    model: object
    params: dict
    tuning_auroc: float
    grid: list[dict]

    def predict_proba(self, x) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x))[:, 1]


def fit_logistic(
    x_train,
    y_train: np.ndarray,
    x_tune,
    y_tune: np.ndarray,
    *,
    grid: Sequence[float] = LOGISTIC_GRID,
    seed: int = 0,
    name: str = "lr",
    scale: bool = True,
) -> FittedModel:
    """L2 logistic regression, ``C`` chosen by AUROC on the tuning split.

    Sparse inputs are scaled with ``with_mean=False``: centring a sparse matrix
    would densify it, and the columns are counts whose zero already means
    "absent", so only the scale is standardised.
    """
    scaler = None
    if scale:
        scaler = StandardScaler(with_mean=not sparse.issparse(x_train)).fit(x_train)
        x_train, x_tune = scaler.transform(x_train), scaler.transform(x_tune)
    best: tuple[float, LogisticRegression, dict] | None = None
    history = []
    for c in grid:
        model = LogisticRegression(
            C=c, max_iter=2000, solver="liblinear", random_state=seed, class_weight=None
        )
        model.fit(x_train, y_train)
        score = auroc(y_tune, model.predict_proba(x_tune)[:, 1])
        history.append({"C": c, "tuning_auroc": score})
        if best is None or (np.isfinite(score) and score > best[0]):
            best = (score, model, {"C": c})
    assert best is not None
    return FittedModel(
        name=name,
        model=_Pipeline(scaler, best[1]),
        params=best[2],
        tuning_auroc=best[0],
        grid=history,
    )


def fit_gbm(
    x_train,
    y_train: np.ndarray,
    x_tune,
    y_tune: np.ndarray,
    *,
    x_predict=None,
    grid: Sequence[Mapping] = GBM_GRID,
    seed: int = 0,
    name: str = "gbm",
) -> FittedModel:
    """Gradient boosting over the same matrix; three settings tuned on ``tuning``.

    The fit happens in a subprocess (see :mod:`ehrjepa.eval` for the OpenMP
    reason), so the returned model is not a live estimator: it carries the
    predictions for ``x_predict`` and will refuse any other input.
    """
    import subprocess
    import sys
    import tempfile

    if x_predict is None:
        x_predict = x_tune
    with tempfile.TemporaryDirectory(prefix="ehrjepa-gbm-") as tmp:
        work = Path(tmp)
        for label, matrix in (
            ("x_train", x_train),
            ("x_tune", x_tune),
            ("x_predict", x_predict),
        ):
            sparse.save_npz(work / f"{label}.npz", sparse.csr_matrix(matrix))
        np.savez(work / "y.npz", train=y_train, tune=y_tune)
        (work / "grid.json").write_text(json.dumps({"grid": [dict(g) for g in grid], "seed": seed}))
        proc = subprocess.run(
            [sys.executable, "-m", "ehrjepa.eval._gbm_worker", str(work)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gbm worker failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
        result = json.loads((work / "result.json").read_text())
        predictions = np.load(work / "p_predict.npy")
    return FittedModel(
        name=name,
        model=_FrozenPredictions(predictions),
        params=result["params"],
        tuning_auroc=result["tuning_auroc"],
        grid=result["grid"],
    )


class _FrozenPredictions:
    """The subprocess' predictions, wearing a ``predict_proba`` face."""

    def __init__(self, positive: np.ndarray) -> None:
        self.positive = np.asarray(positive, dtype=np.float64)

    def predict_proba(self, x) -> np.ndarray:
        if x.shape[0] != self.positive.shape[0]:
            raise ValueError(
                "this model was fit out of process and only holds predictions for the "
                f"{self.positive.shape[0]} rows it was given, not {x.shape[0]}"
            )
        return np.column_stack([1.0 - self.positive, self.positive])


class _Pipeline:
    """Scaler + estimator, small enough not to justify a sklearn Pipeline import."""

    def __init__(self, scaler, estimator) -> None:
        self.scaler = scaler
        self.estimator = estimator

    def predict_proba(self, x) -> np.ndarray:
        return self.estimator.predict_proba(self.scaler.transform(x) if self.scaler else x)


def cache_path(cache_root: Path | str, source: str, task: str) -> Path:
    """Where a built count matrix is memoised."""
    return Path(cache_root) / source / f"counts__{task.replace('/', '__')}.npz"
