"""Discrimination, calibration, and bootstrap confidence intervals.

Every resample is over **subjects**, not rows, which here is the same thing:
:mod:`ehrjepa.eval.tasks` emits exactly one anchor per subject, and that
invariant is checked when a task is built. Two models are compared with a
*paired* bootstrap -- the same resampled subject index applied to both score
vectors -- so the interval on the difference is not inflated by the sampling
variance the two models share.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "METRICS",
    "auprc",
    "auroc",
    "bootstrap_ci",
    "brier",
    "calibration_slope",
    "evaluate",
    "paired_bootstrap",
]

_EPS = 1e-12


def _as_arrays(y: Sequence[int] | np.ndarray, p: Sequence[float] | np.ndarray):
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same shape, got {y.shape} and {p.shape}")
    return y, p


def _degenerate(y: np.ndarray) -> bool:
    """True when the resample has only one class, so ranking metrics are undefined."""
    return y.size == 0 or y.min() == y.max()


def auroc(y, p) -> float:
    y, p = _as_arrays(y, p)
    return float("nan") if _degenerate(y) else float(roc_auc_score(y, p))


def auprc(y, p) -> float:
    """Average precision -- the estimator of AUPRC that does not interpolate."""
    y, p = _as_arrays(y, p)
    return float("nan") if _degenerate(y) else float(average_precision_score(y, p))


def brier(y, p) -> float:
    y, p = _as_arrays(y, p)
    return float("nan") if y.size == 0 else float(np.mean((p - y) ** 2))


def calibration_slope(y, p) -> float:
    """Slope of a logistic recalibration of ``logit(p)`` on ``y``.

    1.0 is perfect; below 1.0 the scores are over-dispersed (too confident),
    above 1.0 under-dispersed. Fitted by plain Newton steps on the one-parameter
    -plus-intercept model so it has no dependence on a solver's regularisation
    default.
    """
    y, p = _as_arrays(y, p)
    if _degenerate(y):
        return float("nan")
    x = np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))
    if np.allclose(x, x[0]):
        return float("nan")
    beta = np.array([0.0, 1.0])
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(100):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(mu * (1 - mu), _EPS, None)
        hessian = design.T @ (design * w[:, None])
        gradient = design.T @ (y - mu)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            return float("nan")
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[1])


#: Metric name -> callable, in report order.
METRICS: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": auroc,
    "auprc": auprc,
    "brier": brier,
    "calibration_slope": calibration_slope,
}


def evaluate(y, p, metrics: Mapping[str, Callable] | None = None) -> dict[str, float]:
    """Point estimates of every metric."""
    y, p = _as_arrays(y, p)
    return {name: fn(y, p) for name, fn in (metrics or METRICS).items()}


def _resample_index(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_ci(
    y,
    p,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    metrics: Mapping[str, Callable] | None = None,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap over subjects.

    Returns ``{metric: {"point", "lo", "hi", "n_boot_valid"}}``. Resamples in
    which a metric is undefined (a draw with one class only) are dropped from
    that metric's percentile, and counted.
    """
    y, p = _as_arrays(y, p)
    metrics = metrics or METRICS
    point = evaluate(y, p, metrics)
    rng = np.random.default_rng(seed)
    index = _resample_index(y.size, n_boot, rng)
    draws: dict[str, list[float]] = {name: [] for name in metrics}
    for row in index:
        ys, ps = y[row], p[row]
        for name, fn in metrics.items():
            draws[name].append(fn(ys, ps))
    out: dict[str, dict[str, float]] = {}
    for name in metrics:
        values = np.asarray(draws[name], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            out[name] = {
                "point": point[name],
                "lo": float("nan"),
                "hi": float("nan"),
                "n_boot_valid": 0,
            }
            continue
        lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        out[name] = {
            "point": point[name],
            "lo": float(lo),
            "hi": float(hi),
            "n_boot_valid": int(values.size),
        }
    return out


def paired_bootstrap(
    y,
    p_a,
    p_b,
    *,
    metric: str | Callable = "auroc",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap the difference ``metric(a) - metric(b)`` on identical subjects.

    ``p_a`` and ``p_b`` must be scores for the same subjects in the same order.
    ``p_greater`` is the two-sided bootstrap tail: the fraction of resamples
    whose difference has the opposite sign to the point difference, doubled.
    """
    fn = METRICS[metric] if isinstance(metric, str) else metric
    y, p_a = _as_arrays(y, p_a)
    _, p_b = _as_arrays(y, p_b)
    point = fn(y, p_a) - fn(y, p_b)
    rng = np.random.default_rng(seed)
    index = _resample_index(y.size, n_boot, rng)
    diffs = []
    for row in index:
        ys = y[row]
        value = fn(ys, p_a[row]) - fn(ys, p_b[row])
        if np.isfinite(value):
            diffs.append(value)
    values = np.asarray(diffs, dtype=np.float64)
    if values.size == 0:  # pragma: no cover - only for empty or single-class labels
        return {"diff": point, "lo": float("nan"), "hi": float("nan"), "p_value": float("nan")}
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    tail = float(np.mean(values <= 0.0)) if point > 0 else float(np.mean(values >= 0.0))
    return {
        "diff": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "p_value": float(min(1.0, 2.0 * tail)),
        "n_boot_valid": int(values.size),
    }
