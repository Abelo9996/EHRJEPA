"""Paired subject-level bootstrap of fine-tuned encoders vs count-GBM (and hybrid vs AR).

Reads the committed ``predictions.parquet`` files of the mimic-ft-1b, mimic-ft-1b-s12 and
mimic-ft-full grids (same held-out split, same anchors, same labels) and bootstraps
AUROC(a) - AUROC(b).

Resampling matches ``ehrjepa.eval.metrics``: draw held-out subjects with replacement
(``rng.integers(0, n, size=(n_boot, n))`` from ``np.random.default_rng(seed)``). Unlike the
harness, which draws per task, a single draw over the union of held-out subjects is applied
jointly to every model and every task, so per-task diffs, the mean-of-6 diff, and the
seed-pooled diff all come from the same resampled cohort. A subject drawn k times contributes
its row in each task k times (implemented as an integer weight; identical to row repetition,
ties counted 1/2).

Usage: python scripts/paired_bootstrap.py [--n-boot 2000] [--seed 0] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "experiments"

TASKS6 = [
    "inpatient_365d",
    "readmission_30d",
    "new_dx_365d/ckd",
    "new_dx_365d/diabetes",
    "new_dx_365d/heart_failure",
    "new_dx_365d/copd",
]
MORT = "mortality_365d"
TASKS = TASKS6 + [MORT]

# cell name -> (grid dir, model id inside that predictions.parquet)
CELLS: dict[str, tuple[str, str]] = {}
for frac in ("8pct", "25pct"):
    for obj in ("hybrid", "ar"):
        CELLS[f"{obj}_{frac}_s0"] = ("mimic-ft-1b", f"ft:{obj}_s0")
        for s in (1, 2):
            CELLS[f"{obj}_{frac}_s{s}"] = ("mimic-ft-1b-s12", f"ft:{obj}_s{s}")
for obj in ("hybrid", "ar"):
    CELLS[f"{obj}_full_s0"] = ("mimic-ft-full", f"ft:{obj}_s0")
BASELINE_CELL = "ar_8pct_s0"  # its eval also holds lr / gbm
BASELINES = ("gbm", "lr")

KEY = ["subject_id", "anchor_time", "task"]


def load() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Return the shared (subject, anchor, task, label) frame and a score Series per model."""
    frames: dict[str, pd.DataFrame] = {}
    for cell, (grid, model) in CELLS.items():
        df = pd.read_parquet(EXP / grid / "eval" / cell / "predictions.parquet")
        sub = df[df.model == model]
        assert len(sub), (cell, model)
        frames[cell] = sub
        if cell == BASELINE_CELL:
            for b in BASELINES:
                frames[b] = df[df.model == b]
    ref = frames["gbm"][KEY + ["label"]].sort_values(KEY).reset_index(drop=True)
    assert not ref.duplicated(["subject_id", "task"]).any(), "one row per subject per task expected"
    scores: dict[str, pd.Series] = {}
    for name, df in frames.items():
        m = ref.merge(df[KEY + ["label", "score"]], on=KEY, how="left", suffixes=("", "_m"))
        assert len(m) == len(ref) and m.score.notna().all(), f"{name}: anchors differ from gbm"
        assert (m.label == m.label_m).all(), f"{name}: labels differ from gbm"
        scores[name] = m.score.to_numpy()
    return ref, scores


class TaskAUC:
    """Weighted AUROC for one task, vectorised over a (n_boot, n_subjects) weight matrix."""

    def __init__(self, subj_idx: np.ndarray, y: np.ndarray):
        self.subj_idx = subj_idx
        self.y = y.astype(bool)

    def __call__(self, p: np.ndarray, w_subj: np.ndarray) -> np.ndarray:
        w = w_subj[:, self.subj_idx].astype(np.float64)  # (B, n_rows)
        order = np.argsort(p, kind="mergesort")
        ps, ys, w = p[order], self.y[order], w[:, order]
        starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
        wpos = np.add.reduceat(w * ys, starts, axis=1)
        wneg = np.add.reduceat(w * ~ys, starts, axis=1)
        neg_below = np.cumsum(wneg, axis=1) - wneg
        num = (wpos * (neg_below + 0.5 * wneg)).sum(axis=1)
        den = wpos.sum(axis=1) * wneg.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den


def summarize(point: float, draws: np.ndarray) -> dict:
    d = draws[np.isfinite(draws)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {
        "diff": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "p_gt0": float(np.mean(d > 0)),
        "n_boot_valid": int(d.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=EXP / "paired-bootstrap")
    args = ap.parse_args()

    ref, scores = load()
    subjects = np.sort(ref.subject_id.unique())
    n = subjects.size
    subj_pos = np.searchsorted(subjects, ref.subject_id.to_numpy())

    rng = np.random.default_rng(args.seed)
    index = rng.integers(0, n, size=(args.n_boot, n))
    w_boot = np.zeros((args.n_boot, n), dtype=np.int32)
    np.add.at(w_boot, (np.arange(args.n_boot)[:, None], index), 1)
    w_point = np.ones((1, n), dtype=np.int32)

    task_rows = {t: np.flatnonzero(ref.task.to_numpy() == t) for t in TASKS}
    aucs = {t: TaskAUC(subj_pos[r], ref.label.to_numpy()[r]) for t, r in task_rows.items()}

    # auc[model][task] = (point, draws[B])
    auc: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
    for name, s in scores.items():
        auc[name] = {}
        for t, r in task_rows.items():
            auc[name][t] = (float(aucs[t](s[r], w_point)[0]), aucs[t](s[r], w_boot))

    def mean6(name: str) -> tuple[float, np.ndarray]:
        return (
            float(np.mean([auc[name][t][0] for t in TASKS6])),
            np.mean([auc[name][t][1] for t in TASKS6], axis=0),
        )

    point_auroc = {
        name: {**{t: auc[name][t][0] for t in TASKS}, "mean6": mean6(name)[0]} for name in auc
    }

    def compare(a: str | list[str], b: str | list[str]) -> dict:
        """a - b; lists are averaged (seed pooling) within each resample."""
        A = a if isinstance(a, list) else [a]
        Bm = b if isinstance(b, list) else [b]
        out = {}
        for t in TASKS + ["mean6"]:
            def get(names):
                if t == "mean6":
                    vals = [mean6(x) for x in names]
                else:
                    vals = [auc[x][t] for x in names]
                return np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals], axis=0)
            pa, da = get(A)
            pb, db = get(Bm)
            out[t] = summarize(pa - pb, da - db)
        return out

    comparisons: dict[str, dict] = {}
    for cell in CELLS:
        comparisons[f"{cell} - gbm"] = compare(cell, "gbm")
    for frac, seeds in (("8pct", (0, 1, 2)), ("25pct", (0, 1, 2)), ("full", (0,))):
        for s in seeds:
            comparisons[f"hybrid_{frac}_s{s} - ar_{frac}_s{s}"] = compare(
                f"hybrid_{frac}_s{s}", f"ar_{frac}_s{s}"
            )
    for frac in ("8pct", "25pct"):
        for obj in ("hybrid", "ar"):
            cells = [f"{obj}_{frac}_s{s}" for s in (0, 1, 2)]
            comparisons[f"pooled3 {obj}_{frac} - gbm"] = compare(cells, "gbm")
        comparisons[f"pooled3 hybrid_{frac} - ar_{frac}"] = compare(
            [f"hybrid_{frac}_s{s}" for s in (0, 1, 2)], [f"ar_{frac}_s{s}" for s in (0, 1, 2)]
        )

    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_subjects": int(n),
        "rows_per_task": {t: int(r.size) for t, r in task_rows.items()},
        "resampling": "joint over held-out subjects; same draw for all models and tasks",
        "ci": "95% percentile",
        "mean6_tasks": TASKS6,
        "point_auroc": point_auroc,
        "comparisons": comparisons,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    def fmt(r):
        return f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] P(>0)={r['p_gt0']:.3f}"

    print(f"n_subjects={n}  n_boot={args.n_boot}  seed={args.seed}")
    for name in ["gbm", "lr"] + list(CELLS):
        print(f"  {name:18s} mean6={point_auroc[name]['mean6']:.4f}  mort={point_auroc[name][MORT]:.4f}")
    for k, v in comparisons.items():
        print(f"{k:34s} mean6 {fmt(v['mean6'])}   mort {fmt(v[MORT])}")


if __name__ == "__main__":
    main()
