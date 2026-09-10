"""``python scripts/plot_scale.py`` -- render the token-scaling figure for
``ar``, ``hybrid``, ``recon_only`` and ``jepa_ema`` on DE-SynPUF sample 1.

Reads seven committed ``summary.md`` files -- nothing plotted is a number not
already committed in one of those tables:

* ``docs/experiments/2026-09-03-pilot-desynpuf/summary.md`` (grid 1, 48M
  tokens, 4L/192d) for ``ar`` and ``jepa_ema``;
* ``docs/experiments/2026-09-04-pilot3-desynpuf/summary.md`` (grid 3, 48M
  tokens, 4L/192d) for ``recon_only``;
* ``docs/experiments/2026-09-04-pilot4-desynpuf/summary.md`` (grid 4, 48M
  tokens, 4L/192d) for ``nextlatent_h1416_recon``, read here as ``hybrid``;
* ``docs/experiments/scale-desynpuf/summary.md`` (200M tokens, 6L/256d) for
  all four cells;
* ``docs/experiments/scale1b-desynpuf/summary.md`` (1B tokens, 6L/256d,
  seed 0) for ``ar`` and ``hybrid`` -- ``recon_only`` and ``jepa_ema`` were
  not run at 1B;
* ``docs/experiments/scale1b-seeds-desynpuf/summary.md`` (1B tokens, 6L/256d,
  seeds 1 and 2) for ``ar`` and ``hybrid``;
* ``docs/experiments/scale1b-final-desynpuf/summary.md`` (1B tokens,
  6L/256d, seeds 0-2, scored on the FULL 11.7k-subject held-out split) for
  ``ar_full`` and ``hybrid_final`` (the repository default:
  ``configs/pretrain_default.yaml``, EMA target, no SIGReg).

Every 48M/200M/subset-1B point above -- everything except the two
``_full`` markers -- is scored on the same seeded 3,000-subject held-out cut.
The two ``_full`` markers are scored on the full held-out split (11,708
subjects) and are drawn as diamonds, offset slightly in x from the subset 1B
circles, so subset and full-split points are never plotted on top of each
other. The 1B point for ``ar``/``hybrid`` (subset) and for
``ar_full``/``hybrid_final`` (full split) is a 3-seed mean (seeds 0, 1, 2) in
both panels; nothing else on the plot is averaged across seeds.

Two panels:

* **left** -- mean held-out AUROC across the seven tasks vs. nominal token
  budget (log x), one line per objective on the 3,000-subject subset (``ar``,
  ``hybrid``, ``recon_only``, ``jepa_ema``), with a dashed horizontal
  reference at the ``gbm`` count-feature baseline; the subset 1B
  ``ar``/``hybrid`` markers carry a vertical min-max error bar across the
  three seeds. Two additional diamond markers at 1B give the full-split
  3-seed means for ``ar_full`` and ``hybrid_final``, also with a min-max
  error bar;
* **right** -- per-task AUROC at 1B tokens on the full held-out split,
  ``ar_full`` vs. ``hybrid_final`` (the repository default), as paired bars,
  each bar plotting the 3-seed mean with a min-max whisker.

Same palette as ``scripts/plot_grids.py``: ``ar`` aqua, ``masked-span jepa``
(``jepa_ema``) yellow, ``recon-only`` green, ``nextlatent`` (``hybrid``)
violet, ``gbm`` blue. The full-split markers reuse the ``ar``/``hybrid``
colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs/figures/scale_desynpuf.png"

TASKS: tuple[str, ...] = (
    "inpatient_365d",
    "mortality_365d",
    "new_dx_365d/ckd",
    "new_dx_365d/copd",
    "new_dx_365d/diabetes",
    "new_dx_365d/heart_failure",
    "readmission_30d",
)
TASK_LABELS: dict[str, str] = {
    "inpatient_365d": "inpatient",
    "mortality_365d": "mortality",
    "new_dx_365d/ckd": "ckd",
    "new_dx_365d/copd": "copd",
    "new_dx_365d/diabetes": "diabetes",
    "new_dx_365d/heart_failure": "heart_failure",
    "readmission_30d": "readmission",
}

GRID1 = REPO / "docs/experiments/2026-09-03-pilot-desynpuf/summary.md"
GRID3 = REPO / "docs/experiments/2026-09-04-pilot3-desynpuf/summary.md"
GRID4 = REPO / "docs/experiments/2026-09-04-pilot4-desynpuf/summary.md"
SCALE_200M = REPO / "docs/experiments/scale-desynpuf/summary.md"
SCALE_1B = REPO / "docs/experiments/scale1b-desynpuf/summary.md"
SCALE_1B_SEEDS = REPO / "docs/experiments/scale1b-seeds-desynpuf/summary.md"
SCALE_1B_FINAL = REPO / "docs/experiments/scale1b-final-desynpuf/summary.md"

# (series label, nominal tokens, source file, row name in that file)
# The 1B entries for ar/hybrid below are placeholders for the file/row used
# to key the 1B token count; their scores are replaced with a 3-seed mean
# (see _load_1b_seed_scores) rather than read as a single row.
SERIES_POINTS: dict[str, list[tuple[float, Path, str]]] = {
    "ar": [
        (48_005_120, GRID1, "ar"),
        (200_015_872, SCALE_200M, "ar"),
        (1_000_013_824, SCALE_1B, "ar"),
    ],
    "hybrid": [
        (48_005_120, GRID4, "nextlatent_h1416_recon"),
        (200_015_872, SCALE_200M, "hybrid"),
        (1_000_013_824, SCALE_1B, "hybrid"),
    ],
    "recon_only": [
        (48_005_120, GRID3, "recon_only"),
        (200_015_872, SCALE_200M, "recon_only"),
    ],
    "jepa_ema": [
        (48_005_120, GRID1, "jepa_ema"),
        (200_015_872, SCALE_200M, "jepa_ema"),
    ],
}

# Cells with a 1B, 3-seed replication grid; row names in SCALE_1B_SEEDS for
# seeds 1 and 2 (seed 0 is the row of the same name in SCALE_1B).
SEED1B_ROWS: dict[str, tuple[str, str]] = {
    "ar": ("ar_s1", "ar_s2"),
    "hybrid": ("hybrid_s1", "hybrid_s2"),
}

# Full-held-out-split, 3-seed 1B cells in SCALE_1B_FINAL. ``ar_full`` is
# next-code AR; ``hybrid_final`` is the repository default
# (configs/pretrain_default.yaml: EMA target, no SIGReg).
FULL1B_ROWS: dict[str, tuple[str, str, str]] = {
    "ar_full": ("ar_s0_full", "ar_s1_full", "ar_s2_full"),
    "hybrid_final": ("hybrid_final_s0", "hybrid_final_s1", "hybrid_final_s2"),
}
FULL1B_SOURCE_SERIES = {"ar_full": "ar", "hybrid_final": "hybrid"}
FULL1B_TOKENS = 1_000_013_824

# Same categorical slots as scripts/plot_grids.py.
COLOR_GBM = "#2a78d6"  # slot 1, blue
SERIES_COLOR: dict[str, str] = {
    "ar": "#1baf7a",  # slot 3, aqua
    "jepa_ema": "#eda100",  # slot 4, yellow (masked-span jepa family)
    "recon_only": "#008300",  # slot 6, green (recon-only family)
    "hybrid": "#4a3aa7",  # slot 7, violet (nextlatent family)
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_COLOR = "#e1e0d9"


def _parse_md_table(text: str, header_marker: str) -> list[dict[str, str]]:
    """Parse the first Markdown table whose header row contains ``header_marker``."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("|") and header_marker in ln)
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for ln in lines[start + 2 :]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def _unbacktick(s: str) -> str:
    return s.strip("`")


def _load(path: Path, header_marker: str, key: str) -> dict[str, dict[str, float]]:
    text = path.read_text()
    rows = _parse_md_table(text, header_marker)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        name = _unbacktick(row[key])
        out[name] = {t: float(row[t]) for t in TASKS if row.get(t) not in (None, "", "--")}
    return out


def _mean(scores: dict[str, float]) -> float:
    return float(np.mean([scores[t] for t in TASKS]))


def _load_1b_seed_scores(name: str) -> list[dict[str, float]]:
    """Per-task scores for each of the 3 seeds (0, 1, 2) for ``name`` (ar|hybrid) at 1B tokens."""
    seed0 = _load(SCALE_1B, "| run |", "run")[name]
    seeds = _load(SCALE_1B_SEEDS, "| run |", "run")
    row_s1, row_s2 = SEED1B_ROWS[name]
    return [seed0, seeds[row_s1], seeds[row_s2]]


def _load_full1b_seed_scores(name: str) -> list[dict[str, float]]:
    """Per-task scores for each of the 3 seeds (0, 1, 2) for a full-held-out-split
    1B cell (``ar_full`` or ``hybrid_final``), from SCALE_1B_FINAL."""
    rows = _load(SCALE_1B_FINAL, "| run |", "run")
    row_s0, row_s1, row_s2 = FULL1B_ROWS[name]
    return [rows[row_s0], rows[row_s1], rows[row_s2]]


def load_full1b_series() -> dict[str, tuple[float, float, float]]:
    """Return {name: (mean, min, max)} of the mean-across-7-tasks AUROC, 3-seed,
    for the full-held-out-split 1B cells (``ar_full``, ``hybrid_final``)."""
    out: dict[str, tuple[float, float, float]] = {}
    for name in FULL1B_ROWS:
        per_seed_means = [_mean(s) for s in _load_full1b_seed_scores(name)]
        out[name] = (float(np.mean(per_seed_means)), min(per_seed_means), max(per_seed_means))
    return out


def load_full1b_per_task() -> tuple[
    dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]
]:
    """Return {name: {task: 3-seed mean}} and {name: {task: (mean-min, max-mean)}}
    for the full-held-out-split 1B cells (``ar_full``, ``hybrid_final``)."""
    means: dict[str, dict[str, float]] = {}
    errs: dict[str, dict[str, tuple[float, float]]] = {}
    for name in FULL1B_ROWS:
        seed_scores = _load_full1b_seed_scores(name)
        task_means = {t: float(np.mean([s[t] for s in seed_scores])) for t in TASKS}
        task_mins = {t: float(np.min([s[t] for s in seed_scores])) for t in TASKS}
        task_maxs = {t: float(np.max([s[t] for s in seed_scores])) for t in TASKS}
        means[name] = task_means
        errs[name] = {
            t: (task_means[t] - task_mins[t], task_maxs[t] - task_means[t]) for t in TASKS
        }
    return means, errs


def load_series() -> tuple[
    dict[str, list[tuple[float, float]]], float, dict[str, tuple[float, float]]
]:
    """Return {series: [(tokens, mean_auroc), ...]}, the gbm reference, and
    {name: (min_mean_auroc, max_mean_auroc)} across the 3 seeds at 1B for
    ar/hybrid (empty for series without a seed grid)."""
    cache: dict[Path, dict[str, dict[str, float]]] = {}

    def runs(path: Path) -> dict[str, dict[str, float]]:
        if path not in cache:
            cache[path] = _load(path, "| run |", "run")
        return cache[path]

    series: dict[str, list[tuple[float, float]]] = {}
    seed1b_range: dict[str, tuple[float, float]] = {}
    for name, points in SERIES_POINTS.items():
        pts: list[tuple[float, float]] = []
        for tokens, path, row in points:
            if path is SCALE_1B and name in SEED1B_ROWS:
                per_seed_means = [_mean(scores) for scores in _load_1b_seed_scores(name)]
                pts.append((tokens, float(np.mean(per_seed_means))))
                seed1b_range[name] = (min(per_seed_means), max(per_seed_means))
            else:
                pts.append((tokens, _mean(runs(path)[row])))
        series[name] = pts

    gbm = _mean(_load(SCALE_200M, "| model |", "model")["gbm"])
    return series, gbm, seed1b_range


def load_1b_per_task() -> tuple[
    dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]
]:
    """Return {name: {task: 3-seed mean}} and {name: {task: (mean-min, max-mean)}}
    for ar/hybrid at 1B tokens."""
    means: dict[str, dict[str, float]] = {}
    errs: dict[str, dict[str, tuple[float, float]]] = {}
    for name in ("ar", "hybrid"):
        seed_scores = _load_1b_seed_scores(name)
        task_means = {t: float(np.mean([s[t] for s in seed_scores])) for t in TASKS}
        task_mins = {t: float(np.min([s[t] for s in seed_scores])) for t in TASKS}
        task_maxs = {t: float(np.max([s[t] for s in seed_scores])) for t in TASKS}
        means[name] = task_means
        errs[name] = {
            t: (task_means[t] - task_mins[t], task_maxs[t] - task_means[t]) for t in TASKS
        }
    return means, errs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    series, gbm, seed1b_range = load_series()
    full1b = load_full1b_series()
    per_task, per_task_err = load_full1b_per_task()

    fig, (ax_scale, ax_task) = plt.subplots(1, 2, figsize=(13, 5))

    for name, points in series.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax_scale.plot(
            xs,
            ys,
            marker="o",
            color=SERIES_COLOR[name],
            linewidth=1.8,
            markersize=6,
            label=name,
        )
        if name in seed1b_range:
            lo, hi = seed1b_range[name]
            tokens_1b, mean_1b = points[-1]
            ax_scale.errorbar(
                [tokens_1b],
                [mean_1b],
                yerr=[[mean_1b - lo], [hi - mean_1b]],
                fmt="none",
                ecolor=SERIES_COLOR[name],
                elinewidth=1.3,
                capsize=4,
                zorder=3,
            )

    # Full-held-out-split 1B markers (diamonds), offset in x so they never sit
    # on top of the subset-1B circles. ar_full pairs with the "ar" series
    # color, hybrid_final with the "hybrid" series color.
    full1b_offsets = {"ar_full": 0.90, "hybrid_final": 1.10}
    for name, (mean_full, lo, hi) in full1b.items():
        color = SERIES_COLOR[FULL1B_SOURCE_SERIES[name]]
        tokens = FULL1B_TOKENS * full1b_offsets[name]
        ax_scale.errorbar(
            [tokens],
            [mean_full],
            yerr=[[mean_full - lo], [hi - mean_full]],
            fmt="D",
            color=color,
            markersize=7,
            markeredgecolor=INK,
            markeredgewidth=0.6,
            elinewidth=1.3,
            capsize=4,
            zorder=4,
            label=f"{name} (full split)",
        )
    ax_scale.axhline(gbm, color=COLOR_GBM, linewidth=1.3, linestyle="--", zorder=0)
    ax_scale.text(48_005_120, gbm, " gbm", color=COLOR_GBM, fontsize=8, va="bottom", ha="left")
    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("nominal token slots (log scale)")
    ax_scale.set_ylabel("mean AUROC across 7 held-out tasks")
    ax_scale.set_title(
        "Mean AUROC vs. token budget\n(circles: 3k-subject subset; diamonds: full split, 1B only)",
        fontsize=9.5,
        color=INK,
    )
    ax_scale.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_scale.set_axisbelow(True)
    ax_scale.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=1)

    x = np.arange(len(TASKS))
    width = 0.38
    ar_vals = [per_task["ar_full"][t] for t in TASKS]
    hybrid_vals = [per_task["hybrid_final"][t] for t in TASKS]
    ar_err = np.array([per_task_err["ar_full"][t] for t in TASKS]).T
    hybrid_err = np.array([per_task_err["hybrid_final"][t] for t in TASKS]).T
    ax_task.bar(
        x - width / 2,
        ar_vals,
        width,
        color=SERIES_COLOR["ar"],
        edgecolor=INK,
        linewidth=0.5,
        label="ar",
        yerr=ar_err,
        error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 3},
    )
    ax_task.bar(
        x + width / 2,
        hybrid_vals,
        width,
        color=SERIES_COLOR["hybrid"],
        edgecolor=INK,
        linewidth=0.5,
        label="hybrid_final (default)",
        yerr=hybrid_err,
        error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 3},
    )
    ax_task.set_xticks(x)
    ax_task.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=30, ha="right", fontsize=8.5)
    ax_task.set_ylabel("held-out AUROC")
    ax_task.set_ylim(0.5, 0.85)
    ax_task.set_title(
        "Per-task AUROC at 1B tokens, full held-out split:\nar vs. hybrid_final, 3-seed mean",
        fontsize=9.5,
        color=INK,
    )
    ax_task.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_task.set_axisbelow(True)
    ax_task.legend(frameon=False, fontsize=8.5, loc="lower right")

    for ax in (ax_scale, ax_task):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)

    fig.suptitle(
        "DE-SynPUF sample 1, RTX 4060 scale grids (6L/256d) -- "
        "left: 3k-subject subset with full-split 1B diamonds; right: full split (11.7k subjects)",
        fontsize=10.5,
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
