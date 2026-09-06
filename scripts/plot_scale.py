"""``python scripts/plot_scale.py`` -- render the token-scaling figure for
``ar``, ``hybrid``, ``recon_only`` and ``jepa_ema`` on DE-SynPUF sample 1.

Reads five committed ``summary.md`` files -- nothing plotted is a number not
already committed in one of those tables:

* ``docs/experiments/2026-09-03-pilot-desynpuf/summary.md`` (grid 1, 48M
  tokens, 4L/192d) for ``ar`` and ``jepa_ema``;
* ``docs/experiments/2026-09-04-pilot3-desynpuf/summary.md`` (grid 3, 48M
  tokens, 4L/192d) for ``recon_only``;
* ``docs/experiments/2026-09-04-pilot4-desynpuf/summary.md`` (grid 4, 48M
  tokens, 4L/192d) for ``nextlatent_h1416_recon``, read here as ``hybrid``;
* ``docs/experiments/scale-desynpuf/summary.md`` (200M tokens, 6L/256d) for
  all four cells;
* ``docs/experiments/scale1b-desynpuf/summary.md`` (1B tokens, 6L/256d) for
  ``ar`` and ``hybrid`` only -- ``recon_only`` and ``jepa_ema`` were not run
  at 1B.

Two panels:

* **left** -- mean held-out AUROC across the seven tasks vs. nominal token
  budget (log x), one line per objective, with a dashed horizontal reference
  at the ``gbm`` count-feature baseline;
* **right** -- per-task AUROC at 1B tokens, ``ar`` vs. ``hybrid``, as paired
  bars (the only two cells trained at that budget).

Same palette as ``scripts/plot_grids.py``: ``ar`` aqua, ``masked-span jepa``
(``jepa_ema``) yellow, ``recon-only`` green, ``nextlatent`` (``hybrid``)
violet, ``gbm`` blue.
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

# (series label, nominal tokens, source file, row name in that file)
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


def load_series() -> tuple[dict[str, list[tuple[float, float]]], float]:
    """Return {series: [(tokens, mean_auroc), ...]} and the gbm reference."""
    cache: dict[Path, dict[str, dict[str, float]]] = {}

    def runs(path: Path) -> dict[str, dict[str, float]]:
        if path not in cache:
            cache[path] = _load(path, "| run |", "run")
        return cache[path]

    series: dict[str, list[tuple[float, float]]] = {}
    for name, points in SERIES_POINTS.items():
        series[name] = [(tokens, _mean(runs(path)[row])) for tokens, path, row in points]

    gbm = _mean(_load(SCALE_200M, "| model |", "model")["gbm"])
    return series, gbm


def load_1b_per_task() -> dict[str, dict[str, float]]:
    runs = _load(SCALE_1B, "| run |", "run")
    return {"ar": runs["ar"], "hybrid": runs["hybrid"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    series, gbm = load_series()
    per_task = load_1b_per_task()

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
    ax_scale.axhline(gbm, color=COLOR_GBM, linewidth=1.3, linestyle="--", zorder=0)
    ax_scale.text(48_005_120, gbm, " gbm", color=COLOR_GBM, fontsize=8, va="bottom", ha="left")
    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("nominal token slots (log scale)")
    ax_scale.set_ylabel("mean AUROC across 7 held-out tasks")
    ax_scale.set_title("Mean AUROC vs. token budget", fontsize=10, color=INK)
    ax_scale.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_scale.set_axisbelow(True)
    ax_scale.legend(frameon=False, fontsize=8.5, loc="lower right")

    x = np.arange(len(TASKS))
    width = 0.38
    ar_vals = [per_task["ar"][t] for t in TASKS]
    hybrid_vals = [per_task["hybrid"][t] for t in TASKS]
    ax_task.bar(
        x - width / 2,
        ar_vals,
        width,
        color=SERIES_COLOR["ar"],
        edgecolor=INK,
        linewidth=0.5,
        label="ar",
    )
    ax_task.bar(
        x + width / 2,
        hybrid_vals,
        width,
        color=SERIES_COLOR["hybrid"],
        edgecolor=INK,
        linewidth=0.5,
        label="hybrid",
    )
    ax_task.set_xticks(x)
    ax_task.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=30, ha="right", fontsize=8.5)
    ax_task.set_ylabel("held-out AUROC")
    ax_task.set_ylim(0.5, 0.85)
    ax_task.set_title("Per-task AUROC at 1B tokens: ar vs. hybrid", fontsize=10, color=INK)
    ax_task.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_task.set_axisbelow(True)
    ax_task.legend(frameon=False, fontsize=8.5, loc="lower right")

    for ax in (ax_scale, ax_task):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)

    fig.suptitle(
        "DE-SynPUF sample 1, held-out 3,000 subjects, RTX 4060 scale grids (6L/256d)",
        fontsize=11,
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
