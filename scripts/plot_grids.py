"""``python scripts/plot_grids.py`` -- render grids 1-4 to one consolidated figure.

Reads the four ``summary.md`` files ablation grids 1-4 wrote
(``docs/experiments/{2026-09-03-pilot,2026-09-03-pilot2,2026-09-04-pilot3,
2026-09-04-pilot4}-desynpuf/summary.md``) and draws two horizontal-bar panels,
one row per trained cell, in the same order in both panels:

* **left** -- mean gain over the cell's own ``random_init`` control (mean AUROC
  across the seven held-out tasks, cell minus control), colored by objective
  family, with a dashed reference line at ``ar``'s gain;
* **right** -- the cell's mean absolute AUROC across the seven tasks, same
  color and order, with dashed reference lines at the count-feature ``gbm``
  and ``lr`` baselines.

A cell's control is the ``random_init@<run>`` row computed for its own
(architecture, pooling) pair -- grid 2's ``jepa_notime`` and grid 3's
``jepa_recon_notime`` share one control value (same untrained 4x192
bidirectional encoder, same seed), as do grid 4's ``nextlatent_h1`` and grid
2's re-scored ``ar_last`` (same untrained causal encoder); each is read from
its own grid's ``summary.md`` rather than deduplicated, so this script never
has to assert two files agree.

Diagnostic re-evaluation rows that train nothing (grid 2's ``ar_last`` and
``jepa_ema_mean``, which re-probe grid 1's checkpoints under different
pooling) are excluded -- they are the same trained weights as grid 1's ``ar``
and ``jepa_ema``, just read differently, and including them would double-count
two cells as four.

No numbers are invented: everything plotted is parsed out of the committed
Markdown tables, and the script fails loudly if a cell or reference model it
expects is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs/figures/pilot_grids_gain.png"

GRID_SUMMARIES: dict[str, Path] = {
    "grid1": REPO / "docs/experiments/2026-09-03-pilot-desynpuf/summary.md",
    "grid2": REPO / "docs/experiments/2026-09-03-pilot2-desynpuf/summary.md",
    "grid3": REPO / "docs/experiments/2026-09-04-pilot3-desynpuf/summary.md",
    "grid4": REPO / "docs/experiments/2026-09-04-pilot4-desynpuf/summary.md",
}

TASKS: tuple[str, ...] = (
    "inpatient_365d",
    "mortality_365d",
    "new_dx_365d/ckd",
    "new_dx_365d/copd",
    "new_dx_365d/diabetes",
    "new_dx_365d/heart_failure",
    "readmission_30d",
)

# Trained cells to plot, per grid -- excludes grid 2's reuse_checkpoint rows
# (ar_last, jepa_ema_mean), which retrain nothing and duplicate grid 1's ar
# and jepa_ema checkpoints under a different pooling.
CELLS_BY_GRID: dict[str, tuple[str, ...]] = {
    "grid1": (
        "ar",
        "jepa_ema",
        "jepa_ema_nosig",
        "jepa_shared_sig",
        "jepa_ema_future",
        "jepa_ema_block",
    ),
    "grid2": (
        "jepa_notime",
        "jepa_content",
        "jepa_content_recon",
        "jepa_recon",
        "jepa_content_future",
    ),
    "grid3": ("jepa_recon_notime", "recon_only", "recon_only_notime", "jepa_recon_nosig"),
    "grid4": (
        "nextlatent_h1",
        "nextlatent_h1416",
        "nextlatent_h1416_recon",
        "window_30_365",
        "window_30_365_recon",
    ),
}

# Each cell's own random_init control, as (grid, control_name) into that
# grid's own reference-model table.
CONTROL_OF: dict[str, tuple[str, str]] = {
    "ar": ("grid1", "random_init@ar"),
    "jepa_ema": ("grid1", "random_init@jepa_ema"),
    "jepa_ema_nosig": ("grid1", "random_init@jepa_ema"),
    "jepa_shared_sig": ("grid1", "random_init@jepa_ema"),
    "jepa_ema_future": ("grid1", "random_init@jepa_ema"),
    "jepa_ema_block": ("grid1", "random_init@jepa_ema"),
    "jepa_notime": ("grid2", "random_init@jepa_notime"),
    "jepa_content": ("grid2", "random_init@jepa_notime"),
    "jepa_content_recon": ("grid2", "random_init@jepa_notime"),
    "jepa_recon": ("grid2", "random_init@jepa_notime"),
    "jepa_content_future": ("grid2", "random_init@jepa_notime"),
    "jepa_recon_notime": ("grid3", "random_init@jepa_recon_notime"),
    "recon_only": ("grid3", "random_init@jepa_recon_notime"),
    "recon_only_notime": ("grid3", "random_init@jepa_recon_notime"),
    "jepa_recon_nosig": ("grid3", "random_init@jepa_recon_notime"),
    "nextlatent_h1": ("grid4", "random_init@nextlatent_h1"),
    "nextlatent_h1416": ("grid4", "random_init@nextlatent_h1"),
    "nextlatent_h1416_recon": ("grid4", "random_init@nextlatent_h1"),
    "window_30_365": ("grid4", "random_init@nextlatent_h1"),
    "window_30_365_recon": ("grid4", "random_init@nextlatent_h1"),
}

# Objective family per cell, for bar color and the legend.
FAMILY_OF: dict[str, str] = {
    "ar": "ar",
    "jepa_ema": "masked-span jepa",
    "jepa_ema_nosig": "masked-span jepa",
    "jepa_shared_sig": "masked-span jepa",
    "jepa_ema_future": "masked-span jepa",
    "jepa_ema_block": "masked-span jepa",
    "jepa_notime": "masked-span jepa",
    "jepa_content": "masked-span jepa",
    "jepa_content_future": "masked-span jepa",
    "jepa_content_recon": "jepa+recon",
    "jepa_recon": "jepa+recon",
    "jepa_recon_notime": "jepa+recon",
    "jepa_recon_nosig": "jepa+recon",
    "recon_only": "recon-only",
    "recon_only_notime": "recon-only",
    "nextlatent_h1": "nextlatent",
    "nextlatent_h1416": "nextlatent",
    "nextlatent_h1416_recon": "nextlatent",
    "window_30_365": "window",
    "window_30_365_recon": "window",
}

# Categorical slots from the project's validated palette (references/palette.md),
# the same one scripts/plot_pilot.py draws from: slot 1 blue = gbm, slot 2
# orange = lr, slot 3 aqua = ar (unchanged from plot_pilot.py); slots 5-8
# extend it to the four objective families plot_pilot.py never had to color.
COLOR_GBM = "#2a78d6"  # slot 1, blue
COLOR_LR = "#eb6834"  # slot 2, orange
FAMILY_COLOR: dict[str, str] = {
    "ar": "#1baf7a",  # slot 3, aqua
    "masked-span jepa": "#eda100",  # slot 4, yellow
    "jepa+recon": "#e87ba4",  # slot 5, magenta
    "recon-only": "#008300",  # slot 6, green
    "nextlatent": "#4a3aa7",  # slot 7, violet
    "window": "#e34948",  # slot 8, red
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


def load_cells(summaries: dict[str, Path]) -> list[dict]:
    """One record per plotted cell: name, grid, family, mean_gain, mean_abs."""
    runs = {g: _load(p, "| run |", "run") for g, p in summaries.items()}
    models = {g: _load(p, "| model |", "model") for g, p in summaries.items()}

    cells = []
    for grid, names in CELLS_BY_GRID.items():
        for name in names:
            scores = runs[grid][name]
            ctrl_grid, ctrl_name = CONTROL_OF[name]
            control = models[ctrl_grid][ctrl_name]
            mean_abs = _mean(scores)
            mean_gain = mean_abs - _mean(control)
            cells.append(
                {
                    "name": name,
                    "grid": grid,
                    "family": FAMILY_OF[name],
                    "mean_gain": mean_gain,
                    "mean_abs": mean_abs,
                }
            )
    baselines = {
        "gbm": _mean(models["grid1"]["gbm"]),
        "lr": _mean(models["grid1"]["lr"]),
        "ar_gain": next(c["mean_gain"] for c in cells if c["name"] == "ar"),
    }
    return cells, baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    cells, baselines = load_cells(GRID_SUMMARIES)
    cells.sort(key=lambda c: c["mean_gain"], reverse=True)

    names = [c["name"] for c in cells]
    colors = [FAMILY_COLOR[c["family"]] for c in cells]
    gains = [c["mean_gain"] for c in cells]
    abs_scores = [c["mean_abs"] for c in cells]
    y = np.arange(len(cells))[::-1]  # top row = first (highest-gain) cell

    fig, (ax_gain, ax_abs) = plt.subplots(1, 2, figsize=(13, 0.34 * len(cells) + 1.6), sharey=True)

    ax_gain.barh(y, gains, color=colors, edgecolor=INK, linewidth=0.5, height=0.72)
    ax_gain.axvline(0, color=MUTED, linewidth=1.0)
    ax_gain.axvline(
        baselines["ar_gain"],
        color=FAMILY_COLOR["ar"],
        linewidth=1.3,
        linestyle="--",
        zorder=0,
    )
    ax_gain.text(
        baselines["ar_gain"],
        len(cells) - 0.3,
        " ar reference",
        color=FAMILY_COLOR["ar"],
        fontsize=8,
        va="bottom",
        ha="left" if baselines["ar_gain"] >= 0 else "right",
    )
    ax_gain.set_xlabel("mean gain over own random_init control (AUROC)")
    ax_gain.set_yticks(y)
    ax_gain.set_yticklabels(names, fontsize=8.5)
    ax_gain.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_gain.set_axisbelow(True)

    ax_abs.barh(y, abs_scores, color=colors, edgecolor=INK, linewidth=0.5, height=0.72)
    ax_abs.axvline(baselines["gbm"], color=COLOR_GBM, linewidth=1.3, linestyle="--", zorder=0)
    ax_abs.axvline(baselines["lr"], color=COLOR_LR, linewidth=1.3, linestyle="--", zorder=0)
    ax_abs.text(
        baselines["gbm"],
        len(cells) - 0.3,
        " gbm",
        color=COLOR_GBM,
        fontsize=8,
        va="bottom",
        ha="left",
    )
    ax_abs.text(
        baselines["lr"],
        len(cells) - 1.3,
        " lr",
        color=COLOR_LR,
        fontsize=8,
        va="bottom",
        ha="left",
    )
    ax_abs.set_xlabel("mean absolute AUROC")
    ax_abs.set_xlim(0.45, 0.85)
    ax_abs.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_abs.set_axisbelow(True)

    for ax in (ax_gain, ax_abs):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)

    fig.suptitle(
        "DE-SynPUF sample 1, held-out 3,000 subjects, 48M-token grids 1-4 (4L/192d)",
        fontsize=11,
        color=INK,
        x=0.02,
        ha="left",
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=family)
        for family, color in FAMILY_COLOR.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02 - 0.012 * max(0, 10 - len(cells))),
        ncol=6,
        frameon=False,
        fontsize=8.5,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
