"""``python scripts/plot_a2.py`` -- render the A2 continuous-state test to one figure.

Reads the four ``summary.md`` files the A2 grids wrote
(``docs/experiments/{a2,a2ft}-physionet{2019,2012}/summary.md``) and draws one
panel per downstream task, four in total: ``sepsis_6h`` and ``sepsis_stay`` on
PhysioNet/CinC 2019, ``mortality_inhospital/24h`` and ``/48h`` on
PhysioNet/CinC 2012.

Each panel has one row per objective and two horizontal bars in that row:

* **solid** -- the frozen-probe 2-seed mean (``_s1``/``_s2``), with a
  black min-max whisker across the two seeds;
* **hatched** -- the end-to-end fine-tuned point on the seed-1 checkpoint,
  a single seed, so no whisker.

``ar_cont`` has a probe bar only: it is a control on the value term and was not
carried into the fine-tuning grids. The bottom row is the pair of untrained
controls -- ``random_init`` under the probe, ``ft_random`` under fine-tuning --
drawn in grey. Dashed vertical lines mark the count-feature ``gbm`` and ``lr``
baselines on that task's own held-out split.

Colors are the categorical slots ``scripts/plot_grids.py`` uses, so an
objective family keeps its color across the project's figures: aqua for the
AR family, violet for the hybrid (a ``nextlatent`` cell), magenta and red for
the two code-loss-free latent objectives.

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
DEFAULT_OUT = REPO / "docs/figures/a2_physionet.png"

# One panel per (source, task): the probe grid, the fine-tune grid, the task
# column name in both summaries, and the panel title.
PANELS: tuple[dict, ...] = (
    {
        "probe": "a2-physionet2019",
        "ft": "a2ft-physionet2019",
        "task": "sepsis_6h",
        "title": "sepsis_6h  (PhysioNet/CinC 2019, 1.7% prevalence)",
    },
    {
        "probe": "a2-physionet2019",
        "ft": "a2ft-physionet2019",
        "task": "sepsis_stay",
        "title": "sepsis_stay  (PhysioNet/CinC 2019, 5.6%)",
    },
    {
        "probe": "a2-physionet2012",
        "ft": "a2ft-physionet2012",
        "task": "mortality_inhospital/24h",
        "title": "mortality_inhospital/24h  (PhysioNet/CinC 2012, 13.9%)",
    },
    {
        "probe": "a2-physionet2012",
        "ft": "a2ft-physionet2012",
        "task": "mortality_inhospital/48h",
        "title": "mortality_inhospital/48h  (PhysioNet/CinC 2012, 13.9%)",
    },
)

# Plotted rows, top to bottom. ``ft`` False means the objective has no
# fine-tuning cell (``ar_cont``, a control on the value term).
OBJECTIVES: tuple[tuple[str, bool], ...] = (
    ("ar_bins", True),
    ("ar_cont", False),
    ("hybrid_bins", True),
    ("latent_cont", True),
    ("latent_only", True),
)

# Categorical slots from the project's validated palette, the same ones
# scripts/plot_grids.py draws from: slot 1 blue = gbm, slot 2 orange = lr,
# slot 3 aqua = the ar family, slot 7 violet = nextlatent (the hybrid), slot 5
# magenta and slot 8 red = the two code-loss-free latent objectives.
COLOR_GBM = "#2a78d6"
COLOR_LR = "#eb6834"
COLOR_OF: dict[str, str] = {
    "ar_bins": "#1baf7a",
    "ar_cont": "#1baf7a",
    "hybrid_bins": "#4a3aa7",
    "latent_cont": "#e87ba4",
    "latent_only": "#e34948",
}
COLOR_CONTROL = "#9a9892"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_COLOR = "#e1e0d9"

HATCH = "///"
XMIN, XMAX = 0.55, 0.90


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


def _table(grid: str, header_marker: str, key: str, task: str) -> dict[str, float]:
    """``{row name: AUROC on task}`` from one grid's ``summary.md``."""
    path = REPO / "docs/experiments" / grid / "summary.md"
    rows = _parse_md_table(path.read_text(), header_marker)
    out: dict[str, float] = {}
    for row in rows:
        if task not in row:
            raise SystemExit(f"{path}: no column {task!r}; found {sorted(row)}")
        cell = row[task]
        if cell in ("", "--"):
            continue
        out[row[key].strip("`")] = float(cell)
    return out


def load_panel(panel: dict) -> dict:
    """Probe seed pairs, fine-tuned points and reference lines for one panel."""
    task = panel["task"]
    probe = _table(panel["probe"], "| run |", "run", task)
    probe_ref = _table(panel["probe"], "| model |", "model", task)
    ft = _table(panel["ft"], "| run |", "run", task)
    ft_ref = _table(panel["ft"], "| model |", "model", task)

    rows = []
    for name, has_ft in OBJECTIVES:
        seeds = [probe[f"{name}_s{s}"] for s in (1, 2)]
        rows.append(
            {
                "name": name,
                "color": COLOR_OF[name],
                "seeds": seeds,
                "probe": float(np.mean(seeds)),
                "ft": ft[f"{name}_s1"] if has_ft else None,
            }
        )
    rows.append(
        {
            "name": "random_init / ft_random",
            "color": COLOR_CONTROL,
            "seeds": None,
            "probe": probe_ref["random_init@ar_bins_s1"],
            "ft": ft_ref["ft_random@ar_bins_s1"],
        }
    )
    return {
        "title": panel["title"],
        "rows": rows,
        "gbm": probe_ref["gbm"],
        "lr": probe_ref["lr"],
    }


def draw(ax, panel: dict, *, show_labels: bool) -> None:
    rows = panel["rows"]
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.34

    for yi, row in zip(y, rows, strict=True):
        ax.barh(
            yi + h / 2,
            row["probe"],
            height=h,
            color=row["color"],
            edgecolor=INK,
            linewidth=0.5,
        )
        if row["seeds"] is not None:
            lo, hi = min(row["seeds"]), max(row["seeds"])
            ax.plot([lo, hi], [yi + h / 2, yi + h / 2], color=INK, linewidth=1.0, zorder=3)
            for x in (lo, hi):
                ax.plot(
                    [x, x],
                    [yi + h / 2 - 0.09, yi + h / 2 + 0.09],
                    color=INK,
                    linewidth=1.0,
                    zorder=3,
                )
        if row["ft"] is not None:
            ax.barh(
                yi - h / 2,
                row["ft"],
                height=h,
                color=row["color"],
                edgecolor=INK,
                linewidth=0.5,
                hatch=HATCH,
            )
            ax.text(
                row["ft"] + 0.005,
                yi - h / 2,
                f"{row['ft']:.3f}",
                fontsize=6.6,
                color=MUTED,
                va="center",
            )
        label_x = row["probe"] if row["seeds"] is None else max(row["seeds"])
        ax.text(
            label_x + 0.005,
            yi + h / 2,
            f"{row['probe']:.3f}",
            fontsize=6.6,
            color=MUTED,
            va="center",
        )

    # When the two baselines are within a hair of each other (``sepsis_stay``:
    # lr 0.768, gbm 0.769) the labels would collide, so lr flips to the left
    # of its own line.
    crowded = abs(panel["gbm"] - panel["lr"]) < 0.01
    refs = (
        (panel["gbm"], COLOR_GBM, " gbm", "left"),
        (panel["lr"], COLOR_LR, "lr " if crowded else " lr", "right" if crowded else "left"),
    )
    for value, color, label, ha in refs:
        ax.axvline(value, color=color, linewidth=1.3, linestyle="--", zorder=0)
        ax.text(
            value,
            len(rows) - 0.42,
            label,
            color=color,
            fontsize=7.5,
            va="bottom",
            ha=ha,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows] if show_labels else [], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_xlim(XMIN, XMAX)
    ax.set_title(panel["title"], fontsize=9, color=INK, loc="left")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    panels = [load_panel(p) for p in PANELS]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.4), sharex=True)
    for ax, panel, show_labels in zip(
        axes.ravel(), panels, (True, False, True, False), strict=True
    ):
        draw(ax, panel, show_labels=show_labels)
    for ax in axes[1]:
        ax.set_xlabel("held-out AUROC")

    fig.suptitle(
        "A2: latent prediction on continuous ICU state -- frozen probe (solid, 2-seed mean, "
        "min-max whisker) and end-to-end fine-tuning (hatched, seed 1)",
        fontsize=10.5,
        color=INK,
        x=0.02,
        ha="left",
    )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#d8d7d1",
            edgecolor=INK,
            linewidth=0.5,
            label="frozen probe (2-seed mean)",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#d8d7d1",
            edgecolor=INK,
            linewidth=0.5,
            hatch=HATCH,
            label="fine-tuned (seed 1)",
        ),
        plt.Line2D([0], [0], color=COLOR_GBM, linestyle="--", label="gbm (count features)"),
        plt.Line2D([0], [0], color=COLOR_LR, linestyle="--", label="lr (count features)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
