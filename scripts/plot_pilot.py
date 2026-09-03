"""``python scripts/plot_pilot.py`` -- render the pilot ablation grid to a PNG.

Reads the two Markdown tables written by :mod:`scripts.ablate` to
``docs/experiments/2026-09-03-pilot-desynpuf/summary.md`` (the ablation grid, and
the reference-model table of ``gbm``/``lr``/``random_init@<run>``) and draws one
grouped bar chart per held-out task: the two count-feature baselines, the ``ar``
run, its ``random_init`` control, the best-scoring ``jepa_*`` run, and *that*
run's ``random_init`` control.

Only ``ar`` and ``jepa_ema`` were probed with an untrained (``random_init``)
control in this grid, so the jepa control shown is always
``random_init@jepa_ema`` regardless of which ``jepa_*`` variant wins a given
task -- it is the architecture's untrained baseline, not a per-variant one.

No numbers are invented: everything plotted is parsed out of the committed
Markdown tables, and the script fails loudly if a task or run it expects is
missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO / "docs/experiments/2026-09-03-pilot-desynpuf/summary.md"
DEFAULT_OUT = REPO / "docs/figures/pilot_desynpuf_auroc.png"

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
    "inpatient_365d": "inpatient\n365d",
    "mortality_365d": "mortality\n365d",
    "new_dx_365d/ckd": "new dx\nckd",
    "new_dx_365d/copd": "new dx\ncopd",
    "new_dx_365d/diabetes": "new dx\ndiabetes",
    "new_dx_365d/heart_failure": "new dx\nheart failure",
    "readmission_30d": "readmission\n30d",
}

# Categorical slots 1-4 from the project's validated palette (references/palette.md).
COLOR_GBM = "#2a78d6"  # blue
COLOR_LR = "#eb6834"  # orange
COLOR_AR = "#1baf7a"  # aqua
COLOR_JEPA = "#eda100"  # yellow
COLOR_CONTROL = "#9a9890"  # neutral grey, hatched
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"


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


def load_summary(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return ``(ablation_runs, reference_models)``, each ``name -> {task: auroc}``."""
    text = path.read_text()
    ablation_rows = _parse_md_table(text, "| run |")
    reference_rows = _parse_md_table(text, "| model |")

    runs: dict[str, dict[str, float]] = {}
    for row in ablation_rows:
        name = _unbacktick(row["run"])
        runs[name] = {t: float(row[t]) for t in TASKS if row.get(t) not in (None, "", "--")}

    models: dict[str, dict[str, float]] = {}
    for row in reference_rows:
        name = _unbacktick(row["model"])
        models[name] = {t: float(row[t]) for t in TASKS if row.get(t) not in (None, "", "--")}

    return runs, models


def best_jepa_per_task(runs: dict[str, dict[str, float]]) -> dict[str, tuple[str, float]]:
    """For each task, the ``(run_name, auroc)`` of the best-scoring ``jepa*`` run."""
    jepa_runs = {name: scores for name, scores in runs.items() if name.startswith("jepa")}
    best: dict[str, tuple[str, float]] = {}
    for task in TASKS:
        candidates = [(name, s[task]) for name, s in jepa_runs.items() if task in s]
        best[task] = max(candidates, key=lambda kv: kv[1])
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    runs, models = load_summary(args.summary)
    best_jepa = best_jepa_per_task(runs)

    bar_defs = [
        ("gbm", COLOR_GBM, False),
        ("lr", COLOR_LR, False),
        ("ar", COLOR_AR, False),
        ("random_init@ar", COLOR_CONTROL, True),
        ("jepa*", COLOR_JEPA, False),
        ("random_init@jepa_ema", COLOR_CONTROL, True),
    ]
    n_bars = len(bar_defs)
    n_tasks = len(TASKS)
    width = 0.8 / n_bars
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (name, color, is_control) in enumerate(bar_defs):
        offsets = x - 0.4 + width * (i + 0.5)
        if name == "jepa*":
            heights = [best_jepa[t][1] for t in TASKS]
        elif name in models:
            heights = [models[name].get(t, np.nan) for t in TASKS]
        else:
            heights = [runs[name].get(t, np.nan) for t in TASKS]
        kwargs = dict(color=color, edgecolor=INK, linewidth=0.6)
        if is_control:
            kwargs.update(hatch="////", facecolor=color, edgecolor=MUTED)
        ax.bar(offsets, heights, width=width * 0.92, **kwargs)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=9)
    ax.set_ylabel("held-out AUROC (3,000 subjects, 200 bootstrap resamples)")
    ax.set_ylim(0.45, 0.85)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=":", zorder=0)
    ax.set_title(
        "DE-SynPUF sample 1, held-out 3,000 subjects, 48M-token pilot (4L/192d)",
        fontsize=11,
        color=INK,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GBM, label="gbm (count features)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_LR, label="lr (count features)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_AR, label="ar (next-code, trained)"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLOR_JEPA,
            edgecolor=INK,
            linewidth=0.6,
            label="best jepa_* (trained)",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLOR_CONTROL,
            edgecolor=MUTED,
            hatch="////",
            label="random_init control (untrained)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
