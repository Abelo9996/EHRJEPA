"""``python scripts/plot_ablate2.py`` -- render the ablate2-desynpuf figure:
model size and hybrid-knob effects on held-out AUROC.

Reads ``docs/experiments/ablate2-desynpuf/summary.md`` -- nothing plotted is a
number not already committed in that table -- and builds each cell's actual
``EHRAR`` / ``EHRNextLatent`` module from its recorded config overrides
(``configs/pretrain_scale.yaml`` plus the ``overrides`` this script applies,
matching ``scripts/ablate.py``'s own construction) to get real trainable
parameter counts, rather than a hand-typed estimate.

Two panels:

* **left** -- mean AUROC across the six non-mortality tasks vs. trainable
  parameter count (log $x$), one line per objective (``ar``, ``hybrid``)
  across the three sizes (small 4L/192d, base 6L/256d, large 8L/384d); the
  small and large points carry a vertical min-max error bar across their two
  seeds, the base point (seed 0 only, re-scored from ``scale-desynpuf``) has
  none.
* **right** -- the base-size hybrid-knob comparison (default, shared target,
  no SIGReg, horizon-1-only, recon 0.3) as horizontal bars, mean AUROC across
  the six non-mortality tasks, with min-max whiskers across the two seeds
  (the default row, seed 0 only, has none).

Same palette as ``scripts/plot_grids.py`` and ``scripts/plot_scale.py``:
``ar`` aqua, ``hybrid`` (``nextlatent`` family) violet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SUMMARY = REPO / "docs/experiments/ablate2-desynpuf/summary.md"
BASE_CONFIG = REPO / "configs/pretrain_scale.yaml"
DEFAULT_OUT = REPO / "docs/figures/ablate2_desynpuf.png"

VOCAB_SIZE = 30_000  # data/cache/desynpuf-s1/meta.json: vocab_size

TASKS: tuple[str, ...] = (
    "inpatient_365d",
    "mortality_365d",
    "new_dx_365d/ckd",
    "new_dx_365d/copd",
    "new_dx_365d/diabetes",
    "new_dx_365d/heart_failure",
    "readmission_30d",
)
TASKS6: tuple[str, ...] = tuple(t for t in TASKS if t != "mortality_365d")

# Same categorical slots as scripts/plot_grids.py / scripts/plot_scale.py.
COLOR_AR = "#1baf7a"  # slot 3, aqua
COLOR_HYBRID = "#4a3aa7"  # slot 7, violet
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_COLOR = "#e1e0d9"

# (size label, ar overrides, hybrid overrides) -- the same overrides
# configs/grids/ablate2_desynpuf.yaml applies for each cell, minus run.seed
# (parameter count does not depend on seed).
_SMALL = {
    "model.dim": 192,
    "model.depth": 4,
    "model.heads": 4,
    "model.pred_dim": 96,
    "model.pred_depth": 2,
    "model.pred_heads": 4,
}
_LARGE = {
    "model.dim": 384,
    "model.depth": 8,
    "model.heads": 6,
    "run.batch_size": 32,
    "optim.accum_steps": 2,
}
_AR_COMMON = {"objective.kind": "ar", "model.causal": True, "model.tie_embeddings": True}
_HYBRID_COMMON = {
    "objective.kind": "nextlatent",
    "model.causal": True,
    "model.target_mode": "ema",
    "objective.horizons": [1, 4, 16],
    "objective.lambda_recon": 0.1,
    "objective.lambda_sigreg": 0.05,
}

SIZE_OVERRIDES: dict[str, tuple[dict, dict]] = {
    "small": ({**_AR_COMMON, **_SMALL}, {**_HYBRID_COMMON, **_SMALL}),
    "base": (dict(_AR_COMMON), dict(_HYBRID_COMMON)),
    "large": ({**_AR_COMMON, **_LARGE}, {**_HYBRID_COMMON, **_LARGE}),
}

# Run-name pairs (seed 1, seed 2) per size, or a single seed-0 name for base.
SIZE_RUNS: dict[str, tuple[str, ...]] = {
    "small": ("ar_small_s1", "ar_small_s2", "hybrid_small_s1", "hybrid_small_s2"),
    "base": ("ar_base_s0", "hybrid_base_s0"),
    "large": ("ar_large_s1", "ar_large_s2", "hybrid_large_s1", "hybrid_large_s2"),
}

KNOB_ROWS: list[tuple[str, str | None]] = [
    ("default (base_s0)", "hybrid_base_s0"),
    ("shared target", "hybrid_shared"),
    ("no SIGReg", "hybrid_nosig"),
    ("horizon [1] only", "hybrid_h1"),
    ("recon 0.3", "hybrid_recon03"),
]


def _parse_md_table(text: str, header_marker: str) -> list[dict[str, str]]:
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


def load_runs() -> dict[str, dict[str, float]]:
    text = SUMMARY.read_text()
    rows = _parse_md_table(text, "| run |")
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        name = _unbacktick(row["run"])
        out[name] = {t: float(row[t]) for t in TASKS if row.get(t) not in (None, "", "--")}
    return out


def mean6(scores: dict[str, float]) -> float:
    return float(np.mean([scores[t] for t in TASKS6]))


def n_params(overrides: dict) -> int:
    """Trainable parameter count for a model built from these overrides,
    matching how scripts/ablate.py / src/ehrjepa/train/pretrain.py build it."""
    import yaml

    from ehrjepa.models.ar import EHRAR
    from ehrjepa.models.latent import LATENT_MODELS
    from ehrjepa.train.config import load_config

    ov_strings = [f"{k}={yaml.safe_dump(v).strip()}" for k, v in overrides.items()]
    cfg = load_config(BASE_CONFIG, ov_strings)
    model_config = cfg.model_config(VOCAB_SIZE)
    model_config.causal = True
    if overrides["objective.kind"] == "ar":
        model = EHRAR(model_config)
    else:
        model = LATENT_MODELS[overrides["objective.kind"]](model_config)
    return int(model.n_parameters()["trainable"])


def size_points(
    runs: dict[str, dict[str, float]],
) -> dict[str, list[tuple[int, float, float, float]]]:
    """{family: [(params, mean, lo, hi), ...]} across the three sizes, in
    small -> base -> large order."""
    out: dict[str, list[tuple[int, float, float, float]]] = {"ar": [], "hybrid": []}
    for size in ("small", "base", "large"):
        ar_ov, hybrid_ov = SIZE_OVERRIDES[size]
        for family, ov in (("ar", ar_ov), ("hybrid", hybrid_ov)):
            params = n_params(ov)
            names = [n for n in SIZE_RUNS[size] if n.startswith(family)]
            means = [mean6(runs[n]) for n in names]
            mean = float(np.mean(means))
            lo, hi = min(means), max(means)
            out[family].append((params, mean, lo, hi))
    return out


def knob_points(runs: dict[str, dict[str, float]]) -> list[tuple[str, float, float, float]]:
    """[(label, mean, lo, hi), ...] for the knob comparison, base-size hybrid."""
    out: list[tuple[str, float, float, float]] = []
    for label, name in KNOB_ROWS:
        if name == "hybrid_base_s0":
            v = mean6(runs[name])
            out.append((label, v, v, v))
        else:
            means = [mean6(runs[f"{name}_s1"]), mean6(runs[f"{name}_s2"])]
            out.append((label, float(np.mean(means)), min(means), max(means)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    runs = load_runs()
    sizes = size_points(runs)
    knobs = knob_points(runs)

    fig, (ax_size, ax_knob) = plt.subplots(1, 2, figsize=(13, 5.2))

    for family, color in (("ar", COLOR_AR), ("hybrid", COLOR_HYBRID)):
        pts = sizes[family]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax_size.plot(xs, ys, marker="o", color=color, linewidth=1.8, markersize=6, label=family)
        lo_err = [y - lo for _, y, lo, _ in pts]
        hi_err = [hi - y for _, y, _, hi in pts]
        ax_size.errorbar(
            xs,
            ys,
            yerr=[lo_err, hi_err],
            fmt="none",
            ecolor=color,
            elinewidth=1.3,
            capsize=4,
            zorder=3,
        )
    ax_size.set_xscale("log")
    ax_size.set_xlabel("trainable parameters (log scale)")
    ax_size.set_ylabel("mean AUROC, 6 non-mortality tasks")
    ax_size.set_title("Mean AUROC vs. parameter count: ar vs. hybrid", fontsize=10, color=INK)
    ax_size.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_size.set_axisbelow(True)
    ax_size.legend(frameon=False, fontsize=8.5, loc="lower right")

    labels = [k[0] for k in knobs]
    means = [k[1] for k in knobs]
    lo_err = [m - lo for _, m, lo, _ in knobs]
    hi_err = [hi - m for _, m, _, hi in knobs]
    y = np.arange(len(knobs))[::-1]
    colors = [COLOR_HYBRID if lbl != "default (base_s0)" else MUTED for lbl in labels]
    ax_knob.barh(y, means, color=colors, edgecolor=INK, linewidth=0.5, height=0.6)
    ax_knob.errorbar(
        means,
        y,
        xerr=[lo_err, hi_err],
        fmt="none",
        ecolor=INK,
        elinewidth=1.2,
        capsize=4,
        zorder=3,
    )
    ax_knob.set_yticks(y)
    ax_knob.set_yticklabels(labels, fontsize=9)
    ax_knob.set_xlabel("mean AUROC, 6 non-mortality tasks")
    ax_knob.set_xlim(0.74, 0.77)
    ax_knob.set_title("Base-size (6L/256d) hybrid-knob comparison, 2-seed", fontsize=10, color=INK)
    ax_knob.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_knob.set_axisbelow(True)

    for ax in (ax_size, ax_knob):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)

    fig.suptitle(
        "DE-SynPUF sample 1, full held-out split (11.7k subjects), 200M tokens/cell",
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
