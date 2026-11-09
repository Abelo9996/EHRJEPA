"""``python scripts/plot_headline.py`` -- the headline per-task figure:
hybrid_final vs AR (both 1B tokens, 3 seeds, seed range shown) vs the GBM and
LR count-feature baselines, on the FULL 11,708-subject held-out split.

Every number is parsed from committed files -- nothing hard-coded:
* ``docs/experiments/scale1b-final-desynpuf/summary.md`` for the six
  ``hybrid_final_s{0,1,2}`` and ``ar_s{0,1,2}_full`` rows (the last seven
  columns are the seven task AUROCs);
* ``docs/experiments/scale1b-final-desynpuf/baselines.json`` for ``gbm``/``lr``.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/experiments/scale1b-final-desynpuf/summary.md"
BASELINES = ROOT / "docs/experiments/scale1b-final-desynpuf/baselines.json"

TASKS = ["inpatient_365d", "mortality_365d", "new_dx_365d/ckd", "new_dx_365d/copd",
         "new_dx_365d/diabetes", "new_dx_365d/heart_failure", "readmission_30d"]
LABELS = ["inpatient\n365d", "mortality\n365d", "new CKD\n365d", "new COPD\n365d",
          "new T2DM\n365d", "new HF\n365d", "readmit\n30d"]

COLOR_GBM = "#2a78d6"
COLOR_LR = "#9a8f80"
COLOR_AR = "#1baf7a"
COLOR_HYBRID = "#4a3aa7"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_COLOR = "#e1e0d9"


def parse_rows(prefix_test):
    """Return {run: [7 task AUROCs]} for rows whose name matches prefix_test."""
    out = {}
    for line in SUMMARY.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        run = cells[0]
        if not prefix_test(run):
            continue
        vals = cells[-7:]
        try:
            out[run] = [float(v) for v in vals]
        except ValueError:
            continue
    return out


def stack(rows):
    """rows: {run: [7]} -> (mean[7], lo[7], hi[7])."""
    a = np.array(list(rows.values()))
    return a.mean(0), a.min(0), a.max(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/headline_desynpuf.png")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    hy = parse_rows(lambda r: r.startswith("hybrid_final_s"))
    ar = parse_rows(lambda r: r.startswith("ar_s") and r.endswith("_full"))
    assert len(hy) == 3 and len(ar) == 3, f"expected 3 seeds each, got hy={len(hy)} ar={len(ar)}"
    base = json.loads(BASELINES.read_text())
    gbm = np.array([base["gbm"][t] for t in TASKS])
    lr = np.array([base["lr"][t] for t in TASKS])

    hy_m, hy_lo, hy_hi = stack(hy)
    ar_m, ar_lo, ar_hi = stack(ar)

    x = np.arange(len(TASKS))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.bar(x - 1.5 * w, gbm, w, label="GBM (counts)", color=COLOR_GBM, edgecolor=INK, linewidth=0.6)
    ax.bar(x - 0.5 * w, lr, w, label="LR (counts)", color=COLOR_LR, edgecolor=INK, linewidth=0.6)
    ax.bar(x + 0.5 * w, ar_m, w, label="AR (1B, 3 seeds)", color=COLOR_AR, edgecolor=INK, linewidth=0.6,
           yerr=[ar_m - ar_lo, ar_hi - ar_m], error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2.5})
    ax.bar(x + 1.5 * w, hy_m, w, label="EHR-JEPA hybrid (1B, 3 seeds)", color=COLOR_HYBRID, edgecolor=INK,
           linewidth=0.6, yerr=[hy_m - hy_lo, hy_hi - hy_m],
           error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2.5})

    # mark where hybrid beats AR
    for i in range(len(TASKS)):
        if hy_lo[i] > ar_hi[i]:  # non-overlapping seed ranges, hybrid higher
            ax.text(x[i] + 1.5 * w, hy_hi[i] + 0.004, "★", ha="center", va="bottom",
                    color=COLOR_HYBRID, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5, color=INK)
    ax.set_ylabel("held-out AUROC", fontsize=10, color=INK)
    ax.set_ylim(0.55, 0.82)
    ax.set_title("EHR-JEPA (hybrid) vs autoregressive vs count baselines — DE-SynPUF, 1B tokens, full held-out split",
                 fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    ax.text(0.0, -0.16, "★ = hybrid's 3-seed range clears AR's on that task (non-overlapping).",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    # also print the numbers for the README
    print("task, gbm, lr, ar_mean, hybrid_mean, hybrid-ar")
    for i, t in enumerate(TASKS):
        print(f"{t}, {gbm[i]:.4f}, {lr[i]:.4f}, {ar_m[i]:.4f}, {hy_m[i]:.4f}, {hy_m[i]-ar_m[i]:+.4f}")
    nonmort = [i for i, t in enumerate(TASKS) if t != "mortality_365d"]
    print(f"mean over 6 non-mortality: ar={ar_m[nonmort].mean():.4f} hybrid={hy_m[nonmort].mean():.4f} "
          f"delta={hy_m[nonmort].mean()-ar_m[nonmort].mean():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
