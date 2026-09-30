"""``python scripts/plot_mimic_result.py`` -- the MIMIC-IV preliminary result:
per-task held-out AUROC for the hybrid vs the AR baseline (each averaged over
whatever seeds finished, seed range shown) vs the GBM/LR count baselines.

Parses ``docs/experiments/mimic-hybrid-ar/summary.md`` (the ablate.py output):
* rows whose name starts ``hybrid`` / ``ar_s`` -> the seven task AUROCs (last 7
  columns);
* the ``## Reference models`` table -> ``gbm`` / ``lr``.
Nothing is hard-coded; if only seed 0 finished, it plots single-seed (no range).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/experiments/mimic-hybrid-ar/summary.md"

TASKS = ["inpatient_365d", "mortality_365d", "new_dx_365d/ckd", "new_dx_365d/copd",
         "new_dx_365d/diabetes", "new_dx_365d/heart_failure", "readmission_30d"]
LABELS = ["inpatient\n365d", "mortality\n365d", "new CKD\n365d", "new COPD\n365d",
          "new T2DM\n365d", "new HF\n365d", "readmit\n30d"]
COLOR_GBM = "#2a78d6"; COLOR_LR = "#9a8f80"; COLOR_AR = "#1baf7a"; COLOR_HYBRID = "#4a3aa7"
INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e1e0d9"


def parse_task_rows(name_test):
    out = {}
    for line in SUMMARY.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        run = cells[0]
        if not name_test(run):
            continue
        try:
            out[run] = [float(v) for v in cells[-7:]]
        except ValueError:
            continue
    return out


def parse_baseline(model):
    for line in SUMMARY.read_text().splitlines():
        if line.startswith(f"| `{model}`"):
            cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            try:
                return np.array([float(v) for v in cells[-7:]])
            except ValueError:
                return None
    return None


def stack(rows):
    a = np.array(list(rows.values()))
    return a.mean(0), a.min(0), a.max(0), a.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/mimic_hybrid_vs_ar.png")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    hy = parse_task_rows(lambda r: r.startswith("hybrid"))
    ar = parse_task_rows(lambda r: r.startswith("ar_s") or r == "ar")
    assert hy and ar, f"no rows parsed (hy={len(hy)} ar={len(ar)}) -- has the grid produced summary rows yet?"
    hy_m, hy_lo, hy_hi, hy_n = stack(hy)
    ar_m, ar_lo, ar_hi, ar_n = stack(ar)
    gbm = parse_baseline("gbm"); lr = parse_baseline("lr")

    x = np.arange(len(TASKS)); w = 0.2
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    if gbm is not None:
        ax.bar(x - 1.5 * w, gbm, w, label="GBM (counts)", color=COLOR_GBM, edgecolor=INK, linewidth=0.6)
    if lr is not None:
        ax.bar(x - 0.5 * w, lr, w, label="LR (counts)", color=COLOR_LR, edgecolor=INK, linewidth=0.6)
    ax.bar(x + 0.5 * w, ar_m, w, label=f"AR ({ar_n} seed{'s' if ar_n > 1 else ''})", color=COLOR_AR,
           edgecolor=INK, linewidth=0.6, yerr=[ar_m - ar_lo, ar_hi - ar_m] if ar_n > 1 else None,
           error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2.5})
    ax.bar(x + 1.5 * w, hy_m, w, label=f"EHR-JEPA hybrid ({hy_n} seed{'s' if hy_n > 1 else ''})",
           color=COLOR_HYBRID, edgecolor=INK, linewidth=0.6,
           yerr=[hy_m - hy_lo, hy_hi - hy_m] if hy_n > 1 else None,
           error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2.5})
    for i in range(len(TASKS)):
        if hy_m[i] > ar_m[i]:
            ax.text(x[i] + 1.5 * w, hy_hi[i] + 0.004, "▲", ha="center", va="bottom",
                    color=COLOR_HYBRID, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8.5, color=INK)
    ax.set_ylabel("held-out AUROC", fontsize=10, color=INK)
    lo = min(0.5, float(np.nanmin([ar_lo.min(), hy_lo.min()])) - 0.03)
    ax.set_ylim(max(0.45, lo), 0.95)
    nonmort = [i for i, t in enumerate(TASKS) if t != "mortality_365d"]
    delta = hy_m[nonmort].mean() - ar_m[nonmort].mean()
    ax.set_title(f"MIMIC-IV v3.1 (~30k subjects, 20M tokens) — EHR-JEPA hybrid vs AR "
                 f"[mean-of-6 Δ = {delta:+.3f}]", fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    ax.text(0.0, -0.16, "▲ = hybrid > AR on that task. Preliminary: subject subset + short budget.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    print("task, gbm, lr, ar, hybrid, hybrid-ar")
    for i, t in enumerate(TASKS):
        g = gbm[i] if gbm is not None else float("nan"); l = lr[i] if lr is not None else float("nan")
        print(f"{t}, {g:.4f}, {l:.4f}, {ar_m[i]:.4f}, {hy_m[i]:.4f}, {hy_m[i]-ar_m[i]:+.4f}")
    print(f"mean-of-6 non-mortality: ar={ar_m[nonmort].mean():.4f} hybrid={hy_m[nonmort].mean():.4f} "
          f"delta={delta:+.4f}  (ar_seeds={ar_n}, hybrid_seeds={hy_n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
