"""``python scripts/plot_mimic.py`` -- MIMIC-IV v3.1 dataset characterization.

Reads ``docs/experiments/mimic-iv/stats.json`` (row counts computed directly
from the v3.1 .csv.gz files) and renders a two-panel scale figure:
  A: table row counts (log), hosp vs icu -- chartevents/labevents dominate.
  B: DE-SynPUF vs MIMIC-IV contrast -- MIMIC adds a lab/vital modality
     (158M lab events) DE-SynPUF entirely lacks.
DE-SynPUF numbers are the committed train-split figures (11.3M events, 0 labs).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "docs/experiments/mimic-iv/stats.json"

COLOR_HOSP = "#2a78d6"
COLOR_ICU = "#eda100"
COLOR_DESYN = "#9a8f80"
COLOR_MIMIC = "#4a3aa7"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"

DESYNPUF_EVENTS = 11_300_000   # train-split events (README/synthesis)
DESYNPUF_LABS = 0             # DE-SynPUF has no labs/vitals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/mimic_iv_scale.png")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()
    d = json.loads(STATS.read_text())
    tables = d["tables"]
    coh = d["cohort"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.55, 1]})

    # --- Panel A: top-14 tables by row count ---
    items = sorted(tables.items(), key=lambda kv: kv[1], reverse=True)[:14][::-1]
    names = [k.split("/")[1] for k, _ in items]
    vals = [v for _, v in items]
    colors = [COLOR_ICU if k.startswith("icu/") else COLOR_HOSP for k, _ in items]
    y = np.arange(len(names))
    axA.barh(y, vals, color=colors, edgecolor=INK, linewidth=0.5)
    axA.set_yticks(y)
    axA.set_yticklabels(names, fontsize=8.5, color=INK)
    axA.set_xscale("log")
    axA.set_xlabel("rows (log scale)", fontsize=9.5, color=INK)
    axA.set_title("MIMIC-IV v3.1 — table scale", fontsize=11, color=INK)
    for yi, v in zip(y, vals):
        axA.text(v * 1.15, yi, f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K",
                 va="center", fontsize=7.5, color=MUTED)
    axA.set_xlim(1e4, 1e9)
    axA.grid(axis="x", color=GRID, linewidth=0.8)
    axA.set_axisbelow(True)
    from matplotlib.patches import Patch
    axA.legend(handles=[Patch(color=COLOR_HOSP, label="hosp"), Patch(color=COLOR_ICU, label="icu")],
               frameon=False, fontsize=8.5, loc="lower right")
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # --- Panel B: DE-SynPUF vs MIMIC contrast ---
    groups = ["total events", "lab / vital events"]
    x = np.arange(len(groups))
    w = 0.36
    desyn = [DESYNPUF_EVENTS, DESYNPUF_LABS + 1]  # +1 so log-bar renders
    mimic = [coh["total_events"], tables["hosp/labevents"] + tables["icu/chartevents"]]
    axB.bar(x - w / 2, desyn, w, label="DE-SynPUF (stages so far)", color=COLOR_DESYN, edgecolor=INK, linewidth=0.6)
    axB.bar(x + w / 2, mimic, w, label="MIMIC-IV v3.1 (stage B)", color=COLOR_MIMIC, edgecolor=INK, linewidth=0.6)
    axB.set_yscale("log")
    axB.set_xticks(x)
    axB.set_xticklabels(groups, fontsize=9, color=INK)
    axB.set_ylabel("events (log scale)", fontsize=9.5, color=INK)
    axB.set_title("Why stage B: real labs & vitals", fontsize=11, color=INK)
    axB.set_ylim(1, 3e9)
    axB.grid(axis="y", color=GRID, linewidth=0.8)
    axB.set_axisbelow(True)
    axB.legend(frameon=False, fontsize=8, loc="upper right")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.annotate("DE-SynPUF: 0 labs", xy=(1 - w / 2, 1.4), fontsize=7.5, color=COLOR_DESYN,
                 ha="center", va="bottom", rotation=90)
    axB.text(0.5, -0.19,
             f"MIMIC-IV: {coh['patients']:,} patients · {coh['admissions']:,} admissions · "
             f"{coh['icustays']:,} ICU stays · {coh['total_events']/1e6:.0f}M events",
             transform=axB.transAxes, fontsize=8, color=MUTED, ha="center")

    fig.suptitle("MIMIC-IV v3.1 acquired & characterized — staged for the first real-EHR pretraining (stage B)",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
