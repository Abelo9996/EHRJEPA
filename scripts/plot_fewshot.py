"""``python scripts/plot_fewshot.py`` -- few-shot AUROC vs. training-set size, DE-SynPUF.

Defaults to the 1B-final, full-held-out-split run:
``docs/experiments/fewshot-1b-final-desynpuf/results.json`` (written by
:mod:`ehrjepa.eval.run` with ``--models
lr,gbm,ckpt:<hybrid_final seeds 0/1/2>,ckpt:<ar seeds 0/1/2>``). Pass
``--results docs/experiments/2026-09-04-fewshot-desynpuf/results.json`` to
regenerate the earlier pilot-era figure (3,000-subject held-out subset,
48M-token checkpoints) instead. Draws one panel per task: x = few-shot
training size (``k`` positives + ``k`` negatives, log scale, with the full
training split plotted at its own size), y = held-out AUROC, one line per
model family (``lr``, ``ar``, ``hybrid``/``hybrid_final``), ``gbm`` as an
unconnected marker at its single full-data point.

A family's point at one ``k`` is the mean over its checkpoints' own few-shot
mean (each already an average over the 5 few-shot subsample seeds
:func:`ehrjepa.eval.probe.few_shot` draws); the shaded band is +/- one
standard deviation of those per-checkpoint means, i.e. seed-to-seed spread
for ``lr`` (one model, so the band is the few-shot-seed std reported
directly) and training-seed spread for ``ar``/``hybrid`` (three checkpoints
each). ``gbm`` has no few-shot line: :func:`ehrjepa.eval.run.evaluate_task`
only fits the few-shot subsamples for ``kind in ("lr", "probe")``, so ``gbm``
carries only its full-data point, plotted alone with a marker and no band.

The hybrid checkpoint names differ between runs (``ckpt:hybrid_final_s0/s1/s2``
for the 1B-final run vs. ``ckpt:nextlatent_h1416_recon``/``hybrid_s1``/
``hybrid_s2`` for the pilot run); which set to read is auto-detected from
the model names present in the loaded ``results.json``.

No numbers are invented: everything plotted is read out of the committed
``results.json`` -- the script fails loudly if a model or task it expects is
missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "docs/experiments/fewshot-1b-final-desynpuf/results.json"
PILOT_RESULTS = REPO / "docs/experiments/2026-09-04-fewshot-desynpuf/results.json"
DEFAULT_OUT = REPO / "docs/figures/fewshot_desynpuf.png"

TASKS: tuple[str, ...] = (
    "inpatient_365d",
    "mortality_365d",
    "new_dx_365d/ckd",
    "new_dx_365d/copd",
    "new_dx_365d/diabetes",
    "new_dx_365d/heart_failure",
    "readmission_30d",
)

FEW_SHOT_KS: tuple[int | None, ...] = (32, 128, 512, None)

# Model names in results.json per family, keyed by which run produced the
# file. lr and gbm are single models; ar and hybrid are averaged over their
# three training seeds. Auto-detected in main() from the models actually
# present.
FAMILY_MODELS_FINAL1B: dict[str, tuple[str, ...]] = {
    "lr": ("lr",),
    "gbm": ("gbm",),
    "ar": ("ckpt:ar", "ckpt:ar_s1", "ckpt:ar_s2"),
    "hybrid_final": ("ckpt:hybrid_final_s0", "ckpt:hybrid_final_s1", "ckpt:hybrid_final_s2"),
}
FAMILY_MODELS_PILOT: dict[str, tuple[str, ...]] = {
    "lr": ("lr",),
    "gbm": ("gbm",),
    "ar": ("ckpt:ar", "ckpt:ar_s1", "ckpt:ar_s2"),
    "hybrid": ("ckpt:nextlatent_h1416_recon", "ckpt:hybrid_s1", "ckpt:hybrid_s2"),
}

# Same categorical slots as scripts/plot_grids.py / scripts/plot_pilot.py:
# slot 1 blue = gbm, slot 2 orange = lr, slot 3 aqua = ar, slot 7 violet =
# the nextlatent/hybrid family (hybrid_final reuses the same slot).
FAMILY_COLOR: dict[str, str] = {
    "gbm": "#2a78d6",
    "lr": "#eb6834",
    "ar": "#1baf7a",
    "hybrid": "#4a3aa7",
    "hybrid_final": "#4a3aa7",
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_COLOR = "#e1e0d9"


def _few_shot_points(model_entry: dict) -> dict[int | None, dict]:
    """``{k: row}`` from one model's ``few_shot`` list, keyed by ``k``."""
    return {row["k"]: row for row in model_entry.get("few_shot", [])}


def _family_series(
    task_entry: dict, models: tuple[str, ...]
) -> tuple[list[float], list[float], list[float]]:
    """``(x, mean, std)`` at each of ``FEW_SHOT_KS``.

    ``x`` uses each checkpoint's own reported ``n_train`` (identical across
    checkpoints for a given ``k`` and task, since every model in a run scores
    the same anchor frame) so the "all" point sits at the task's actual
    training-set size rather than an arbitrary sentinel.
    """
    xs, means, stds = [], [], []
    for k in FEW_SHOT_KS:
        vals = []
        n_train = None
        for name in models:
            model_entry = task_entry["models"].get(name)
            if model_entry is None:
                raise KeyError(f"model {name!r} missing from results.json for this task")
            points = _few_shot_points(model_entry)
            row = points.get(k)
            if row is None or not np.isfinite(row.get("auroc_mean", float("nan"))):
                continue
            vals.append(row["auroc_mean"])
            n_train = row["n_train"]
        if not vals:
            continue
        xs.append(n_train)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else vals[0] * 0.0)
    return xs, means, stds


def _gbm_point(task_entry: dict) -> tuple[float, float] | None:
    """The full-data AUROC for ``gbm``, which has no few-shot fits at all."""
    model_entry = task_entry["models"].get("gbm")
    if model_entry is None:
        return None
    n_train = task_entry.get("counts", {}).get("train")
    auroc = model_entry.get("metrics", {}).get("auroc", {}).get("point")
    if n_train is None or auroc is None:
        return None
    return float(n_train), float(auroc)


def _resolve_family_models(tasks: dict) -> dict[str, tuple[str, ...]]:
    """Pick the family/model map matching the checkpoint names in ``tasks``.

    The 1B-final run names its hybrid checkpoints ``ckpt:hybrid_final_s*``;
    the pilot run names them ``ckpt:nextlatent_h1416_recon``/``hybrid_s1``/
    ``hybrid_s2``. Detected from whichever set is present in the first task
    entry so ``--results`` alone is enough to switch datasets.
    """
    any_task = next(iter(tasks.values()))
    models_present = set(any_task.get("models", {}))
    if "ckpt:hybrid_final_s0" in models_present:
        return FAMILY_MODELS_FINAL1B
    if "ckpt:nextlatent_h1416_recon" in models_present:
        return FAMILY_MODELS_PILOT
    raise KeyError(
        "results.json has neither the 1B-final hybrid checkpoints "
        "(ckpt:hybrid_final_s0/s1/s2) nor the pilot ones "
        "(ckpt:nextlatent_h1416_recon/hybrid_s1/hybrid_s2) -- "
        f"models present: {sorted(models_present)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=(
            "path to a results.json from ehrjepa.eval.run. Defaults to the "
            f"1B-final, full-held-out-split run ({DEFAULT_RESULTS.relative_to(REPO)}). "
            f"Pass --results {PILOT_RESULTS.relative_to(REPO)} for the pilot-era "
            "figure (3,000-subject subset, 48M-token checkpoints)."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    tasks = results["tasks"]
    missing = [t for t in TASKS if t not in tasks]
    if missing:
        raise KeyError(f"results.json is missing tasks: {missing}")
    family_models = _resolve_family_models(tasks)
    eval_subject_limit = results.get("eval_subject_limit")
    held_out_desc = (
        f"held-out {eval_subject_limit:,} subjects" if eval_subject_limit else "full held-out split"
    )

    ncols = 4
    nrows = -(-len(TASKS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, task in zip(flat_axes, TASKS, strict=False):
        task_entry = tasks[task]
        for family, models in family_models.items():
            if family == "gbm":
                point = _gbm_point(task_entry)
                if point is not None:
                    ax.scatter(
                        [point[0]],
                        [point[1]],
                        color=FAMILY_COLOR["gbm"],
                        marker="D",
                        s=28,
                        zorder=3,
                        label="gbm" if ax is flat_axes[0] else None,
                    )
                continue
            xs, means, stds = _family_series(task_entry, models)
            if not xs:
                continue
            xs_a, means_a, stds_a = map(np.array, (xs, means, stds))
            order = np.argsort(xs_a)
            xs_a, means_a, stds_a = xs_a[order], means_a[order], stds_a[order]
            color = FAMILY_COLOR[family]
            ax.plot(
                xs_a,
                means_a,
                color=color,
                linewidth=1.4,
                marker="o",
                markersize=3.5,
                label=family if ax is flat_axes[0] else None,
                zorder=3,
            )
            ax.fill_between(
                xs_a,
                means_a - stds_a,
                means_a + stds_a,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        ax.set_xscale("log")
        ax.set_title(task, fontsize=9, color=INK)
        ax.grid(axis="both", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(labelsize=7.5)

    for ax in flat_axes[len(TASKS) :]:
        ax.axis("off")

    fig.supxlabel("training examples (k positives + k negatives, or the full split)", fontsize=9.5)
    fig.supylabel("held-out AUROC", fontsize=9.5)
    fig.suptitle(
        f"DE-SynPUF sample 1, {held_out_desc} -- few-shot AUROC by model family",
        fontsize=11,
        color=INK,
    )

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(family_models),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
