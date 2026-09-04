# 2026-09-04 — phase-5e seed spread on DE-SynPUF

Grids 1–4 report each cell at one seed. Every gap they show — AR's +5 over its
own control, the hybrid's whatever-it-turns-out-to-be over AR — is a single
draw, and nothing in those grids says how much of that gap is the objective
and how much is which seed happened to be used. This grid trains the two most
load-bearing cells so far, `ar` (grid 1) and `nextlatent_h1416_recon` (grid
4's hybrid, cell 3), at two more seeds each, same everything else.

**Question, stated before the runs started: what is the seed-to-seed spread of
held-out AUROC for `ar` and for the hybrid at this budget, and are the
differences grids 1–4 report bigger than that spread?** A gap between AR and
the hybrid that is smaller than the spread across `ar_s0`/`ar_s1`/`ar_s2` (or
across the hybrid's three seeds) is not a finding about the objective — it is
noise this grid exists to name.

Nothing in this README is a result. The numbers land in
[`summary.md`](summary.md) and `summary.json`, appended by `scripts/ablate.py`
as each cell finishes.

## The cells

| run | design | seed | source |
|---|---|---|---|
| `ar` (grid 1) | next-code AR | 0 | `docs/experiments/2026-09-03-pilot-desynpuf/` |
| `ar_s1` | next-code AR | 1 | this grid |
| `ar_s2` | next-code AR | 2 | this grid |
| `nextlatent_h1416_recon` (grid 4) | AR+JEPA hybrid | 0 | `docs/experiments/2026-09-04-pilot4-desynpuf/` |
| `hybrid_s1` | AR+JEPA hybrid | 1 | this grid |
| `hybrid_s2` | AR+JEPA hybrid | 2 | this grid |

`ar_s1`/`ar_s2` overrides are grid 1's `ar` cell verbatim
(`objective.kind: ar`, `model.causal: true`, `model.tie_embeddings: true`)
plus `run.seed`. `hybrid_s1`/`hybrid_s2` overrides are grid 4's
`nextlatent_h1416_recon` verbatim (`objective.kind: nextlatent`,
`model.causal: true`, `model.target_mode: ema`,
`objective.horizons: [1, 4, 16]`, `objective.lambda_sigreg: 0.05`,
`objective.lambda_recon: 0.1`) plus `run.seed`. Same base config
(`configs/pretrain_pilot.yaml`), same 48M-token budget, same source
(`desynpuf-s1`), same held-out subset, same 200 bootstrap resamples, same
`probe_features: auto` (resolves to `last@final` for both — both
architectures are causal).

Run order in the grid file, and the order cells train in:
`ar_s1, hybrid_s1, ar_s2, hybrid_s2`.

## What `run.seed` controls, and what stays fixed

`Trainer.__init__` (`src/ehrjepa/train/pretrain.py`) calls
`seed_everything(run.seed)` before the model is constructed and before the
dataset, mask generator or dataloader are built. One override therefore seeds
the model's weight init *and* every training-time stream (the mask and
diagnostics generators are `run.seed + 1` / `run.seed + 2`, the dataloader is
`run.seed + 3`) — there is no separate "model init seed" to set, and none was
added.

Everything else that could confound a seed comparison is a grid-level field,
so it is one value shared by all four cells in this file, not four
independent draws:

- **Held-out subject subset**: `eval_subject_seed: 0` (below) is the seed
  `restrict_eval_split` hashes subject IDs against
  (`src/ehrjepa/eval/run.py`). It is unrelated to `run.seed` and is set once
  for the grid, so `ar_s1`, `hybrid_s1`, `ar_s2` and `hybrid_s2` — and grids
  1–4 — all score the same 3,000 subjects. The runner has no per-cell knob
  for this field, so there is nothing to override per run; stating it
  explicitly here is the whole guarantee.
- **Anchors**: drawn from `tasks.ANCHOR_SEED = 20260903`
  (`src/ehrjepa/eval/tasks.py`), a module constant. No grid field or CLI flag
  reaches it — `scripts/ablate.py` never passes an anchor seed to
  `ehrjepa.eval.run` — so it is identical across every cell of every grid
  this repo has run, this one included. Documented here rather than
  overridden, because there is nothing in the runner to override.
- **Bootstrap / probe-fit seed**: the grid-level `seed: 0` is the `--seed`
  `scripts/ablate.py` hands to `ehrjepa.eval.run` for every cell
  (`eval_one`), which seeds the 200 bootstrap resamples and the `lr`/`gbm`/
  linear-probe fits. Fixed at 0 for the whole grid, same as grids 1–4, so the
  eval side of a row never differs because of which seed trained the
  checkpoint.

## The `random_init` control

`control_runs: [ar_s1, hybrid_s1]` — two controls, one per architecture, not
four. `load_encoder` (`src/ehrjepa/eval/probe.py`) seeds an untrained
control's weights from the eval `--seed` (grid-level, fixed at 0 above), not
from the training run's `run.seed`; the only thing that changes a control is
which checkpoint's architecture it copies. `ar_s1` and `ar_s2` build the same
4×192 causal, tied-embedding, no-recon-head stack; `hybrid_s1` and `hybrid_s2`
build the same stack plus the tied recon head `lambda_recon: 0.1` adds.
Computing a second `ar` control off `ar_s2`, or a second hybrid control off
`hybrid_s2`, would repeat the identical `torch.manual_seed(0)` draw of the
identical shape — not a second control, the first one under a different
cache key. So: **the runner keys `random_init` by architecture, not by
training seed, and this grid reuses the one control each architecture gets**
— `random_init@ar_s1` for both AR cells (and, read against grid 1's own
`random_init@ar` control, for grid 1's `ar` row too), `random_init@hybrid_s1`
for both hybrid cells (and, read against grid 4's `random_init@nextlatent_h1`,
for grid 4's hybrid row).

## How to read this table once it fills in

Compute the AUROC range across `{ar (grid 1), ar_s1, ar_s2}` per task, and
across `{nextlatent_h1416_recon (grid 4), hybrid_s1, hybrid_s2}` per task.
Those two ranges are the seed-to-seed spread this grid exists to measure. A
per-task gap between the AR family and the hybrid family that is smaller than
either range is not attributable to the objective at this budget. This file
does not compute or report that comparison — it states the question and the
protocol that makes the comparison answerable from `summary.json` once all
four cells finish.

## Provenance

- Grid file: `configs/grids/pilot5_seeds_desynpuf.yaml`
- Base config: `configs/pretrain_pilot.yaml`
- Cells being reseeded: `docs/experiments/2026-09-03-pilot-desynpuf/` (`ar`),
  `docs/experiments/2026-09-04-pilot4-desynpuf/` (`nextlatent_h1416_recon`)
- Log: `runs/2026-09-04-pilot5-seeds-desynpuf/ablate.log`
- Queued behind grid 4's `scripts/ablate.py` process via
  `scripts/queue_after.sh`, since one M4 has one GPU.
