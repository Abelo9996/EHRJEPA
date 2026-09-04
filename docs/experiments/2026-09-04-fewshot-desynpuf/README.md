# 2026-09-04 — few-shot evaluation on DE-SynPUF sample 1

Same held-out cohort, anchors and bootstrap protocol as grids 1–5
(`docs/experiments/PILOT_RESULTS.md`), read at few-shot training sizes
instead of only the full training split: how much labeled data does each
model family need before its held-out AUROC stops moving?

Nothing in this README is a result — the numbers are in
[`results.md`](results.md) and [`results.json`](results.json).

## What ran

```bash
python -m ehrjepa.eval.run \
  --source desynpuf-s1 \
  --tasks all \
  --models lr,gbm,ckpt:runs/2026-09-03-pilot-desynpuf/ar/final.pt,\
ckpt:runs/2026-09-04-pilot5-seeds-desynpuf/ar_s1/final.pt,\
ckpt:runs/2026-09-04-pilot5-seeds-desynpuf/ar_s2/final.pt,\
ckpt:runs/2026-09-04-pilot4-desynpuf/nextlatent_h1416_recon/final.pt,\
ckpt:runs/2026-09-04-pilot5-seeds-desynpuf/hybrid_s1/final.pt,\
ckpt:runs/2026-09-04-pilot5-seeds-desynpuf/hybrid_s2/final.pt \
  --out docs/experiments/2026-09-04-fewshot-desynpuf/ \
  --eval-subject-limit 3000 --eval-subject-seed 0 \
  --bootstrap 200 --seed 0 --probe-features auto --device mps
```

One `ehrjepa.eval.run` invocation, 8 models × 7 tasks, on the same 3,000
held-out subject subset (`eval_subject_seed: 0`), same seeded anchors
(`tasks.ANCHOR_SEED`), same 200 bootstrap resamples, `probe_features: auto`
(resolves to `last@final` for every checkpoint here — all six are causal).
Ran in 422.6 s end to end on Apple Silicon MPS, commit `d64d7ab`. Every
checkpoint's embeddings (`emb__ckpt-{ar,ar_s1,ar_s2,nextlatent_h1416_recon,
hybrid_s1,hybrid_s2}__last__final.parquet`) and every task's count matrix
(`counts__<task>.npz`) were already cached under `data/eval_cache/desynpuf-s1/`
from grids 1–5, so this run embedded nothing new — the wall time is fitting
`lr`, `gbm`, and 6 linear probes, each at 4 few-shot sizes × 5 seeds, on top
of their full-data fit.

## The four families

| family | models | training seeds |
|---|---|---|
| `lr` | `lr` (count features) | — (deterministic) |
| `gbm` | `gbm` (count features) | — (deterministic) |
| `ar` | `ckpt:ar` (grid 1), `ckpt:ar_s1`, `ckpt:ar_s2` (grid 5) | 0, 1, 2 |
| `hybrid` | `ckpt:nextlatent_h1416_recon` (grid 4), `ckpt:hybrid_s1`, `ckpt:hybrid_s2` (grid 5) | 0, 1, 2 |

## Few-shot mechanics — what already existed vs. what this run used

`ehrjepa.eval.probe.few_shot` and `ehrjepa.eval.run.evaluate_task` already
implement the whole few-shot path — this run added no new flags or code:

- Sizes: `probe.FEW_SHOT_K = (32, 128, 512, None)` (`None` = the full train
  split). This report only tables `k=32`, `k=128`, and `k=all`, per the
  brief; `k=512` is in `results.json` for anyone who wants it.
- Seeds: `probe.FEW_SHOT_SEEDS = (0, 1, 2, 3, 4)` — 5 draws per `(task, k,
  checkpoint)`. `k=None` always takes every training row, so it only runs
  once regardless of seed count.
- **Only `spec.kind in ("lr", "probe")` gets few-shot fits**
  (`ehrjepa.eval.run.evaluate_task`, the `if few_shot and spec.kind in
  ("lr", "probe")` guard) — `gbm` is excluded. `baselines.fit_gbm` fits out
  of process and only returns predictions for the `x_predict` matrix it was
  given at fit time; `probe.few_shot` calls its `fit_fn` without ever
  passing `x_predict`, so a `gbm` few-shot fit would score against the wrong
  matrix. Rather than patch that path for this run, `gbm`'s k=32/k=128 cells
  in `results.md` are reported as unavailable, and `gbm`'s k=all cell is its
  ordinary full-data fit (a single point — no seed variation exists to
  report for a deterministic model fit once).

## How family numbers are built

`ar` and `hybrid` are three independently trained checkpoints each (grid
1/4's seed-0 cell plus grid 5's two reseeds). For each `(task, k)`:

1. Read each checkpoint's own few-shot row — already a mean over 5 few-shot
   seeds (`auroc_mean` in `results.json`).
2. Mean and population std of those 3 numbers is the family's row in
   `results.md`.

So the reported `±` for `ar`/`hybrid` is **training-seed spread**, not
few-shot-sampling spread (that variance is already averaged out inside each
checkpoint's own `auroc_mean`). `lr`'s `±` is different in kind — one model,
so it is the std over the 5 few-shot seeds directly, reported as-is from
`results.json`.

## Figure

`scripts/plot_fewshot.py` renders one panel per task, x = training size (log
scale, `k=32`, `k=128`, and the task's actual full-train-split size), y =
held-out AUROC, one line per family with a ±1 std band (`gbm` as an
unconnected marker at its single full-data point). Same categorical palette
as `scripts/plot_grids.py` / `scripts/plot_pilot.py`: blue = `gbm`, orange =
`lr`, aqua = `ar`, violet = `hybrid` (the `nextlatent` slot).

```bash
python scripts/plot_fewshot.py
```

Output: `docs/figures/fewshot_desynpuf.png`.

## Provenance

- Checkpoints: `docs/experiments/2026-09-03-pilot-desynpuf/` (`ar`),
  `docs/experiments/2026-09-04-pilot4-desynpuf/` (`nextlatent_h1416_recon`),
  `docs/experiments/2026-09-04-pilot5-seeds-desynpuf/` (`ar_s1`, `ar_s2`,
  `hybrid_s1`, `hybrid_s2`).
- Eval command: `src/ehrjepa/eval/run.py`.
- Few-shot implementation: `src/ehrjepa/eval/probe.py` (`few_shot`,
  `subsample`, `FEW_SHOT_K`, `FEW_SHOT_SEEDS`).
- Raw report: [`eval_report.md`](eval_report.md) (the standard
  `ehrjepa.eval.report` markdown for this run, unedited — every model's
  AUROC/AUPRC/Brier/calibration, paired bootstrap, and the full k=32/128/
  512/all few-shot table per checkpoint rather than per family).
