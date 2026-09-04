# 2026-09-04 — micro-grid 4: the pipeline test for the two causal latent objectives

**This is not an experiment.** It is the smoke test that had to pass before
[the fourth pilot grid](../2026-09-04-pilot4-desynpuf/README.md) was queued, kept
for the same reason micro-grids 2 and 3 were: "the runner worked once, end to
end, on this commit" is a claim worth being able to check.

```bash
python scripts/ablate.py configs/grids/micro4_desynpuf.yaml
```

Two cells, 50 steps each — one per new `objective.kind`, both in their
`lambda_recon` variant because that builds strictly more than the plain one:

| cell | objective | overrides |
|---|---|---|
| `micro4_nextlatent` | `nextlatent` | `horizons: [1, 4, 16]`, `lambda_recon: 0.1`, `target_mode: ema` |
| `micro4_window` | `window` | `window_horizons: [30, 365]`, `window_anchors: 8`, `lambda_recon: 0.1`, `target_mode: ema` |

50 steps is 819,200 nominal token slots, 1.7% of a grid-4 cell's budget.
**The AUROCs are what a barely-initialised encoder scores** and are not a result.

## What it establishes

* **Both objectives run end to end on MPS at the pilot's real batch shape**
  (64 × 256), through training, checkpointing, and the full seven-task
  evaluation with a `random_init` control.
* **The probe rebuilds each class from the checkpoint.** `auto` pooling resolved
  to `last@final` for both cells, which is the causal branch of
  `ehrjepa.eval.probe.default_features` — the new `model_config` fields
  (`build_predictor`, `horizons`, `window_horizons`) round-tripped through
  `torch.save` and `load_encoder` rebuilt `EHRNextLatent` / `EHRWindowLatent`
  with `strict=True` state loading.
* **The logged total is exactly the sum of its terms.** At step 50,
  `nextlatent`: `0.3404 + 0.05 × (0.3216 + 0.0352) + 0.1 × 9.6071 = 1.3190`,
  which is the logged `loss` to four digits. `window`:
  `0.3217 + 0.05 × (0.3026 + 0.0566) + 0.1 × 0.4831 = 0.3880`, likewise.
* **The AR term inside `nextlatent` is learning.** Its `ce` fell from 10.1081 at
  step 25 to 9.6071 at step 50, off the `log(30000) = 10.31` floor, with `top10`
  rising 0.067 → 0.127 in 25 steps.
* **The window skip rule fires at the rate predicted before the run.** Offline,
  on 512 sampled windows at `max_len: 256` with 8 anchors, 71% of anchors had an
  observed non-empty 30-day horizon and 47% a 365-day one — a mean kept fraction
  of 0.59. The run logged `skipped_frac` 0.360 at step 25 and 0.411 at step 50.
* **The multi-label head has something to predict.**
  `positives_per_anchor` — *distinct* codes inside the horizon, not events —
  averaged 18.9 over the two horizons' kept anchors.
* **One `random_init` control does cover both architectures.**
  `random_init@micro4_nextlatent` scored 0.6469 / 0.5339 / 0.6580 / 0.6620 /
  0.7100 / 0.6660 / 0.5781, which is `random_init@ar_last` from grid 2 to four
  decimal places on all seven tasks. An untrained causal 4×192 encoder is the
  same encoder whichever head is bolted on after it, so grid 4 computes this
  control once.

## Peak memory, which is why this run measures at the real batch shape

| cell | peak (MB) | tok/s | wall (s) |
|---|---|---|---|
| `micro4_nextlatent` | 7,863.9 | 269.8¹ | 811 |
| `micro4_window` | 3,296.3 | 6,747 | 69 |

Both are under the 9 GB ceiling this machine was budgeted against, so grid 4
runs at batch 64 with no gradient accumulation. `nextlatent` is the expensive
one for the same reason the AR baseline was (5.5 GB): the chunked 30,000-way
softmax over every scored position, here on top of a second (EMA) encoder pass
and three horizon heads.

¹ The two `tok/s` numbers are not comparable to each other or to anything else:
this run shared the M4's one GPU with the phase-5c grid throughout, and
`micro4_nextlatent` overlapped that grid's evaluation phase. `peak_memory_mb` is
per-process and is not affected.

## Results

Full table in [`summary.md`](summary.md).

| run | probe | steps | loss | pred | recon | inpatient | mortality | ckd | copd | diabetes | heart failure | readmission |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `micro4_nextlatent` | `last@final` | 50 | 1.3190 | 0.3404 | 9.6071 | 0.6509 | 0.5451 | 0.6594 | 0.6634 | 0.7136 | 0.6697 | 0.5654 |
| `micro4_window` | `last@final` | 50 | 0.3880 | 0.3217 | 0.4831 | 0.6424 | 0.5395 | 0.6486 | 0.6647 | 0.7035 | 0.6613 | 0.5737 |
| `random_init@micro4_nextlatent` | `last@final` | — | — | — | — | 0.6469 | 0.5339 | 0.6580 | 0.6620 | 0.7100 | 0.6660 | 0.5781 |

At 50 steps both checkpoints are within noise of the untrained control on every
task, which is what 1.7% of budget buys and says nothing about whether either
objective moves anything at 48M tokens. That question belongs to grid 4.

## Setup

| | |
|---|---|
| base config | `configs/pretrain_pilot.yaml` — 4×192 encoder, `max_len` 256, batch 64 |
| grid | `configs/grids/micro4_desynpuf.yaml` |
| budget | 819,200 nominal token slots = 50 steps × 64 × 256 per cell |
| data | `data/cache/desynpuf-s1`, `train` split, 78,997 subjects, vocabulary 30,000 |
| parameters | `nextlatent` 8,012,848 trainable (embedding 5,892,288 / encoder 1,781,056 / heads 222,336 / AR head 30,384); `window` 7,778,224 (heads 74,496 / BCE head 30,384) |
| evaluation | 3,000-subject `held_out` cut (seed 0), 200 bootstrap resamples, `probe_features: auto`, layer `final`, no few-shot |
| baselines | `lr`/`gbm` reused from `../2026-09-03-eval-desynpuf/predictions.parquet`; `random_init` computed once against `micro4_nextlatent`'s architecture |
| hardware | Apple M4, 16 GB, MPS, float32, torch 2.14 — **shared with the phase-5c grid throughout** |

Results: [`summary.md`](summary.md), `summary.json`, `baselines.json`, and
per-cell evaluation output under `eval/`.
