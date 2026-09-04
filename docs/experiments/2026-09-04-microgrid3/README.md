# 2026-09-04 — micro-grid 3: the pipeline test for `lambda_pred: 0`

**This is not an experiment.** It is the one-cell smoke test that had to pass
before [the third pilot grid](../2026-09-04-pilot3-desynpuf/README.md) was
launched, kept for the same reason micro-grid 2 was: "the runner worked once,
end to end, on this commit" is a claim worth being able to check.

```bash
python scripts/ablate.py configs/grids/micro3_desynpuf.yaml
```

One cell, `micro3_lambda_pred0`, 50 steps: grid 3's `recon_only_notime` row
(`objective.lambda_pred: 0`, `objective.lambda_recon: 0.1`,
`predictor.mask_token_time: false`, `model.target_mode: ema`) at 1.7% of its
budget. It combines the two paths worth checking at the pilot's real batch and
window rather than in a unit test: the target-encoder-forward skip running
under `target_mode: ema` (the EMA modules exist but the run never calls or
updates them), and the chunked reconstruction cross-entropy as the only term
carrying gradient into the predictor.

50 steps is 819,200 nominal token slots, 1.7% of a grid-3 cell's budget.
**The AUROCs are what a barely-initialised encoder scores** and are not a
result.

## What it establishes

* **the target encoder is never called.** `ema_momentum` is logged `nan` on
  every row of `metrics.csv` — the EMA-update guard in
  `ehrjepa.train.pretrain.Trainer.train` held for both logged steps, so
  `update_ema` never ran. Unit coverage
  (`tests/test_models.py::test_compute_targets_false_never_calls_the_target_encoder`)
  pins this with a call counter on `target_encoder.forward`; this run is the
  same claim at 64x256 on MPS instead of a synthetic batch.
* **`pred_loss` logs `NaN`, not a number computed against a placeholder.**
  Both logged steps (25, 50) show `pred_loss = nan` in `metrics.csv`, exactly as
  documented in `JEPAObjective.forward`.
* **the total loss is exactly `lambda_recon * recon_loss + lambda_sigreg *
  (sigreg_tokens + sigreg_cls)`, with no term for the latent prediction.** At
  step 50: `recon_loss = 9.6073513`, `sigreg_tokens = 0.0225951`,
  `sigreg_cls = 0.0402607`, and
  `0.1 * 9.6073513 + 0.05 * (0.0225951 + 0.0402607) = 0.9638779`, which is
  `metrics.csv`'s logged `loss` to seven digits.
* **the reconstruction head still learns with the latent term gone.**
  `recon_loss` (chunked code cross-entropy) fell from 10.0325 at step 25 to
  9.6074 at step 50, off the `log(30000) = 10.31` floor, in 25 steps — the
  auxiliary head is attached to something that learns even with nothing pulling
  on the latent itself.
* **the runner, model and eval harness go end to end on this commit.** Training
  finished in 59.6s wall, `auto` pooling resolved `mean@final` for this
  bidirectional cell, and the full seven-task evaluation plus a `random_init`
  control completed and landed in `summary.md`/`summary.json`.

## Results

Full table in [`summary.md`](summary.md); the AUROC columns for the cell and
its control:

| run | probe | steps | loss | pred | recon | inpatient | mortality | ckd | copd | diabetes | heart failure | readmission |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `micro3_lambda_pred0` | `mean@final` | 50 | 0.9639 | `nan` | 9.6074 | 0.6758 | 0.5762 | 0.6893 | 0.6954 | 0.7228 | 0.6870 | 0.6488 |
| `random_init@micro3_lambda_pred0` | `mean@final` | — | — | — | — | 0.6775 | 0.5699 | 0.6962 | 0.6961 | 0.7252 | 0.6882 | 0.6619 |

At 50 steps the checkpoint and its untrained control are within noise of each
other on every task, which is expected at 1.7% of budget and says nothing
about whether `lambda_pred: 0` moves anything at the full 48M-token budget —
that question belongs to grid 3.

This run shared the M4's one GPU with the phase-5a grid throughout, so `tok/s`
(7,337) and `wall_s` describe a contended machine, not a clean measurement.

## Setup

| | |
|---|---|
| base config | `configs/pretrain_pilot.yaml` — 4x192 encoder, 2x96 predictor, `max_len` 256, batch 64 |
| grid | `configs/grids/micro3_desynpuf.yaml` |
| budget | 819,200 nominal token slots = 50 steps x 64 x 256 |
| data | `data/cache/desynpuf-s1`, `train` split, 78,997 subjects, vocabulary 30,000 |
| evaluation | 3,000-subject `held_out` cut (seed 0), 200 bootstrap resamples, `probe_features: auto`, layer `final`, no few-shot |
| baselines | `lr`/`gbm` reused from `../2026-09-03-eval-desynpuf/predictions.parquet`; `random_init` computed once against `micro3_lambda_pred0`'s architecture |
| hardware | Apple M4, 16 GB, MPS, float32, torch 2.14 — **shared with the phase-5a grid throughout** |

Results: [`summary.md`](summary.md), `summary.json`, `baselines.json`, and
per-cell evaluation output under `eval/micro3_lambda_pred0/`.
