# 2026-09-03 — micro-grid 2: the pipeline test for the phase-5b flags

**This is not an experiment.** It is the two-cell smoke test that had to pass
before [the second pilot grid](../2026-09-03-pilot2-desynpuf/README.md) was
launched, kept because "the runner worked once, end to end, on this commit" is a
claim worth being able to check.

```bash
python scripts/ablate.py configs/grids/micro2_desynpuf.yaml
```

Two cells:

* **`micro2_all_flags`** — 100 steps with every flag any grid-2 cell uses turned
  on at once, plus `predictor.mask_token_time: false`. A superset of every single
  grid-2 cell, so a cell that trains with all of them trains with any subset. The
  two paths worth smoke-testing at *scale* rather than in a unit test are the
  span-only target encoder's gather/scatter on MPS and the chunked reconstruction
  softmax's peak memory; both are exercised here at the pilot's real batch and
  window. (`objective.recon_value` is the one flag not exercised: grid 2 does not
  use it, and it is covered by `tests/test_objectives.py`.)
* **`micro2_ar_last`** — trains nothing. It re-scores grid 1's `ar` checkpoint
  through `reuse_checkpoint:`, which is the mechanism grid 2's last two rows use.

100 steps is 1,638,400 nominal token slots, 3.4% of a grid-2 cell's budget.
**The `micro2_all_flags` AUROCs are what a barely-initialised encoder scores**,
and the two rows are not comparable to each other — one is a 1.6M-token cell
sitting next to a fully trained 48M-token checkpoint, present only to prove the
runner can place it there.

Both cells ran while the phase-5a grid was still training on the same GPU, so the
`tok/s` and `wall_s` figures are contended and describe nothing.

## What it establishes

* **the new flags train.** `micro2_all_flags` completed 100 steps with
  content-only span-only targets, a time-free mask token, 30% time-feature
  dropout and `lambda_recon: 0.1`, at **7.09 GB peak** (against 3.3 GB for a
  plain JEPA cell) and 5,599 tok/s under contention. The reconstruction
  cross-entropy fell from 10.14 at step 25 to 8.69 at step 100, off the
  `log(30000) = 10.31` floor, so the auxiliary head is attached to something that
  learns;
* **`reuse_checkpoint` works.** `micro2_ar_last` skipped training, read
  `steps=2930` and `tokens=48,005,120` out of grid 1's own `config.json` rather
  than out of this grid's base, took its `loss`/`ce`/`top1` from the `metrics.csv`
  beside the checkpoint, recorded `wall_s` 0, and wrote nothing into grid 1's
  directory;
* **`auto` pooling resolves per checkpoint.** The bidirectional cell was probed
  at `mean@final` and the causal one at `last@final`, in the same grid, from one
  setting — and each row's `probe` column says which;
* **the control follows the architecture *and* the pooling.**
  `random_init@micro2_all_flags` is an untrained 4×192 bidirectional encoder
  probed at `mean@final`, not grid 1's `cls_mean@final` control;
* the count-baseline reuse still reproduces the phase-4 `lr`/`gbm` numbers
  exactly (`gbm` .744 / .572 / .765 / .767 / .772 / .789 / .672).

## Results

Full table in [`summary.md`](summary.md); the AUROC columns for the two cells and
the control:

| run | probe | steps | inpatient | mortality | ckd | copd | diabetes | heart failure | readmission |
|---|---|---|---|---|---|---|---|---|---|
| `micro2_all_flags` | `mean@final` | 100 | 0.6762 | 0.5573 | 0.6943 | 0.6948 | 0.7227 | 0.6878 | 0.6612 |
| `random_init@micro2_all_flags` | `mean@final` | — | 0.6775 | 0.5699 | 0.6962 | 0.6961 | 0.7252 | 0.6882 | 0.6619 |
| `micro2_ar_last` | `last@final` | 2,930 | 0.7404 | 0.5951 | 0.7508 | 0.7613 | 0.7627 | 0.7714 | 0.6738 |

## One thing the pipeline test found

The re-evaluation gives the first full-cohort measurement of the pooling change
that motivated it. Against grid 1's `cls_mean@final` row for the *same* `ar`
checkpoint, on the same 3,000-subject cut:

| task | `cls_mean@final` | `last@final` | Δ (points) |
|---|---|---|---|
| `inpatient_365d` | 0.7265 | 0.7404 | **+1.39** |
| `mortality_365d` | 0.6060 | 0.5951 | −1.09 |
| `new_dx_365d/ckd` | 0.7586 | 0.7508 | −0.78 |
| `new_dx_365d/copd` | 0.7552 | 0.7613 | +0.61 |
| `new_dx_365d/diabetes` | 0.7590 | 0.7627 | +0.37 |
| `new_dx_365d/heart_failure` | 0.7437 | 0.7714 | **+2.77** |
| `readmission_30d` | 0.6946 | 0.6738 | −2.07 |
| **mean** | | | **+0.17** |

`last` wins four of seven and the mean difference is +0.17 points, not the
2.5–3.3 that the diagnostic reported. Whatever subset that figure was measured
on, it is not what the full seven-task held-out cohort shows. The pooling default
still changes — `last` is the row a causal encoder actually summarises history
into, and `cls_mean`'s CLS half is provably a constant column there, which is an
argument from the architecture rather than from these seven numbers — but the
*size* of the effect claimed for it does not survive measurement, and grid 2's
`ar_last` and `jepa_ema_mean` rows exist precisely so this is separable from the
objective changes rather than folded into them.

No bootstrap interval is quoted for those deltas because none was computed: the
two evaluations are paired on the same anchors, so the paired bootstrap in each
run's `results.json` is the right instrument and it was not run across grids.

## Setup

| | |
|---|---|
| base config | `configs/pretrain_pilot.yaml` — 4×192 encoder, 2×96 predictor, `max_len` 256, batch 64 |
| grid | `configs/grids/micro2_desynpuf.yaml` |
| budget | 1,638,400 nominal token slots for the trained cell = 100 steps × 64 × 256 |
| data | `data/cache/desynpuf-s1`, `train` split, 78,997 subjects, vocabulary 30,000 |
| evaluation | 3,000-subject `held_out` cut (seed 0), 200 bootstrap resamples, `probe_features: auto`, layer `final`, no few-shot |
| baselines | `lr`/`gbm` reused from `../2026-09-03-eval-desynpuf/predictions.parquet`; `random_init` computed once per (architecture, pooling) |
| hardware | Apple M4, 16 GB, MPS, float32, torch 2.14 — **shared with the phase-5a grid throughout** |

Results: [`summary.md`](summary.md), `summary.json`, `baselines.json`, and
per-cell evaluation output under `eval/<run>/`.
