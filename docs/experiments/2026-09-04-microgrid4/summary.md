# Ablation grid -- 2026-09-04-microgrid4

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-04T07:07:17+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `micro4_nextlatent` | nextlatent | last@final | ema | 0.05 | -- | 50 | 819,200 | 1.319 | 0.3404 | 9.607 | 0.0726 | 0.1267 | 153.2 | -0.0004674 | 269.8 | 810.9 | 0.6509 | 0.5451 | 0.6594 | 0.6634 | 0.7136 | 0.6697 | 0.5654 |
| `micro4_window` | window | last@final | ema | 0.05 | -- | 50 | 819,200 | 0.388 | 0.3217 | -- | -- | -- | 155.7 | 0.001321 | 6,747 | 68.7 | 0.6424 | 0.5395 | 0.6486 | 0.6647 | 0.7035 | 0.6613 | 0.5737 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@micro4_nextlatent` | 0.6469 | 0.5339 | 0.658 | 0.662 | 0.71 | 0.666 | 0.5781 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
