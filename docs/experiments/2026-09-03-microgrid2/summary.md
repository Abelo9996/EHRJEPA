# Ablation grid -- 2026-09-03-microgrid2

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-03T21:15:41+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `micro2_all_flags` | jepa | mean@final | ema | 0.05 | 0.6 | 100 | 1,638,400 | 1.113 | 0.2413 | -- | -- | -- | 142.4 | 0.00618 | 5,599 | 179.4 | 0.6762 | 0.5573 | 0.6943 | 0.6948 | 0.7227 | 0.6878 | 0.6612 |
| `micro2_ar_last` | ar | last@final | -- | -- | -- | 2,930 | 48,005,120 | 6.318 | -- | 6.318 | 0.1809 | 0.3459 | 117.5 | 0 | 7,428 | 0 | 0.7404 | 0.5951 | 0.7508 | 0.7613 | 0.7627 | 0.7714 | 0.6738 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@micro2_all_flags` | 0.6775 | 0.5699 | 0.6962 | 0.6961 | 0.7252 | 0.6882 | 0.6619 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
