# Ablation grid -- 2026-09-03-microgrid

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `cls_mean@final`.

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-03T16:41:42+00:00.

| run | objective | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `micro_ar` | ar | -- | -- | -- | 200 | 3,276,800 | 7.486 | -- | 7.486 | 0.09345 | 0.2064 | 89.18 | 0 | 7,227 | 224.5 | 0.6938 | 0.6135 | 0.7052 | 0.6991 | 0.7375 | 0.704 | 0.6385 |
| `micro_jepa_ema` | jepa | ema | 0.05 | 0.6 | 200 | 3,276,800 | 0.2105 | 0.2094 | -- | -- | -- | 143.7 | 0.08986 | 9,011 | 183.5 | 0.6701 | 0.6162 | 0.6918 | 0.6906 | 0.7148 | 0.6896 | 0.6267 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@micro_ar` | 0.6723 | 0.5913 | 0.7004 | 0.7002 | 0.7372 | 0.6993 | 0.6522 |
| `random_init@micro_jepa_ema` | 0.6723 | 0.5913 | 0.7004 | 0.7002 | 0.7372 | 0.6993 | 0.6522 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
