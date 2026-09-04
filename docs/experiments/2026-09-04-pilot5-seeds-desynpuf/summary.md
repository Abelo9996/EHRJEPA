# Ablation grid -- 2026-09-04-pilot5-seeds-desynpuf

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-04T18:55:01+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s1` | ar | last@final | -- | -- | -- | 2,930 | 48,005,120 | 6.321 | -- | 6.321 | 0.1718 | 0.3486 | 116.4 | 0 | 9,488 | 2,502 | 0.7392 | 0.6094 | 0.7419 | 0.7592 | 0.7574 | 0.7562 | 0.689 |
| `hybrid_s1` | nextlatent | last@final | ema | 0.05 | -- | 2,930 | 48,005,120 | 0.8282 | 0.1902 | 6.36 | 0.1706 | 0.3425 | 121.7 | -0.02869 | 4,770 | 5,286 | 0.7479 | 0.6072 | 0.7543 | 0.7544 | 0.7621 | 0.7584 | 0.6996 |
| `ar_s2` | ar | last@final | -- | -- | -- | 2,930 | 48,005,120 | 5.881 | -- | 5.881 | 0.1888 | 0.394 | 116.3 | 0 | 9,240 | 2,566 | 0.7455 | 0.6057 | 0.7514 | 0.7606 | 0.7652 | 0.7703 | 0.6844 |
| `hybrid_s2` | nextlatent | last@final | ema | 0.05 | -- | 2,930 | 48,005,120 | 0.8134 | 0.2191 | 5.93 | 0.1877 | 0.386 | 124 | -0.03733 | 6,639 | 3,572 | 0.7438 | 0.6254 | 0.7501 | 0.7514 | 0.7689 | 0.7683 | 0.6926 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@ar_s1` | 0.6469 | 0.5339 | 0.658 | 0.662 | 0.71 | 0.666 | 0.5781 |
| `random_init@hybrid_s1` | 0.6469 | 0.5339 | 0.658 | 0.662 | 0.71 | 0.666 | 0.5781 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
