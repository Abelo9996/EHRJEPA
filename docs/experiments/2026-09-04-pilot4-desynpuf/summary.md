# Ablation grid -- 2026-09-04-pilot4-desynpuf

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-04T10:46:20+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nextlatent_h1` | nextlatent | last@final | ema | 0.05 | -- | 2,930 | 48,005,120 | 0.07247 | 0.07074 | -- | -- | -- | 140.3 | -0.003873 | 14,892 | 1,588 | 0.6601 | 0.6108 | 0.6829 | 0.6854 | 0.7198 | 0.6867 | 0.6225 |
| `nextlatent_h1416` | nextlatent | last@final | ema | 0.05 | -- | 2,930 | 48,005,120 | 0.07623 | 0.07469 | -- | -- | -- | 124.3 | 0.002952 | 13,035 | 1,823 | 0.6687 | 0.6063 | 0.6886 | 0.6823 | 0.7198 | 0.6868 | 0.6401 |
| `nextlatent_h1416_recon` | nextlatent | last@final | ema | 0.05 | -- | 2,930 | 48,005,120 | 0.8393 | 0.2007 | 6.369 | 0.1782 | 0.3384 | 123.2 | -0.034 | 5,873 | 4,079 | 0.7417 | 0.6332 | 0.7531 | 0.7487 | 0.7618 | 0.7642 | 0.6985 |
| `window_30_365` | window | last@final | ema | 0.05 | -- | 2,930 | 48,005,120 | 0.04397 | 0.04247 | -- | -- | -- | 114.3 | 0.1757 | 16,223 | 1,462 | 0.6725 | 0.586 | 0.6859 | 0.7028 | 0.7279 | 0.6901 | 0.6259 |
| `window_30_365_recon` | window | last@final | ema | 0.05 | -- | 2,930 | 48,005,120 | 0.04568 | 0.04373 | -- | -- | -- | 114.7 | 0.1771 | 15,027 | 1,574 | 0.6764 | 0.5944 | 0.6805 | 0.7054 | 0.7274 | 0.6951 | 0.6209 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@nextlatent_h1` | 0.6469 | 0.5339 | 0.658 | 0.662 | 0.71 | 0.666 | 0.5781 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
