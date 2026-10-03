# Ablation grid -- mimic-hybrid-ar-100m

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-30T11:19:44+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s0` | probe | ar | last@final | -- | -- | -- | -- | -- | 3,052 | 100,007,936 | 1.21 | -- | 1.21 | 0.74 | 0.9049 | 207.6 | 0 | 60,123 | 925.8 | 0.6543 | 0.9097 | 0.8402 | 0.7328 | 0.7845 | 0.8411 | 0.6018 |
| `hybrid_s0` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.344 | 0.2181 | 1.259 | 0.7388 | 0.8975 | 210.9 | 0.0691 | 47,130 | 1,180 | 0.6557 | 0.9003 | 0.8417 | 0.7209 | 0.7816 | 0.8286 | 0.63 |
| `ar_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 3,052 | 100,007,936 | 1.236 | -- | 1.236 | 0.7451 | 0.9035 | 210.3 | 0 | 60,318 | 922.3 | 0.6435 | 0.9059 | 0.8524 | 0.7122 | 0.7959 | 0.8476 | 0.6037 |
| `hybrid_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.3519 | 0.2229 | 1.29 | 0.7349 | 0.8973 | 213.1 | 0.06323 | 47,152 | 1,179 | 0.6742 | 0.8976 | 0.8324 | 0.7469 | 0.7905 | 0.8453 | 0.6291 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |
| `random_init@ar_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
