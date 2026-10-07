# Ablation grid -- mimic-scale-1b

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-07T12:13:12+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s0` | probe | ar | last@final | -- | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.7538 | -- | 0.7538 | 0.8117 | 0.9619 | 224.8 | 0 | 60,324 | 9,184 | 0.6594 | 0.918 | 0.8963 | 0.7619 | 0.8485 | 0.8648 | 0.5989 |
| `hybrid_s0` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2375 | 0.1472 | 0.896 | 0.7817 | 0.9468 | 213.7 | 0.01649 | 45,486 | 12,179 | 0.7228 | 0.9347 | 0.9142 | 0.7494 | 0.8897 | 0.8737 | 0.6401 |
| `ar_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.7596 | -- | 0.7596 | 0.8101 | 0.9632 | 224.1 | 0 | 58,325 | 2,678 | 0.6497 | 0.9162 | 0.9082 | 0.7439 | 0.8502 | 0.8612 | 0.6063 |
| `hybrid_s1` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2341 | 0.1443 | 0.8921 | 0.7832 | 0.9477 | 216.5 | 0.01802 | 44,816 | 2,702 | 0.7128 | 0.9325 | 0.9065 | 0.7367 | 0.8952 | 0.8715 | 0.6241 |
| `ar_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.7942 | -- | 0.7942 | 0.7975 | 0.9601 | 224.8 | 0 | 59,112 | 2,655 | 0.6528 | 0.9133 | 0.8933 | 0.7302 | 0.8558 | 0.869 | 0.5998 |
| `hybrid_s2` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2403 | 0.1473 | 0.9239 | 0.7751 | 0.9449 | 216.3 | 0.01833 | 45,346 | 12,218 | 0.7144 | 0.9309 | 0.9103 | 0.7419 | 0.8869 | 0.8816 | 0.6258 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |
| `random_init@ar_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
