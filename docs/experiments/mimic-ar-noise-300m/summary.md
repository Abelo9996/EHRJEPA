# Ablation grid -- mimic-ar-noise-300m

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-03T12:08:18+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s0` | probe | ar | last@final | -- | -- | -- | -- | -- | 9,156 | 300,023,808 | 0.9209 | -- | 0.9209 | 0.7786 | 0.945 | 221.3 | 0 | 59,079 | 2,820 | 0.6623 | 0.9169 | 0.895 | 0.7467 | 0.8251 | 0.8578 | 0.6111 |
| `ar_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 9,156 | 300,023,808 | 0.9454 | -- | 0.9454 | 0.774 | 0.9416 | 220.3 | 0 | 59,068 | 2,818 | 0.6652 | 0.9208 | 0.8851 | 0.718 | 0.8384 | 0.8656 | 0.5919 |
| `ar_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 9,156 | 300,023,808 | 0.9209 | -- | 0.9209 | 0.778 | 0.9463 | 219.5 | 0 | 60,121 | 2,768 | 0.6719 | 0.9222 | 0.892 | 0.7333 | 0.8494 | 0.87 | 0.6025 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |
| `random_init@ar_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
