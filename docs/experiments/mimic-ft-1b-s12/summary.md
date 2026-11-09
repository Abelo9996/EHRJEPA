# Ablation grid -- mimic-ft-1b-s12

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling). Evaluation modes: `finetune`.

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-22T23:43:49+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_8pct_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2341 | 0.1443 | 0.8921 | 0.7832 | 0.9477 | 216.5 | 0.01802 | 44,816 | 0 | 0.7113 | 0.9429 | 0.9371 | 0.7849 | 0.9201 | 0.8882 | 0.6093 |
| `ar_8pct_s1` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.7596 | -- | 0.7596 | 0.8101 | 0.9632 | 224.1 | 0 | 58,325 | 0 | 0.7108 | 0.9408 | 0.9335 | 0.7919 | 0.9127 | 0.884 | 0.6261 |
| `hybrid_8pct_s2` | finetune | nextlatent | last@final | 64 | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2403 | 0.1473 | 0.9239 | 0.7751 | 0.9449 | 216.3 | 0.01833 | 45,346 | 0 | 0.724 | 0.9435 | 0.9372 | 0.7797 | 0.9021 | 0.8871 | 0.6341 |
| `ar_8pct_s2` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.7942 | -- | 0.7942 | 0.7975 | 0.9601 | 224.8 | 0 | 59,112 | 0 | 0.7099 | 0.936 | 0.9402 | 0.7906 | 0.9038 | 0.8828 | 0.6067 |
| `hybrid_25pct_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.234 | 0.1405 | 0.9289 | 0.7828 | 0.9387 | 210.1 | 0.02712 | 42,014 | 0 | 0.7234 | 0.9461 | 0.9415 | 0.8003 | 0.9181 | 0.8957 | 0.6416 |
| `ar_25pct_s1` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.8575 | -- | 0.8575 | 0.7951 | 0.946 | 218.3 | 0 | 58,146 | 0 | 0.7217 | 0.9455 | 0.9359 | 0.7814 | 0.9126 | 0.8849 | 0.6222 |
| `hybrid_25pct_s2` | finetune | nextlatent | last@final | 64 | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2423 | 0.1415 | 1.002 | 0.7734 | 0.9327 | 212.6 | 0.02544 | 44,857 | 0 | 0.7302 | 0.9447 | 0.9461 | 0.8136 | 0.9103 | 0.8954 | 0.6518 |
| `ar_25pct_s2` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.9295 | -- | 0.9295 | 0.7893 | 0.9388 | 221.6 | 0 | 59,711 | 0 | 0.7139 | 0.9436 | 0.941 | 0.8141 | 0.92 | 0.8882 | 0.6155 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
