# Ablation grid -- scale1b-seeds-desynpuf

Base config `configs/pretrain_scale.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-07T12:29:25+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s1` | ar | last@final | -- | -- | -- | 30,518 | 1,000,013,824 | 5.589 | -- | 5.589 | 0.2069 | 0.4001 | 225.5 | 0 | 35,008 | 7,738 | 0.741 | 0.6031 | 0.7584 | 0.7738 | 0.7639 | 0.7796 | 0.6742 |
| `hybrid_s1` | nextlatent | last@final | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.7335 | 0.1613 | 5.715 | 0.2051 | 0.3954 | 196 | -0.01246 | 24,516 | 12,517 | 0.7491 | 0.5862 | 0.7717 | 0.7798 | 0.7787 | 0.8015 | 0.6846 |
| `ar_s2` | ar | last@final | -- | -- | -- | 30,518 | 1,000,013,824 | 5.405 | -- | 5.405 | 0.22 | 0.4264 | 222.8 | 0 | 34,001 | 7,967 | 0.7441 | 0.5933 | 0.7581 | 0.7656 | 0.7589 | 0.7787 | 0.6679 |
| `hybrid_s2` | nextlatent | last@final | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.7189 | 0.166 | 5.52 | 0.2198 | 0.4176 | 197.7 | -0.01065 | 25,582 | 10,582 | 0.7556 | 0.5969 | 0.7752 | 0.776 | 0.7784 | 0.795 | 0.684 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@ar_s1` | 0.6686 | 0.5469 | 0.6799 | 0.6815 | 0.7158 | 0.6763 | 0.6136 |
| `random_init@hybrid_s1` | 0.6686 | 0.5469 | 0.6799 | 0.6815 | 0.7158 | 0.6763 | 0.6136 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
