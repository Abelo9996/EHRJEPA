# Ablation grid -- scale1b-desynpuf

Base config `configs/pretrain_scale.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-06T10:44:02+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar` | ar | last@final | -- | -- | -- | 30,518 | 1,000,013,824 | 5.528 | -- | 5.528 | 0.2073 | 0.4079 | 222.7 | 0 | 34,680 | 0 | 0.7506 | 0.6028 | 0.7586 | 0.7629 | 0.7618 | 0.7804 | 0.649 |
| `hybrid` | nextlatent | last@final | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.7304 | 0.1654 | 5.641 | 0.2021 | 0.401 | 194.6 | -0.01678 | 26,090 | 10,375 | 0.7553 | 0.5767 | 0.7741 | 0.7781 | 0.7779 | 0.7997 | 0.6874 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@ar` | 0.6686 | 0.5469 | 0.6799 | 0.6815 | 0.7158 | 0.6763 | 0.6136 |
| `random_init@hybrid` | 0.6686 | 0.5469 | 0.6799 | 0.6815 | 0.7158 | 0.6763 | 0.6136 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
