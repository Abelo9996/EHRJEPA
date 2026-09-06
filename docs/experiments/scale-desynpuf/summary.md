# Ablation grid -- scale-desynpuf

Base config `configs/pretrain_scale.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-06T05:21:02+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar` | ar | last@final | -- | -- | -- | 6,104 | 200,015,872 | 6.032 | -- | 6.032 | 0.1842 | 0.3706 | 165.8 | 0 | 31,944 | 0 | 0.7515 | 0.5946 | 0.7624 | 0.7747 | 0.7714 | 0.7908 | 0.6674 |
| `hybrid` | nextlatent | last@final | ema | 0.05 | -- | 6,104 | 200,015,872 | 0.7966 | 0.1883 | 6.068 | 0.1838 | 0.3677 | 171.2 | -0.02253 | 20,112 | 2,704 | 0.7511 | 0.5919 | 0.7761 | 0.7676 | 0.7729 | 0.7767 | 0.693 |
| `recon_only` | jepa | mean@final | ema | 0.05 | 0.6 | 6,104 | 200,015,872 | 0.6933 | -- | -- | -- | -- | 186.2 | 0 | 28,463 | 2,105 | 0.7326 | 0.6508 | 0.7336 | 0.7242 | 0.7568 | 0.7453 | 0.6894 |
| `jepa_ema` | jepa | mean@final | ema | 0.05 | 0.6 | 6,104 | 200,015,872 | 0.1576 | 0.1569 | -- | -- | -- | 183.5 | 0.2945 | 27,061 | 2,286 | 0.6948 | 0.6413 | 0.7108 | 0.7189 | 0.7499 | 0.7159 | 0.658 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@ar` | 0.6686 | 0.5469 | 0.6799 | 0.6815 | 0.7158 | 0.6763 | 0.6136 |
| `random_init@jepa_ema` | 0.6672 | 0.5974 | 0.685 | 0.6993 | 0.7282 | 0.6936 | 0.6486 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
