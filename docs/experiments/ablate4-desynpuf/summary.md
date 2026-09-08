# Ablation grid -- ablate4-desynpuf

Base config `configs/pretrain_scale.yaml`, source `desynpuf-s1`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-08T14:40:24+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_nosig_frozen_s1` | nextlatent | last@final | frozen | 0 | -- | 6,104 | 200,015,872 | 0.8553 | 0.2819 | 5.734 | 0.2101 | 0.4005 | 181.5 | 0.002167 | 22,942 | 2,362 | 0.7563 | 0.6175 | 0.7812 | 0.7711 | 0.7768 | 0.7958 | 0.6923 |
| `hybrid_nosig_frozen_s2` | nextlatent | last@final | frozen | 0 | -- | 6,104 | 200,015,872 | 0.8548 | 0.2807 | 5.741 | 0.2213 | 0.4056 | 181.1 | -0.001391 | 26,937 | 2,014 | 0.7589 | 0.6134 | 0.7796 | 0.7714 | 0.7759 | 0.7956 | 0.6941 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7455 | 0.5564 | 0.7713 | 0.7677 | 0.7709 | 0.784 | 0.667 |
| `lr` | 0.7124 | 0.5659 | 0.739 | 0.7326 | 0.7368 | 0.7407 | 0.6529 |
| `random_init@hybrid_nosig_frozen_s1` | 0.6742 | 0.5406 | 0.6857 | 0.6906 | 0.7185 | 0.6897 | 0.6166 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
