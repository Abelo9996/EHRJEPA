# Ablation grid -- scale1b-final-desynpuf

Base config `configs/pretrain_scale.yaml`, source `desynpuf-s1`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-10T15:03:30+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_final_s0` | nextlatent | last@final | ema | 0 | -- | 30,518 | 1,000,013,824 | 0.739 | 0.1768 | 5.622 | 0.2018 | 0.4044 | 198.8 | -0.0287 | 26,281 | 10,302 | 0.7595 | 0.6011 | 0.7838 | 0.7724 | 0.7792 | 0.798 | 0.6865 |
| `hybrid_final_s1` | nextlatent | last@final | ema | 0 | -- | 30,518 | 1,000,013,824 | 0.7421 | 0.1724 | 5.697 | 0.2066 | 0.399 | 201.7 | -0.0186 | 23,776 | 8,092 | 0.7567 | 0.6025 | 0.7844 | 0.772 | 0.7778 | 0.797 | 0.6951 |
| `hybrid_final_s2` | nextlatent | last@final | ema | 0 | -- | 30,518 | 1,000,013,824 | 0.7225 | 0.1726 | 5.499 | 0.2211 | 0.4221 | 201.4 | -0.01708 | 21,745 | 12,629 | 0.7558 | 0.5997 | 0.7821 | 0.7719 | 0.7762 | 0.7997 | 0.6827 |
| `ar_s0_full` | ar | last@final | -- | -- | -- | 30,518 | 1,000,013,824 | 5.528 | -- | 5.528 | 0.2073 | 0.4079 | 222.7 | 0 | 34,680 | 0 | 0.7425 | 0.6037 | 0.7701 | 0.7594 | 0.7633 | 0.7691 | 0.6438 |
| `hybrid_s0_full` | nextlatent | last@final | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.7304 | 0.1654 | 5.641 | 0.2021 | 0.401 | 194.6 | -0.01678 | 26,090 | 0 | 0.7583 | 0.603 | 0.7845 | 0.7747 | 0.777 | 0.7958 | 0.6839 |
| `ar_s1_full` | ar | last@final | -- | -- | -- | 30,518 | 1,000,013,824 | 5.589 | -- | 5.589 | 0.2069 | 0.4001 | 225.5 | 0 | 35,008 | 0 | 0.7399 | 0.6116 | 0.7657 | 0.7634 | 0.7637 | 0.7714 | 0.6655 |
| `hybrid_s1_full` | nextlatent | last@final | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.7335 | 0.1613 | 5.715 | 0.2051 | 0.3954 | 196 | -0.01246 | 24,516 | 0 | 0.7534 | 0.5929 | 0.785 | 0.7734 | 0.7783 | 0.7965 | 0.6826 |
| `ar_s2_full` | ar | last@final | -- | -- | -- | 30,518 | 1,000,013,824 | 5.405 | -- | 5.405 | 0.22 | 0.4264 | 222.8 | 0 | 34,001 | 0 | 0.7435 | 0.6006 | 0.7663 | 0.7581 | 0.7606 | 0.7741 | 0.6648 |
| `hybrid_s2_full` | nextlatent | last@final | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.7189 | 0.166 | 5.52 | 0.2198 | 0.4176 | 197.7 | -0.01065 | 25,582 | 0 | 0.7559 | 0.6014 | 0.784 | 0.7694 | 0.7788 | 0.7908 | 0.685 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7455 | 0.5564 | 0.7713 | 0.7677 | 0.7709 | 0.784 | 0.667 |
| `lr` | 0.7124 | 0.5659 | 0.739 | 0.7326 | 0.7368 | 0.7407 | 0.6529 |
| `random_init@hybrid_final_s0` | 0.6742 | 0.5406 | 0.6857 | 0.6906 | 0.7185 | 0.6897 | 0.6166 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
