# Ablation grid -- ablate3-desynpuf

Base config `configs/pretrain_scale.yaml`, source `desynpuf-s1`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-08T13:03:36+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_frozen_ar_s1` | nextlatent | last@final | frozen | 0.05 | -- | 6,104 | 200,015,872 | 0.8587 | 0.2827 | 5.751 | 0.2078 | 0.4015 | 185.4 | 0.001442 | 20,482 | 2,645 | 0.756 | 0.6178 | 0.7828 | 0.7702 | 0.7764 | 0.7939 | 0.698 |
| `hybrid_frozen_ar_init_s1` | nextlatent | last@final | frozen | 0.05 | -- | 6,104 | 200,015,872 | 0.8292 | 0.2775 | 5.509 | 0.2143 | 0.4159 | 221.8 | -0.02248 | 25,925 | 2,092 | 0.7524 | 0.6155 | 0.7793 | 0.7674 | 0.7734 | 0.7868 | 0.6731 |
| `hybrid_textinit_s1` | nextlatent | last@final | ema | 0.05 | -- | 6,104 | 200,015,872 | 0.749 | 0.1725 | 5.756 | 0.2054 | 0.3973 | 181.7 | -0.01836 | 25,692 | 2,110 | 0.7537 | 0.6243 | 0.7791 | 0.7631 | 0.773 | 0.7708 | 0.6938 |
| `hybrid_textinit_frozen_s1` | nextlatent | last@final | ema | 0.05 | -- | 6,104 | 200,015,872 | 0.8733 | 0.1773 | 6.932 | 0.1342 | 0.2899 | 145.8 | -0.03956 | 26,137 | 2,074 | 0.7534 | 0.6166 | 0.7717 | 0.7661 | 0.7776 | 0.7691 | 0.6928 |
| `ar_textinit_s1` | ar | last@final | -- | -- | -- | 6,104 | 200,015,872 | 5.71 | -- | 5.71 | 0.2089 | 0.4017 | 186.5 | 0 | 34,427 | 1,576 | 0.752 | 0.5985 | 0.7723 | 0.7606 | 0.7687 | 0.7727 | 0.6752 |
| `hybrid_frozen_ar_s2` | nextlatent | last@final | frozen | 0.05 | -- | 6,104 | 200,015,872 | 0.8564 | 0.281 | 5.743 | 0.2214 | 0.4051 | 185.9 | -0.0008504 | 25,894 | 2,095 | 0.7581 | 0.6099 | 0.7826 | 0.7698 | 0.7758 | 0.7938 | 0.6799 |
| `hybrid_frozen_ar_init_s2` | nextlatent | last@final | frozen | 0.05 | -- | 6,104 | 200,015,872 | 0.8297 | 0.2766 | 5.523 | 0.2268 | 0.4147 | 223.1 | -0.02543 | 25,889 | 2,096 | 0.7532 | 0.6165 | 0.7804 | 0.768 | 0.7726 | 0.7868 | 0.6775 |
| `hybrid_textinit_s2` | nextlatent | last@final | ema | 0.05 | -- | 6,104 | 200,015,872 | 0.7399 | 0.1633 | 5.755 | 0.2194 | 0.4049 | 182.6 | -0.0206 | 25,644 | 2,115 | 0.7524 | 0.617 | 0.7808 | 0.7651 | 0.7704 | 0.7733 | 0.6971 |
| `hybrid_textinit_frozen_s2` | nextlatent | last@final | ema | 0.05 | -- | 6,104 | 200,015,872 | 0.8694 | 0.1671 | 6.993 | 0.1304 | 0.3009 | 143.7 | -0.04402 | 26,099 | 2,077 | 0.7541 | 0.6092 | 0.782 | 0.7568 | 0.7727 | 0.7711 | 0.688 |
| `ar_textinit_s2` | ar | last@final | -- | -- | -- | 6,104 | 200,015,872 | 5.715 | -- | 5.715 | 0.2237 | 0.4089 | 186.2 | 0 | 34,430 | 1,576 | 0.7484 | 0.6105 | 0.7698 | 0.7618 | 0.7732 | 0.7751 | 0.6581 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7455 | 0.5564 | 0.7713 | 0.7677 | 0.7709 | 0.784 | 0.667 |
| `lr` | 0.7124 | 0.5659 | 0.739 | 0.7326 | 0.7368 | 0.7407 | 0.6529 |
| `random_init@hybrid_frozen_ar_s1` | 0.6742 | 0.5406 | 0.6857 | 0.6906 | 0.7185 | 0.6897 | 0.6166 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
