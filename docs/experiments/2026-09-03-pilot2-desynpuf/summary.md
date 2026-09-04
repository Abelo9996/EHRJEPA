# Ablation grid -- 2026-09-03-pilot2-desynpuf

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-04T05:01:34+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jepa_notime` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.2698 | 0.2689 | -- | -- | -- | 141.9 | 0.02528 | 11,484 | 2,141 | 0.692 | 0.6332 | 0.7121 | 0.7167 | 0.7477 | 0.7089 | 0.659 |
| `jepa_content` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.2548 | 0.2542 | -- | -- | -- | 144 | 0.05602 | 14,601 | 1,626 | 0.6914 | 0.5998 | 0.6984 | 0.701 | 0.7307 | 0.7019 | 0.6488 |
| `jepa_content_recon` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.9819 | 0.2813 | -- | -- | -- | 142 | 0.1099 | 12,283 | 1,925 | 0.7085 | 0.6265 | 0.7167 | 0.7141 | 0.7432 | 0.7154 | 0.6747 |
| `jepa_recon` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.894 | 0.1824 | -- | -- | -- | 144.3 | 0.2968 | 11,021 | 2,143 | 0.7078 | 0.5954 | 0.7169 | 0.7112 | 0.741 | 0.7122 | 0.6611 |
| `jepa_content_future` | jepa | mean@final | ema | 0.05 | 1 | 2,930 | 48,005,120 | 0.2403 | 0.2396 | -- | -- | -- | 146.3 | 0.0694 | 13,827 | 1,838 | 0.6784 | 0.5776 | 0.6947 | 0.6975 | 0.7277 | 0.696 | 0.6528 |
| `ar_last` | ar | last@final | -- | -- | -- | 2,930 | 48,005,120 | 6.318 | -- | 6.318 | 0.1809 | 0.3459 | 117.5 | 0 | 7,428 | 0 | 0.7404 | 0.5951 | 0.7508 | 0.7613 | 0.7627 | 0.7714 | 0.6738 |
| `jepa_ema_mean` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.1402 | 0.1395 | -- | -- | -- | 146.5 | 0.2632 | 13,673 | 0 | 0.6857 | 0.6239 | 0.6997 | 0.7052 | 0.7426 | 0.7026 | 0.6501 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@ar_last` | 0.6469 | 0.5339 | 0.658 | 0.662 | 0.71 | 0.666 | 0.5781 |
| `random_init@jepa_notime` | 0.6775 | 0.5699 | 0.6962 | 0.6961 | 0.7252 | 0.6882 | 0.6619 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
