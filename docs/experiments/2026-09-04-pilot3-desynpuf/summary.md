# Ablation grid -- 2026-09-04-pilot3-desynpuf

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-04T07:32:34+00:00.

| run | objective | probe | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jepa_recon_notime` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 1.059 | 0.2751 | -- | -- | -- | 143.7 | 0.02367 | 10,495 | 2,256 | 0.7043 | 0.6492 | 0.7116 | 0.7173 | 0.7538 | 0.7144 | 0.6714 |
| `recon_only` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.683 | -- | -- | -- | -- | 149.5 | 0 | 13,433 | 1,762 | 0.7077 | 0.6068 | 0.7193 | 0.7155 | 0.7441 | 0.714 | 0.6854 |
| `recon_only_notime` | jepa | mean@final | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.7777 | -- | -- | -- | -- | 148.8 | 0 | 13,365 | 1,955 | 0.7009 | 0.6173 | 0.7216 | 0.7117 | 0.7445 | 0.7069 | 0.6804 |
| `jepa_recon_nosig` | jepa | mean@final | ema | 0 | 0.6 | 2,930 | 48,005,120 | 0.9802 | 0.1965 | -- | -- | -- | 136.3 | 0.01957 | 11,980 | 2,088 | 0.6947 | 0.5923 | 0.711 | 0.7138 | 0.7473 | 0.7092 | 0.6703 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@jepa_recon_notime` | 0.6775 | 0.5699 | 0.6962 | 0.6961 | 0.7252 | 0.6882 | 0.6619 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
