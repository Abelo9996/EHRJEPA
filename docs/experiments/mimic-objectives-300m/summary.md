# Ablation grid -- mimic-objectives-300m

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-03T21:53:18+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s0` | probe | ar | last@final | -- | -- | -- | -- | -- | 9,156 | 300,023,808 | 0.921 | -- | 0.921 | 0.7788 | 0.9445 | 221.2 | 0 | 60,191 | 2,768 | 0.6642 | 0.9239 | 0.8934 | 0.7745 | 0.8393 | 0.8619 | 0.6186 |
| `jepa_ema_s0` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.4385 | 0.2245 | -- | -- | -- | 186.2 | 0.183 | 51,863 | 3,208 | 0.5907 | 0.8719 | 0.7871 | 0.7631 | 0.7387 | 0.8179 | 0.5788 |
| `recon_only_s0` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.1897 | -- | -- | -- | -- | 166.5 | 0 | 63,147 | 2,636 | 0.61 | 0.8516 | 0.7403 | 0.6918 | 0.7485 | 0.7903 | 0.6172 |
| `nextlatent_s0` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.02958 | 0.02815 | -- | -- | -- | 134.9 | 0.001981 | 62,833 | 2,650 | 0.5371 | 0.8698 | 0.745 | 0.6825 | 0.6836 | 0.7883 | 0.6115 |
| `hybrid_s0` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.2858 | 0.1845 | 1.006 | 0.7643 | 0.9352 | 223.4 | 0.02031 | 44,873 | 3,708 | 0.702 | 0.9245 | 0.8773 | 0.7314 | 0.8437 | 0.858 | 0.6168 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |
| `random_init@ar_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |
| `random_init@jepa_ema_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
