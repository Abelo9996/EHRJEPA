# Ablation grid -- mimic-ft-1b

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling). Evaluation modes: `finetune`.

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-22T18:19:15+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_8pct_s0` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.7538 | -- | 0.7538 | 0.8117 | 0.9619 | 224.8 | 0 | 60,324 | 0 | 0.6917 | 0.9411 | 0.9365 | 0.7902 | 0.9004 | 0.8925 | 0.6082 |
| `hybrid_8pct_s0` | finetune | nextlatent | last@final | 64 | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2375 | 0.1472 | 0.896 | 0.7817 | 0.9468 | 213.7 | 0.01649 | 45,486 | 0 | 0.7232 | 0.9443 | 0.9386 | 0.8121 | 0.9145 | 0.8967 | 0.6289 |
| `ar_25pct_s0` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.9036 | -- | 0.9036 | 0.7798 | 0.9481 | 222.4 | 0 | 59,199 | 0 | 0.719 | 0.9488 | 0.944 | 0.7873 | 0.9141 | 0.8937 | 0.6281 |
| `hybrid_25pct_s0` | finetune | nextlatent | last@final | 64 | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2463 | 0.1471 | 0.9848 | 0.7667 | 0.9411 | 213.3 | 0.01375 | 38,803 | 0 | 0.7316 | 0.9434 | 0.9327 | 0.8107 | 0.9155 | 0.8922 | 0.6269 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `ft_random@ar_8pct_s0` | 0.5377 | 0.8739 | 0.7543 | 0.7241 | 0.6936 | 0.8083 | 0.6143 |
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.

`ft_random@<run>` is that architecture *fine-tuned from untrained weights* on each task -- the train-from-scratch arm the `finetune` rows are read against.
