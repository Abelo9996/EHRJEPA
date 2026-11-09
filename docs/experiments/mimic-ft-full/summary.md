# Ablation grid -- mimic-ft-full

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling). Evaluation modes: `finetune`.

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-23T14:11:14+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_full_s0` | finetune | nextlatent | last@final | 64 | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2447 | 0.1413 | 1.027 | 0.7678 | 0.9286 | 211.4 | 0.02342 | 43,401 | 0 | 0.732 | 0.9437 | 0.9388 | 0.8009 | 0.9247 | 0.9036 | 0.6314 |
| `ar_full_s0` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.958 | -- | 0.958 | 0.7751 | 0.9373 | 216.7 | 0 | 57,261 | 0 | 0.7165 | 0.9475 | 0.9372 | 0.7956 | 0.9137 | 0.8934 | 0.6275 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
