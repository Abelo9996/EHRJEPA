# Ablation grid -- mimic-hybrid-ar

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-30T09:48:09+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s0` | probe | ar | last@final | -- | -- | -- | -- | -- | 611 | 20,021,248 | 2 | -- | 2 | 0.6336 | 0.8303 | 190.5 | 0 | 59,094 | 192.7 | 0.5931 | 0.8705 | 0.7491 | 0.712 | 0.7638 | 0.7962 | 0.5832 |
| `hybrid_s0` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 611 | 20,021,248 | 0.3095 | 0.1047 | 2.048 | 0.6249 | 0.8236 | 185.7 | 0.02501 | 37,074 | 303.3 | 0.5944 | 0.8785 | 0.7707 | 0.7129 | 0.757 | 0.8174 | 0.5801 |
| `ar_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 611 | 20,021,248 | 1.709 | -- | 1.709 | 0.6745 | 0.8562 | 189.4 | 0 | 59,006 | 191.8 | 0.6106 | 0.851 | 0.7822 | 0.7125 | 0.7074 | 0.7883 | 0.5751 |
| `hybrid_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 611 | 20,021,248 | 0.2807 | 0.1062 | 1.745 | 0.6698 | 0.8528 | 184.3 | 0.02504 | 46,121 | 244.4 | 0.5976 | 0.865 | 0.7892 | 0.7123 | 0.7242 | 0.8017 | 0.5843 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |
| `random_init@ar_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
