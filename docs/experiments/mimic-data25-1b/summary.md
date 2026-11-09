# Ablation grid -- mimic-data25-1b

Base config `configs/pretrain_default.yaml`, source `mimic-iv-25`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-22T15:41:05+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s0` | probe | ar | last@final | -- | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.9036 | -- | 0.9036 | 0.7798 | 0.9481 | 222.4 | 0 | 59,199 | 9,398 | 0.6891 | 0.9285 | 0.8949 | 0.8181 | 0.8734 | 0.9 | 0.6638 |
| `hybrid_s0` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2463 | 0.1471 | 0.9848 | 0.7667 | 0.9411 | 213.3 | 0.01375 | 38,803 | 14,326 | 0.736 | 0.9333 | 0.8997 | 0.8227 | 0.877 | 0.8994 | 0.6706 |
| `ar_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.8575 | -- | 0.8575 | 0.7951 | 0.946 | 218.3 | 0 | 58,146 | 11,442 | 0.6953 | 0.9278 | 0.8994 | 0.8193 | 0.8647 | 0.8967 | 0.66 |
| `hybrid_s1` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.234 | 0.1405 | 0.9289 | 0.7828 | 0.9387 | 210.1 | 0.02712 | 42,014 | 21,078 | 0.7333 | 0.9274 | 0.8964 | 0.8214 | 0.8725 | 0.8963 | 0.6694 |
| `ar_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 30,518 | 1,000,013,824 | 0.9295 | -- | 0.9295 | 0.7893 | 0.9388 | 221.6 | 0 | 59,711 | 9,319 | 0.683 | 0.9294 | 0.8999 | 0.8265 | 0.8661 | 0.8977 | 0.6416 |
| `hybrid_s2` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 30,518 | 1,000,013,824 | 0.2423 | 0.1415 | 1.002 | 0.7734 | 0.9327 | 212.6 | 0.02544 | 44,857 | 1,870 | 0.7327 | 0.9299 | 0.9013 | 0.8094 | 0.8698 | 0.8976 | 0.6717 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7235 | 0.9401 | 0.9401 | 0.8562 | 0.9193 | 0.9106 | 0.6966 |
| `lr` | 0.7075 | 0.939 | 0.9082 | 0.8539 | 0.9063 | 0.9142 | 0.6894 |
| `random_init@ar_s0` | 0.559 | 0.8481 | 0.7997 | 0.7429 | 0.7345 | 0.8272 | 0.574 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
