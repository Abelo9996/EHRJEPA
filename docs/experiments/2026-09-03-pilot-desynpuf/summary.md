# Ablation grid -- 2026-09-03-pilot-desynpuf

Base config `configs/pretrain_pilot.yaml`, source `desynpuf-s1`, held-out AUROC on a 3000-subject subset (seed 0), 200 bootstrap resamples, probe `cls_mean@final`.

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-04T01:52:07+00:00.

| run | objective | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar` | ar | -- | -- | -- | 2,930 | 48,005,120 | 6.318 | -- | 6.318 | 0.1809 | 0.3459 | 117.5 | 0 | 7,428 | 3,300 | 0.7265 | 0.606 | 0.7586 | 0.7552 | 0.759 | 0.7437 | 0.6946 |
| `jepa_ema` | jepa | ema | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.1402 | 0.1395 | -- | -- | -- | 146.5 | 0.2632 | 13,673 | 1,748 | 0.6798 | 0.6096 | 0.6944 | 0.7059 | 0.7418 | 0.7027 | 0.646 |
| `jepa_ema_nosig` | jepa | ema | 0 | 0.6 | 2,930 | 48,005,120 | 0.08467 | 0.08467 | -- | -- | -- | 129.1 | 0.1126 | 14,604 | 1,623 | 0.6927 | 0.6409 | 0.6968 | 0.6997 | 0.7379 | 0.7031 | 0.6597 |
| `jepa_shared_sig` | jepa | shared | 0.05 | 0.6 | 2,930 | 48,005,120 | 0.1414 | 0.1399 | -- | -- | -- | 142.2 | 0.1894 | 10,946 | 2,191 | 0.6784 | 0.6573 | 0.6865 | 0.6948 | 0.7292 | 0.6955 | 0.6293 |
| `jepa_ema_future` | jepa | ema | 0.05 | 1 | 2,930 | 48,005,120 | 0.1394 | 0.1386 | -- | -- | -- | 146.8 | 0.2626 | 8,271 | 3,164 | 0.6819 | 0.5837 | 0.6943 | 0.7048 | 0.7382 | 0.7 | 0.6612 |
| `jepa_ema_block` | jepa | ema | 0.05 | 0 | 2,930 | 48,005,120 | 0.146 | 0.1454 | -- | -- | -- | 148.8 | 0.2448 | 11,131 | 2,230 | 0.6763 | 0.5859 | 0.6939 | 0.7035 | 0.7295 | 0.7097 | 0.6442 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.744 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 |
| `lr` | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 |
| `random_init@ar` | 0.6751 | 0.616 | 0.694 | 0.689 | 0.7211 | 0.6904 | 0.6464 |
| `random_init@jepa_ema` | 0.685 | 0.6021 | 0.7005 | 0.6933 | 0.7264 | 0.7001 | 0.6675 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
