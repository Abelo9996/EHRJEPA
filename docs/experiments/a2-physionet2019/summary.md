# Ablation grid -- a2-physionet2019

Base config `configs/pretrain_default.yaml`, source `physionet2019`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-14T15:50:48+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | sepsis_6h | sepsis_stay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_bins_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 3,052 | 100,007,936 | 1.646 | -- | 0.2824 | 0.9047 | 0.9997 | 171.8 | 0 | 86,802 | 649.5 | 0.8207 | 0.7676 |
| `ar_cont_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 3,052 | 100,007,936 | 0.4246 | -- | 0.275 | 0.9053 | 0.9997 | 159.2 | 0 | 86,203 | 653.4 | 0.8185 | 0.7487 |
| `hybrid_bins_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.3422 | 0.1733 | 0.2943 | 0.8995 | 0.9997 | 183.8 | 0.2101 | 61,426 | 915.1 | 0.8224 | 0.7656 |
| `latent_cont_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.3482 | 0.1069 | -- | -- | -- | 138.1 | 0.05527 | 64,238 | 875.4 | 0.7751 | 0.6828 |
| `latent_only_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.02729 | 0.02729 | -- | -- | -- | 114.3 | 0.001857 | 65,059 | 864.4 | 0.624 | 0.5796 |
| `ar_bins_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 3,052 | 100,007,936 | 1.69 | -- | 0.2949 | 0.9016 | 0.9994 | 172.6 | 0 | 88,188 | 639.8 | 0.8143 | 0.745 |
| `ar_cont_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 3,052 | 100,007,936 | 0.4281 | -- | 0.2848 | 0.9029 | 0.9995 | 160.3 | 0 | 88,026 | 640.1 | 0.8227 | 0.7495 |
| `hybrid_bins_s2` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.3491 | 0.1753 | 0.3096 | 0.8974 | 0.999 | 183.8 | 0.1972 | 62,513 | 899.1 | 0.8327 | 0.7658 |
| `latent_cont_s2` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.3313 | 0.1032 | -- | -- | -- | 136 | 0.04913 | 65,270 | 861.3 | 0.7823 | 0.7088 |
| `latent_only_s2` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.02663 | 0.02663 | -- | -- | -- | 111.5 | 0.00193 | 67,341 | 835.2 | 0.6246 | 0.6066 |

## Reference models

| model | sepsis_6h | sepsis_stay |
|---|---|---|
| `gbm` | 0.8363 | 0.7693 |
| `lr` | 0.7852 | 0.7681 |
| `random_init@ar_bins_s1` | 0.6861 | 0.6269 |
| `random_init@hybrid_bins_s1` | 0.6861 | 0.6269 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
