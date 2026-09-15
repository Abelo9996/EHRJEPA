# Ablation grid -- a2ft-physionet2019

Base config `configs/pretrain_default.yaml`, source `physionet2019`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling). Evaluation modes: `finetune`.

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-15T14:51:29+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | sepsis_6h | sepsis_stay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_bins_s1` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 3,052 | 100,007,936 | 1.646 | -- | 0.2824 | 0.9047 | 0.9997 | 171.8 | 0 | 86,802 | 0 | 0.8264 | 0.7861 |
| `hybrid_bins_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.3422 | 0.1733 | 0.2943 | 0.8995 | 0.9997 | 183.8 | 0.2101 | 61,426 | 0 | 0.8403 | 0.7726 |
| `latent_cont_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.3482 | 0.1069 | -- | -- | -- | 138.1 | 0.05527 | 64,238 | 0 | 0.8102 | 0.7521 |
| `latent_only_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0 | -- | 3,052 | 100,007,936 | 0.02729 | 0.02729 | -- | -- | -- | 114.3 | 0.001857 | 65,059 | 0 | 0.727 | 0.6182 |

## Reference models

| model | sepsis_6h | sepsis_stay |
|---|---|---|
| `ft_random@ar_bins_s1` | 0.7438 | 0.6359 |
| `gbm` | 0.8363 | 0.7693 |
| `lr` | 0.7852 | 0.7681 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.

`ft_random@<run>` is that architecture *fine-tuned from untrained weights* on each task -- the train-from-scratch arm the `finetune` rows are read against.
