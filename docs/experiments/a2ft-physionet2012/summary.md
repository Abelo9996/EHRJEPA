# Ablation grid -- a2ft-physionet2012

Base config `configs/pretrain_default.yaml`, source `physionet2012`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling). Evaluation modes: `finetune`.

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-15T15:15:07+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | mortality_inhospital/24h | mortality_inhospital/48h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_bins_s1` | finetune | ar | last@final | 64 | -- | -- | -- | -- | 1,221 | 40,009,728 | 1.931 | -- | 0.4667 | 0.8551 | 0.9963 | 169.9 | 0 | 105,378 | 0 | 0.7999 | 0.8355 |
| `hybrid_bins_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.4015 | 0.2039 | 0.4753 | 0.8557 | 0.9959 | 170.9 | 0.1465 | 88,444 | 0 | 0.7969 | 0.8304 |
| `latent_cont_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.3312 | 0.07741 | -- | -- | -- | 126.9 | 0.006382 | 92,186 | 0 | 0.77 | 0.8103 |
| `latent_only_s1` | finetune | nextlatent | last@final | 64 | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.01752 | 0.01752 | -- | -- | -- | 96.77 | 0.002473 | 94,466 | 0 | 0.6443 | 0.6596 |

## Reference models

| model | mortality_inhospital/24h | mortality_inhospital/48h |
|---|---|---|
| `ft_random@ar_bins_s1` | 0.654 | 0.6843 |
| `gbm` | 0.8292 | 0.8589 |
| `lr` | 0.7894 | 0.8281 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.

`ft_random@<run>` is that architecture *fine-tuned from untrained weights* on each task -- the train-from-scratch arm the `finetune` rows are read against.
