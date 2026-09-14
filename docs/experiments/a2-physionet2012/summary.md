# Ablation grid -- a2-physionet2012

Base config `configs/pretrain_default.yaml`, source `physionet2012`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-09-14T16:58:30+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | mortality_inhospital/24h | mortality_inhospital/48h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_bins_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 1,221 | 40,009,728 | 1.931 | -- | 0.4667 | 0.8551 | 0.9963 | 169.9 | 0 | 105,378 | 260.8 | 0.7669 | 0.8083 |
| `ar_cont_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 1,221 | 40,009,728 | 0.6088 | -- | 0.4481 | 0.8604 | 0.9971 | 159.4 | 0 | 127,247 | 262.1 | 0.7768 | 0.814 |
| `hybrid_bins_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.4015 | 0.2039 | 0.4753 | 0.8557 | 0.9959 | 170.9 | 0.1465 | 88,444 | 375.6 | 0.7631 | 0.7813 |
| `latent_cont_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.3312 | 0.07741 | -- | -- | -- | 126.9 | 0.006382 | 92,186 | 360.6 | 0.6994 | 0.7375 |
| `latent_only_s1` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.01752 | 0.01752 | -- | -- | -- | 96.77 | 0.002473 | 94,466 | 351.7 | 0.6597 | 0.7022 |
| `ar_bins_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 1,221 | 40,009,728 | 1.895 | -- | 0.4652 | 0.8538 | 0.9962 | 169.4 | 0 | 125,525 | 265.6 | 0.7595 | 0.7941 |
| `ar_cont_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 1,221 | 40,009,728 | 0.6004 | -- | 0.4459 | 0.8579 | 0.9968 | 160 | 0 | 124,550 | 267.6 | 0.7682 | 0.8129 |
| `hybrid_bins_s2` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.407 | 0.213 | 0.4777 | 0.8538 | 0.996 | 170.9 | 0.1722 | 88,504 | 375.2 | 0.7689 | 0.7883 |
| `latent_cont_s2` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.3273 | 0.08589 | -- | -- | -- | 129.3 | 0.007543 | 93,973 | 353.6 | 0.7365 | 0.7777 |
| `latent_only_s2` | probe | nextlatent | last@final | -- | -- | ema | 0 | -- | 1,221 | 40,009,728 | 0.01671 | 0.01671 | -- | -- | -- | 96.32 | 0.002375 | 92,420 | 359.7 | 0.6498 | 0.7096 |

## Reference models

| model | mortality_inhospital/24h | mortality_inhospital/48h |
|---|---|---|
| `gbm` | 0.8292 | 0.8589 |
| `lr` | 0.7894 | 0.8281 |
| `random_init@ar_bins_s1` | 0.6475 | 0.7032 |
| `random_init@hybrid_bins_s1` | 0.6475 | 0.7032 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
