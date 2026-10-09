# Ablation grid -- mimic-objectives-300m

Base config `configs/pretrain_default.yaml`, source `mimic-iv`, held-out AUROC on the full held-out split, 200 bootstrap resamples, probe `auto@final` (the `probe` column gives each row's resolved pooling).

Rows are appended by `scripts/ablate.py` as each run finishes. Last update: 2026-10-04T14:33:00+00:00.

| run | mode | objective | probe | ft_bs | eval_subj | target | lambda | p_future | steps | tokens | loss | pred | ce | top1 | top10 | rank | cos_gap | tok/s | wall_s | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ar_s0` | probe | ar | last@final | -- | -- | -- | -- | -- | 9,156 | 300,023,808 | 0.921 | -- | 0.921 | 0.7788 | 0.9445 | 221.2 | 0 | 60,191 | 2,768 | 0.6642 | 0.9239 | 0.8934 | 0.7745 | 0.8393 | 0.8619 | 0.6186 |
| `jepa_ema_s0` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.4385 | 0.2245 | -- | -- | -- | 186.2 | 0.183 | 51,863 | 3,208 | 0.5907 | 0.8719 | 0.7871 | 0.7631 | 0.7387 | 0.8179 | 0.5788 |
| `recon_only_s0` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.1897 | -- | -- | -- | -- | 166.5 | 0 | 63,147 | 2,636 | 0.61 | 0.8516 | 0.7403 | 0.6918 | 0.7485 | 0.7903 | 0.6172 |
| `nextlatent_s0` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.02958 | 0.02815 | -- | -- | -- | 134.9 | 0.001981 | 62,833 | 2,650 | 0.5371 | 0.8698 | 0.745 | 0.6825 | 0.6836 | 0.7883 | 0.6115 |
| `hybrid_s0` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.2858 | 0.1845 | 1.006 | 0.7643 | 0.9352 | 223.4 | 0.02031 | 44,873 | 3,708 | 0.702 | 0.9245 | 0.8773 | 0.7314 | 0.8437 | 0.858 | 0.6168 |
| `window_s0` | probe | window | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.0345 | 0.03337 | -- | -- | -- | 147.1 | 0.1797 | 65,269 | 2,550 | 0.5782 | 0.8624 | 0.7504 | 0.7412 | 0.7215 | 0.8084 | 0.6054 |
| `ar_s1` | probe | ar | last@final | -- | -- | -- | -- | -- | 9,156 | 300,023,808 | 0.947 | -- | 0.947 | 0.7729 | 0.9419 | 220.2 | 0 | 59,266 | 2,808 | 0.6722 | 0.9165 | 0.8853 | 0.7351 | 0.8409 | 0.865 | 0.5844 |
| `jepa_ema_s1` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.4796 | 0.2398 | -- | -- | -- | 186.1 | 0.1917 | 51,795 | 3,212 | 0.5872 | 0.868 | 0.7933 | 0.7084 | 0.7475 | 0.805 | 0.5881 |
| `recon_only_s1` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.2168 | -- | -- | -- | -- | 168.6 | 0 | 62,983 | 2,643 | 0.5859 | 0.8658 | 0.7905 | 0.699 | 0.7196 | 0.8191 | 0.5928 |
| `nextlatent_s1` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.02898 | 0.02741 | -- | -- | -- | 130.8 | 0.001941 | 62,511 | 2,664 | 0.5417 | 0.8652 | 0.7681 | 0.7076 | 0.7332 | 0.7792 | 0.5975 |
| `hybrid_s1` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.2882 | 0.1846 | 1.027 | 0.7601 | 0.9333 | 225.2 | 0.02156 | 44,671 | 3,724 | 0.6951 | 0.925 | 0.8634 | 0.714 | 0.8489 | 0.8652 | 0.6072 |
| `window_s1` | probe | window | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.03012 | 0.02891 | -- | -- | -- | 146.6 | 0.1969 | 65,085 | 2,558 | 0.5702 | 0.8714 | 0.7606 | 0.7439 | 0.7293 | 0.7953 | 0.6036 |
| `ar_s2` | probe | ar | last@final | -- | -- | -- | -- | -- | 9,156 | 300,023,808 | 0.9202 | -- | 0.9202 | 0.7777 | 0.9467 | 219.4 | 0 | 59,120 | 2,814 | 0.6606 | 0.9206 | 0.8802 | 0.7404 | 0.8443 | 0.8663 | 0.5976 |
| `jepa_ema_s2` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.4574 | 0.2221 | -- | -- | -- | 183.8 | 0.1881 | 51,766 | 3,213 | 0.5856 | 0.8851 | 0.7939 | 0.7251 | 0.7148 | 0.7743 | 0.5893 |
| `recon_only_s2` | probe | jepa | last@final | -- | -- | ema | 0.05 | 0.6 | 9,156 | 300,023,808 | 0.2162 | -- | -- | -- | -- | 167.8 | 0 | 62,599 | 2,656 | 0.594 | 0.8555 | 0.7854 | 0.7074 | 0.7192 | 0.7976 | 0.597 |
| `nextlatent_s2` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.0299 | 0.02838 | -- | -- | -- | 130.1 | 0.001898 | 61,610 | 2,703 | 0.543 | 0.8613 | 0.7752 | 0.6934 | 0.7092 | 0.7714 | 0.6017 |
| `hybrid_s2` | probe | nextlatent | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.2851 | 0.1841 | 1.003 | 0.7657 | 0.9387 | 223.4 | 0.02305 | 38,858 | 4,283 | 0.6964 | 0.9243 | 0.8781 | 0.7389 | 0.8415 | 0.8646 | 0.6171 |
| `window_s2` | probe | window | last@final | -- | -- | ema | 0.05 | -- | 9,156 | 300,023,808 | 0.03738 | 0.03622 | -- | -- | -- | 145.5 | 0.2006 | 66,238 | 2,513 | 0.5763 | 0.8612 | 0.745 | 0.7465 | 0.7435 | 0.7921 | 0.595 |

## Reference models

| model | inpatient_365d | mortality_365d | new_dx_365d/ckd | new_dx_365d/copd | new_dx_365d/diabetes | new_dx_365d/heart_failure | readmission_30d |
|---|---|---|---|---|---|---|---|
| `gbm` | 0.7004 | 0.9354 | 0.9296 | 0.7889 | 0.9129 | 0.8995 | 0.6171 |
| `lr` | 0.6802 | 0.9393 | 0.8895 | 0.7963 | 0.9057 | 0.8752 | 0.6255 |
| `random_init@ar_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |
| `random_init@jepa_ema_s0` | 0.5107 | 0.8473 | 0.7555 | 0.6791 | 0.6909 | 0.8051 | 0.5568 |

`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so their scores are reused from an earlier run's `predictions.parquet`. `random_init@<run>` is that run's own architecture with untrained weights, probed identically -- the control for a causal encoder is an untrained causal encoder.
