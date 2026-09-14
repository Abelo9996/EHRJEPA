# Downstream evaluation -- physionet2012

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2012` |
| MEDS | `data/meds/physionet2012` |
| cache | `data/cache/physionet2012` |
| tasks | `data/tasks/physionet2012` |
| anchor seed | 20260903 |
| commit | `039c63c` |
| created | 2026-09-14T15:55:17+00:00 |
| runtime (s) | 142.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/ar_bins_s1/final.pt` | last@final |
| `ckpt:ar_bins_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/ar_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 0.789 [0.756, 0.825] | 0.829 [0.797, 0.864] | 0.648 [0.599, 0.688] | 0.767 [0.734, 0.796] |
| `mortality_inhospital/48h` | 0.828 [0.797, 0.866] | 0.859 [0.829, 0.890] | 0.703 [0.662, 0.750] | 0.808 [0.777, 0.839] |

## AUPRC

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 0.394 [0.319, 0.472] | 0.465 [0.385, 0.550] | 0.242 [0.184, 0.312] | 0.346 [0.268, 0.413] |
| `mortality_inhospital/48h` | 0.457 [0.389, 0.535] | 0.516 [0.435, 0.597] | 0.283 [0.231, 0.358] | 0.400 [0.331, 0.478] |

## BRIER

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 0.1019 [0.0896, 0.1137] | 0.0948 [0.0835, 0.1062] | 0.1157 [0.1021, 0.1296] | 0.1068 [0.0942, 0.1197] |
| `mortality_inhospital/48h` | 0.0963 [0.0839, 0.1091] | 0.0907 [0.0794, 0.1030] | 0.1117 [0.0979, 0.1241] | 0.1006 [0.0897, 0.1142] |

## CALIBRATION SLOPE

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 1.162 [0.995, 1.400] | 0.877 [0.749, 1.024] | 0.703 [0.468, 0.910] | 0.900 [0.754, 1.042] |
| `mortality_inhospital/48h` | 1.376 [1.207, 1.630] | 0.875 [0.758, 1.014] | 0.924 [0.729, 1.171] | 0.945 [0.815, 1.125] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_inhospital/24h` | `lr` - `gbm` | -0.040 | [-0.064, -0.016] | 0.000 |
| `mortality_inhospital/24h` | `lr` - `random_init` | 0.142 | [0.095, 0.187] | 0.000 |
| `mortality_inhospital/24h` | `lr` - `ckpt:ar_bins_s1` | 0.022 | [-0.012, 0.054] | 0.200 |
| `mortality_inhospital/24h` | `gbm` - `random_init` | 0.182 | [0.139, 0.226] | 0.000 |
| `mortality_inhospital/24h` | `gbm` - `ckpt:ar_bins_s1` | 0.062 | [0.030, 0.092] | 0.000 |
| `mortality_inhospital/24h` | `random_init` - `ckpt:ar_bins_s1` | -0.119 | [-0.164, -0.079] | 0.000 |
| `mortality_inhospital/48h` | `lr` - `gbm` | -0.031 | [-0.050, -0.010] | 0.000 |
| `mortality_inhospital/48h` | `lr` - `random_init` | 0.125 | [0.084, 0.164] | 0.000 |
| `mortality_inhospital/48h` | `lr` - `ckpt:ar_bins_s1` | 0.020 | [-0.008, 0.048] | 0.190 |
| `mortality_inhospital/48h` | `gbm` - `random_init` | 0.156 | [0.108, 0.195] | 0.000 |
| `mortality_inhospital/48h` | `gbm` - `ckpt:ar_bins_s1` | 0.051 | [0.025, 0.074] | 0.000 |
| `mortality_inhospital/48h` | `random_init` - `ckpt:ar_bins_s1` | -0.105 | [-0.150, -0.055] | 0.000 |

## Skipped

| task | reason |
|---|---|
| `mortality_365d` | death |
| `inpatient_365d` | inpatient_admission |
| `readmission_30d` | inpatient_admission |
| `new_dx_365d/diabetes` | dx_diabetes |
| `new_dx_365d/heart_failure` | dx_heart_failure |
| `new_dx_365d/ckd` | dx_ckd |
| `new_dx_365d/copd` | dx_copd |
| `sepsis_6h` | not defined on source family 'physionet2012' |
| `sepsis_stay` | not defined on source family 'physionet2012' |
