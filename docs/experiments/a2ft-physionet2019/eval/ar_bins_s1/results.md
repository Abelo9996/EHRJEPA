# Downstream evaluation -- physionet2019

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2019` |
| MEDS | `data/meds/physionet2019` |
| cache | `data/cache/physionet2019` |
| tasks | `data/tasks/physionet2019` |
| anchor seed | 20260903 |
| commit | `ce0692e` |
| created | 2026-09-14T23:18:40+00:00 |
| runtime (s) | 6967.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `ft_random` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/ar_bins_s1/final.pt` | last@final |
| `ft:ar_bins_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/ar_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 0.785 [0.766, 0.805] | 0.836 [0.820, 0.855] | 0.744 [0.721, 0.764] | 0.826 [0.807, 0.841] |
| `sepsis_stay` | 0.768 [0.737, 0.799] | 0.769 [0.735, 0.800] | 0.636 [0.601, 0.668] | 0.786 [0.755, 0.817] |

## AUPRC

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 0.068 [0.058, 0.083] | 0.095 [0.081, 0.110] | 0.055 [0.047, 0.066] | 0.076 [0.067, 0.087] |
| `sepsis_stay` | 0.194 [0.157, 0.246] | 0.186 [0.151, 0.238] | 0.085 [0.072, 0.105] | 0.195 [0.166, 0.253] |

## BRIER

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 0.0165 [0.0151, 0.0178] | 0.0163 [0.0150, 0.0176] | 0.0166 [0.0152, 0.0179] | 0.0170 [0.0157, 0.0183] |
| `sepsis_stay` | 0.0488 [0.0435, 0.0535] | 0.0493 [0.0441, 0.0543] | 0.0521 [0.0463, 0.0573] | 0.0488 [0.0433, 0.0538] |

## CALIBRATION SLOPE

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 1.026 [0.950, 1.105] | 0.996 [0.931, 1.053] | 0.794 [0.718, 0.865] | 0.632 [0.591, 0.666] |
| `sepsis_stay` | 1.100 [0.957, 1.248] | 0.885 [0.783, 1.002] | 0.873 [0.661, 1.090] | 0.707 [0.625, 0.806] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `sepsis_6h` | `lr` - `gbm` | -0.051 | [-0.065, -0.035] | 0.000 |
| `sepsis_6h` | `lr` - `ft_random` | 0.041 | [0.023, 0.066] | 0.000 |
| `sepsis_6h` | `lr` - `ft:ar_bins_s1` | -0.041 | [-0.057, -0.022] | 0.000 |
| `sepsis_6h` | `gbm` - `ft_random` | 0.093 | [0.074, 0.112] | 0.000 |
| `sepsis_6h` | `gbm` - `ft:ar_bins_s1` | 0.010 | [-0.002, 0.022] | 0.100 |
| `sepsis_6h` | `ft_random` - `ft:ar_bins_s1` | -0.083 | [-0.104, -0.064] | 0.000 |
| `sepsis_stay` | `lr` - `gbm` | -0.001 | [-0.023, 0.022] | 0.970 |
| `sepsis_stay` | `lr` - `ft_random` | 0.132 | [0.098, 0.170] | 0.000 |
| `sepsis_stay` | `lr` - `ft:ar_bins_s1` | -0.018 | [-0.038, 0.005] | 0.170 |
| `sepsis_stay` | `gbm` - `ft_random` | 0.133 | [0.100, 0.169] | 0.000 |
| `sepsis_stay` | `gbm` - `ft:ar_bins_s1` | -0.017 | [-0.039, 0.004] | 0.120 |
| `sepsis_stay` | `ft_random` - `ft:ar_bins_s1` | -0.150 | [-0.190, -0.116] | 0.000 |

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
| `mortality_inhospital/24h` | not defined on source family 'physionet2019' |
| `mortality_inhospital/48h` | not defined on source family 'physionet2019' |
