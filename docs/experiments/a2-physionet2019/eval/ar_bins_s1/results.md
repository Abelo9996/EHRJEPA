# Downstream evaluation -- physionet2019

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2019` |
| MEDS | `data/meds/physionet2019` |
| cache | `data/cache/physionet2019` |
| tasks | `data/tasks/physionet2019` |
| anchor seed | 20260903 |
| commit | `89bdf94` |
| created | 2026-09-11T02:07:32+00:00 |
| runtime (s) | 1173.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/ar_bins_s1/final.pt` | last@final |
| `ckpt:ar_bins_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/ar_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 0.785 [0.766, 0.805] | 0.836 [0.820, 0.855] | 0.686 [0.663, 0.707] | 0.821 [0.799, 0.837] |
| `sepsis_stay` | 0.768 [0.737, 0.799] | 0.769 [0.735, 0.800] | 0.627 [0.593, 0.661] | 0.768 [0.737, 0.796] |

## AUPRC

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 0.068 [0.058, 0.083] | 0.095 [0.081, 0.110] | 0.041 [0.035, 0.050] | 0.079 [0.067, 0.090] |
| `sepsis_stay` | 0.194 [0.157, 0.246] | 0.186 [0.151, 0.238] | 0.096 [0.078, 0.126] | 0.176 [0.145, 0.224] |

## BRIER

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 0.0165 [0.0151, 0.0178] | 0.0163 [0.0150, 0.0176] | 0.0167 [0.0153, 0.0181] | 0.0164 [0.0150, 0.0178] |
| `sepsis_stay` | 0.0488 [0.0435, 0.0535] | 0.0493 [0.0441, 0.0543] | 0.0519 [0.0462, 0.0573] | 0.0493 [0.0437, 0.0542] |

## CALIBRATION SLOPE

| task | lr | gbm | random_init | ckpt:ar_bins_s1 |
|---|---|---|---|---|
| `sepsis_6h` | 1.026 [0.950, 1.105] | 0.996 [0.931, 1.053] | 1.270 [1.123, 1.418] | 1.037 [0.957, 1.106] |
| `sepsis_stay` | 1.100 [0.957, 1.248] | 0.885 [0.783, 1.002] | 0.744 [0.539, 0.943] | 1.018 [0.878, 1.167] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `sepsis_6h` | `lr` - `gbm` | -0.051 | [-0.065, -0.035] | 0.000 |
| `sepsis_6h` | `lr` - `random_init` | 0.099 | [0.076, 0.123] | 0.000 |
| `sepsis_6h` | `lr` - `ckpt:ar_bins_s1` | -0.035 | [-0.051, -0.017] | 0.000 |
| `sepsis_6h` | `gbm` - `random_init` | 0.150 | [0.127, 0.171] | 0.000 |
| `sepsis_6h` | `gbm` - `ckpt:ar_bins_s1` | 0.016 | [0.006, 0.027] | 0.010 |
| `sepsis_6h` | `random_init` - `ckpt:ar_bins_s1` | -0.135 | [-0.151, -0.115] | 0.000 |
| `sepsis_stay` | `lr` - `gbm` | -0.001 | [-0.023, 0.022] | 0.970 |
| `sepsis_stay` | `lr` - `random_init` | 0.141 | [0.105, 0.178] | 0.000 |
| `sepsis_stay` | `lr` - `ckpt:ar_bins_s1` | 0.001 | [-0.021, 0.021] | 0.890 |
| `sepsis_stay` | `gbm` - `random_init` | 0.142 | [0.102, 0.179] | 0.000 |
| `sepsis_stay` | `gbm` - `ckpt:ar_bins_s1` | 0.002 | [-0.021, 0.026] | 0.960 |
| `sepsis_stay` | `random_init` - `ckpt:ar_bins_s1` | -0.141 | [-0.178, -0.106] | 0.000 |

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
