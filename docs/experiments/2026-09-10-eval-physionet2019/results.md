# Downstream evaluation -- physionet2019

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2019` |
| MEDS | `data/meds/physionet2019` |
| cache | `data/cache/physionet2019` |
| tasks | `data/tasks/physionet2019` |
| anchor seed | 20260903 |
| commit | `bf1af4f` |
| created | 2026-09-10T17:45:27+00:00 |
| runtime (s) | 59.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | lr | gbm |
|---|---|---|
| `sepsis_6h` | 0.785 [0.766, 0.805] | 0.836 [0.820, 0.855] |
| `sepsis_stay` | 0.768 [0.737, 0.799] | 0.769 [0.735, 0.800] |

## AUPRC

| task | lr | gbm |
|---|---|---|
| `sepsis_6h` | 0.068 [0.058, 0.083] | 0.095 [0.081, 0.110] |
| `sepsis_stay` | 0.194 [0.157, 0.246] | 0.186 [0.151, 0.238] |

## BRIER

| task | lr | gbm |
|---|---|---|
| `sepsis_6h` | 0.0165 [0.0151, 0.0178] | 0.0163 [0.0150, 0.0176] |
| `sepsis_stay` | 0.0488 [0.0435, 0.0535] | 0.0493 [0.0441, 0.0543] |

## CALIBRATION SLOPE

| task | lr | gbm |
|---|---|---|
| `sepsis_6h` | 1.026 [0.950, 1.105] | 0.996 [0.931, 1.053] |
| `sepsis_stay` | 1.100 [0.957, 1.248] | 0.885 [0.783, 1.002] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `sepsis_6h` | `lr` - `gbm` | -0.051 | [-0.065, -0.035] | 0.000 |
| `sepsis_stay` | `lr` - `gbm` | -0.001 | [-0.023, 0.022] | 0.970 |

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
