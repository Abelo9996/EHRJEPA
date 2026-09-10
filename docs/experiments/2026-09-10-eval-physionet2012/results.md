# Downstream evaluation -- physionet2012

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2012` |
| MEDS | `data/meds/physionet2012` |
| cache | `data/cache/physionet2012` |
| tasks | `data/tasks/physionet2012` |
| anchor seed | 20260903 |
| commit | `bf1af4f` |
| created | 2026-09-10T17:45:13+00:00 |
| runtime (s) | 11.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | lr | gbm |
|---|---|---|
| `mortality_inhospital/24h` | 0.789 [0.756, 0.825] | 0.829 [0.797, 0.864] |
| `mortality_inhospital/48h` | 0.828 [0.797, 0.866] | 0.859 [0.829, 0.890] |

## AUPRC

| task | lr | gbm |
|---|---|---|
| `mortality_inhospital/24h` | 0.394 [0.319, 0.472] | 0.465 [0.385, 0.550] |
| `mortality_inhospital/48h` | 0.457 [0.389, 0.535] | 0.516 [0.435, 0.597] |

## BRIER

| task | lr | gbm |
|---|---|---|
| `mortality_inhospital/24h` | 0.1019 [0.0896, 0.1137] | 0.0948 [0.0835, 0.1062] |
| `mortality_inhospital/48h` | 0.0963 [0.0839, 0.1091] | 0.0907 [0.0794, 0.1030] |

## CALIBRATION SLOPE

| task | lr | gbm |
|---|---|---|
| `mortality_inhospital/24h` | 1.162 [0.995, 1.400] | 0.877 [0.749, 1.024] |
| `mortality_inhospital/48h` | 1.376 [1.207, 1.630] | 0.875 [0.758, 1.014] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_inhospital/24h` | `lr` - `gbm` | -0.040 | [-0.064, -0.016] | 0.000 |
| `mortality_inhospital/48h` | `lr` - `gbm` | -0.031 | [-0.050, -0.010] | 0.000 |

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
