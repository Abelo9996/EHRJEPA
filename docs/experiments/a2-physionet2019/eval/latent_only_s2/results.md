# Downstream evaluation -- physionet2019

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2019` |
| MEDS | `data/meds/physionet2019` |
| cache | `data/cache/physionet2019` |
| tasks | `data/tasks/physionet2019` |
| anchor seed | 20260903 |
| commit | `039c63c` |
| created | 2026-09-14T15:42:06+00:00 |
| runtime (s) | 521.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:latent_only_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/latent_only_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:latent_only_s2 |
|---|---|
| `sepsis_6h` | 0.625 [0.603, 0.645] |
| `sepsis_stay` | 0.607 [0.570, 0.648] |

## AUPRC

| task | ckpt:latent_only_s2 |
|---|---|
| `sepsis_6h` | 0.036 [0.027, 0.047] |
| `sepsis_stay` | 0.080 [0.068, 0.099] |

## BRIER

| task | ckpt:latent_only_s2 |
|---|---|
| `sepsis_6h` | 0.0167 [0.0154, 0.0181] |
| `sepsis_stay` | 0.0523 [0.0466, 0.0577] |

## CALIBRATION SLOPE

| task | ckpt:latent_only_s2 |
|---|---|
| `sepsis_6h` | 1.041 [0.844, 1.205] |
| `sepsis_stay` | 0.696 [0.420, 0.975] |

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
