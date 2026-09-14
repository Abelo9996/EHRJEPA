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
| created | 2026-09-14T14:31:59+00:00 |
| runtime (s) | 537.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_cont_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/ar_cont_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:ar_cont_s2 |
|---|---|
| `sepsis_6h` | 0.823 [0.803, 0.840] |
| `sepsis_stay` | 0.749 [0.721, 0.777] |

## AUPRC

| task | ckpt:ar_cont_s2 |
|---|---|
| `sepsis_6h` | 0.087 [0.073, 0.103] |
| `sepsis_stay` | 0.152 [0.129, 0.194] |

## BRIER

| task | ckpt:ar_cont_s2 |
|---|---|
| `sepsis_6h` | 0.0162 [0.0149, 0.0175] |
| `sepsis_stay` | 0.0504 [0.0449, 0.0553] |

## CALIBRATION SLOPE

| task | ckpt:ar_cont_s2 |
|---|---|
| `sepsis_6h` | 1.046 [0.965, 1.123] |
| `sepsis_stay` | 0.868 [0.744, 0.994] |

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
