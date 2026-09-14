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
| created | 2026-09-11T02:38:01+00:00 |
| runtime (s) | 549.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_cont_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/ar_cont_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:ar_cont_s1 |
|---|---|
| `sepsis_6h` | 0.819 [0.799, 0.835] |
| `sepsis_stay` | 0.749 [0.715, 0.783] |

## AUPRC

| task | ckpt:ar_cont_s1 |
|---|---|
| `sepsis_6h` | 0.076 [0.065, 0.091] |
| `sepsis_stay` | 0.173 [0.143, 0.218] |

## BRIER

| task | ckpt:ar_cont_s1 |
|---|---|
| `sepsis_6h` | 0.0164 [0.0150, 0.0177] |
| `sepsis_stay` | 0.0494 [0.0437, 0.0544] |

## CALIBRATION SLOPE

| task | ckpt:ar_cont_s1 |
|---|---|
| `sepsis_6h` | 1.044 [0.972, 1.118] |
| `sepsis_stay` | 1.061 [0.910, 1.214] |

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
