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
| created | 2026-09-14T14:55:58+00:00 |
| runtime (s) | 529.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_bins_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/hybrid_bins_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `sepsis_6h` | 0.833 [0.817, 0.850] |
| `sepsis_stay` | 0.766 [0.735, 0.798] |

## AUPRC

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `sepsis_6h` | 0.097 [0.082, 0.114] |
| `sepsis_stay` | 0.181 [0.152, 0.233] |

## BRIER

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `sepsis_6h` | 0.0161 [0.0147, 0.0175] |
| `sepsis_stay` | 0.0491 [0.0437, 0.0542] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `sepsis_6h` | 1.072 [1.005, 1.140] |
| `sepsis_stay` | 0.959 [0.842, 1.081] |

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
