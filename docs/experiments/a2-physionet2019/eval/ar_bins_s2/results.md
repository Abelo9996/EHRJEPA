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
| created | 2026-09-14T14:12:19+00:00 |
| runtime (s) | 537.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_bins_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/ar_bins_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:ar_bins_s2 |
|---|---|
| `sepsis_6h` | 0.814 [0.794, 0.832] |
| `sepsis_stay` | 0.745 [0.712, 0.777] |

## AUPRC

| task | ckpt:ar_bins_s2 |
|---|---|
| `sepsis_6h` | 0.080 [0.068, 0.093] |
| `sepsis_stay` | 0.161 [0.135, 0.206] |

## BRIER

| task | ckpt:ar_bins_s2 |
|---|---|
| `sepsis_6h` | 0.0163 [0.0149, 0.0177] |
| `sepsis_stay` | 0.0497 [0.0440, 0.0548] |

## CALIBRATION SLOPE

| task | ckpt:ar_bins_s2 |
|---|---|
| `sepsis_6h` | 1.054 [0.977, 1.134] |
| `sepsis_stay` | 0.954 [0.820, 1.108] |

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
