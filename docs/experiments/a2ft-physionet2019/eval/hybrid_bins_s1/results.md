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
| created | 2026-09-15T01:14:51+00:00 |
| runtime (s) | 2355.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_bins_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/hybrid_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ft:hybrid_bins_s1 |
|---|---|
| `sepsis_6h` | 0.840 [0.822, 0.856] |
| `sepsis_stay` | 0.773 [0.737, 0.805] |

## AUPRC

| task | ft:hybrid_bins_s1 |
|---|---|
| `sepsis_6h` | 0.090 [0.077, 0.104] |
| `sepsis_stay` | 0.208 [0.172, 0.267] |

## BRIER

| task | ft:hybrid_bins_s1 |
|---|---|
| `sepsis_6h` | 0.0161 [0.0147, 0.0174] |
| `sepsis_stay` | 0.0481 [0.0426, 0.0532] |

## CALIBRATION SLOPE

| task | ft:hybrid_bins_s1 |
|---|---|
| `sepsis_6h` | 1.200 [1.112, 1.286] |
| `sepsis_stay` | 0.805 [0.706, 0.910] |

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
