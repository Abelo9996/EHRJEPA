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
| created | 2026-09-15T02:44:57+00:00 |
| runtime (s) | 2312.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:latent_only_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/latent_only_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ft:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.727 [0.702, 0.746] |
| `sepsis_stay` | 0.618 [0.586, 0.655] |

## AUPRC

| task | ft:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.042 [0.036, 0.050] |
| `sepsis_stay` | 0.093 [0.073, 0.127] |

## BRIER

| task | ft:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.0167 [0.0154, 0.0180] |
| `sepsis_stay` | 0.0520 [0.0463, 0.0574] |

## CALIBRATION SLOPE

| task | ft:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.921 [0.826, 0.992] |
| `sepsis_stay` | 0.692 [0.500, 0.905] |

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
