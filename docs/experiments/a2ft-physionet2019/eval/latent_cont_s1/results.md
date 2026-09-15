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
| created | 2026-09-15T01:54:10+00:00 |
| runtime (s) | 3044.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:latent_cont_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/latent_cont_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ft:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 0.810 [0.793, 0.826] |
| `sepsis_stay` | 0.752 [0.719, 0.788] |

## AUPRC

| task | ft:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 0.068 [0.059, 0.079] |
| `sepsis_stay` | 0.177 [0.149, 0.223] |

## BRIER

| task | ft:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 0.0169 [0.0156, 0.0182] |
| `sepsis_stay` | 0.0493 [0.0437, 0.0542] |

## CALIBRATION SLOPE

| task | ft:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 0.674 [0.626, 0.717] |
| `sepsis_stay` | 0.814 [0.703, 0.959] |

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
