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
| created | 2026-09-14T15:19:12+00:00 |
| runtime (s) | 535.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:latent_cont_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/latent_cont_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:latent_cont_s2 |
|---|---|
| `sepsis_6h` | 0.782 [0.762, 0.802] |
| `sepsis_stay` | 0.709 [0.675, 0.747] |

## AUPRC

| task | ckpt:latent_cont_s2 |
|---|---|
| `sepsis_6h` | 0.071 [0.060, 0.084] |
| `sepsis_stay` | 0.139 [0.112, 0.185] |

## BRIER

| task | ckpt:latent_cont_s2 |
|---|---|
| `sepsis_6h` | 0.0164 [0.0150, 0.0177] |
| `sepsis_stay` | 0.0506 [0.0445, 0.0556] |

## CALIBRATION SLOPE

| task | ckpt:latent_cont_s2 |
|---|---|
| `sepsis_6h` | 1.032 [0.950, 1.110] |
| `sepsis_stay` | 0.899 [0.741, 1.068] |

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
