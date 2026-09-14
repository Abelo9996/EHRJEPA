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
| created | 2026-09-11T03:58:50+00:00 |
| runtime (s) | 540.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:latent_only_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/latent_only_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.624 [0.602, 0.644] |
| `sepsis_stay` | 0.580 [0.542, 0.624] |

## AUPRC

| task | ckpt:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.026 [0.023, 0.030] |
| `sepsis_stay` | 0.075 [0.060, 0.099] |

## BRIER

| task | ckpt:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.0168 [0.0154, 0.0182] |
| `sepsis_stay` | 0.0523 [0.0466, 0.0576] |

## CALIBRATION SLOPE

| task | ckpt:latent_only_s1 |
|---|---|
| `sepsis_6h` | 0.942 [0.759, 1.084] |
| `sepsis_stay` | 0.762 [0.406, 1.216] |

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
