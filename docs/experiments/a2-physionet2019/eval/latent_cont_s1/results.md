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
| created | 2026-09-11T03:35:13+00:00 |
| runtime (s) | 549.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:latent_cont_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/latent_cont_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | ckpt:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 0.775 [0.752, 0.795] |
| `sepsis_stay` | 0.683 [0.653, 0.720] |

## AUPRC

| task | ckpt:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 0.082 [0.065, 0.100] |
| `sepsis_stay` | 0.123 [0.102, 0.164] |

## BRIER

| task | ckpt:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 0.0163 [0.0149, 0.0177] |
| `sepsis_stay` | 0.0514 [0.0457, 0.0570] |

## CALIBRATION SLOPE

| task | ckpt:latent_cont_s1 |
|---|---|
| `sepsis_6h` | 1.019 [0.932, 1.092] |
| `sepsis_stay` | 0.736 [0.591, 0.878] |

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
