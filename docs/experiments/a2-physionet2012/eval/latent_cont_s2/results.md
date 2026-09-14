# Downstream evaluation -- physionet2012

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2012` |
| MEDS | `data/meds/physionet2012` |
| cache | `data/cache/physionet2012` |
| tasks | `data/tasks/physionet2012` |
| anchor seed | 20260903 |
| commit | `039c63c` |
| created | 2026-09-14T16:50:10+00:00 |
| runtime (s) | 68.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:latent_cont_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/latent_cont_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ckpt:latent_cont_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.736 [0.700, 0.780] |
| `mortality_inhospital/48h` | 0.778 [0.747, 0.819] |

## AUPRC

| task | ckpt:latent_cont_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.312 [0.253, 0.397] |
| `mortality_inhospital/48h` | 0.385 [0.325, 0.468] |

## BRIER

| task | ckpt:latent_cont_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.1096 [0.0961, 0.1221] |
| `mortality_inhospital/48h` | 0.1036 [0.0910, 0.1160] |

## CALIBRATION SLOPE

| task | ckpt:latent_cont_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.806 [0.671, 1.003] |
| `mortality_inhospital/48h` | 0.983 [0.862, 1.190] |

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
| `sepsis_6h` | not defined on source family 'physionet2012' |
| `sepsis_stay` | not defined on source family 'physionet2012' |
