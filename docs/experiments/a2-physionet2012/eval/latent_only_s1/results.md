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
| created | 2026-09-14T16:24:42+00:00 |
| runtime (s) | 65.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:latent_only_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/latent_only_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ckpt:latent_only_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.660 [0.607, 0.705] |
| `mortality_inhospital/48h` | 0.702 [0.664, 0.744] |

## AUPRC

| task | ckpt:latent_only_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.277 [0.210, 0.346] |
| `mortality_inhospital/48h` | 0.306 [0.249, 0.384] |

## BRIER

| task | ckpt:latent_only_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.1132 [0.0987, 0.1271] |
| `mortality_inhospital/48h` | 0.1106 [0.0964, 0.1253] |

## CALIBRATION SLOPE

| task | ckpt:latent_only_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.815 [0.537, 1.064] |
| `mortality_inhospital/48h` | 1.025 [0.799, 1.294] |

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
