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
| created | 2026-09-14T16:57:20+00:00 |
| runtime (s) | 68.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:latent_only_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/latent_only_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ckpt:latent_only_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.650 [0.598, 0.697] |
| `mortality_inhospital/48h` | 0.710 [0.666, 0.752] |

## AUPRC

| task | ckpt:latent_only_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.277 [0.211, 0.347] |
| `mortality_inhospital/48h` | 0.323 [0.258, 0.405] |

## BRIER

| task | ckpt:latent_only_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.1136 [0.1001, 0.1283] |
| `mortality_inhospital/48h` | 0.1094 [0.0960, 0.1233] |

## CALIBRATION SLOPE

| task | ckpt:latent_only_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.759 [0.514, 0.987] |
| `mortality_inhospital/48h` | 1.005 [0.763, 1.246] |

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
