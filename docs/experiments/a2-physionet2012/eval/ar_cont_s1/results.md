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
| created | 2026-09-14T16:02:04+00:00 |
| runtime (s) | 63.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_cont_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/ar_cont_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ckpt:ar_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.777 [0.748, 0.813] |
| `mortality_inhospital/48h` | 0.814 [0.785, 0.848] |

## AUPRC

| task | ckpt:ar_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.342 [0.272, 0.432] |
| `mortality_inhospital/48h` | 0.397 [0.333, 0.480] |

## BRIER

| task | ckpt:ar_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.1057 [0.0929, 0.1166] |
| `mortality_inhospital/48h` | 0.1010 [0.0893, 0.1130] |

## CALIBRATION SLOPE

| task | ckpt:ar_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.928 [0.792, 1.112] |
| `mortality_inhospital/48h` | 0.883 [0.779, 1.053] |

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
