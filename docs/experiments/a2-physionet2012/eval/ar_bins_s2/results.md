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
| created | 2026-09-14T16:30:16+00:00 |
| runtime (s) | 63.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_bins_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/ar_bins_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ckpt:ar_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.760 [0.726, 0.792] |
| `mortality_inhospital/48h` | 0.794 [0.762, 0.824] |

## AUPRC

| task | ckpt:ar_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.320 [0.257, 0.390] |
| `mortality_inhospital/48h` | 0.381 [0.325, 0.457] |

## BRIER

| task | ckpt:ar_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.1081 [0.0950, 0.1207] |
| `mortality_inhospital/48h` | 0.1028 [0.0918, 0.1174] |

## CALIBRATION SLOPE

| task | ckpt:ar_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.852 [0.728, 1.005] |
| `mortality_inhospital/48h` | 0.867 [0.760, 0.994] |

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
