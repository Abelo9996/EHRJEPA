# Downstream evaluation -- mimic-iv-25

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `mimic-iv-25` |
| MEDS | `data/meds/mimic-iv-25` |
| cache | `data/cache/mimic-iv-25` |
| tasks | `data/tasks/mimic-iv-25` |
| anchor seed | 20260903 |
| commit | `b35a09f` |
| created | 2026-10-09T05:26:38+00:00 |
| runtime (s) | 175.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 27644 (0.2133) | 3492 (0.2153) | 3331 (0.2017) |  |
| `inpatient_365d` | 27644 (0.3697) | 3492 (0.3691) | 3331 (0.3723) |  |
| `readmission_30d` | 25686 (0.1365) | 3201 (0.1340) | 3091 (0.1330) |  |
| `new_dx_365d/diabetes` | 24926 (0.1025) | 3170 (0.1000) | 3025 (0.1055) |  |
| `new_dx_365d/heart_failure` | 26017 (0.0852) | 3306 (0.0796) | 3132 (0.0843) |  |
| `new_dx_365d/ckd` | 25867 (0.0748) | 3297 (0.0673) | 3119 (0.0792) |  |
| `new_dx_365d/copd` | 26621 (0.0465) | 3374 (0.0492) | 3223 (0.0428) |  |

## AUROC

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.929 [0.920, 0.939] |
| `inpatient_365d` | 0.683 [0.666, 0.700] |
| `readmission_30d` | 0.642 [0.616, 0.666] |
| `new_dx_365d/diabetes` | 0.866 [0.845, 0.888] |
| `new_dx_365d/heart_failure` | 0.898 [0.879, 0.913] |
| `new_dx_365d/ckd` | 0.900 [0.882, 0.913] |
| `new_dx_365d/copd` | 0.827 [0.797, 0.858] |

## AUPRC

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.766 [0.738, 0.797] |
| `inpatient_365d` | 0.575 [0.543, 0.605] |
| `readmission_30d` | 0.251 [0.219, 0.288] |
| `new_dx_365d/diabetes` | 0.545 [0.495, 0.600] |
| `new_dx_365d/heart_failure` | 0.505 [0.439, 0.560] |
| `new_dx_365d/ckd` | 0.455 [0.385, 0.519] |
| `new_dx_365d/copd` | 0.164 [0.123, 0.218] |

## BRIER

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.0827 [0.0768, 0.0886] |
| `inpatient_365d` | 0.2100 [0.2051, 0.2152] |
| `readmission_30d` | 0.1108 [0.1034, 0.1181] |
| `new_dx_365d/diabetes` | 0.0677 [0.0611, 0.0750] |
| `new_dx_365d/heart_failure` | 0.0560 [0.0506, 0.0614] |
| `new_dx_365d/ckd` | 0.0552 [0.0490, 0.0601] |
| `new_dx_365d/copd` | 0.0382 [0.0337, 0.0432] |

## CALIBRATION SLOPE

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 1.077 [1.011, 1.169] |
| `inpatient_365d` | 0.857 [0.764, 0.946] |
| `readmission_30d` | 0.717 [0.592, 0.845] |
| `new_dx_365d/diabetes` | 1.099 [1.002, 1.225] |
| `new_dx_365d/heart_failure` | 1.022 [0.919, 1.128] |
| `new_dx_365d/ckd` | 1.112 [1.010, 1.224] |
| `new_dx_365d/copd` | 1.048 [0.888, 1.222] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
