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
| created | 2026-10-09T15:59:41+00:00 |
| runtime (s) | 181.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.930 [0.921, 0.939] |
| `inpatient_365d` | 0.733 [0.716, 0.748] |
| `readmission_30d` | 0.672 [0.646, 0.695] |
| `new_dx_365d/diabetes` | 0.870 [0.852, 0.889] |
| `new_dx_365d/heart_failure` | 0.898 [0.878, 0.913] |
| `new_dx_365d/ckd` | 0.901 [0.885, 0.915] |
| `new_dx_365d/copd` | 0.809 [0.781, 0.839] |

## AUPRC

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.768 [0.738, 0.798] |
| `inpatient_365d` | 0.643 [0.617, 0.670] |
| `readmission_30d` | 0.284 [0.247, 0.321] |
| `new_dx_365d/diabetes` | 0.549 [0.498, 0.604] |
| `new_dx_365d/heart_failure` | 0.516 [0.449, 0.579] |
| `new_dx_365d/ckd` | 0.467 [0.398, 0.527] |
| `new_dx_365d/copd` | 0.138 [0.108, 0.195] |

## BRIER

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.0826 [0.0765, 0.0883] |
| `inpatient_365d` | 0.1951 [0.1893, 0.2011] |
| `readmission_30d` | 0.1081 [0.1006, 0.1150] |
| `new_dx_365d/diabetes` | 0.0673 [0.0604, 0.0746] |
| `new_dx_365d/heart_failure` | 0.0556 [0.0501, 0.0619] |
| `new_dx_365d/ckd` | 0.0548 [0.0481, 0.0598] |
| `new_dx_365d/copd` | 0.0392 [0.0346, 0.0440] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 1.083 [1.010, 1.162] |
| `inpatient_365d` | 0.935 [0.854, 1.007] |
| `readmission_30d` | 0.886 [0.753, 1.020] |
| `new_dx_365d/diabetes` | 1.099 [1.001, 1.242] |
| `new_dx_365d/heart_failure` | 1.066 [0.952, 1.176] |
| `new_dx_365d/ckd` | 1.094 [1.000, 1.213] |
| `new_dx_365d/copd` | 0.852 [0.722, 0.997] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
