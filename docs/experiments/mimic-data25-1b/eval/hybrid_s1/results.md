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
| created | 2026-10-09T02:48:09+00:00 |
| runtime (s) | 185.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.927 [0.917, 0.937] |
| `inpatient_365d` | 0.733 [0.716, 0.749] |
| `readmission_30d` | 0.669 [0.642, 0.694] |
| `new_dx_365d/diabetes` | 0.872 [0.852, 0.892] |
| `new_dx_365d/heart_failure` | 0.896 [0.877, 0.914] |
| `new_dx_365d/ckd` | 0.896 [0.882, 0.912] |
| `new_dx_365d/copd` | 0.821 [0.791, 0.853] |

## AUPRC

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.771 [0.743, 0.800] |
| `inpatient_365d` | 0.648 [0.621, 0.676] |
| `readmission_30d` | 0.274 [0.242, 0.313] |
| `new_dx_365d/diabetes` | 0.542 [0.489, 0.595] |
| `new_dx_365d/heart_failure` | 0.507 [0.449, 0.576] |
| `new_dx_365d/ckd` | 0.460 [0.397, 0.527] |
| `new_dx_365d/copd` | 0.154 [0.117, 0.203] |

## BRIER

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.0826 [0.0763, 0.0891] |
| `inpatient_365d` | 0.1945 [0.1885, 0.2007] |
| `readmission_30d` | 0.1086 [0.1011, 0.1156] |
| `new_dx_365d/diabetes` | 0.0676 [0.0610, 0.0744] |
| `new_dx_365d/heart_failure` | 0.0561 [0.0508, 0.0621] |
| `new_dx_365d/ckd` | 0.0551 [0.0486, 0.0602] |
| `new_dx_365d/copd` | 0.0384 [0.0338, 0.0437] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.942 [0.863, 1.021] |
| `inpatient_365d` | 0.954 [0.868, 1.046] |
| `readmission_30d` | 0.867 [0.735, 1.000] |
| `new_dx_365d/diabetes` | 1.124 [1.017, 1.251] |
| `new_dx_365d/heart_failure` | 0.947 [0.858, 1.066] |
| `new_dx_365d/ckd` | 1.072 [0.984, 1.201] |
| `new_dx_365d/copd` | 1.077 [0.919, 1.249] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
