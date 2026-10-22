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
| created | 2026-10-08T16:57:58+00:00 |
| runtime (s) | 167.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s0/final.pt` | last@final |

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

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.933 [0.924, 0.942] |
| `inpatient_365d` | 0.736 [0.720, 0.752] |
| `readmission_30d` | 0.671 [0.642, 0.691] |
| `new_dx_365d/diabetes` | 0.877 [0.857, 0.899] |
| `new_dx_365d/heart_failure` | 0.899 [0.880, 0.915] |
| `new_dx_365d/ckd` | 0.900 [0.883, 0.914] |
| `new_dx_365d/copd` | 0.823 [0.792, 0.848] |

## AUPRC

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.780 [0.752, 0.811] |
| `inpatient_365d` | 0.647 [0.619, 0.674] |
| `readmission_30d` | 0.272 [0.239, 0.311] |
| `new_dx_365d/diabetes` | 0.572 [0.521, 0.622] |
| `new_dx_365d/heart_failure` | 0.505 [0.440, 0.566] |
| `new_dx_365d/ckd` | 0.473 [0.407, 0.538] |
| `new_dx_365d/copd` | 0.156 [0.118, 0.207] |

## BRIER

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.0801 [0.0746, 0.0855] |
| `inpatient_365d` | 0.1941 [0.1886, 0.1997] |
| `readmission_30d` | 0.1086 [0.1011, 0.1159] |
| `new_dx_365d/diabetes` | 0.0652 [0.0582, 0.0719] |
| `new_dx_365d/heart_failure` | 0.0561 [0.0505, 0.0621] |
| `new_dx_365d/ckd` | 0.0545 [0.0480, 0.0596] |
| `new_dx_365d/copd` | 0.0383 [0.0337, 0.0434] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 1.020 [0.945, 1.104] |
| `inpatient_365d` | 0.929 [0.858, 1.013] |
| `readmission_30d` | 0.859 [0.719, 0.984] |
| `new_dx_365d/diabetes` | 1.023 [0.930, 1.157] |
| `new_dx_365d/heart_failure` | 1.073 [0.973, 1.178] |
| `new_dx_365d/ckd` | 1.091 [0.983, 1.210] |
| `new_dx_365d/copd` | 1.099 [0.926, 1.278] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
