# Downstream evaluation -- mimic-iv

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `mimic-iv` |
| MEDS | `data/meds/mimic-iv` |
| cache | `data/cache/mimic-iv` |
| tasks | `data/tasks/mimic-iv` |
| anchor seed | 20260903 |
| commit | `b35a09f` |
| created | 2026-10-23T10:55:24+00:00 |
| runtime (s) | 101.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-full-1b/hybrid_s0/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 9044 (0.2074) | 1184 (0.2095) | 1120 (0.1875) |  |
| `inpatient_365d` | 9044 (0.3638) | 1184 (0.3910) | 1120 (0.3839) |  |
| `readmission_30d` | 8472 (0.1327) | 1063 (0.1251) | 1038 (0.1204) |  |
| `new_dx_365d/diabetes` | 8130 (0.1000) | 1076 (0.1115) | 1025 (0.1093) |  |
| `new_dx_365d/heart_failure` | 8509 (0.0885) | 1128 (0.0878) | 1044 (0.0805) |  |
| `new_dx_365d/ckd` | 8463 (0.0740) | 1124 (0.0783) | 1039 (0.0751) |  |
| `new_dx_365d/copd` | 8697 (0.0455) | 1146 (0.0489) | 1082 (0.0416) |  |

## AUROC

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.926 [0.911, 0.941] |
| `inpatient_365d` | 0.723 [0.691, 0.755] |
| `readmission_30d` | 0.646 [0.592, 0.689] |
| `new_dx_365d/diabetes` | 0.895 [0.862, 0.922] |
| `new_dx_365d/heart_failure` | 0.886 [0.853, 0.912] |
| `new_dx_365d/ckd` | 0.915 [0.889, 0.935] |
| `new_dx_365d/copd` | 0.752 [0.690, 0.812] |

## AUPRC

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.725 [0.669, 0.777] |
| `inpatient_365d` | 0.644 [0.594, 0.688] |
| `readmission_30d` | 0.228 [0.168, 0.298] |
| `new_dx_365d/diabetes` | 0.638 [0.535, 0.713] |
| `new_dx_365d/heart_failure` | 0.423 [0.328, 0.518] |
| `new_dx_365d/ckd` | 0.491 [0.380, 0.610] |
| `new_dx_365d/copd` | 0.130 [0.072, 0.213] |

## BRIER

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.0835 [0.0731, 0.0941] |
| `inpatient_365d` | 0.1997 [0.1889, 0.2099] |
| `readmission_30d` | 0.1033 [0.0891, 0.1178] |
| `new_dx_365d/diabetes` | 0.0646 [0.0535, 0.0732] |
| `new_dx_365d/heart_failure` | 0.0575 [0.0480, 0.0672] |
| `new_dx_365d/ckd` | 0.0507 [0.0403, 0.0606] |
| `new_dx_365d/copd` | 0.0386 [0.0290, 0.0489] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 1.045 [0.920, 1.186] |
| `inpatient_365d` | 0.967 [0.810, 1.142] |
| `readmission_30d` | 0.765 [0.490, 1.010] |
| `new_dx_365d/diabetes` | 1.570 [1.336, 1.907] |
| `new_dx_365d/heart_failure` | 1.122 [0.940, 1.350] |
| `new_dx_365d/ckd` | 1.304 [1.094, 1.589] |
| `new_dx_365d/copd` | 0.885 [0.629, 1.226] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
