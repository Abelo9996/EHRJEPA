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
| created | 2026-10-22T21:30:58+00:00 |
| runtime (s) | 870.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/hybrid_s1/final.pt` | last@final |

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

| task | ft:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.943 [0.928, 0.956] |
| `inpatient_365d` | 0.711 [0.683, 0.742] |
| `readmission_30d` | 0.609 [0.553, 0.663] |
| `new_dx_365d/diabetes` | 0.920 [0.890, 0.943] |
| `new_dx_365d/heart_failure` | 0.888 [0.852, 0.913] |
| `new_dx_365d/ckd` | 0.937 [0.916, 0.957] |
| `new_dx_365d/copd` | 0.785 [0.727, 0.849] |

## AUPRC

| task | ft:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.803 [0.757, 0.847] |
| `inpatient_365d` | 0.604 [0.557, 0.651] |
| `readmission_30d` | 0.198 [0.147, 0.268] |
| `new_dx_365d/diabetes` | 0.715 [0.623, 0.780] |
| `new_dx_365d/heart_failure` | 0.480 [0.364, 0.575] |
| `new_dx_365d/ckd` | 0.602 [0.493, 0.718] |
| `new_dx_365d/copd` | 0.131 [0.087, 0.207] |

## BRIER

| task | ft:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.0714 [0.0622, 0.0826] |
| `inpatient_365d` | 0.2146 [0.1995, 0.2285] |
| `readmission_30d` | 0.1147 [0.1009, 0.1279] |
| `new_dx_365d/diabetes` | 0.0537 [0.0444, 0.0640] |
| `new_dx_365d/heart_failure` | 0.0569 [0.0481, 0.0674] |
| `new_dx_365d/ckd` | 0.0456 [0.0370, 0.0561] |
| `new_dx_365d/copd` | 0.0406 [0.0319, 0.0512] |

## CALIBRATION SLOPE

| task | ft:hybrid_s1 |
|---|---|
| `mortality_365d` | 1.014 [0.902, 1.140] |
| `inpatient_365d` | 0.533 [0.448, 0.630] |
| `readmission_30d` | 0.378 [0.204, 0.553] |
| `new_dx_365d/diabetes` | 0.952 [0.825, 1.088] |
| `new_dx_365d/heart_failure` | 0.791 [0.662, 0.926] |
| `new_dx_365d/ckd` | 0.768 [0.658, 0.898] |
| `new_dx_365d/copd` | 0.527 [0.406, 0.690] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
