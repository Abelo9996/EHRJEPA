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
| created | 2026-10-22T22:46:21+00:00 |
| runtime (s) | 1153.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:ar_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s1/final.pt` | last@final |

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

| task | ft:ar_s1 |
|---|---|
| `mortality_365d` | 0.945 [0.933, 0.961] |
| `inpatient_365d` | 0.722 [0.693, 0.754] |
| `readmission_30d` | 0.622 [0.566, 0.667] |
| `new_dx_365d/diabetes` | 0.913 [0.878, 0.939] |
| `new_dx_365d/heart_failure` | 0.885 [0.851, 0.911] |
| `new_dx_365d/ckd` | 0.936 [0.914, 0.955] |
| `new_dx_365d/copd` | 0.781 [0.726, 0.842] |

## AUPRC

| task | ft:ar_s1 |
|---|---|
| `mortality_365d` | 0.800 [0.755, 0.849] |
| `inpatient_365d` | 0.627 [0.574, 0.673] |
| `readmission_30d` | 0.220 [0.158, 0.297] |
| `new_dx_365d/diabetes` | 0.710 [0.610, 0.774] |
| `new_dx_365d/heart_failure` | 0.437 [0.349, 0.539] |
| `new_dx_365d/ckd` | 0.561 [0.465, 0.677] |
| `new_dx_365d/copd` | 0.175 [0.097, 0.294] |

## BRIER

| task | ft:ar_s1 |
|---|---|
| `mortality_365d` | 0.0726 [0.0620, 0.0823] |
| `inpatient_365d` | 0.2064 [0.1901, 0.2193] |
| `readmission_30d` | 0.1112 [0.0969, 0.1246] |
| `new_dx_365d/diabetes` | 0.0540 [0.0447, 0.0637] |
| `new_dx_365d/heart_failure` | 0.0591 [0.0485, 0.0699] |
| `new_dx_365d/ckd` | 0.0473 [0.0378, 0.0588] |
| `new_dx_365d/copd` | 0.0392 [0.0307, 0.0485] |

## CALIBRATION SLOPE

| task | ft:ar_s1 |
|---|---|
| `mortality_365d` | 0.968 [0.861, 1.130] |
| `inpatient_365d` | 0.619 [0.531, 0.736] |
| `readmission_30d` | 0.462 [0.262, 0.650] |
| `new_dx_365d/diabetes` | 0.909 [0.782, 1.047] |
| `new_dx_365d/heart_failure` | 0.746 [0.637, 0.868] |
| `new_dx_365d/ckd` | 0.784 [0.668, 0.948] |
| `new_dx_365d/copd` | 0.536 [0.406, 0.707] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
