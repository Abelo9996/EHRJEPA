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
| created | 2026-10-22T17:48:12+00:00 |
| runtime (s) | 925.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:ar_s0` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s0/final.pt` | last@final |

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

| task | ft:ar_s0 |
|---|---|
| `mortality_365d` | 0.949 [0.935, 0.963] |
| `inpatient_365d` | 0.719 [0.694, 0.750] |
| `readmission_30d` | 0.628 [0.565, 0.676] |
| `new_dx_365d/diabetes` | 0.914 [0.883, 0.937] |
| `new_dx_365d/heart_failure` | 0.894 [0.863, 0.920] |
| `new_dx_365d/ckd` | 0.944 [0.924, 0.959] |
| `new_dx_365d/copd` | 0.787 [0.728, 0.853] |

## AUPRC

| task | ft:ar_s0 |
|---|---|
| `mortality_365d` | 0.818 [0.774, 0.864] |
| `inpatient_365d` | 0.629 [0.582, 0.676] |
| `readmission_30d` | 0.210 [0.155, 0.285] |
| `new_dx_365d/diabetes` | 0.707 [0.604, 0.772] |
| `new_dx_365d/heart_failure` | 0.528 [0.425, 0.610] |
| `new_dx_365d/ckd` | 0.614 [0.497, 0.724] |
| `new_dx_365d/copd` | 0.189 [0.108, 0.315] |

## BRIER

| task | ft:ar_s0 |
|---|---|
| `mortality_365d` | 0.0690 [0.0590, 0.0773] |
| `inpatient_365d` | 0.2076 [0.1927, 0.2200] |
| `readmission_30d` | 0.1110 [0.0991, 0.1240] |
| `new_dx_365d/diabetes` | 0.0546 [0.0462, 0.0639] |
| `new_dx_365d/heart_failure` | 0.0544 [0.0460, 0.0643] |
| `new_dx_365d/ckd` | 0.0456 [0.0360, 0.0563] |
| `new_dx_365d/copd` | 0.0377 [0.0292, 0.0482] |

## CALIBRATION SLOPE

| task | ft:ar_s0 |
|---|---|
| `mortality_365d` | 1.028 [0.905, 1.200] |
| `inpatient_365d` | 0.609 [0.529, 0.724] |
| `readmission_30d` | 0.467 [0.248, 0.643] |
| `new_dx_365d/diabetes` | 0.904 [0.782, 1.029] |
| `new_dx_365d/heart_failure` | 0.741 [0.635, 0.854] |
| `new_dx_365d/ckd` | 0.762 [0.657, 0.886] |
| `new_dx_365d/copd` | 0.595 [0.448, 0.783] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
