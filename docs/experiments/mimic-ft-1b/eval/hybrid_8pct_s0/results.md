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
| created | 2026-10-22T17:33:56+00:00 |
| runtime (s) | 847.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_s0` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/hybrid_s0/final.pt` | last@final |

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

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.944 [0.930, 0.958] |
| `inpatient_365d` | 0.723 [0.692, 0.753] |
| `readmission_30d` | 0.629 [0.576, 0.677] |
| `new_dx_365d/diabetes` | 0.915 [0.886, 0.940] |
| `new_dx_365d/heart_failure` | 0.897 [0.863, 0.924] |
| `new_dx_365d/ckd` | 0.939 [0.918, 0.957] |
| `new_dx_365d/copd` | 0.812 [0.758, 0.871] |

## AUPRC

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.797 [0.741, 0.842] |
| `inpatient_365d` | 0.605 [0.554, 0.653] |
| `readmission_30d` | 0.227 [0.164, 0.317] |
| `new_dx_365d/diabetes` | 0.699 [0.609, 0.767] |
| `new_dx_365d/heart_failure` | 0.479 [0.381, 0.566] |
| `new_dx_365d/ckd` | 0.579 [0.473, 0.697] |
| `new_dx_365d/copd` | 0.184 [0.107, 0.289] |

## BRIER

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.0731 [0.0632, 0.0844] |
| `inpatient_365d` | 0.2118 [0.1967, 0.2256] |
| `readmission_30d` | 0.1131 [0.1010, 0.1255] |
| `new_dx_365d/diabetes` | 0.0559 [0.0470, 0.0647] |
| `new_dx_365d/heart_failure` | 0.0568 [0.0471, 0.0669] |
| `new_dx_365d/ckd` | 0.0457 [0.0370, 0.0553] |
| `new_dx_365d/copd` | 0.0375 [0.0291, 0.0478] |

## CALIBRATION SLOPE

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.926 [0.823, 1.057] |
| `inpatient_365d` | 0.546 [0.453, 0.653] |
| `readmission_30d` | 0.431 [0.266, 0.596] |
| `new_dx_365d/diabetes` | 1.206 [1.039, 1.389] |
| `new_dx_365d/heart_failure` | 0.784 [0.665, 0.931] |
| `new_dx_365d/ckd` | 0.772 [0.664, 0.911] |
| `new_dx_365d/copd` | 0.678 [0.524, 0.856] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
