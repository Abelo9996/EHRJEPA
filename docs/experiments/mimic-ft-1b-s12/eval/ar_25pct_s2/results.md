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
| created | 2026-10-22T23:26:06+00:00 |
| runtime (s) | 1058.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:ar_s2` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s2/final.pt` | last@final |

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

| task | ft:ar_s2 |
|---|---|
| `mortality_365d` | 0.944 [0.930, 0.958] |
| `inpatient_365d` | 0.714 [0.686, 0.749] |
| `readmission_30d` | 0.615 [0.548, 0.664] |
| `new_dx_365d/diabetes` | 0.920 [0.894, 0.941] |
| `new_dx_365d/heart_failure` | 0.888 [0.857, 0.914] |
| `new_dx_365d/ckd` | 0.941 [0.922, 0.957] |
| `new_dx_365d/copd` | 0.814 [0.760, 0.873] |

## AUPRC

| task | ft:ar_s2 |
|---|---|
| `mortality_365d` | 0.789 [0.733, 0.843] |
| `inpatient_365d` | 0.619 [0.572, 0.661] |
| `readmission_30d` | 0.220 [0.154, 0.303] |
| `new_dx_365d/diabetes` | 0.697 [0.601, 0.762] |
| `new_dx_365d/heart_failure` | 0.509 [0.403, 0.582] |
| `new_dx_365d/ckd` | 0.575 [0.468, 0.692] |
| `new_dx_365d/copd` | 0.191 [0.115, 0.309] |

## BRIER

| task | ft:ar_s2 |
|---|---|
| `mortality_365d` | 0.0736 [0.0633, 0.0843] |
| `inpatient_365d` | 0.2097 [0.1930, 0.2224] |
| `readmission_30d` | 0.1102 [0.0968, 0.1237] |
| `new_dx_365d/diabetes` | 0.0565 [0.0480, 0.0672] |
| `new_dx_365d/heart_failure` | 0.0582 [0.0474, 0.0689] |
| `new_dx_365d/ckd` | 0.0461 [0.0381, 0.0563] |
| `new_dx_365d/copd` | 0.0387 [0.0302, 0.0492] |

## CALIBRATION SLOPE

| task | ft:ar_s2 |
|---|---|
| `mortality_365d` | 0.955 [0.854, 1.098] |
| `inpatient_365d` | 0.587 [0.500, 0.711] |
| `readmission_30d` | 0.460 [0.237, 0.642] |
| `new_dx_365d/diabetes` | 0.911 [0.794, 1.020] |
| `new_dx_365d/heart_failure` | 0.631 [0.538, 0.729] |
| `new_dx_365d/ckd` | 0.768 [0.652, 0.893] |
| `new_dx_365d/copd` | 0.644 [0.490, 0.868] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
