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
| created | 2026-10-22T21:45:37+00:00 |
| runtime (s) | 870.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:ar_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/ar_s1/final.pt` | last@final |

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
| `mortality_365d` | 0.941 [0.927, 0.957] |
| `inpatient_365d` | 0.711 [0.682, 0.744] |
| `readmission_30d` | 0.626 [0.569, 0.676] |
| `new_dx_365d/diabetes` | 0.913 [0.885, 0.935] |
| `new_dx_365d/heart_failure` | 0.884 [0.852, 0.914] |
| `new_dx_365d/ckd` | 0.934 [0.912, 0.954] |
| `new_dx_365d/copd` | 0.792 [0.740, 0.852] |

## AUPRC

| task | ft:ar_s1 |
|---|---|
| `mortality_365d` | 0.784 [0.724, 0.833] |
| `inpatient_365d` | 0.606 [0.560, 0.658] |
| `readmission_30d` | 0.216 [0.160, 0.298] |
| `new_dx_365d/diabetes` | 0.660 [0.559, 0.733] |
| `new_dx_365d/heart_failure` | 0.424 [0.323, 0.529] |
| `new_dx_365d/ckd` | 0.551 [0.444, 0.667] |
| `new_dx_365d/copd` | 0.214 [0.123, 0.326] |

## BRIER

| task | ft:ar_s1 |
|---|---|
| `mortality_365d` | 0.0748 [0.0649, 0.0852] |
| `inpatient_365d` | 0.2075 [0.1942, 0.2189] |
| `readmission_30d` | 0.1113 [0.0971, 0.1250] |
| `new_dx_365d/diabetes` | 0.0607 [0.0505, 0.0705] |
| `new_dx_365d/heart_failure` | 0.0590 [0.0487, 0.0694] |
| `new_dx_365d/ckd` | 0.0494 [0.0415, 0.0600] |
| `new_dx_365d/copd` | 0.0383 [0.0295, 0.0485] |

## CALIBRATION SLOPE

| task | ft:ar_s1 |
|---|---|
| `mortality_365d` | 0.953 [0.840, 1.112] |
| `inpatient_365d` | 0.784 [0.661, 0.940] |
| `readmission_30d` | 0.441 [0.255, 0.630] |
| `new_dx_365d/diabetes` | 0.820 [0.724, 0.938] |
| `new_dx_365d/heart_failure` | 0.761 [0.634, 0.907] |
| `new_dx_365d/ckd` | 0.712 [0.607, 0.835] |
| `new_dx_365d/copd` | 0.540 [0.412, 0.691] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
