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
| created | 2026-10-22T22:29:28+00:00 |
| runtime (s) | 1002.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s1/final.pt` | last@final |

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
| `mortality_365d` | 0.946 [0.932, 0.960] |
| `inpatient_365d` | 0.723 [0.696, 0.753] |
| `readmission_30d` | 0.642 [0.584, 0.685] |
| `new_dx_365d/diabetes` | 0.918 [0.889, 0.942] |
| `new_dx_365d/heart_failure` | 0.896 [0.863, 0.920] |
| `new_dx_365d/ckd` | 0.941 [0.922, 0.960] |
| `new_dx_365d/copd` | 0.800 [0.740, 0.855] |

## AUPRC

| task | ft:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.797 [0.743, 0.848] |
| `inpatient_365d` | 0.627 [0.577, 0.674] |
| `readmission_30d` | 0.226 [0.165, 0.296] |
| `new_dx_365d/diabetes` | 0.703 [0.617, 0.765] |
| `new_dx_365d/heart_failure` | 0.509 [0.395, 0.586] |
| `new_dx_365d/ckd` | 0.633 [0.511, 0.735] |
| `new_dx_365d/copd` | 0.214 [0.121, 0.326] |

## BRIER

| task | ft:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.0727 [0.0625, 0.0823] |
| `inpatient_365d` | 0.2072 [0.1914, 0.2197] |
| `readmission_30d` | 0.1031 [0.0875, 0.1183] |
| `new_dx_365d/diabetes` | 0.0554 [0.0443, 0.0647] |
| `new_dx_365d/heart_failure` | 0.0552 [0.0463, 0.0647] |
| `new_dx_365d/ckd` | 0.0436 [0.0354, 0.0537] |
| `new_dx_365d/copd` | 0.0374 [0.0289, 0.0466] |

## CALIBRATION SLOPE

| task | ft:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.942 [0.831, 1.082] |
| `inpatient_365d` | 0.586 [0.508, 0.690] |
| `readmission_30d` | 0.683 [0.420, 0.900] |
| `new_dx_365d/diabetes` | 0.936 [0.815, 1.078] |
| `new_dx_365d/heart_failure` | 0.766 [0.648, 0.892] |
| `new_dx_365d/ckd` | 0.819 [0.709, 0.958] |
| `new_dx_365d/copd` | 0.628 [0.478, 0.799] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
