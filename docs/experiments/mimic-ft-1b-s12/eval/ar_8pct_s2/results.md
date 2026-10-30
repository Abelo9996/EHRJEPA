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
| created | 2026-10-22T22:14:06+00:00 |
| runtime (s) | 913.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:ar_s2` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/ar_s2/final.pt` | last@final |

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
| `mortality_365d` | 0.936 [0.922, 0.951] |
| `inpatient_365d` | 0.710 [0.682, 0.739] |
| `readmission_30d` | 0.607 [0.555, 0.655] |
| `new_dx_365d/diabetes` | 0.904 [0.872, 0.933] |
| `new_dx_365d/heart_failure` | 0.883 [0.855, 0.911] |
| `new_dx_365d/ckd` | 0.940 [0.922, 0.956] |
| `new_dx_365d/copd` | 0.791 [0.731, 0.862] |

## AUPRC

| task | ft:ar_s2 |
|---|---|
| `mortality_365d` | 0.776 [0.727, 0.822] |
| `inpatient_365d` | 0.610 [0.564, 0.662] |
| `readmission_30d` | 0.202 [0.154, 0.278] |
| `new_dx_365d/diabetes` | 0.656 [0.564, 0.726] |
| `new_dx_365d/heart_failure` | 0.450 [0.345, 0.541] |
| `new_dx_365d/ckd` | 0.577 [0.469, 0.695] |
| `new_dx_365d/copd` | 0.151 [0.089, 0.238] |

## BRIER

| task | ft:ar_s2 |
|---|---|
| `mortality_365d` | 0.0801 [0.0699, 0.0902] |
| `inpatient_365d` | 0.2104 [0.1967, 0.2238] |
| `readmission_30d` | 0.1123 [0.0981, 0.1274] |
| `new_dx_365d/diabetes` | 0.0604 [0.0510, 0.0694] |
| `new_dx_365d/heart_failure` | 0.0600 [0.0491, 0.0712] |
| `new_dx_365d/ckd` | 0.0460 [0.0370, 0.0558] |
| `new_dx_365d/copd` | 0.0401 [0.0313, 0.0503] |

## CALIBRATION SLOPE

| task | ft:ar_s2 |
|---|---|
| `mortality_365d` | 0.968 [0.860, 1.122] |
| `inpatient_365d` | 0.618 [0.520, 0.730] |
| `readmission_30d` | 0.384 [0.210, 0.573] |
| `new_dx_365d/diabetes` | 1.077 [0.940, 1.240] |
| `new_dx_365d/heart_failure` | 0.613 [0.526, 0.728] |
| `new_dx_365d/ckd` | 0.770 [0.669, 0.879] |
| `new_dx_365d/copd` | 0.513 [0.378, 0.710] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
