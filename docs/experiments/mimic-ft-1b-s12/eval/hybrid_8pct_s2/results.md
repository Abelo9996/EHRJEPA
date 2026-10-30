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
| created | 2026-10-22T22:00:16+00:00 |
| runtime (s) | 821.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_s2` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/hybrid_s2/final.pt` | last@final |

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

| task | ft:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.944 [0.930, 0.957] |
| `inpatient_365d` | 0.724 [0.698, 0.757] |
| `readmission_30d` | 0.634 [0.578, 0.676] |
| `new_dx_365d/diabetes` | 0.902 [0.867, 0.929] |
| `new_dx_365d/heart_failure` | 0.887 [0.856, 0.915] |
| `new_dx_365d/ckd` | 0.937 [0.917, 0.956] |
| `new_dx_365d/copd` | 0.780 [0.724, 0.839] |

## AUPRC

| task | ft:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.790 [0.743, 0.831] |
| `inpatient_365d` | 0.616 [0.573, 0.666] |
| `readmission_30d` | 0.212 [0.156, 0.278] |
| `new_dx_365d/diabetes` | 0.686 [0.591, 0.760] |
| `new_dx_365d/heart_failure` | 0.471 [0.352, 0.558] |
| `new_dx_365d/ckd` | 0.609 [0.503, 0.718] |
| `new_dx_365d/copd` | 0.130 [0.084, 0.223] |

## BRIER

| task | ft:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.0725 [0.0627, 0.0827] |
| `inpatient_365d` | 0.2085 [0.1926, 0.2213] |
| `readmission_30d` | 0.1036 [0.0882, 0.1188] |
| `new_dx_365d/diabetes` | 0.0566 [0.0460, 0.0664] |
| `new_dx_365d/heart_failure` | 0.0577 [0.0490, 0.0689] |
| `new_dx_365d/ckd` | 0.0442 [0.0354, 0.0546] |
| `new_dx_365d/copd` | 0.0396 [0.0307, 0.0499] |

## CALIBRATION SLOPE

| task | ft:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.966 [0.855, 1.105] |
| `inpatient_365d` | 0.580 [0.491, 0.690] |
| `readmission_30d` | 0.682 [0.412, 0.926] |
| `new_dx_365d/diabetes` | 1.040 [0.894, 1.208] |
| `new_dx_365d/heart_failure` | 0.792 [0.681, 0.942] |
| `new_dx_365d/ckd` | 0.789 [0.679, 0.941] |
| `new_dx_365d/copd` | 0.572 [0.436, 0.765] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
