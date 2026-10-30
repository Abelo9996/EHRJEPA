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
| created | 2026-10-23T13:56:35+00:00 |
| runtime (s) | 879.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:ar_s0` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-full-1b/ar_s0/final.pt` | last@final |

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
| `mortality_365d` | 0.947 [0.935, 0.962] |
| `inpatient_365d` | 0.716 [0.685, 0.751] |
| `readmission_30d` | 0.627 [0.570, 0.676] |
| `new_dx_365d/diabetes` | 0.914 [0.881, 0.935] |
| `new_dx_365d/heart_failure` | 0.893 [0.864, 0.919] |
| `new_dx_365d/ckd` | 0.937 [0.916, 0.954] |
| `new_dx_365d/copd` | 0.796 [0.739, 0.855] |

## AUPRC

| task | ft:ar_s0 |
|---|---|
| `mortality_365d` | 0.801 [0.749, 0.846] |
| `inpatient_365d` | 0.624 [0.574, 0.675] |
| `readmission_30d` | 0.213 [0.154, 0.289] |
| `new_dx_365d/diabetes` | 0.687 [0.587, 0.757] |
| `new_dx_365d/heart_failure` | 0.505 [0.407, 0.588] |
| `new_dx_365d/ckd` | 0.610 [0.488, 0.709] |
| `new_dx_365d/copd` | 0.190 [0.116, 0.304] |

## BRIER

| task | ft:ar_s0 |
|---|---|
| `mortality_365d` | 0.0727 [0.0623, 0.0827] |
| `inpatient_365d` | 0.2078 [0.1937, 0.2198] |
| `readmission_30d` | 0.1122 [0.0995, 0.1248] |
| `new_dx_365d/diabetes` | 0.0581 [0.0498, 0.0661] |
| `new_dx_365d/heart_failure` | 0.0551 [0.0462, 0.0648] |
| `new_dx_365d/ckd` | 0.0456 [0.0361, 0.0569] |
| `new_dx_365d/copd` | 0.0389 [0.0305, 0.0489] |

## CALIBRATION SLOPE

| task | ft:ar_s0 |
|---|---|
| `mortality_365d` | 1.006 [0.899, 1.174] |
| `inpatient_365d` | 0.648 [0.550, 0.770] |
| `readmission_30d` | 0.476 [0.282, 0.655] |
| `new_dx_365d/diabetes` | 1.204 [1.032, 1.405] |
| `new_dx_365d/heart_failure` | 0.748 [0.637, 0.862] |
| `new_dx_365d/ckd` | 0.782 [0.676, 0.904] |
| `new_dx_365d/copd` | 0.571 [0.437, 0.746] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
