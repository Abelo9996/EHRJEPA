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
| created | 2026-10-23T13:39:05+00:00 |
| runtime (s) | 100.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-full-1b/ar_s0/final.pt` | last@final |

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

| task | ckpt:ar_s0 |
|---|---|
| `mortality_365d` | 0.934 [0.919, 0.949] |
| `inpatient_365d` | 0.691 [0.659, 0.730] |
| `readmission_30d` | 0.617 [0.559, 0.658] |
| `new_dx_365d/diabetes` | 0.883 [0.845, 0.910] |
| `new_dx_365d/heart_failure` | 0.876 [0.841, 0.901] |
| `new_dx_365d/ckd` | 0.913 [0.885, 0.935] |
| `new_dx_365d/copd` | 0.747 [0.686, 0.818] |

## AUPRC

| task | ckpt:ar_s0 |
|---|---|
| `mortality_365d` | 0.753 [0.693, 0.805] |
| `inpatient_365d` | 0.593 [0.541, 0.638] |
| `readmission_30d` | 0.201 [0.153, 0.256] |
| `new_dx_365d/diabetes` | 0.594 [0.495, 0.680] |
| `new_dx_365d/heart_failure` | 0.423 [0.323, 0.512] |
| `new_dx_365d/ckd` | 0.474 [0.365, 0.591] |
| `new_dx_365d/copd` | 0.101 [0.068, 0.159] |

## BRIER

| task | ckpt:ar_s0 |
|---|---|
| `mortality_365d` | 0.0801 [0.0701, 0.0890] |
| `inpatient_365d` | 0.2104 [0.1980, 0.2200] |
| `readmission_30d` | 0.1055 [0.0917, 0.1205] |
| `new_dx_365d/diabetes` | 0.0679 [0.0563, 0.0767] |
| `new_dx_365d/heart_failure` | 0.0584 [0.0491, 0.0687] |
| `new_dx_365d/ckd` | 0.0516 [0.0419, 0.0617] |
| `new_dx_365d/copd` | 0.0390 [0.0294, 0.0496] |

## CALIBRATION SLOPE

| task | ckpt:ar_s0 |
|---|---|
| `mortality_365d` | 1.261 [1.112, 1.443] |
| `inpatient_365d` | 1.000 [0.821, 1.218] |
| `readmission_30d` | 0.599 [0.313, 0.838] |
| `new_dx_365d/diabetes` | 1.431 [1.210, 1.678] |
| `new_dx_365d/heart_failure` | 1.025 [0.842, 1.197] |
| `new_dx_365d/ckd` | 1.312 [1.093, 1.573] |
| `new_dx_365d/copd` | 0.822 [0.603, 1.150] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
