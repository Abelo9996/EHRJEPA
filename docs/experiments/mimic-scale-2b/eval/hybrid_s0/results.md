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
| created | 2026-10-06T07:17:22+00:00 |
| runtime (s) | 73.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-2b/hybrid_s0/final.pt` | last@final |

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

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.936 [0.922, 0.952] |
| `inpatient_365d` | 0.717 [0.689, 0.746] |
| `readmission_30d` | 0.604 [0.550, 0.654] |
| `new_dx_365d/diabetes` | 0.906 [0.875, 0.932] |
| `new_dx_365d/heart_failure` | 0.871 [0.833, 0.901] |
| `new_dx_365d/ckd` | 0.920 [0.893, 0.941] |
| `new_dx_365d/copd` | 0.768 [0.708, 0.833] |

## AUPRC

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.750 [0.699, 0.802] |
| `inpatient_365d` | 0.614 [0.567, 0.658] |
| `readmission_30d` | 0.201 [0.144, 0.284] |
| `new_dx_365d/diabetes` | 0.660 [0.572, 0.742] |
| `new_dx_365d/heart_failure` | 0.457 [0.359, 0.539] |
| `new_dx_365d/ckd` | 0.502 [0.396, 0.623] |
| `new_dx_365d/copd` | 0.116 [0.081, 0.194] |

## BRIER

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.0776 [0.0677, 0.0873] |
| `inpatient_365d` | 0.2045 [0.1935, 0.2157] |
| `readmission_30d` | 0.1058 [0.0915, 0.1199] |
| `new_dx_365d/diabetes` | 0.0638 [0.0537, 0.0730] |
| `new_dx_365d/heart_failure` | 0.0575 [0.0479, 0.0673] |
| `new_dx_365d/ckd` | 0.0495 [0.0394, 0.0597] |
| `new_dx_365d/copd` | 0.0384 [0.0291, 0.0490] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 1.233 [1.108, 1.439] |
| `inpatient_365d` | 0.826 [0.695, 0.970] |
| `readmission_30d` | 0.563 [0.290, 0.818] |
| `new_dx_365d/diabetes` | 1.713 [1.442, 2.079] |
| `new_dx_365d/heart_failure` | 1.107 [0.919, 1.298] |
| `new_dx_365d/ckd` | 1.371 [1.128, 1.699] |
| `new_dx_365d/copd` | 0.971 [0.724, 1.351] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
