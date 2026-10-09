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
| created | 2026-10-07T12:12:00+00:00 |
| runtime (s) | 72.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/hybrid_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.931 [0.915, 0.945] |
| `inpatient_365d` | 0.714 [0.683, 0.742] |
| `readmission_30d` | 0.626 [0.566, 0.667] |
| `new_dx_365d/diabetes` | 0.887 [0.850, 0.917] |
| `new_dx_365d/heart_failure` | 0.882 [0.846, 0.911] |
| `new_dx_365d/ckd` | 0.910 [0.884, 0.935] |
| `new_dx_365d/copd` | 0.742 [0.683, 0.807] |

## AUPRC

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.733 [0.675, 0.785] |
| `inpatient_365d` | 0.613 [0.565, 0.664] |
| `readmission_30d` | 0.208 [0.150, 0.278] |
| `new_dx_365d/diabetes` | 0.636 [0.533, 0.712] |
| `new_dx_365d/heart_failure` | 0.463 [0.360, 0.561] |
| `new_dx_365d/ckd` | 0.498 [0.395, 0.621] |
| `new_dx_365d/copd` | 0.099 [0.067, 0.176] |

## BRIER

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.0813 [0.0726, 0.0905] |
| `inpatient_365d` | 0.2048 [0.1943, 0.2159] |
| `readmission_30d` | 0.1046 [0.0904, 0.1192] |
| `new_dx_365d/diabetes` | 0.0656 [0.0547, 0.0745] |
| `new_dx_365d/heart_failure` | 0.0570 [0.0478, 0.0675] |
| `new_dx_365d/ckd` | 0.0501 [0.0397, 0.0595] |
| `new_dx_365d/copd` | 0.0390 [0.0295, 0.0493] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 1.244 [1.103, 1.416] |
| `inpatient_365d` | 0.854 [0.717, 1.007] |
| `readmission_30d` | 0.655 [0.371, 0.902] |
| `new_dx_365d/diabetes` | 1.579 [1.310, 1.915] |
| `new_dx_365d/heart_failure` | 1.160 [0.963, 1.383] |
| `new_dx_365d/ckd` | 1.281 [1.078, 1.601] |
| `new_dx_365d/copd` | 0.863 [0.619, 1.205] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
