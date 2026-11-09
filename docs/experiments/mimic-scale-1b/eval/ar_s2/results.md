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
| created | 2026-10-07T08:47:12+00:00 |
| runtime (s) | 66.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/ar_s2/final.pt` | last@final |

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

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.913 [0.895, 0.929] |
| `inpatient_365d` | 0.653 [0.622, 0.689] |
| `readmission_30d` | 0.600 [0.539, 0.644] |
| `new_dx_365d/diabetes` | 0.856 [0.822, 0.886] |
| `new_dx_365d/heart_failure` | 0.869 [0.831, 0.897] |
| `new_dx_365d/ckd` | 0.893 [0.858, 0.922] |
| `new_dx_365d/copd` | 0.730 [0.668, 0.808] |

## AUPRC

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.699 [0.635, 0.761] |
| `inpatient_365d` | 0.547 [0.496, 0.597] |
| `readmission_30d` | 0.184 [0.134, 0.244] |
| `new_dx_365d/diabetes` | 0.495 [0.406, 0.571] |
| `new_dx_365d/heart_failure` | 0.406 [0.310, 0.494] |
| `new_dx_365d/ckd` | 0.429 [0.315, 0.572] |
| `new_dx_365d/copd` | 0.108 [0.071, 0.165] |

## BRIER

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.0898 [0.0806, 0.0994] |
| `inpatient_365d` | 0.2201 [0.2100, 0.2291] |
| `readmission_30d` | 0.1061 [0.0915, 0.1218] |
| `new_dx_365d/diabetes` | 0.0745 [0.0636, 0.0844] |
| `new_dx_365d/heart_failure` | 0.0593 [0.0488, 0.0697] |
| `new_dx_365d/ckd` | 0.0541 [0.0442, 0.0640] |
| `new_dx_365d/copd` | 0.0388 [0.0295, 0.0497] |

## CALIBRATION SLOPE

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 1.162 [1.014, 1.312] |
| `inpatient_365d` | 0.832 [0.654, 1.019] |
| `readmission_30d` | 0.571 [0.268, 0.823] |
| `new_dx_365d/diabetes` | 1.305 [1.099, 1.516] |
| `new_dx_365d/heart_failure` | 1.095 [0.906, 1.275] |
| `new_dx_365d/ckd` | 1.425 [1.178, 1.743] |
| `new_dx_365d/copd` | 0.803 [0.561, 1.174] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
