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
| created | 2026-10-22T23:05:45+00:00 |
| runtime (s) | 1209.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_s2` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s2/final.pt` | last@final |

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
| `mortality_365d` | 0.945 [0.931, 0.958] |
| `inpatient_365d` | 0.730 [0.702, 0.761] |
| `readmission_30d` | 0.652 [0.592, 0.699] |
| `new_dx_365d/diabetes` | 0.910 [0.880, 0.936] |
| `new_dx_365d/heart_failure` | 0.895 [0.864, 0.919] |
| `new_dx_365d/ckd` | 0.946 [0.926, 0.961] |
| `new_dx_365d/copd` | 0.814 [0.764, 0.871] |

## AUPRC

| task | ft:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.797 [0.751, 0.844] |
| `inpatient_365d` | 0.632 [0.589, 0.678] |
| `readmission_30d` | 0.240 [0.172, 0.319] |
| `new_dx_365d/diabetes` | 0.704 [0.614, 0.771] |
| `new_dx_365d/heart_failure` | 0.481 [0.377, 0.577] |
| `new_dx_365d/ckd` | 0.627 [0.515, 0.734] |
| `new_dx_365d/copd` | 0.159 [0.100, 0.250] |

## BRIER

| task | ft:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.0731 [0.0643, 0.0828] |
| `inpatient_365d` | 0.2028 [0.1889, 0.2156] |
| `readmission_30d` | 0.1080 [0.0939, 0.1218] |
| `new_dx_365d/diabetes` | 0.0547 [0.0442, 0.0647] |
| `new_dx_365d/heart_failure` | 0.0564 [0.0472, 0.0665] |
| `new_dx_365d/ckd` | 0.0422 [0.0332, 0.0524] |
| `new_dx_365d/copd` | 0.0385 [0.0297, 0.0498] |

## CALIBRATION SLOPE

| task | ft:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.992 [0.879, 1.149] |
| `inpatient_365d` | 0.674 [0.575, 0.784] |
| `readmission_30d` | 0.573 [0.360, 0.746] |
| `new_dx_365d/diabetes` | 0.886 [0.770, 1.015] |
| `new_dx_365d/heart_failure` | 0.749 [0.645, 0.858] |
| `new_dx_365d/ckd` | 0.878 [0.756, 1.034] |
| `new_dx_365d/copd` | 0.695 [0.549, 0.916] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
