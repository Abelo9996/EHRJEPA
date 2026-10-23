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
| created | 2026-10-23T13:40:58+00:00 |
| runtime (s) | 932.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_s0` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-full-1b/hybrid_s0/final.pt` | last@final |

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

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.944 [0.930, 0.957] |
| `inpatient_365d` | 0.732 [0.702, 0.763] |
| `readmission_30d` | 0.631 [0.571, 0.680] |
| `new_dx_365d/diabetes` | 0.925 [0.897, 0.949] |
| `new_dx_365d/heart_failure` | 0.904 [0.871, 0.926] |
| `new_dx_365d/ckd` | 0.939 [0.914, 0.958] |
| `new_dx_365d/copd` | 0.801 [0.752, 0.862] |

## AUPRC

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.791 [0.738, 0.839] |
| `inpatient_365d` | 0.640 [0.590, 0.679] |
| `readmission_30d` | 0.218 [0.155, 0.292] |
| `new_dx_365d/diabetes` | 0.743 [0.656, 0.806] |
| `new_dx_365d/heart_failure` | 0.509 [0.398, 0.594] |
| `new_dx_365d/ckd` | 0.628 [0.519, 0.725] |
| `new_dx_365d/copd` | 0.189 [0.110, 0.307] |

## BRIER

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.0736 [0.0639, 0.0838] |
| `inpatient_365d` | 0.2024 [0.1883, 0.2163] |
| `readmission_30d` | 0.1088 [0.0940, 0.1229] |
| `new_dx_365d/diabetes` | 0.0507 [0.0410, 0.0600] |
| `new_dx_365d/heart_failure` | 0.0543 [0.0451, 0.0646] |
| `new_dx_365d/ckd` | 0.0430 [0.0351, 0.0524] |
| `new_dx_365d/copd` | 0.0376 [0.0290, 0.0480] |

## CALIBRATION SLOPE

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.985 [0.874, 1.137] |
| `inpatient_365d` | 0.648 [0.539, 0.756] |
| `readmission_30d` | 0.509 [0.295, 0.696] |
| `new_dx_365d/diabetes` | 1.000 [0.871, 1.165] |
| `new_dx_365d/heart_failure` | 0.825 [0.700, 0.964] |
| `new_dx_365d/ckd` | 0.864 [0.727, 1.012] |
| `new_dx_365d/copd` | 0.717 [0.562, 0.960] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
