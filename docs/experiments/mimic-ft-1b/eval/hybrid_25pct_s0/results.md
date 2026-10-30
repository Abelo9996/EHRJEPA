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
| created | 2026-10-22T18:03:45+00:00 |
| runtime (s) | 925.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_s0` | finetune | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s0/final.pt` | last@final |

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
| `mortality_365d` | 0.943 [0.930, 0.958] |
| `inpatient_365d` | 0.732 [0.703, 0.764] |
| `readmission_30d` | 0.627 [0.572, 0.677] |
| `new_dx_365d/diabetes` | 0.915 [0.887, 0.942] |
| `new_dx_365d/heart_failure` | 0.892 [0.859, 0.915] |
| `new_dx_365d/ckd` | 0.933 [0.911, 0.950] |
| `new_dx_365d/copd` | 0.811 [0.760, 0.864] |

## AUPRC

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.796 [0.747, 0.847] |
| `inpatient_365d` | 0.641 [0.595, 0.683] |
| `readmission_30d` | 0.238 [0.170, 0.314] |
| `new_dx_365d/diabetes` | 0.729 [0.645, 0.791] |
| `new_dx_365d/heart_failure` | 0.492 [0.386, 0.571] |
| `new_dx_365d/ckd` | 0.612 [0.497, 0.715] |
| `new_dx_365d/copd` | 0.205 [0.120, 0.337] |

## BRIER

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.0733 [0.0630, 0.0837] |
| `inpatient_365d` | 0.2021 [0.1870, 0.2159] |
| `readmission_30d` | 0.1090 [0.0961, 0.1221] |
| `new_dx_365d/diabetes` | 0.0526 [0.0430, 0.0630] |
| `new_dx_365d/heart_failure` | 0.0570 [0.0481, 0.0670] |
| `new_dx_365d/ckd` | 0.0463 [0.0371, 0.0569] |
| `new_dx_365d/copd` | 0.0370 [0.0287, 0.0470] |

## CALIBRATION SLOPE

| task | ft:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.937 [0.835, 1.084] |
| `inpatient_365d` | 0.669 [0.573, 0.797] |
| `readmission_30d` | 0.498 [0.288, 0.691] |
| `new_dx_365d/diabetes` | 0.913 [0.810, 1.050] |
| `new_dx_365d/heart_failure` | 0.735 [0.624, 0.837] |
| `new_dx_365d/ckd` | 0.774 [0.669, 0.893] |
| `new_dx_365d/copd` | 0.729 [0.562, 0.952] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
