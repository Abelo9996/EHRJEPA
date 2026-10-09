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
| created | 2026-10-04T20:44:00+00:00 |
| runtime (s) | 75.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/hybrid_s0/final.pt` | last@final |

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
| `mortality_365d` | 0.935 [0.921, 0.950] |
| `inpatient_365d` | 0.723 [0.688, 0.754] |
| `readmission_30d` | 0.640 [0.587, 0.683] |
| `new_dx_365d/diabetes` | 0.890 [0.854, 0.922] |
| `new_dx_365d/heart_failure` | 0.874 [0.836, 0.905] |
| `new_dx_365d/ckd` | 0.914 [0.887, 0.939] |
| `new_dx_365d/copd` | 0.749 [0.693, 0.810] |

## AUPRC

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.762 [0.712, 0.811] |
| `inpatient_365d` | 0.618 [0.569, 0.664] |
| `readmission_30d` | 0.222 [0.163, 0.287] |
| `new_dx_365d/diabetes` | 0.651 [0.558, 0.732] |
| `new_dx_365d/heart_failure` | 0.453 [0.364, 0.565] |
| `new_dx_365d/ckd` | 0.512 [0.398, 0.623] |
| `new_dx_365d/copd` | 0.111 [0.067, 0.202] |

## BRIER

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 0.0787 [0.0701, 0.0885] |
| `inpatient_365d` | 0.2025 [0.1894, 0.2141] |
| `readmission_30d` | 0.1037 [0.0896, 0.1180] |
| `new_dx_365d/diabetes` | 0.0640 [0.0527, 0.0730] |
| `new_dx_365d/heart_failure` | 0.0578 [0.0478, 0.0687] |
| `new_dx_365d/ckd` | 0.0501 [0.0400, 0.0604] |
| `new_dx_365d/copd` | 0.0389 [0.0299, 0.0495] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s0 |
|---|---|
| `mortality_365d` | 1.270 [1.140, 1.483] |
| `inpatient_365d` | 0.890 [0.746, 1.046] |
| `readmission_30d` | 0.716 [0.466, 0.960] |
| `new_dx_365d/diabetes` | 1.550 [1.275, 1.879] |
| `new_dx_365d/heart_failure` | 1.077 [0.909, 1.290] |
| `new_dx_365d/ckd` | 1.306 [1.075, 1.575] |
| `new_dx_365d/copd` | 0.883 [0.633, 1.169] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
