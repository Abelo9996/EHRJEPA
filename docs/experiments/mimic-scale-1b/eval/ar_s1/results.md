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
| created | 2026-10-06T18:08:13+00:00 |
| runtime (s) | 68.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/ar_s1/final.pt` | last@final |

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

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.916 [0.896, 0.935] |
| `inpatient_365d` | 0.650 [0.618, 0.687] |
| `readmission_30d` | 0.606 [0.542, 0.660] |
| `new_dx_365d/diabetes` | 0.850 [0.811, 0.879] |
| `new_dx_365d/heart_failure` | 0.861 [0.818, 0.901] |
| `new_dx_365d/ckd` | 0.908 [0.878, 0.935] |
| `new_dx_365d/copd` | 0.744 [0.675, 0.812] |

## AUPRC

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.740 [0.686, 0.791] |
| `inpatient_365d` | 0.553 [0.508, 0.606] |
| `readmission_30d` | 0.190 [0.138, 0.256] |
| `new_dx_365d/diabetes` | 0.460 [0.365, 0.545] |
| `new_dx_365d/heart_failure` | 0.388 [0.285, 0.485] |
| `new_dx_365d/ckd` | 0.485 [0.373, 0.607] |
| `new_dx_365d/copd` | 0.123 [0.077, 0.222] |

## BRIER

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.0864 [0.0757, 0.0973] |
| `inpatient_365d` | 0.2204 [0.2092, 0.2298] |
| `readmission_30d` | 0.1055 [0.0912, 0.1203] |
| `new_dx_365d/diabetes` | 0.0766 [0.0636, 0.0856] |
| `new_dx_365d/heart_failure` | 0.0592 [0.0495, 0.0692] |
| `new_dx_365d/ckd` | 0.0517 [0.0417, 0.0608] |
| `new_dx_365d/copd` | 0.0386 [0.0290, 0.0494] |

## CALIBRATION SLOPE

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 1.219 [1.068, 1.403] |
| `inpatient_365d` | 0.793 [0.638, 1.004] |
| `readmission_30d` | 0.588 [0.261, 0.856] |
| `new_dx_365d/diabetes` | 1.296 [1.079, 1.505] |
| `new_dx_365d/heart_failure` | 1.057 [0.842, 1.274] |
| `new_dx_365d/ckd` | 1.461 [1.218, 1.790] |
| `new_dx_365d/copd` | 0.906 [0.627, 1.270] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
