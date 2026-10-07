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
| created | 2026-10-07T02:42:33+00:00 |
| runtime (s) | 76.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/hybrid_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.932 [0.916, 0.947] |
| `inpatient_365d` | 0.713 [0.685, 0.743] |
| `readmission_30d` | 0.624 [0.571, 0.672] |
| `new_dx_365d/diabetes` | 0.895 [0.865, 0.924] |
| `new_dx_365d/heart_failure` | 0.871 [0.839, 0.899] |
| `new_dx_365d/ckd` | 0.906 [0.874, 0.930] |
| `new_dx_365d/copd` | 0.737 [0.678, 0.809] |

## AUPRC

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.752 [0.699, 0.805] |
| `inpatient_365d` | 0.623 [0.575, 0.670] |
| `readmission_30d` | 0.205 [0.151, 0.270] |
| `new_dx_365d/diabetes` | 0.604 [0.487, 0.686] |
| `new_dx_365d/heart_failure` | 0.435 [0.329, 0.526] |
| `new_dx_365d/ckd` | 0.498 [0.390, 0.610] |
| `new_dx_365d/copd` | 0.101 [0.066, 0.171] |

## BRIER

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.0806 [0.0716, 0.0899] |
| `inpatient_365d` | 0.2041 [0.1930, 0.2136] |
| `readmission_30d` | 0.1051 [0.0909, 0.1192] |
| `new_dx_365d/diabetes` | 0.0664 [0.0558, 0.0755] |
| `new_dx_365d/heart_failure` | 0.0580 [0.0493, 0.0683] |
| `new_dx_365d/ckd` | 0.0508 [0.0408, 0.0606] |
| `new_dx_365d/copd` | 0.0394 [0.0299, 0.0501] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 1.270 [1.131, 1.433] |
| `inpatient_365d` | 0.866 [0.739, 1.003] |
| `readmission_30d` | 0.654 [0.369, 0.919] |
| `new_dx_365d/diabetes` | 1.534 [1.294, 1.900] |
| `new_dx_365d/heart_failure` | 1.081 [0.910, 1.283] |
| `new_dx_365d/ckd` | 1.342 [1.103, 1.677] |
| `new_dx_365d/copd` | 0.793 [0.566, 1.106] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
