# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `3ffe71b` |
| created | 2026-09-04T17:00:47+00:00 |
| runtime (s) | 350.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot5-seeds-desynpuf/hybrid_s1/final.pt` | last@final |
| `ckpt:hybrid_s1` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot5-seeds-desynpuf/hybrid_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 3000 (0.0220) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 3000 (0.2010) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3000 (0.0493) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 3000 (0.2240) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 3000 (0.1370) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 3000 (0.0970) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 3000 (0.1303) |  |

## AUROC

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.534 [0.474, 0.603] | 0.607 [0.545, 0.679] |
| `inpatient_365d` | 0.647 [0.625, 0.669] | 0.748 [0.726, 0.769] |
| `readmission_30d` | 0.578 [0.531, 0.628] | 0.700 [0.662, 0.741] |
| `new_dx_365d/diabetes` | 0.710 [0.690, 0.732] | 0.762 [0.745, 0.783] |
| `new_dx_365d/heart_failure` | 0.666 [0.641, 0.692] | 0.758 [0.738, 0.780] |
| `new_dx_365d/ckd` | 0.658 [0.627, 0.685] | 0.754 [0.731, 0.781] |
| `new_dx_365d/copd` | 0.662 [0.634, 0.686] | 0.754 [0.729, 0.777] |

## AUPRC

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.027 [0.020, 0.043] | 0.030 [0.023, 0.047] |
| `inpatient_365d` | 0.292 [0.264, 0.326] | 0.413 [0.376, 0.453] |
| `readmission_30d` | 0.074 [0.054, 0.109] | 0.109 [0.084, 0.149] |
| `new_dx_365d/diabetes` | 0.384 [0.356, 0.420] | 0.457 [0.420, 0.500] |
| `new_dx_365d/heart_failure` | 0.224 [0.195, 0.254] | 0.308 [0.268, 0.358] |
| `new_dx_365d/ckd` | 0.154 [0.129, 0.184] | 0.248 [0.209, 0.297] |
| `new_dx_365d/copd` | 0.204 [0.179, 0.232] | 0.296 [0.251, 0.337] |

## BRIER

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1541 [0.1462, 0.1625] | 0.1405 [0.1336, 0.1473] |
| `readmission_30d` | 0.0468 [0.0408, 0.0532] | 0.0457 [0.0398, 0.0513] |
| `new_dx_365d/diabetes` | 0.1581 [0.1506, 0.1661] | 0.1479 [0.1411, 0.1544] |
| `new_dx_365d/heart_failure` | 0.1137 [0.1046, 0.1214] | 0.1064 [0.0992, 0.1130] |
| `new_dx_365d/ckd` | 0.0856 [0.0773, 0.0931] | 0.0806 [0.0727, 0.0871] |
| `new_dx_365d/copd` | 0.1096 [0.1007, 0.1185] | 0.1025 [0.0953, 0.1099] |

## CALIBRATION SLOPE

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.365 [-0.396, 1.258] | 0.615 [0.262, 1.070] |
| `inpatient_365d` | 0.929 [0.777, 1.088] | 1.106 [0.971, 1.254] |
| `readmission_30d` | 0.668 [0.306, 1.101] | 1.033 [0.829, 1.276] |
| `new_dx_365d/diabetes` | 1.116 [0.989, 1.241] | 0.974 [0.872, 1.108] |
| `new_dx_365d/heart_failure` | 0.928 [0.773, 1.121] | 1.023 [0.923, 1.154] |
| `new_dx_365d/ckd` | 0.857 [0.679, 1.024] | 0.962 [0.848, 1.119] |
| `new_dx_365d/copd` | 0.914 [0.743, 1.085] | 1.015 [0.863, 1.155] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:hybrid_s1` | -0.073 | [-0.153, 0.017] | 0.100 |
| `inpatient_365d` | `random_init` - `ckpt:hybrid_s1` | -0.101 | [-0.123, -0.076] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:hybrid_s1` | -0.122 | [-0.187, -0.060] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:hybrid_s1` | -0.052 | [-0.070, -0.036] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:hybrid_s1` | -0.092 | [-0.117, -0.071] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:hybrid_s1` | -0.096 | [-0.123, -0.071] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:hybrid_s1` | -0.092 | [-0.116, -0.069] | 0.000 |
