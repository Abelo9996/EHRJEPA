# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `95c8e35` |
| created | 2026-09-06T12:57:32+00:00 |
| runtime (s) | 364.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/ar_s1/final.pt` | last@final |
| `ckpt:ar_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/ar_s1/final.pt` | last@final |

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

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.547 [0.490, 0.614] | 0.603 [0.527, 0.676] |
| `inpatient_365d` | 0.669 [0.643, 0.691] | 0.741 [0.719, 0.762] |
| `readmission_30d` | 0.614 [0.568, 0.653] | 0.674 [0.627, 0.714] |
| `new_dx_365d/diabetes` | 0.716 [0.693, 0.737] | 0.764 [0.748, 0.785] |
| `new_dx_365d/heart_failure` | 0.676 [0.650, 0.704] | 0.780 [0.757, 0.800] |
| `new_dx_365d/ckd` | 0.680 [0.654, 0.706] | 0.758 [0.733, 0.788] |
| `new_dx_365d/copd` | 0.681 [0.653, 0.702] | 0.774 [0.749, 0.796] |

## AUPRC

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.024 [0.019, 0.033] | 0.033 [0.024, 0.053] |
| `inpatient_365d` | 0.315 [0.286, 0.352] | 0.385 [0.349, 0.422] |
| `readmission_30d` | 0.073 [0.056, 0.093] | 0.090 [0.072, 0.125] |
| `new_dx_365d/diabetes` | 0.389 [0.357, 0.430] | 0.453 [0.413, 0.493] |
| `new_dx_365d/heart_failure` | 0.239 [0.209, 0.272] | 0.306 [0.263, 0.348] |
| `new_dx_365d/ckd` | 0.159 [0.138, 0.184] | 0.253 [0.219, 0.300] |
| `new_dx_365d/copd` | 0.214 [0.189, 0.251] | 0.328 [0.281, 0.376] |

## BRIER

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0267] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1516 [0.1440, 0.1592] | 0.1425 [0.1358, 0.1492] |
| `readmission_30d` | 0.0468 [0.0409, 0.0529] | 0.0464 [0.0405, 0.0523] |
| `new_dx_365d/diabetes` | 0.1569 [0.1501, 0.1651] | 0.1481 [0.1413, 0.1552] |
| `new_dx_365d/heart_failure` | 0.1129 [0.1033, 0.1195] | 0.1056 [0.0978, 0.1123] |
| `new_dx_365d/ckd` | 0.0850 [0.0767, 0.0924] | 0.0804 [0.0722, 0.0865] |
| `new_dx_365d/copd` | 0.1085 [0.1001, 0.1175] | 0.0999 [0.0919, 0.1072] |

## CALIBRATION SLOPE

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.485 [-0.072, 1.187] | 0.646 [0.201, 1.077] |
| `inpatient_365d` | 1.034 [0.864, 1.207] | 0.989 [0.881, 1.118] |
| `readmission_30d` | 0.725 [0.415, 1.002] | 0.873 [0.619, 1.100] |
| `new_dx_365d/diabetes` | 1.046 [0.910, 1.187] | 0.953 [0.858, 1.070] |
| `new_dx_365d/heart_failure` | 0.886 [0.751, 1.060] | 1.047 [0.934, 1.177] |
| `new_dx_365d/ckd` | 0.920 [0.768, 1.086] | 0.993 [0.861, 1.144] |
| `new_dx_365d/copd` | 0.977 [0.807, 1.132] | 1.024 [0.891, 1.166] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:ar_s1` | -0.056 | [-0.134, 0.024] | 0.200 |
| `inpatient_365d` | `random_init` - `ckpt:ar_s1` | -0.072 | [-0.096, -0.051] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar_s1` | -0.061 | [-0.111, -0.007] | 0.020 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar_s1` | -0.048 | [-0.067, -0.033] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar_s1` | -0.103 | [-0.127, -0.079] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar_s1` | -0.079 | [-0.109, -0.052] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar_s1` | -0.092 | [-0.118, -0.066] | 0.000 |
