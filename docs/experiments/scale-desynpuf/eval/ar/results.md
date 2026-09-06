# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `12478aa` |
| created | 2026-09-06T01:40:34+00:00 |
| runtime (s) | 466.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/ar/final.pt` | last@final |
| `ckpt:ar` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/ar/final.pt` | last@final |

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

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.547 [0.490, 0.614] | 0.595 [0.512, 0.681] |
| `inpatient_365d` | 0.669 [0.643, 0.691] | 0.752 [0.730, 0.769] |
| `readmission_30d` | 0.614 [0.568, 0.653] | 0.667 [0.620, 0.706] |
| `new_dx_365d/diabetes` | 0.716 [0.693, 0.737] | 0.771 [0.755, 0.791] |
| `new_dx_365d/heart_failure` | 0.676 [0.650, 0.704] | 0.791 [0.771, 0.812] |
| `new_dx_365d/ckd` | 0.680 [0.654, 0.706] | 0.762 [0.735, 0.788] |
| `new_dx_365d/copd` | 0.681 [0.653, 0.702] | 0.775 [0.750, 0.795] |

## AUPRC

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.024 [0.019, 0.033] | 0.065 [0.024, 0.121] |
| `inpatient_365d` | 0.315 [0.286, 0.352] | 0.411 [0.372, 0.454] |
| `readmission_30d` | 0.073 [0.056, 0.093] | 0.105 [0.076, 0.154] |
| `new_dx_365d/diabetes` | 0.389 [0.357, 0.430] | 0.464 [0.427, 0.505] |
| `new_dx_365d/heart_failure` | 0.239 [0.209, 0.272] | 0.340 [0.296, 0.390] |
| `new_dx_365d/ckd` | 0.159 [0.138, 0.184] | 0.265 [0.224, 0.316] |
| `new_dx_365d/copd` | 0.214 [0.189, 0.251] | 0.335 [0.288, 0.378] |

## BRIER

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0267] | 0.0214 [0.0172, 0.0265] |
| `inpatient_365d` | 0.1516 [0.1440, 0.1592] | 0.1403 [0.1336, 0.1476] |
| `readmission_30d` | 0.0468 [0.0409, 0.0529] | 0.0461 [0.0403, 0.0519] |
| `new_dx_365d/diabetes` | 0.1569 [0.1501, 0.1651] | 0.1461 [0.1394, 0.1524] |
| `new_dx_365d/heart_failure` | 0.1129 [0.1033, 0.1195] | 0.1029 [0.0952, 0.1093] |
| `new_dx_365d/ckd` | 0.0850 [0.0767, 0.0924] | 0.0797 [0.0722, 0.0864] |
| `new_dx_365d/copd` | 0.1085 [0.1001, 0.1175] | 0.0999 [0.0926, 0.1066] |

## CALIBRATION SLOPE

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.485 [-0.072, 1.187] | 0.671 [0.148, 1.209] |
| `inpatient_365d` | 1.034 [0.864, 1.207] | 1.051 [0.924, 1.178] |
| `readmission_30d` | 0.725 [0.415, 1.002] | 0.913 [0.632, 1.162] |
| `new_dx_365d/diabetes` | 1.046 [0.910, 1.187] | 0.994 [0.904, 1.110] |
| `new_dx_365d/heart_failure` | 0.886 [0.751, 1.060] | 1.089 [0.976, 1.212] |
| `new_dx_365d/ckd` | 0.920 [0.768, 1.086] | 0.984 [0.850, 1.122] |
| `new_dx_365d/copd` | 0.977 [0.807, 1.132] | 0.978 [0.861, 1.087] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:ar` | -0.048 | [-0.132, 0.050] | 0.210 |
| `inpatient_365d` | `random_init` - `ckpt:ar` | -0.083 | [-0.106, -0.061] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar` | -0.054 | [-0.104, -0.002] | 0.040 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar` | -0.056 | [-0.073, -0.040] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar` | -0.114 | [-0.140, -0.088] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar` | -0.082 | [-0.109, -0.057] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar` | -0.093 | [-0.115, -0.068] | 0.000 |
