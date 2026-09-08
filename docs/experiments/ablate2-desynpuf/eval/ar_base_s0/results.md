# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `7bfd19f` |
| created | 2026-09-08T06:24:58+00:00 |
| runtime (s) | 426.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/ar/final.pt` | last@final |
| `ckpt:ar` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/ar/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 7358 (0.0190) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 7358 (0.2098) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3116 (0.0501) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 4939 (0.2264) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 6164 (0.1308) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 6487 (0.0999) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 6009 (0.1340) |  |

## AUROC

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.541 [0.493, 0.580] | 0.616 [0.572, 0.652] |
| `inpatient_365d` | 0.674 [0.660, 0.687] | 0.750 [0.735, 0.762] |
| `readmission_30d` | 0.617 [0.571, 0.655] | 0.661 [0.622, 0.706] |
| `new_dx_365d/diabetes` | 0.719 [0.700, 0.736] | 0.773 [0.758, 0.787] |
| `new_dx_365d/heart_failure` | 0.690 [0.671, 0.708] | 0.781 [0.766, 0.796] |
| `new_dx_365d/ckd` | 0.686 [0.667, 0.704] | 0.772 [0.752, 0.791] |
| `new_dx_365d/copd` | 0.691 [0.674, 0.709] | 0.767 [0.747, 0.784] |

## AUPRC

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.021 [0.017, 0.027] | 0.031 [0.024, 0.051] |
| `inpatient_365d` | 0.327 [0.303, 0.348] | 0.425 [0.399, 0.451] |
| `readmission_30d` | 0.074 [0.061, 0.094] | 0.104 [0.076, 0.142] |
| `new_dx_365d/diabetes` | 0.387 [0.365, 0.414] | 0.465 [0.436, 0.494] |
| `new_dx_365d/heart_failure` | 0.233 [0.213, 0.258] | 0.327 [0.295, 0.357] |
| `new_dx_365d/ckd` | 0.172 [0.158, 0.194] | 0.274 [0.247, 0.307] |
| `new_dx_365d/copd` | 0.232 [0.208, 0.256] | 0.339 [0.302, 0.381] |

## BRIER

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.0187 [0.0156, 0.0219] | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1557 [0.1509, 0.1606] | 0.1440 [0.1393, 0.1482] |
| `readmission_30d` | 0.0474 [0.0412, 0.0534] | 0.0468 [0.0410, 0.0524] |
| `new_dx_365d/diabetes` | 0.1577 [0.1519, 0.1636] | 0.1469 [0.1420, 0.1523] |
| `new_dx_365d/heart_failure` | 0.1079 [0.1016, 0.1130] | 0.0998 [0.0946, 0.1042] |
| `new_dx_365d/ckd` | 0.0867 [0.0810, 0.0917] | 0.0810 [0.0758, 0.0853] |
| `new_dx_365d/copd` | 0.1102 [0.1033, 0.1160] | 0.1020 [0.0962, 0.1073] |

## CALIBRATION SLOPE

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.483 [0.020, 0.900] | 0.757 [0.446, 1.033] |
| `inpatient_365d` | 1.049 [0.959, 1.141] | 1.030 [0.941, 1.097] |
| `readmission_30d` | 0.748 [0.465, 1.018] | 0.877 [0.646, 1.127] |
| `new_dx_365d/diabetes` | 1.049 [0.950, 1.153] | 0.995 [0.917, 1.077] |
| `new_dx_365d/heart_failure` | 0.950 [0.852, 1.068] | 1.021 [0.932, 1.118] |
| `new_dx_365d/ckd` | 0.948 [0.840, 1.057] | 0.979 [0.881, 1.077] |
| `new_dx_365d/copd` | 1.027 [0.919, 1.146] | 0.958 [0.872, 1.050] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:ar` | -0.075 | [-0.126, -0.025] | 0.010 |
| `inpatient_365d` | `random_init` - `ckpt:ar` | -0.076 | [-0.089, -0.062] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar` | -0.045 | [-0.092, 0.007] | 0.110 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar` | -0.054 | [-0.065, -0.043] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar` | -0.092 | [-0.109, -0.074] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar` | -0.087 | [-0.106, -0.069] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar` | -0.076 | [-0.094, -0.058] | 0.000 |
