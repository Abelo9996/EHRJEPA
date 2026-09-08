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
| created | 2026-09-08T07:20:28+00:00 |
| runtime (s) | 888.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_frozen_ar_s1/final.pt` | last@final |
| `ckpt:hybrid_frozen_ar_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_frozen_ar_s1/final.pt` | last@final |

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

| task | lr | gbm | random_init | ckpt:hybrid_frozen_ar_s1 |
|---|---|---|---|---|
| `mortality_365d` | 0.566 [0.518, 0.616] | 0.556 [0.514, 0.603] | 0.541 [0.493, 0.580] | 0.618 [0.570, 0.654] |
| `inpatient_365d` | 0.712 [0.697, 0.724] | 0.746 [0.731, 0.755] | 0.674 [0.660, 0.687] | 0.756 [0.741, 0.768] |
| `readmission_30d` | 0.653 [0.611, 0.704] | 0.667 [0.625, 0.712] | 0.617 [0.571, 0.655] | 0.698 [0.659, 0.740] |
| `new_dx_365d/diabetes` | 0.737 [0.721, 0.753] | 0.771 [0.756, 0.784] | 0.719 [0.700, 0.736] | 0.776 [0.764, 0.791] |
| `new_dx_365d/heart_failure` | 0.741 [0.724, 0.758] | 0.784 [0.769, 0.800] | 0.690 [0.671, 0.708] | 0.794 [0.779, 0.808] |
| `new_dx_365d/ckd` | 0.739 [0.718, 0.762] | 0.771 [0.751, 0.790] | 0.686 [0.667, 0.704] | 0.783 [0.765, 0.801] |
| `new_dx_365d/copd` | 0.733 [0.715, 0.748] | 0.768 [0.748, 0.787] | 0.691 [0.674, 0.709] | 0.770 [0.752, 0.789] |

## AUPRC

| task | lr | gbm | random_init | ckpt:hybrid_frozen_ar_s1 |
|---|---|---|---|---|
| `mortality_365d` | 0.024 [0.018, 0.036] | 0.023 [0.019, 0.030] | 0.021 [0.017, 0.027] | 0.027 [0.022, 0.035] |
| `inpatient_365d` | 0.377 [0.353, 0.400] | 0.425 [0.397, 0.452] | 0.327 [0.303, 0.348] | 0.441 [0.414, 0.468] |
| `readmission_30d` | 0.097 [0.076, 0.131] | 0.108 [0.080, 0.141] | 0.074 [0.061, 0.094] | 0.109 [0.081, 0.147] |
| `new_dx_365d/diabetes` | 0.421 [0.394, 0.451] | 0.469 [0.442, 0.497] | 0.387 [0.365, 0.414] | 0.474 [0.445, 0.503] |
| `new_dx_365d/heart_failure` | 0.263 [0.241, 0.290] | 0.335 [0.305, 0.368] | 0.233 [0.213, 0.258] | 0.351 [0.319, 0.384] |
| `new_dx_365d/ckd` | 0.246 [0.220, 0.278] | 0.265 [0.237, 0.293] | 0.172 [0.158, 0.194] | 0.291 [0.261, 0.325] |
| `new_dx_365d/copd` | 0.310 [0.278, 0.342] | 0.347 [0.313, 0.392] | 0.232 [0.208, 0.256] | 0.348 [0.312, 0.387] |

## BRIER

| task | lr | gbm | random_init | ckpt:hybrid_frozen_ar_s1 |
|---|---|---|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] | 0.0187 [0.0157, 0.0220] | 0.0187 [0.0156, 0.0219] | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1515 [0.1470, 0.1559] | 0.1442 [0.1400, 0.1488] | 0.1557 [0.1509, 0.1606] | 0.1425 [0.1382, 0.1466] |
| `readmission_30d` | 0.0468 [0.0408, 0.0527] | 0.0468 [0.0409, 0.0527] | 0.0474 [0.0412, 0.0534] | 0.0466 [0.0410, 0.0521] |
| `new_dx_365d/diabetes` | 0.1554 [0.1497, 0.1616] | 0.1470 [0.1414, 0.1532] | 0.1577 [0.1519, 0.1636] | 0.1460 [0.1408, 0.1513] |
| `new_dx_365d/heart_failure` | 0.1060 [0.0998, 0.1110] | 0.0994 [0.0943, 0.1035] | 0.1079 [0.1016, 0.1130] | 0.0981 [0.0931, 0.1027] |
| `new_dx_365d/ckd` | 0.0835 [0.0779, 0.0887] | 0.0813 [0.0760, 0.0861] | 0.0867 [0.0810, 0.0917] | 0.0800 [0.0752, 0.0848] |
| `new_dx_365d/copd` | 0.1062 [0.0999, 0.1124] | 0.1011 [0.0949, 0.1067] | 0.1102 [0.1033, 0.1160] | 0.1010 [0.0953, 0.1059] |

## CALIBRATION SLOPE

| task | lr | gbm | random_init | ckpt:hybrid_frozen_ar_s1 |
|---|---|---|---|---|
| `mortality_365d` | 3.109 [0.490, 5.414] | 0.413 [0.100, 0.708] | 0.483 [0.020, 0.900] | 0.681 [0.383, 0.908] |
| `inpatient_365d` | 1.169 [1.070, 1.256] | 1.064 [0.985, 1.128] | 1.049 [0.959, 1.141] | 1.034 [0.951, 1.103] |
| `readmission_30d` | 1.033 [0.764, 1.339] | 0.717 [0.541, 0.914] | 0.748 [0.465, 1.018] | 0.892 [0.707, 1.116] |
| `new_dx_365d/diabetes` | 0.990 [0.904, 1.075] | 1.018 [0.942, 1.087] | 1.049 [0.950, 1.153] | 0.942 [0.876, 1.016] |
| `new_dx_365d/heart_failure` | 1.249 [1.137, 1.364] | 1.021 [0.940, 1.111] | 0.950 [0.852, 1.068] | 0.970 [0.892, 1.059] |
| `new_dx_365d/ckd` | 1.048 [0.952, 1.155] | 0.990 [0.898, 1.072] | 0.948 [0.840, 1.057] | 1.023 [0.930, 1.122] |
| `new_dx_365d/copd` | 1.406 [1.295, 1.522] | 1.017 [0.925, 1.099] | 1.027 [0.919, 1.146] | 0.968 [0.882, 1.065] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | 0.009 | [-0.037, 0.056] | 0.710 |
| `mortality_365d` | `lr` - `random_init` | 0.025 | [-0.036, 0.084] | 0.360 |
| `mortality_365d` | `lr` - `ckpt:hybrid_frozen_ar_s1` | -0.052 | [-0.097, 0.005] | 0.070 |
| `mortality_365d` | `gbm` - `random_init` | 0.016 | [-0.054, 0.082] | 0.670 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_frozen_ar_s1` | -0.061 | [-0.105, -0.011] | 0.020 |
| `mortality_365d` | `random_init` - `ckpt:hybrid_frozen_ar_s1` | -0.077 | [-0.134, -0.020] | 0.010 |
| `inpatient_365d` | `lr` - `gbm` | -0.033 | [-0.043, -0.026] | 0.000 |
| `inpatient_365d` | `lr` - `random_init` | 0.038 | [0.025, 0.049] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_frozen_ar_s1` | -0.044 | [-0.054, -0.035] | 0.000 |
| `inpatient_365d` | `gbm` - `random_init` | 0.071 | [0.058, 0.085] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_frozen_ar_s1` | -0.010 | [-0.018, -0.005] | 0.000 |
| `inpatient_365d` | `random_init` - `ckpt:hybrid_frozen_ar_s1` | -0.082 | [-0.094, -0.069] | 0.000 |
| `readmission_30d` | `lr` - `gbm` | -0.014 | [-0.044, 0.021] | 0.390 |
| `readmission_30d` | `lr` - `random_init` | 0.036 | [-0.009, 0.094] | 0.090 |
| `readmission_30d` | `lr` - `ckpt:hybrid_frozen_ar_s1` | -0.045 | [-0.073, -0.017] | 0.000 |
| `readmission_30d` | `gbm` - `random_init` | 0.050 | [-0.005, 0.109] | 0.080 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_frozen_ar_s1` | -0.031 | [-0.062, 0.003] | 0.060 |
| `readmission_30d` | `random_init` - `ckpt:hybrid_frozen_ar_s1` | -0.081 | [-0.122, -0.031] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.034 | [-0.045, -0.024] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `random_init` | 0.018 | [0.007, 0.031] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_frozen_ar_s1` | -0.040 | [-0.050, -0.031] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `random_init` | 0.052 | [0.039, 0.068] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_frozen_ar_s1` | -0.005 | [-0.013, 0.002] | 0.140 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:hybrid_frozen_ar_s1` | -0.058 | [-0.073, -0.046] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | -0.043 | [-0.053, -0.031] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `random_init` | 0.051 | [0.036, 0.068] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_frozen_ar_s1` | -0.053 | [-0.065, -0.040] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `random_init` | 0.094 | [0.077, 0.111] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_frozen_ar_s1` | -0.010 | [-0.019, -0.001] | 0.030 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:hybrid_frozen_ar_s1` | -0.104 | [-0.122, -0.084] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.032 | [-0.043, -0.020] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `random_init` | 0.053 | [0.031, 0.072] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_frozen_ar_s1` | -0.044 | [-0.055, -0.031] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `random_init` | 0.086 | [0.065, 0.104] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_frozen_ar_s1` | -0.011 | [-0.020, -0.004] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:hybrid_frozen_ar_s1` | -0.097 | [-0.117, -0.078] | 0.000 |
| `new_dx_365d/copd` | `lr` - `gbm` | -0.035 | [-0.045, -0.023] | 0.000 |
| `new_dx_365d/copd` | `lr` - `random_init` | 0.042 | [0.027, 0.059] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_frozen_ar_s1` | -0.038 | [-0.049, -0.028] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `random_init` | 0.077 | [0.057, 0.097] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_frozen_ar_s1` | -0.003 | [-0.011, 0.005] | 0.540 |
| `new_dx_365d/copd` | `random_init` - `ckpt:hybrid_frozen_ar_s1` | -0.080 | [-0.100, -0.063] | 0.000 |
