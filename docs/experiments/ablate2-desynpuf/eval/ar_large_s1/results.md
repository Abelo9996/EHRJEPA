# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `fdc4f65` |
| created | 2026-09-07T14:13:53+00:00 |
| runtime (s) | 784.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/ar_large_s1/final.pt` | last@final |
| `ckpt:ar_large_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/ar_large_s1/final.pt` | last@final |

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

| task | random_init | ckpt:ar_large_s1 |
|---|---|---|
| `mortality_365d` | 0.564 [0.514, 0.604] | 0.611 [0.564, 0.649] |
| `inpatient_365d` | 0.676 [0.661, 0.689] | 0.753 [0.737, 0.764] |
| `readmission_30d` | 0.626 [0.579, 0.671] | 0.676 [0.634, 0.713] |
| `new_dx_365d/diabetes` | 0.730 [0.712, 0.746] | 0.774 [0.760, 0.789] |
| `new_dx_365d/heart_failure` | 0.702 [0.683, 0.719] | 0.779 [0.764, 0.795] |
| `new_dx_365d/ckd` | 0.692 [0.676, 0.711] | 0.775 [0.756, 0.796] |
| `new_dx_365d/copd` | 0.696 [0.679, 0.714] | 0.766 [0.748, 0.784] |

## AUPRC

| task | random_init | ckpt:ar_large_s1 |
|---|---|---|
| `mortality_365d` | 0.024 [0.018, 0.037] | 0.026 [0.021, 0.034] |
| `inpatient_365d` | 0.328 [0.308, 0.349] | 0.437 [0.409, 0.464] |
| `readmission_30d` | 0.094 [0.067, 0.126] | 0.093 [0.073, 0.121] |
| `new_dx_365d/diabetes` | 0.406 [0.377, 0.435] | 0.475 [0.441, 0.504] |
| `new_dx_365d/heart_failure` | 0.242 [0.219, 0.271] | 0.315 [0.285, 0.345] |
| `new_dx_365d/ckd` | 0.180 [0.164, 0.205] | 0.279 [0.253, 0.312] |
| `new_dx_365d/copd` | 0.237 [0.214, 0.266] | 0.341 [0.309, 0.380] |

## BRIER

| task | random_init | ckpt:ar_large_s1 |
|---|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1553 [0.1505, 0.1602] | 0.1431 [0.1387, 0.1474] |
| `readmission_30d` | 0.0474 [0.0413, 0.0534] | 0.0469 [0.0411, 0.0526] |
| `new_dx_365d/diabetes` | 0.1556 [0.1493, 0.1613] | 0.1464 [0.1408, 0.1515] |
| `new_dx_365d/heart_failure` | 0.1071 [0.1010, 0.1118] | 0.1005 [0.0953, 0.1052] |
| `new_dx_365d/ckd` | 0.0863 [0.0807, 0.0911] | 0.0807 [0.0757, 0.0854] |
| `new_dx_365d/copd` | 0.1097 [0.1033, 0.1154] | 0.1019 [0.0960, 0.1073] |

## CALIBRATION SLOPE

| task | random_init | ckpt:ar_large_s1 |
|---|---|---|
| `mortality_365d` | 0.368 [0.071, 0.625] | 0.654 [0.376, 0.884] |
| `inpatient_365d` | 1.038 [0.939, 1.133] | 1.048 [0.967, 1.123] |
| `readmission_30d` | 0.557 [0.335, 0.780] | 0.875 [0.677, 1.094] |
| `new_dx_365d/diabetes` | 1.013 [0.919, 1.107] | 0.961 [0.891, 1.042] |
| `new_dx_365d/heart_failure` | 1.005 [0.880, 1.119] | 0.980 [0.905, 1.078] |
| `new_dx_365d/ckd` | 0.954 [0.850, 1.079] | 0.989 [0.901, 1.100] |
| `new_dx_365d/copd` | 1.016 [0.914, 1.133] | 0.960 [0.875, 1.056] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:ar_large_s1` | -0.047 | [-0.098, 0.004] | 0.080 |
| `inpatient_365d` | `random_init` - `ckpt:ar_large_s1` | -0.077 | [-0.089, -0.063] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar_large_s1` | -0.050 | [-0.094, -0.002] | 0.040 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar_large_s1` | -0.045 | [-0.059, -0.033] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar_large_s1` | -0.077 | [-0.095, -0.062] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar_large_s1` | -0.084 | [-0.099, -0.067] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar_large_s1` | -0.070 | [-0.086, -0.053] | 0.000 |
