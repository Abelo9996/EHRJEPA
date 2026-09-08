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
| created | 2026-09-07T15:25:06+00:00 |
| runtime (s) | 827.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_large_s1/final.pt` | last@final |
| `ckpt:hybrid_large_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_large_s1/final.pt` | last@final |

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

| task | random_init | ckpt:hybrid_large_s1 |
|---|---|---|
| `mortality_365d` | 0.564 [0.514, 0.604] | 0.607 [0.561, 0.645] |
| `inpatient_365d` | 0.676 [0.661, 0.689] | 0.759 [0.743, 0.771] |
| `readmission_30d` | 0.626 [0.579, 0.671] | 0.691 [0.652, 0.730] |
| `new_dx_365d/diabetes` | 0.730 [0.712, 0.746] | 0.777 [0.765, 0.792] |
| `new_dx_365d/heart_failure` | 0.702 [0.683, 0.719] | 0.783 [0.769, 0.799] |
| `new_dx_365d/ckd` | 0.692 [0.676, 0.711] | 0.786 [0.767, 0.804] |
| `new_dx_365d/copd` | 0.696 [0.679, 0.714] | 0.766 [0.749, 0.784] |

## AUPRC

| task | random_init | ckpt:hybrid_large_s1 |
|---|---|---|
| `mortality_365d` | 0.024 [0.018, 0.037] | 0.030 [0.022, 0.047] |
| `inpatient_365d` | 0.328 [0.308, 0.349] | 0.446 [0.416, 0.472] |
| `readmission_30d` | 0.094 [0.067, 0.126] | 0.106 [0.078, 0.137] |
| `new_dx_365d/diabetes` | 0.406 [0.377, 0.435] | 0.482 [0.454, 0.512] |
| `new_dx_365d/heart_failure` | 0.242 [0.219, 0.271] | 0.327 [0.297, 0.357] |
| `new_dx_365d/ckd` | 0.180 [0.164, 0.205] | 0.294 [0.263, 0.324] |
| `new_dx_365d/copd` | 0.237 [0.214, 0.266] | 0.362 [0.325, 0.401] |

## BRIER

| task | random_init | ckpt:hybrid_large_s1 |
|---|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1553 [0.1505, 0.1602] | 0.1418 [0.1371, 0.1459] |
| `readmission_30d` | 0.0474 [0.0413, 0.0534] | 0.0467 [0.0413, 0.0525] |
| `new_dx_365d/diabetes` | 0.1556 [0.1493, 0.1613] | 0.1454 [0.1397, 0.1509] |
| `new_dx_365d/heart_failure` | 0.1071 [0.1010, 0.1118] | 0.0999 [0.0947, 0.1043] |
| `new_dx_365d/ckd` | 0.0863 [0.0807, 0.0911] | 0.0798 [0.0750, 0.0844] |
| `new_dx_365d/copd` | 0.1097 [0.1033, 0.1154] | 0.1010 [0.0948, 0.1061] |

## CALIBRATION SLOPE

| task | random_init | ckpt:hybrid_large_s1 |
|---|---|---|
| `mortality_365d` | 0.368 [0.071, 0.625] | 0.609 [0.333, 0.855] |
| `inpatient_365d` | 1.038 [0.939, 1.133] | 1.038 [0.956, 1.122] |
| `readmission_30d` | 0.557 [0.335, 0.780] | 0.812 [0.635, 0.997] |
| `new_dx_365d/diabetes` | 1.013 [0.919, 1.107] | 0.956 [0.886, 1.038] |
| `new_dx_365d/heart_failure` | 1.005 [0.880, 1.119] | 0.945 [0.865, 1.035] |
| `new_dx_365d/ckd` | 0.954 [0.850, 1.079] | 1.037 [0.939, 1.136] |
| `new_dx_365d/copd` | 1.016 [0.914, 1.133] | 0.947 [0.863, 1.040] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:hybrid_large_s1` | -0.043 | [-0.099, 0.011] | 0.120 |
| `inpatient_365d` | `random_init` - `ckpt:hybrid_large_s1` | -0.083 | [-0.095, -0.069] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:hybrid_large_s1` | -0.065 | [-0.107, -0.013] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:hybrid_large_s1` | -0.048 | [-0.062, -0.035] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:hybrid_large_s1` | -0.081 | [-0.098, -0.068] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:hybrid_large_s1` | -0.094 | [-0.109, -0.077] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:hybrid_large_s1` | -0.070 | [-0.086, -0.054] | 0.000 |
