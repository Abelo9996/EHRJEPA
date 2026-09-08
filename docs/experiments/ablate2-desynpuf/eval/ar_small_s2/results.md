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
| created | 2026-09-08T00:56:54+00:00 |
| runtime (s) | 185.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_small_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/ar_small_s2/final.pt` | last@final |

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

| task | ckpt:ar_small_s2 |
|---|---|
| `mortality_365d` | 0.604 [0.552, 0.646] |
| `inpatient_365d` | 0.747 [0.729, 0.758] |
| `readmission_30d` | 0.676 [0.632, 0.718] |
| `new_dx_365d/diabetes` | 0.770 [0.756, 0.786] |
| `new_dx_365d/heart_failure` | 0.762 [0.747, 0.777] |
| `new_dx_365d/ckd` | 0.770 [0.749, 0.788] |
| `new_dx_365d/copd` | 0.756 [0.739, 0.774] |

## AUPRC

| task | ckpt:ar_small_s2 |
|---|---|
| `mortality_365d` | 0.034 [0.022, 0.052] |
| `inpatient_365d` | 0.429 [0.403, 0.456] |
| `readmission_30d` | 0.114 [0.080, 0.160] |
| `new_dx_365d/diabetes` | 0.473 [0.441, 0.505] |
| `new_dx_365d/heart_failure` | 0.306 [0.273, 0.337] |
| `new_dx_365d/ckd` | 0.270 [0.245, 0.303] |
| `new_dx_365d/copd` | 0.330 [0.300, 0.366] |

## BRIER

| task | ckpt:ar_small_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1442 [0.1398, 0.1482] |
| `readmission_30d` | 0.0465 [0.0407, 0.0522] |
| `new_dx_365d/diabetes` | 0.1469 [0.1413, 0.1521] |
| `new_dx_365d/heart_failure` | 0.1022 [0.0967, 0.1068] |
| `new_dx_365d/ckd` | 0.0814 [0.0761, 0.0858] |
| `new_dx_365d/copd` | 0.1030 [0.0972, 0.1086] |

## CALIBRATION SLOPE

| task | ckpt:ar_small_s2 |
|---|---|
| `mortality_365d` | 0.718 [0.365, 1.024] |
| `inpatient_365d` | 1.082 [0.990, 1.159] |
| `readmission_30d` | 1.068 [0.766, 1.358] |
| `new_dx_365d/diabetes` | 0.984 [0.906, 1.072] |
| `new_dx_365d/heart_failure` | 0.921 [0.845, 1.008] |
| `new_dx_365d/ckd` | 0.967 [0.873, 1.068] |
| `new_dx_365d/copd` | 0.980 [0.896, 1.085] |
