# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `5353d33` |
| created | 2026-09-10T15:00:36+00:00 |
| runtime (s) | 53.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/hybrid_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.593 [0.546, 0.633] |
| `inpatient_365d` | 0.753 [0.739, 0.766] |
| `readmission_30d` | 0.683 [0.639, 0.726] |
| `new_dx_365d/diabetes` | 0.778 [0.765, 0.793] |
| `new_dx_365d/heart_failure` | 0.796 [0.782, 0.812] |
| `new_dx_365d/ckd` | 0.785 [0.767, 0.803] |
| `new_dx_365d/copd` | 0.773 [0.752, 0.791] |

## AUPRC

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.029 [0.022, 0.046] |
| `inpatient_365d` | 0.447 [0.419, 0.472] |
| `readmission_30d` | 0.100 [0.078, 0.128] |
| `new_dx_365d/diabetes` | 0.486 [0.456, 0.514] |
| `new_dx_365d/heart_failure` | 0.350 [0.318, 0.381] |
| `new_dx_365d/ckd` | 0.294 [0.267, 0.329] |
| `new_dx_365d/copd` | 0.362 [0.326, 0.398] |

## BRIER

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1424 [0.1378, 0.1464] |
| `readmission_30d` | 0.0470 [0.0413, 0.0528] |
| `new_dx_365d/diabetes` | 0.1452 [0.1395, 0.1507] |
| `new_dx_365d/heart_failure` | 0.0980 [0.0932, 0.1026] |
| `new_dx_365d/ckd` | 0.0798 [0.0747, 0.0845] |
| `new_dx_365d/copd` | 0.1003 [0.0939, 0.1055] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s1 |
|---|---|
| `mortality_365d` | 0.580 [0.295, 0.862] |
| `inpatient_365d` | 0.996 [0.915, 1.069] |
| `readmission_30d` | 0.761 [0.582, 0.959] |
| `new_dx_365d/diabetes` | 0.954 [0.881, 1.023] |
| `new_dx_365d/heart_failure` | 0.962 [0.880, 1.051] |
| `new_dx_365d/ckd` | 1.022 [0.933, 1.120] |
| `new_dx_365d/copd` | 0.963 [0.866, 1.047] |
