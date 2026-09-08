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
| created | 2026-09-08T06:32:08+00:00 |
| runtime (s) | 217.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/hybrid/final.pt` | last@final |

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

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.609 [0.566, 0.646] |
| `inpatient_365d` | 0.752 [0.739, 0.764] |
| `readmission_30d` | 0.690 [0.652, 0.733] |
| `new_dx_365d/diabetes` | 0.772 [0.758, 0.787] |
| `new_dx_365d/heart_failure` | 0.774 [0.759, 0.788] |
| `new_dx_365d/ckd` | 0.780 [0.761, 0.797] |
| `new_dx_365d/copd` | 0.764 [0.749, 0.785] |

## AUPRC

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.028 [0.022, 0.038] |
| `inpatient_365d` | 0.447 [0.421, 0.470] |
| `readmission_30d` | 0.100 [0.081, 0.133] |
| `new_dx_365d/diabetes` | 0.482 [0.449, 0.515] |
| `new_dx_365d/heart_failure` | 0.327 [0.298, 0.355] |
| `new_dx_365d/ckd` | 0.285 [0.255, 0.321] |
| `new_dx_365d/copd` | 0.347 [0.312, 0.385] |

## BRIER

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1427 [0.1383, 0.1467] |
| `readmission_30d` | 0.0469 [0.0413, 0.0527] |
| `new_dx_365d/diabetes` | 0.1463 [0.1405, 0.1517] |
| `new_dx_365d/heart_failure` | 0.1006 [0.0953, 0.1049] |
| `new_dx_365d/ckd` | 0.0804 [0.0756, 0.0851] |
| `new_dx_365d/copd` | 0.1019 [0.0961, 0.1073] |

## CALIBRATION SLOPE

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.668 [0.388, 0.909] |
| `inpatient_365d` | 1.041 [0.958, 1.116] |
| `readmission_30d` | 0.782 [0.608, 0.982] |
| `new_dx_365d/diabetes` | 0.956 [0.886, 1.032] |
| `new_dx_365d/heart_failure` | 0.983 [0.894, 1.064] |
| `new_dx_365d/ckd` | 0.983 [0.883, 1.080] |
| `new_dx_365d/copd` | 0.997 [0.914, 1.102] |
