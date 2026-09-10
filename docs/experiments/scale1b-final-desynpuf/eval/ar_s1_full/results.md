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
| created | 2026-09-09T23:26:04+00:00 |
| runtime (s) | 136.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/ar_s1/final.pt` | last@final |

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

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.612 [0.565, 0.653] |
| `inpatient_365d` | 0.740 [0.725, 0.752] |
| `readmission_30d` | 0.665 [0.619, 0.708] |
| `new_dx_365d/diabetes` | 0.764 [0.749, 0.780] |
| `new_dx_365d/heart_failure` | 0.771 [0.756, 0.787] |
| `new_dx_365d/ckd` | 0.766 [0.750, 0.786] |
| `new_dx_365d/copd` | 0.763 [0.746, 0.783] |

## AUPRC

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.028 [0.023, 0.037] |
| `inpatient_365d` | 0.409 [0.383, 0.434] |
| `readmission_30d` | 0.088 [0.067, 0.115] |
| `new_dx_365d/diabetes` | 0.454 [0.424, 0.484] |
| `new_dx_365d/heart_failure` | 0.302 [0.276, 0.328] |
| `new_dx_365d/ckd` | 0.259 [0.236, 0.287] |
| `new_dx_365d/copd` | 0.334 [0.303, 0.375] |

## BRIER

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1460 [0.1417, 0.1508] |
| `readmission_30d` | 0.0471 [0.0411, 0.0526] |
| `new_dx_365d/diabetes` | 0.1491 [0.1434, 0.1543] |
| `new_dx_365d/heart_failure` | 0.1018 [0.0965, 0.1062] |
| `new_dx_365d/ckd` | 0.0821 [0.0765, 0.0869] |
| `new_dx_365d/copd` | 0.1023 [0.0961, 0.1075] |

## CALIBRATION SLOPE

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.675 [0.393, 0.955] |
| `inpatient_365d` | 0.963 [0.878, 1.032] |
| `readmission_30d` | 0.833 [0.601, 1.065] |
| `new_dx_365d/diabetes` | 0.947 [0.873, 1.027] |
| `new_dx_365d/heart_failure` | 1.005 [0.932, 1.087] |
| `new_dx_365d/ckd` | 1.001 [0.924, 1.102] |
| `new_dx_365d/copd` | 0.989 [0.905, 1.082] |
