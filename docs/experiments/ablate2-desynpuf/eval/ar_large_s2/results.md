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
| created | 2026-09-08T02:16:34+00:00 |
| runtime (s) | 443.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_large_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/ar_large_s2/final.pt` | last@final |

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

| task | ckpt:ar_large_s2 |
|---|---|
| `mortality_365d` | 0.612 [0.562, 0.647] |
| `inpatient_365d` | 0.752 [0.737, 0.764] |
| `readmission_30d` | 0.683 [0.643, 0.724] |
| `new_dx_365d/diabetes` | 0.777 [0.763, 0.791] |
| `new_dx_365d/heart_failure` | 0.769 [0.754, 0.785] |
| `new_dx_365d/ckd` | 0.773 [0.752, 0.791] |
| `new_dx_365d/copd` | 0.770 [0.750, 0.787] |

## AUPRC

| task | ckpt:ar_large_s2 |
|---|---|
| `mortality_365d` | 0.027 [0.021, 0.033] |
| `inpatient_365d` | 0.429 [0.403, 0.457] |
| `readmission_30d` | 0.100 [0.075, 0.134] |
| `new_dx_365d/diabetes` | 0.484 [0.454, 0.513] |
| `new_dx_365d/heart_failure` | 0.309 [0.282, 0.342] |
| `new_dx_365d/ckd` | 0.276 [0.245, 0.305] |
| `new_dx_365d/copd` | 0.337 [0.304, 0.374] |

## BRIER

| task | ckpt:ar_large_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1435 [0.1392, 0.1477] |
| `readmission_30d` | 0.0468 [0.0409, 0.0527] |
| `new_dx_365d/diabetes` | 0.1454 [0.1404, 0.1503] |
| `new_dx_365d/heart_failure` | 0.1016 [0.0961, 0.1060] |
| `new_dx_365d/ckd` | 0.0809 [0.0756, 0.0855] |
| `new_dx_365d/copd` | 0.1017 [0.0958, 0.1067] |

## CALIBRATION SLOPE

| task | ckpt:ar_large_s2 |
|---|---|
| `mortality_365d` | 0.678 [0.374, 0.906] |
| `inpatient_365d` | 1.039 [0.955, 1.110] |
| `readmission_30d` | 0.894 [0.711, 1.128] |
| `new_dx_365d/diabetes` | 0.967 [0.891, 1.036] |
| `new_dx_365d/heart_failure` | 0.947 [0.866, 1.041] |
| `new_dx_365d/ckd` | 0.966 [0.870, 1.063] |
| `new_dx_365d/copd` | 0.945 [0.854, 1.030] |
