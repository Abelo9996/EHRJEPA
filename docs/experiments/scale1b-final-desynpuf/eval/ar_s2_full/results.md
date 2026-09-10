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
| created | 2026-09-10T15:01:32+00:00 |
| runtime (s) | 53.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/ar_s2/final.pt` | last@final |

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

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.601 [0.555, 0.640] |
| `inpatient_365d` | 0.743 [0.730, 0.756] |
| `readmission_30d` | 0.665 [0.617, 0.708] |
| `new_dx_365d/diabetes` | 0.761 [0.747, 0.775] |
| `new_dx_365d/heart_failure` | 0.774 [0.756, 0.788] |
| `new_dx_365d/ckd` | 0.766 [0.748, 0.787] |
| `new_dx_365d/copd` | 0.758 [0.738, 0.777] |

## AUPRC

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.026 [0.020, 0.035] |
| `inpatient_365d` | 0.425 [0.399, 0.449] |
| `readmission_30d` | 0.102 [0.075, 0.136] |
| `new_dx_365d/diabetes` | 0.446 [0.414, 0.471] |
| `new_dx_365d/heart_failure` | 0.319 [0.291, 0.351] |
| `new_dx_365d/ckd` | 0.271 [0.243, 0.302] |
| `new_dx_365d/copd` | 0.316 [0.281, 0.350] |

## BRIER

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1447 [0.1402, 0.1490] |
| `readmission_30d` | 0.0468 [0.0412, 0.0526] |
| `new_dx_365d/diabetes` | 0.1498 [0.1442, 0.1554] |
| `new_dx_365d/heart_failure` | 0.1007 [0.0955, 0.1052] |
| `new_dx_365d/ckd` | 0.0814 [0.0762, 0.0860] |
| `new_dx_365d/copd` | 0.1036 [0.0974, 0.1089] |

## CALIBRATION SLOPE

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.589 [0.304, 0.841] |
| `inpatient_365d` | 1.024 [0.946, 1.096] |
| `readmission_30d` | 0.854 [0.612, 1.092] |
| `new_dx_365d/diabetes` | 0.964 [0.887, 1.039] |
| `new_dx_365d/heart_failure` | 1.010 [0.915, 1.083] |
| `new_dx_365d/ckd` | 1.008 [0.925, 1.110] |
| `new_dx_365d/copd` | 0.960 [0.857, 1.063] |
