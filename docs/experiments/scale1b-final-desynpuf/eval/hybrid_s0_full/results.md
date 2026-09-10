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
| created | 2026-09-09T23:24:49+00:00 |
| runtime (s) | 70.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-desynpuf/hybrid/final.pt` | last@final |

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
| `mortality_365d` | 0.603 [0.554, 0.641] |
| `inpatient_365d` | 0.758 [0.743, 0.771] |
| `readmission_30d` | 0.684 [0.643, 0.722] |
| `new_dx_365d/diabetes` | 0.777 [0.764, 0.793] |
| `new_dx_365d/heart_failure` | 0.796 [0.780, 0.810] |
| `new_dx_365d/ckd` | 0.784 [0.766, 0.804] |
| `new_dx_365d/copd` | 0.775 [0.757, 0.793] |

## AUPRC

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.029 [0.023, 0.042] |
| `inpatient_365d` | 0.451 [0.424, 0.476] |
| `readmission_30d` | 0.103 [0.082, 0.138] |
| `new_dx_365d/diabetes` | 0.485 [0.456, 0.515] |
| `new_dx_365d/heart_failure` | 0.353 [0.318, 0.385] |
| `new_dx_365d/ckd` | 0.303 [0.271, 0.342] |
| `new_dx_365d/copd` | 0.355 [0.320, 0.394] |

## BRIER

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1417 [0.1374, 0.1456] |
| `readmission_30d` | 0.0469 [0.0411, 0.0527] |
| `new_dx_365d/diabetes` | 0.1456 [0.1398, 0.1507] |
| `new_dx_365d/heart_failure` | 0.0982 [0.0932, 0.1023] |
| `new_dx_365d/ckd` | 0.0794 [0.0747, 0.0840] |
| `new_dx_365d/copd` | 0.1004 [0.0943, 0.1056] |

## CALIBRATION SLOPE

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.641 [0.367, 0.901] |
| `inpatient_365d` | 1.032 [0.952, 1.105] |
| `readmission_30d` | 0.774 [0.601, 0.973] |
| `new_dx_365d/diabetes` | 0.949 [0.881, 1.028] |
| `new_dx_365d/heart_failure` | 1.010 [0.928, 1.094] |
| `new_dx_365d/ckd` | 0.950 [0.858, 1.047] |
| `new_dx_365d/copd` | 0.961 [0.872, 1.048] |
