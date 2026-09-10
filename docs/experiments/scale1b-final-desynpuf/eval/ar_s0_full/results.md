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
| created | 2026-09-09T23:23:33+00:00 |
| runtime (s) | 72.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-desynpuf/ar/final.pt` | last@final |

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

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.604 [0.549, 0.647] |
| `inpatient_365d` | 0.743 [0.727, 0.756] |
| `readmission_30d` | 0.644 [0.598, 0.686] |
| `new_dx_365d/diabetes` | 0.763 [0.748, 0.780] |
| `new_dx_365d/heart_failure` | 0.769 [0.752, 0.783] |
| `new_dx_365d/ckd` | 0.770 [0.752, 0.790] |
| `new_dx_365d/copd` | 0.759 [0.741, 0.777] |

## AUPRC

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.027 [0.021, 0.036] |
| `inpatient_365d` | 0.423 [0.396, 0.451] |
| `readmission_30d` | 0.086 [0.064, 0.119] |
| `new_dx_365d/diabetes` | 0.454 [0.425, 0.485] |
| `new_dx_365d/heart_failure` | 0.317 [0.286, 0.346] |
| `new_dx_365d/ckd` | 0.275 [0.249, 0.303] |
| `new_dx_365d/copd` | 0.341 [0.306, 0.382] |

## BRIER

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1449 [0.1402, 0.1490] |
| `readmission_30d` | 0.0472 [0.0415, 0.0530] |
| `new_dx_365d/diabetes` | 0.1492 [0.1437, 0.1550] |
| `new_dx_365d/heart_failure` | 0.1012 [0.0962, 0.1060] |
| `new_dx_365d/ckd` | 0.0812 [0.0761, 0.0860] |
| `new_dx_365d/copd` | 0.1024 [0.0965, 0.1079] |

## CALIBRATION SLOPE

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.649 [0.337, 0.925] |
| `inpatient_365d` | 1.016 [0.936, 1.100] |
| `readmission_30d` | 0.790 [0.524, 1.025] |
| `new_dx_365d/diabetes` | 0.904 [0.833, 0.976] |
| `new_dx_365d/heart_failure` | 0.985 [0.900, 1.064] |
| `new_dx_365d/ckd` | 1.026 [0.943, 1.118] |
| `new_dx_365d/copd` | 0.984 [0.891, 1.072] |
