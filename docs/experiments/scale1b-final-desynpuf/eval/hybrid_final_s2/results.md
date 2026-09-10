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
| created | 2026-09-09T23:20:07+00:00 |
| runtime (s) | 202.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_final_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-final-desynpuf/hybrid_final_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_final_s2 |
|---|---|
| `mortality_365d` | 0.600 [0.553, 0.636] |
| `inpatient_365d` | 0.756 [0.742, 0.768] |
| `readmission_30d` | 0.683 [0.643, 0.725] |
| `new_dx_365d/diabetes` | 0.776 [0.763, 0.792] |
| `new_dx_365d/heart_failure` | 0.800 [0.785, 0.813] |
| `new_dx_365d/ckd` | 0.782 [0.765, 0.802] |
| `new_dx_365d/copd` | 0.772 [0.754, 0.790] |

## AUPRC

| task | ckpt:hybrid_final_s2 |
|---|---|
| `mortality_365d` | 0.029 [0.022, 0.042] |
| `inpatient_365d` | 0.446 [0.419, 0.473] |
| `readmission_30d` | 0.099 [0.077, 0.130] |
| `new_dx_365d/diabetes` | 0.481 [0.451, 0.514] |
| `new_dx_365d/heart_failure` | 0.350 [0.320, 0.381] |
| `new_dx_365d/ckd` | 0.297 [0.268, 0.330] |
| `new_dx_365d/copd` | 0.365 [0.332, 0.404] |

## BRIER

| task | ckpt:hybrid_final_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1422 [0.1378, 0.1466] |
| `readmission_30d` | 0.0470 [0.0413, 0.0529] |
| `new_dx_365d/diabetes` | 0.1459 [0.1402, 0.1514] |
| `new_dx_365d/heart_failure` | 0.0979 [0.0929, 0.1024] |
| `new_dx_365d/ckd` | 0.0798 [0.0750, 0.0847] |
| `new_dx_365d/copd` | 0.1000 [0.0941, 0.1049] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_final_s2 |
|---|---|
| `mortality_365d` | 0.632 [0.316, 0.884] |
| `inpatient_365d` | 0.971 [0.903, 1.035] |
| `readmission_30d` | 0.756 [0.582, 0.947] |
| `new_dx_365d/diabetes` | 0.893 [0.829, 0.963] |
| `new_dx_365d/heart_failure` | 0.960 [0.884, 1.035] |
| `new_dx_365d/ckd` | 1.001 [0.912, 1.105] |
| `new_dx_365d/copd` | 0.964 [0.881, 1.042] |
