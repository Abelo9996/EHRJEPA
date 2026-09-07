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
| created | 2026-09-07T09:26:56+00:00 |
| runtime (s) | 177.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/ar_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 3000 (0.0220) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 3000 (0.2010) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3000 (0.0493) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 3000 (0.2240) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 3000 (0.1370) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 3000 (0.0970) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 3000 (0.1303) |  |

## AUROC

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.593 [0.527, 0.671] |
| `inpatient_365d` | 0.744 [0.720, 0.764] |
| `readmission_30d` | 0.668 [0.625, 0.711] |
| `new_dx_365d/diabetes` | 0.759 [0.740, 0.780] |
| `new_dx_365d/heart_failure` | 0.779 [0.755, 0.803] |
| `new_dx_365d/ckd` | 0.758 [0.729, 0.782] |
| `new_dx_365d/copd` | 0.766 [0.739, 0.787] |

## AUPRC

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.035 [0.024, 0.066] |
| `inpatient_365d` | 0.404 [0.363, 0.440] |
| `readmission_30d` | 0.103 [0.077, 0.150] |
| `new_dx_365d/diabetes` | 0.438 [0.400, 0.480] |
| `new_dx_365d/heart_failure` | 0.335 [0.289, 0.390] |
| `new_dx_365d/ckd` | 0.247 [0.215, 0.300] |
| `new_dx_365d/copd` | 0.319 [0.276, 0.373] |

## BRIER

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1413 [0.1344, 0.1482] |
| `readmission_30d` | 0.0462 [0.0403, 0.0519] |
| `new_dx_365d/diabetes` | 0.1494 [0.1426, 0.1567] |
| `new_dx_365d/heart_failure` | 0.1039 [0.0960, 0.1110] |
| `new_dx_365d/ckd` | 0.0804 [0.0732, 0.0870] |
| `new_dx_365d/copd` | 0.1008 [0.0927, 0.1080] |

## CALIBRATION SLOPE

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.569 [0.144, 1.070] |
| `inpatient_365d` | 1.040 [0.909, 1.155] |
| `readmission_30d` | 0.876 [0.655, 1.144] |
| `new_dx_365d/diabetes` | 0.952 [0.856, 1.057] |
| `new_dx_365d/heart_failure` | 1.057 [0.944, 1.211] |
| `new_dx_365d/ckd` | 0.965 [0.826, 1.101] |
| `new_dx_365d/copd` | 0.999 [0.870, 1.121] |
