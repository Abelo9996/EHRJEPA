# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `406ff81` |
| created | 2026-09-06T10:41:04+00:00 |
| runtime (s) | 177.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-desynpuf/ar/final.pt` | last@final |

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

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.603 [0.539, 0.669] |
| `inpatient_365d` | 0.751 [0.728, 0.769] |
| `readmission_30d` | 0.649 [0.605, 0.689] |
| `new_dx_365d/diabetes` | 0.762 [0.744, 0.783] |
| `new_dx_365d/heart_failure` | 0.780 [0.760, 0.803] |
| `new_dx_365d/ckd` | 0.759 [0.732, 0.785] |
| `new_dx_365d/copd` | 0.763 [0.735, 0.787] |

## AUPRC

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.034 [0.026, 0.061] |
| `inpatient_365d` | 0.424 [0.381, 0.465] |
| `readmission_30d` | 0.087 [0.063, 0.121] |
| `new_dx_365d/diabetes` | 0.445 [0.412, 0.482] |
| `new_dx_365d/heart_failure` | 0.340 [0.299, 0.385] |
| `new_dx_365d/ckd` | 0.262 [0.217, 0.308] |
| `new_dx_365d/copd` | 0.328 [0.283, 0.370] |

## BRIER

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.0215 [0.0172, 0.0265] |
| `inpatient_365d` | 0.1396 [0.1331, 0.1463] |
| `readmission_30d` | 0.0465 [0.0406, 0.0523] |
| `new_dx_365d/diabetes` | 0.1489 [0.1419, 0.1556] |
| `new_dx_365d/heart_failure` | 0.1039 [0.0962, 0.1103] |
| `new_dx_365d/ckd` | 0.0801 [0.0724, 0.0864] |
| `new_dx_365d/copd` | 0.1006 [0.0929, 0.1077] |

## CALIBRATION SLOPE

| task | ckpt:ar |
|---|---|
| `mortality_365d` | 0.677 [0.282, 1.127] |
| `inpatient_365d` | 1.092 [0.972, 1.208] |
| `readmission_30d` | 0.821 [0.560, 1.057] |
| `new_dx_365d/diabetes` | 0.894 [0.807, 1.001] |
| `new_dx_365d/heart_failure` | 1.068 [0.960, 1.216] |
| `new_dx_365d/ckd` | 0.999 [0.867, 1.135] |
| `new_dx_365d/copd` | 0.981 [0.851, 1.102] |
