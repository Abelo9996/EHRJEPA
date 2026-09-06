# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `12478aa` |
| created | 2026-09-06T03:50:11+00:00 |
| runtime (s) | 236.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/hybrid/final.pt` | last@final |

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

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.592 [0.525, 0.661] |
| `inpatient_365d` | 0.751 [0.727, 0.770] |
| `readmission_30d` | 0.693 [0.660, 0.729] |
| `new_dx_365d/diabetes` | 0.773 [0.756, 0.795] |
| `new_dx_365d/heart_failure` | 0.777 [0.756, 0.799] |
| `new_dx_365d/ckd` | 0.776 [0.751, 0.798] |
| `new_dx_365d/copd` | 0.768 [0.746, 0.791] |

## AUPRC

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.035 [0.022, 0.061] |
| `inpatient_365d` | 0.425 [0.380, 0.468] |
| `readmission_30d` | 0.100 [0.080, 0.132] |
| `new_dx_365d/diabetes` | 0.483 [0.446, 0.525] |
| `new_dx_365d/heart_failure` | 0.325 [0.285, 0.374] |
| `new_dx_365d/ckd` | 0.278 [0.232, 0.327] |
| `new_dx_365d/copd` | 0.328 [0.283, 0.382] |

## BRIER

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1394 [0.1332, 0.1460] |
| `readmission_30d` | 0.0463 [0.0403, 0.0521] |
| `new_dx_365d/diabetes` | 0.1450 [0.1386, 0.1516] |
| `new_dx_365d/heart_failure` | 0.1047 [0.0971, 0.1113] |
| `new_dx_365d/ckd` | 0.0789 [0.0715, 0.0852] |
| `new_dx_365d/copd` | 0.1004 [0.0933, 0.1074] |

## CALIBRATION SLOPE

| task | ckpt:hybrid |
|---|---|
| `mortality_365d` | 0.569 [0.150, 1.065] |
| `inpatient_365d` | 1.059 [0.923, 1.188] |
| `readmission_30d` | 0.794 [0.644, 0.966] |
| `new_dx_365d/diabetes` | 0.964 [0.870, 1.088] |
| `new_dx_365d/heart_failure` | 1.012 [0.912, 1.141] |
| `new_dx_365d/ckd` | 0.972 [0.842, 1.105] |
| `new_dx_365d/copd` | 1.004 [0.882, 1.156] |
