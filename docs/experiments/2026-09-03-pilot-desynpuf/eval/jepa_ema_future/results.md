# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `e4a0e2d` |
| created | 2026-09-03T21:44:48+00:00 |
| runtime (s) | 207.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_ema_future` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/jepa_ema_future/final.pt` | cls_mean@final |

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

| task | ckpt:jepa_ema_future |
|---|---|
| `mortality_365d` | 0.584 [0.523, 0.645] |
| `inpatient_365d` | 0.682 [0.659, 0.706] |
| `readmission_30d` | 0.661 [0.614, 0.706] |
| `new_dx_365d/diabetes` | 0.738 [0.719, 0.760] |
| `new_dx_365d/heart_failure` | 0.700 [0.675, 0.724] |
| `new_dx_365d/ckd` | 0.694 [0.670, 0.721] |
| `new_dx_365d/copd` | 0.705 [0.680, 0.728] |

## AUPRC

| task | ckpt:jepa_ema_future |
|---|---|
| `mortality_365d` | 0.027 [0.021, 0.039] |
| `inpatient_365d` | 0.335 [0.297, 0.371] |
| `readmission_30d` | 0.101 [0.075, 0.141] |
| `new_dx_365d/diabetes` | 0.414 [0.383, 0.455] |
| `new_dx_365d/heart_failure` | 0.249 [0.216, 0.290] |
| `new_dx_365d/ckd` | 0.180 [0.157, 0.214] |
| `new_dx_365d/copd` | 0.233 [0.205, 0.273] |

## BRIER

| task | ckpt:jepa_ema_future |
|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] |
| `inpatient_365d` | 0.1496 [0.1426, 0.1572] |
| `readmission_30d` | 0.0461 [0.0404, 0.0520] |
| `new_dx_365d/diabetes` | 0.1528 [0.1457, 0.1598] |
| `new_dx_365d/heart_failure` | 0.1113 [0.1025, 0.1183] |
| `new_dx_365d/ckd` | 0.0841 [0.0761, 0.0911] |
| `new_dx_365d/copd` | 0.1071 [0.0982, 0.1156] |

## CALIBRATION SLOPE

| task | ckpt:jepa_ema_future |
|---|---|
| `mortality_365d` | 0.575 [0.197, 0.982] |
| `inpatient_365d` | 1.005 [0.844, 1.183] |
| `readmission_30d` | 0.985 [0.672, 1.233] |
| `new_dx_365d/diabetes` | 0.969 [0.868, 1.094] |
| `new_dx_365d/heart_failure` | 0.956 [0.840, 1.112] |
| `new_dx_365d/ckd` | 0.867 [0.724, 1.040] |
| `new_dx_365d/copd` | 0.944 [0.794, 1.107] |
