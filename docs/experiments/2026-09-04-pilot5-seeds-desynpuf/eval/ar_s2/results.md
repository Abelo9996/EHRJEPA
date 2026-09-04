# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `3ffe71b` |
| created | 2026-09-04T17:49:26+00:00 |
| runtime (s) | 184.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s2` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot5-seeds-desynpuf/ar_s2/final.pt` | last@final |

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
| `mortality_365d` | 0.606 [0.532, 0.689] |
| `inpatient_365d` | 0.746 [0.724, 0.765] |
| `readmission_30d` | 0.684 [0.646, 0.727] |
| `new_dx_365d/diabetes` | 0.765 [0.748, 0.786] |
| `new_dx_365d/heart_failure` | 0.770 [0.748, 0.789] |
| `new_dx_365d/ckd` | 0.751 [0.725, 0.774] |
| `new_dx_365d/copd` | 0.761 [0.737, 0.781] |

## AUPRC

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.034 [0.024, 0.055] |
| `inpatient_365d` | 0.408 [0.370, 0.451] |
| `readmission_30d` | 0.121 [0.093, 0.174] |
| `new_dx_365d/diabetes` | 0.464 [0.430, 0.507] |
| `new_dx_365d/heart_failure` | 0.307 [0.271, 0.351] |
| `new_dx_365d/ckd` | 0.246 [0.204, 0.293] |
| `new_dx_365d/copd` | 0.314 [0.267, 0.359] |

## BRIER

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.0215 [0.0172, 0.0265] |
| `inpatient_365d` | 0.1410 [0.1336, 0.1477] |
| `readmission_30d` | 0.0456 [0.0398, 0.0512] |
| `new_dx_365d/diabetes` | 0.1471 [0.1405, 0.1535] |
| `new_dx_365d/heart_failure` | 0.1058 [0.0979, 0.1122] |
| `new_dx_365d/ckd` | 0.0807 [0.0730, 0.0873] |
| `new_dx_365d/copd` | 0.1016 [0.0939, 0.1096] |

## CALIBRATION SLOPE

| task | ckpt:ar_s2 |
|---|---|
| `mortality_365d` | 0.813 [0.243, 1.452] |
| `inpatient_365d` | 1.138 [1.012, 1.294] |
| `readmission_30d` | 1.096 [0.846, 1.384] |
| `new_dx_365d/diabetes` | 1.017 [0.910, 1.148] |
| `new_dx_365d/heart_failure` | 1.042 [0.930, 1.163] |
| `new_dx_365d/ckd` | 1.007 [0.869, 1.178] |
| `new_dx_365d/copd` | 1.099 [0.956, 1.259] |
