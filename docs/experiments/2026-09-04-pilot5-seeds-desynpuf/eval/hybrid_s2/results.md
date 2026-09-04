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
| created | 2026-09-04T18:52:05+00:00 |
| runtime (s) | 175.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s2` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot5-seeds-desynpuf/hybrid_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.625 [0.558, 0.697] |
| `inpatient_365d` | 0.744 [0.723, 0.763] |
| `readmission_30d` | 0.693 [0.656, 0.735] |
| `new_dx_365d/diabetes` | 0.769 [0.753, 0.791] |
| `new_dx_365d/heart_failure` | 0.768 [0.750, 0.791] |
| `new_dx_365d/ckd` | 0.750 [0.726, 0.774] |
| `new_dx_365d/copd` | 0.751 [0.729, 0.775] |

## AUPRC

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.040 [0.024, 0.086] |
| `inpatient_365d` | 0.405 [0.369, 0.443] |
| `readmission_30d` | 0.126 [0.095, 0.188] |
| `new_dx_365d/diabetes` | 0.472 [0.436, 0.514] |
| `new_dx_365d/heart_failure` | 0.314 [0.269, 0.360] |
| `new_dx_365d/ckd` | 0.247 [0.209, 0.295] |
| `new_dx_365d/copd` | 0.275 [0.238, 0.315] |

## BRIER

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1413 [0.1343, 0.1478] |
| `readmission_30d` | 0.0456 [0.0398, 0.0511] |
| `new_dx_365d/diabetes` | 0.1462 [0.1394, 0.1530] |
| `new_dx_365d/heart_failure` | 0.1056 [0.0978, 0.1120] |
| `new_dx_365d/ckd` | 0.0806 [0.0730, 0.0873] |
| `new_dx_365d/copd` | 0.1036 [0.0958, 0.1111] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.777 [0.361, 1.187] |
| `inpatient_365d` | 1.091 [0.956, 1.237] |
| `readmission_30d` | 1.046 [0.829, 1.318] |
| `new_dx_365d/diabetes` | 0.996 [0.901, 1.131] |
| `new_dx_365d/heart_failure` | 1.031 [0.921, 1.165] |
| `new_dx_365d/ckd` | 0.961 [0.839, 1.116] |
| `new_dx_365d/copd` | 1.013 [0.861, 1.171] |
