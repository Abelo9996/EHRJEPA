# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `22cb91c` |
| created | 2026-09-04T04:58:31+00:00 |
| runtime (s) | 182.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_ema` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/jepa_ema/final.pt` | mean@final |

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

| task | ckpt:jepa_ema |
|---|---|
| `mortality_365d` | 0.624 [0.562, 0.695] |
| `inpatient_365d` | 0.686 [0.662, 0.707] |
| `readmission_30d` | 0.650 [0.603, 0.697] |
| `new_dx_365d/diabetes` | 0.743 [0.724, 0.764] |
| `new_dx_365d/heart_failure` | 0.703 [0.678, 0.725] |
| `new_dx_365d/ckd` | 0.700 [0.677, 0.731] |
| `new_dx_365d/copd` | 0.705 [0.679, 0.728] |

## AUPRC

| task | ckpt:jepa_ema |
|---|---|
| `mortality_365d` | 0.031 [0.023, 0.044] |
| `inpatient_365d` | 0.343 [0.310, 0.378] |
| `readmission_30d` | 0.088 [0.069, 0.118] |
| `new_dx_365d/diabetes` | 0.427 [0.395, 0.471] |
| `new_dx_365d/heart_failure` | 0.247 [0.215, 0.284] |
| `new_dx_365d/ckd` | 0.193 [0.165, 0.230] |
| `new_dx_365d/copd` | 0.234 [0.204, 0.270] |

## BRIER

| task | ckpt:jepa_ema |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1491 [0.1416, 0.1564] |
| `readmission_30d` | 0.0464 [0.0406, 0.0522] |
| `new_dx_365d/diabetes` | 0.1519 [0.1447, 0.1588] |
| `new_dx_365d/heart_failure` | 0.1112 [0.1022, 0.1183] |
| `new_dx_365d/ckd` | 0.0836 [0.0755, 0.0909] |
| `new_dx_365d/copd` | 0.1069 [0.0985, 0.1153] |

## CALIBRATION SLOPE

| task | ckpt:jepa_ema |
|---|---|
| `mortality_365d` | 0.873 [0.438, 1.371] |
| `inpatient_365d` | 1.057 [0.905, 1.224] |
| `readmission_30d` | 0.956 [0.686, 1.209] |
| `new_dx_365d/diabetes` | 0.990 [0.869, 1.119] |
| `new_dx_365d/heart_failure` | 0.989 [0.870, 1.132] |
| `new_dx_365d/ckd` | 0.924 [0.771, 1.123] |
| `new_dx_365d/copd` | 0.983 [0.820, 1.131] |
