# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `1c43622` |
| created | 2026-09-04T08:36:15+00:00 |
| runtime (s) | 195.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:nextlatent_h1416` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot4-desynpuf/nextlatent_h1416/final.pt` | last@final |

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

| task | ckpt:nextlatent_h1416 |
|---|---|
| `mortality_365d` | 0.606 [0.546, 0.675] |
| `inpatient_365d` | 0.669 [0.647, 0.689] |
| `readmission_30d` | 0.640 [0.589, 0.695] |
| `new_dx_365d/diabetes` | 0.720 [0.700, 0.740] |
| `new_dx_365d/heart_failure` | 0.687 [0.662, 0.712] |
| `new_dx_365d/ckd` | 0.689 [0.661, 0.716] |
| `new_dx_365d/copd` | 0.682 [0.658, 0.707] |

## AUPRC

| task | ckpt:nextlatent_h1416 |
|---|---|
| `mortality_365d` | 0.029 [0.022, 0.043] |
| `inpatient_365d` | 0.315 [0.284, 0.347] |
| `readmission_30d` | 0.092 [0.070, 0.131] |
| `new_dx_365d/diabetes` | 0.398 [0.361, 0.432] |
| `new_dx_365d/heart_failure` | 0.235 [0.210, 0.268] |
| `new_dx_365d/ckd` | 0.167 [0.144, 0.196] |
| `new_dx_365d/copd` | 0.208 [0.187, 0.239] |

## BRIER

| task | ckpt:nextlatent_h1416 |
|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] |
| `inpatient_365d` | 0.1514 [0.1441, 0.1594] |
| `readmission_30d` | 0.0464 [0.0406, 0.0523] |
| `new_dx_365d/diabetes` | 0.1558 [0.1492, 0.1631] |
| `new_dx_365d/heart_failure` | 0.1122 [0.1030, 0.1193] |
| `new_dx_365d/ckd` | 0.0844 [0.0762, 0.0917] |
| `new_dx_365d/copd` | 0.1089 [0.1001, 0.1178] |

## CALIBRATION SLOPE

| task | ckpt:nextlatent_h1416 |
|---|---|
| `mortality_365d` | 0.586 [0.263, 0.974] |
| `inpatient_365d` | 0.996 [0.852, 1.168] |
| `readmission_30d` | 0.911 [0.567, 1.297] |
| `new_dx_365d/diabetes` | 0.948 [0.848, 1.062] |
| `new_dx_365d/heart_failure` | 0.989 [0.863, 1.170] |
| `new_dx_365d/ckd` | 0.984 [0.810, 1.164] |
| `new_dx_365d/copd` | 0.848 [0.714, 1.000] |
