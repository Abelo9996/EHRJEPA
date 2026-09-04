# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `1b448f9` |
| created | 2026-09-04T10:14:39+00:00 |
| runtime (s) | 156.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:window_30_365` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot4-desynpuf/window_30_365/final.pt` | last@final |

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

| task | ckpt:window_30_365 |
|---|---|
| `mortality_365d` | 0.586 [0.527, 0.651] |
| `inpatient_365d` | 0.673 [0.648, 0.694] |
| `readmission_30d` | 0.626 [0.577, 0.672] |
| `new_dx_365d/diabetes` | 0.728 [0.707, 0.747] |
| `new_dx_365d/heart_failure` | 0.690 [0.666, 0.714] |
| `new_dx_365d/ckd` | 0.686 [0.657, 0.711] |
| `new_dx_365d/copd` | 0.703 [0.675, 0.729] |

## AUPRC

| task | ckpt:window_30_365 |
|---|---|
| `mortality_365d` | 0.028 [0.021, 0.042] |
| `inpatient_365d` | 0.319 [0.289, 0.353] |
| `readmission_30d` | 0.086 [0.065, 0.116] |
| `new_dx_365d/diabetes` | 0.411 [0.373, 0.452] |
| `new_dx_365d/heart_failure` | 0.242 [0.211, 0.273] |
| `new_dx_365d/ckd` | 0.172 [0.147, 0.206] |
| `new_dx_365d/copd` | 0.228 [0.199, 0.266] |

## BRIER

| task | ckpt:window_30_365 |
|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] |
| `inpatient_365d` | 0.1511 [0.1440, 0.1589] |
| `readmission_30d` | 0.0465 [0.0407, 0.0523] |
| `new_dx_365d/diabetes` | 0.1545 [0.1482, 0.1612] |
| `new_dx_365d/heart_failure` | 0.1120 [0.1026, 0.1192] |
| `new_dx_365d/ckd` | 0.0844 [0.0764, 0.0920] |
| `new_dx_365d/copd` | 0.1074 [0.0985, 0.1161] |

## CALIBRATION SLOPE

| task | ckpt:window_30_365 |
|---|---|
| `mortality_365d` | 0.537 [0.210, 0.964] |
| `inpatient_365d` | 0.988 [0.840, 1.160] |
| `readmission_30d` | 0.735 [0.420, 1.028] |
| `new_dx_365d/diabetes` | 0.948 [0.842, 1.064] |
| `new_dx_365d/heart_failure` | 0.963 [0.850, 1.102] |
| `new_dx_365d/ckd` | 0.924 [0.783, 1.106] |
| `new_dx_365d/copd` | 0.975 [0.817, 1.139] |
