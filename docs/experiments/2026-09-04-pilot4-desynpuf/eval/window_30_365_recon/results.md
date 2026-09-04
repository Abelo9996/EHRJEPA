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
| created | 2026-09-04T10:43:32+00:00 |
| runtime (s) | 167.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:window_30_365_recon` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot4-desynpuf/window_30_365_recon/final.pt` | last@final |

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

| task | ckpt:window_30_365_recon |
|---|---|
| `mortality_365d` | 0.594 [0.527, 0.651] |
| `inpatient_365d` | 0.676 [0.653, 0.696] |
| `readmission_30d` | 0.621 [0.568, 0.666] |
| `new_dx_365d/diabetes` | 0.727 [0.709, 0.749] |
| `new_dx_365d/heart_failure` | 0.695 [0.670, 0.720] |
| `new_dx_365d/ckd` | 0.681 [0.652, 0.709] |
| `new_dx_365d/copd` | 0.705 [0.675, 0.730] |

## AUPRC

| task | ckpt:window_30_365_recon |
|---|---|
| `mortality_365d` | 0.028 [0.021, 0.041] |
| `inpatient_365d` | 0.324 [0.292, 0.358] |
| `readmission_30d` | 0.083 [0.062, 0.117] |
| `new_dx_365d/diabetes` | 0.412 [0.376, 0.453] |
| `new_dx_365d/heart_failure` | 0.249 [0.218, 0.283] |
| `new_dx_365d/ckd` | 0.168 [0.141, 0.197] |
| `new_dx_365d/copd` | 0.236 [0.203, 0.273] |

## BRIER

| task | ckpt:window_30_365_recon |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1507 [0.1437, 0.1584] |
| `readmission_30d` | 0.0466 [0.0409, 0.0525] |
| `new_dx_365d/diabetes` | 0.1546 [0.1478, 0.1614] |
| `new_dx_365d/heart_failure` | 0.1116 [0.1024, 0.1185] |
| `new_dx_365d/ckd` | 0.0848 [0.0771, 0.0925] |
| `new_dx_365d/copd` | 0.1070 [0.0984, 0.1159] |

## CALIBRATION SLOPE

| task | ckpt:window_30_365_recon |
|---|---|
| `mortality_365d` | 0.638 [0.223, 1.021] |
| `inpatient_365d` | 1.026 [0.885, 1.192] |
| `readmission_30d` | 0.739 [0.382, 1.102] |
| `new_dx_365d/diabetes` | 0.924 [0.819, 1.049] |
| `new_dx_365d/heart_failure` | 0.945 [0.827, 1.094] |
| `new_dx_365d/ckd` | 0.848 [0.698, 1.027] |
| `new_dx_365d/copd` | 0.974 [0.817, 1.131] |
