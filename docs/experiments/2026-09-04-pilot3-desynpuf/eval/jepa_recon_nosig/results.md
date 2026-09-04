# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `54b9a7a` |
| created | 2026-09-04T07:29:51+00:00 |
| runtime (s) | 162.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_recon_nosig` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot3-desynpuf/jepa_recon_nosig/final.pt` | mean@final |

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

| task | ckpt:jepa_recon_nosig |
|---|---|
| `mortality_365d` | 0.592 [0.513, 0.664] |
| `inpatient_365d` | 0.695 [0.673, 0.714] |
| `readmission_30d` | 0.670 [0.625, 0.714] |
| `new_dx_365d/diabetes` | 0.747 [0.728, 0.771] |
| `new_dx_365d/heart_failure` | 0.709 [0.686, 0.733] |
| `new_dx_365d/ckd` | 0.711 [0.683, 0.739] |
| `new_dx_365d/copd` | 0.714 [0.686, 0.736] |

## AUPRC

| task | ckpt:jepa_recon_nosig |
|---|---|
| `mortality_365d` | 0.030 [0.021, 0.048] |
| `inpatient_365d` | 0.341 [0.310, 0.376] |
| `readmission_30d` | 0.103 [0.077, 0.140] |
| `new_dx_365d/diabetes` | 0.448 [0.414, 0.490] |
| `new_dx_365d/heart_failure` | 0.255 [0.227, 0.290] |
| `new_dx_365d/ckd` | 0.190 [0.165, 0.225] |
| `new_dx_365d/copd` | 0.252 [0.220, 0.290] |

## BRIER

| task | ckpt:jepa_recon_nosig |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1486 [0.1415, 0.1559] |
| `readmission_30d` | 0.0460 [0.0405, 0.0518] |
| `new_dx_365d/diabetes` | 0.1503 [0.1432, 0.1582] |
| `new_dx_365d/heart_failure` | 0.1108 [0.1019, 0.1176] |
| `new_dx_365d/ckd` | 0.0835 [0.0753, 0.0909] |
| `new_dx_365d/copd` | 0.1062 [0.0982, 0.1150] |

## CALIBRATION SLOPE

| task | ckpt:jepa_recon_nosig |
|---|---|
| `mortality_365d` | 0.654 [0.075, 1.199] |
| `inpatient_365d` | 0.988 [0.868, 1.111] |
| `readmission_30d` | 1.049 [0.782, 1.327] |
| `new_dx_365d/diabetes` | 0.983 [0.873, 1.123] |
| `new_dx_365d/heart_failure` | 0.900 [0.793, 1.038] |
| `new_dx_365d/ckd` | 0.976 [0.815, 1.165] |
| `new_dx_365d/copd` | 0.946 [0.803, 1.092] |
