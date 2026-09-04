# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `4828dcb` |
| created | 2026-09-03T20:05:26+00:00 |
| runtime (s) | 210.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_ema_nosig` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/jepa_ema_nosig/final.pt` | cls_mean@final |

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

| task | ckpt:jepa_ema_nosig |
|---|---|
| `mortality_365d` | 0.641 [0.578, 0.705] |
| `inpatient_365d` | 0.693 [0.670, 0.713] |
| `readmission_30d` | 0.660 [0.609, 0.703] |
| `new_dx_365d/diabetes` | 0.738 [0.719, 0.760] |
| `new_dx_365d/heart_failure` | 0.703 [0.679, 0.728] |
| `new_dx_365d/ckd` | 0.697 [0.672, 0.722] |
| `new_dx_365d/copd` | 0.700 [0.676, 0.723] |

## AUPRC

| task | ckpt:jepa_ema_nosig |
|---|---|
| `mortality_365d` | 0.034 [0.025, 0.055] |
| `inpatient_365d` | 0.358 [0.323, 0.394] |
| `readmission_30d` | 0.107 [0.077, 0.150] |
| `new_dx_365d/diabetes` | 0.417 [0.380, 0.455] |
| `new_dx_365d/heart_failure` | 0.263 [0.228, 0.302] |
| `new_dx_365d/ckd` | 0.187 [0.161, 0.226] |
| `new_dx_365d/copd` | 0.234 [0.204, 0.269] |

## BRIER

| task | ckpt:jepa_ema_nosig |
|---|---|
| `mortality_365d` | 0.0214 [0.0172, 0.0265] |
| `inpatient_365d` | 0.1481 [0.1407, 0.1558] |
| `readmission_30d` | 0.0461 [0.0404, 0.0519] |
| `new_dx_365d/diabetes` | 0.1528 [0.1460, 0.1603] |
| `new_dx_365d/heart_failure` | 0.1108 [0.1021, 0.1180] |
| `new_dx_365d/ckd` | 0.0838 [0.0756, 0.0907] |
| `new_dx_365d/copd` | 0.1072 [0.0991, 0.1153] |

## CALIBRATION SLOPE

| task | ckpt:jepa_ema_nosig |
|---|---|
| `mortality_365d` | 1.111 [0.637, 1.657] |
| `inpatient_365d` | 1.084 [0.916, 1.277] |
| `readmission_30d` | 1.161 [0.778, 1.521] |
| `new_dx_365d/diabetes` | 0.991 [0.882, 1.126] |
| `new_dx_365d/heart_failure` | 0.996 [0.873, 1.159] |
| `new_dx_365d/ckd` | 0.915 [0.754, 1.108] |
| `new_dx_365d/copd` | 0.896 [0.768, 1.040] |
