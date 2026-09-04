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
| created | 2026-09-04T04:52:09+00:00 |
| runtime (s) | 181.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_content_future` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot2-desynpuf/jepa_content_future/final.pt` | mean@final |

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

| task | ckpt:jepa_content_future |
|---|---|
| `mortality_365d` | 0.578 [0.512, 0.655] |
| `inpatient_365d` | 0.678 [0.654, 0.699] |
| `readmission_30d` | 0.653 [0.606, 0.703] |
| `new_dx_365d/diabetes` | 0.728 [0.706, 0.748] |
| `new_dx_365d/heart_failure` | 0.696 [0.671, 0.726] |
| `new_dx_365d/ckd` | 0.695 [0.667, 0.724] |
| `new_dx_365d/copd` | 0.697 [0.674, 0.721] |

## AUPRC

| task | ckpt:jepa_content_future |
|---|---|
| `mortality_365d` | 0.027 [0.020, 0.039] |
| `inpatient_365d` | 0.336 [0.304, 0.370] |
| `readmission_30d` | 0.101 [0.072, 0.140] |
| `new_dx_365d/diabetes` | 0.404 [0.371, 0.444] |
| `new_dx_365d/heart_failure` | 0.256 [0.224, 0.297] |
| `new_dx_365d/ckd` | 0.189 [0.162, 0.227] |
| `new_dx_365d/copd` | 0.226 [0.200, 0.265] |

## BRIER

| task | ckpt:jepa_content_future |
|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] |
| `inpatient_365d` | 0.1502 [0.1429, 0.1577] |
| `readmission_30d` | 0.0462 [0.0405, 0.0520] |
| `new_dx_365d/diabetes` | 0.1551 [0.1483, 0.1638] |
| `new_dx_365d/heart_failure` | 0.1114 [0.1026, 0.1185] |
| `new_dx_365d/ckd` | 0.0839 [0.0762, 0.0909] |
| `new_dx_365d/copd` | 0.1076 [0.0991, 0.1158] |

## CALIBRATION SLOPE

| task | ckpt:jepa_content_future |
|---|---|
| `mortality_365d` | 0.524 [0.076, 1.083] |
| `inpatient_365d` | 1.055 [0.911, 1.194] |
| `readmission_30d` | 1.072 [0.726, 1.415] |
| `new_dx_365d/diabetes` | 0.908 [0.803, 1.029] |
| `new_dx_365d/heart_failure` | 0.954 [0.826, 1.115] |
| `new_dx_365d/ckd` | 0.917 [0.775, 1.089] |
| `new_dx_365d/copd` | 0.969 [0.827, 1.119] |
