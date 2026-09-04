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
| created | 2026-09-03T17:57:42+00:00 |
| runtime (s) | 3534.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/ar/final.pt` | cls_mean@final |
| `ckpt:ar` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/ar/final.pt` | cls_mean@final |

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

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.616 [0.541, 0.678] | 0.606 [0.533, 0.681] |
| `inpatient_365d` | 0.675 [0.651, 0.694] | 0.727 [0.703, 0.745] |
| `readmission_30d` | 0.646 [0.593, 0.695] | 0.695 [0.656, 0.736] |
| `new_dx_365d/diabetes` | 0.721 [0.699, 0.741] | 0.759 [0.740, 0.781] |
| `new_dx_365d/heart_failure` | 0.690 [0.667, 0.720] | 0.744 [0.720, 0.769] |
| `new_dx_365d/ckd` | 0.694 [0.665, 0.721] | 0.759 [0.735, 0.784] |
| `new_dx_365d/copd` | 0.689 [0.665, 0.713] | 0.755 [0.731, 0.778] |

## AUPRC

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.032 [0.023, 0.047] | 0.032 [0.023, 0.048] |
| `inpatient_365d` | 0.335 [0.301, 0.368] | 0.386 [0.347, 0.429] |
| `readmission_30d` | 0.093 [0.067, 0.123] | 0.116 [0.088, 0.165] |
| `new_dx_365d/diabetes` | 0.402 [0.370, 0.439] | 0.456 [0.418, 0.498] |
| `new_dx_365d/heart_failure` | 0.243 [0.215, 0.281] | 0.282 [0.244, 0.325] |
| `new_dx_365d/ckd` | 0.175 [0.151, 0.211] | 0.245 [0.207, 0.295] |
| `new_dx_365d/copd` | 0.221 [0.196, 0.256] | 0.308 [0.262, 0.354] |

## BRIER

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1507 [0.1436, 0.1590] | 0.1438 [0.1367, 0.1502] |
| `readmission_30d` | 0.0463 [0.0402, 0.0521] | 0.0457 [0.0399, 0.0515] |
| `new_dx_365d/diabetes` | 0.1558 [0.1486, 0.1630] | 0.1484 [0.1416, 0.1552] |
| `new_dx_365d/heart_failure` | 0.1121 [0.1032, 0.1189] | 0.1084 [0.1001, 0.1145] |
| `new_dx_365d/ckd` | 0.0843 [0.0759, 0.0917] | 0.0806 [0.0732, 0.0868] |
| `new_dx_365d/copd` | 0.1081 [0.0999, 0.1168] | 0.1019 [0.0949, 0.1096] |

## CALIBRATION SLOPE

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.750 [0.309, 1.159] | 0.656 [0.242, 1.122] |
| `inpatient_365d` | 1.067 [0.924, 1.208] | 1.031 [0.894, 1.169] |
| `readmission_30d` | 1.158 [0.761, 1.558] | 1.017 [0.809, 1.274] |
| `new_dx_365d/diabetes` | 0.982 [0.864, 1.094] | 0.997 [0.886, 1.126] |
| `new_dx_365d/heart_failure` | 0.915 [0.795, 1.094] | 0.932 [0.827, 1.079] |
| `new_dx_365d/ckd` | 0.957 [0.806, 1.129] | 0.996 [0.873, 1.152] |
| `new_dx_365d/copd` | 0.951 [0.798, 1.116] | 1.004 [0.874, 1.146] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:ar` | 0.010 | [-0.052, 0.069] | 0.890 |
| `inpatient_365d` | `random_init` - `ckpt:ar` | -0.051 | [-0.070, -0.032] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar` | -0.048 | [-0.101, -0.009] | 0.020 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar` | -0.038 | [-0.051, -0.021] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar` | -0.053 | [-0.074, -0.034] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar` | -0.065 | [-0.086, -0.047] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar` | -0.066 | [-0.085, -0.045] | 0.000 |
