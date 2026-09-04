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
| created | 2026-09-04T11:28:46+00:00 |
| runtime (s) | 411.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot5-seeds-desynpuf/ar_s1/final.pt` | last@final |
| `ckpt:ar_s1` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot5-seeds-desynpuf/ar_s1/final.pt` | last@final |

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

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.534 [0.474, 0.603] | 0.609 [0.545, 0.679] |
| `inpatient_365d` | 0.647 [0.625, 0.669] | 0.739 [0.717, 0.759] |
| `readmission_30d` | 0.578 [0.531, 0.628] | 0.689 [0.648, 0.735] |
| `new_dx_365d/diabetes` | 0.710 [0.690, 0.732] | 0.757 [0.741, 0.780] |
| `new_dx_365d/heart_failure` | 0.666 [0.641, 0.692] | 0.756 [0.735, 0.779] |
| `new_dx_365d/ckd` | 0.658 [0.627, 0.685] | 0.742 [0.716, 0.771] |
| `new_dx_365d/copd` | 0.662 [0.634, 0.686] | 0.759 [0.730, 0.780] |

## AUPRC

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.027 [0.020, 0.043] | 0.031 [0.023, 0.045] |
| `inpatient_365d` | 0.292 [0.264, 0.326] | 0.388 [0.355, 0.429] |
| `readmission_30d` | 0.074 [0.054, 0.109] | 0.113 [0.086, 0.160] |
| `new_dx_365d/diabetes` | 0.384 [0.356, 0.420] | 0.456 [0.418, 0.499] |
| `new_dx_365d/heart_failure` | 0.224 [0.195, 0.254] | 0.301 [0.261, 0.345] |
| `new_dx_365d/ckd` | 0.154 [0.129, 0.184] | 0.246 [0.207, 0.302] |
| `new_dx_365d/copd` | 0.204 [0.179, 0.232] | 0.306 [0.262, 0.352] |

## BRIER

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1541 [0.1462, 0.1625] | 0.1425 [0.1358, 0.1493] |
| `readmission_30d` | 0.0468 [0.0408, 0.0532] | 0.0458 [0.0400, 0.0514] |
| `new_dx_365d/diabetes` | 0.1581 [0.1506, 0.1661] | 0.1485 [0.1415, 0.1553] |
| `new_dx_365d/heart_failure` | 0.1137 [0.1046, 0.1214] | 0.1068 [0.0988, 0.1132] |
| `new_dx_365d/ckd` | 0.0856 [0.0773, 0.0931] | 0.0810 [0.0728, 0.0878] |
| `new_dx_365d/copd` | 0.1096 [0.1007, 0.1185] | 0.1018 [0.0936, 0.1094] |

## CALIBRATION SLOPE

| task | random_init | ckpt:ar_s1 |
|---|---|---|
| `mortality_365d` | 0.365 [-0.396, 1.258] | 0.679 [0.273, 1.110] |
| `inpatient_365d` | 0.929 [0.777, 1.088] | 1.056 [0.920, 1.194] |
| `readmission_30d` | 0.668 [0.306, 1.101] | 1.036 [0.818, 1.314] |
| `new_dx_365d/diabetes` | 1.116 [0.989, 1.241] | 1.003 [0.898, 1.137] |
| `new_dx_365d/heart_failure` | 0.928 [0.773, 1.121] | 1.031 [0.913, 1.179] |
| `new_dx_365d/ckd` | 0.857 [0.679, 1.024] | 0.942 [0.809, 1.122] |
| `new_dx_365d/copd` | 0.914 [0.743, 1.085] | 0.998 [0.855, 1.115] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:ar_s1` | -0.075 | [-0.146, 0.005] | 0.080 |
| `inpatient_365d` | `random_init` - `ckpt:ar_s1` | -0.092 | [-0.113, -0.069] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar_s1` | -0.111 | [-0.180, -0.049] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar_s1` | -0.047 | [-0.065, -0.031] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar_s1` | -0.090 | [-0.115, -0.065] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar_s1` | -0.084 | [-0.109, -0.062] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar_s1` | -0.097 | [-0.123, -0.070] | 0.000 |
