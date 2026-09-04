# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `844e1c1` |
| created | 2026-09-04T06:51:14+00:00 |
| runtime (s) | 579.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-microgrid4/micro4_nextlatent/final.pt` | last@final |
| `ckpt:micro4_nextlatent` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-microgrid4/micro4_nextlatent/final.pt` | last@final |

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

| task | random_init | ckpt:micro4_nextlatent |
|---|---|---|
| `mortality_365d` | 0.534 [0.474, 0.603] | 0.545 [0.480, 0.617] |
| `inpatient_365d` | 0.647 [0.625, 0.669] | 0.651 [0.628, 0.673] |
| `readmission_30d` | 0.578 [0.531, 0.628] | 0.565 [0.520, 0.611] |
| `new_dx_365d/diabetes` | 0.710 [0.690, 0.732] | 0.714 [0.695, 0.734] |
| `new_dx_365d/heart_failure` | 0.666 [0.641, 0.692] | 0.670 [0.645, 0.699] |
| `new_dx_365d/ckd` | 0.658 [0.627, 0.685] | 0.659 [0.627, 0.687] |
| `new_dx_365d/copd` | 0.662 [0.634, 0.686] | 0.663 [0.633, 0.686] |

## AUPRC

| task | random_init | ckpt:micro4_nextlatent |
|---|---|---|
| `mortality_365d` | 0.027 [0.020, 0.043] | 0.027 [0.020, 0.044] |
| `inpatient_365d` | 0.292 [0.264, 0.326] | 0.290 [0.264, 0.318] |
| `readmission_30d` | 0.074 [0.054, 0.109] | 0.070 [0.053, 0.098] |
| `new_dx_365d/diabetes` | 0.384 [0.356, 0.420] | 0.383 [0.353, 0.417] |
| `new_dx_365d/heart_failure` | 0.224 [0.195, 0.254] | 0.224 [0.197, 0.256] |
| `new_dx_365d/ckd` | 0.154 [0.129, 0.184] | 0.151 [0.128, 0.178] |
| `new_dx_365d/copd` | 0.204 [0.179, 0.232] | 0.202 [0.179, 0.232] |

## BRIER

| task | random_init | ckpt:micro4_nextlatent |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1541 [0.1462, 0.1625] | 0.1536 [0.1459, 0.1618] |
| `readmission_30d` | 0.0468 [0.0408, 0.0532] | 0.0469 [0.0409, 0.0533] |
| `new_dx_365d/diabetes` | 0.1581 [0.1506, 0.1661] | 0.1572 [0.1503, 0.1647] |
| `new_dx_365d/heart_failure` | 0.1137 [0.1046, 0.1214] | 0.1134 [0.1043, 0.1211] |
| `new_dx_365d/ckd` | 0.0856 [0.0773, 0.0931] | 0.0856 [0.0775, 0.0931] |
| `new_dx_365d/copd` | 0.1096 [0.1007, 0.1185] | 0.1094 [0.1004, 0.1184] |

## CALIBRATION SLOPE

| task | random_init | ckpt:micro4_nextlatent |
|---|---|---|
| `mortality_365d` | 0.365 [-0.396, 1.258] | 0.386 [-0.426, 1.492] |
| `inpatient_365d` | 0.929 [0.777, 1.088] | 0.960 [0.807, 1.142] |
| `readmission_30d` | 0.668 [0.306, 1.101] | 0.573 [0.166, 0.989] |
| `new_dx_365d/diabetes` | 1.116 [0.989, 1.241] | 1.048 [0.931, 1.177] |
| `new_dx_365d/heart_failure` | 0.928 [0.773, 1.121] | 0.939 [0.788, 1.138] |
| `new_dx_365d/ckd` | 0.857 [0.679, 1.024] | 0.832 [0.649, 0.998] |
| `new_dx_365d/copd` | 0.914 [0.743, 1.085] | 0.922 [0.747, 1.081] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:micro4_nextlatent` | -0.011 | [-0.025, 0.005] | 0.180 |
| `inpatient_365d` | `random_init` - `ckpt:micro4_nextlatent` | -0.004 | [-0.012, 0.005] | 0.350 |
| `readmission_30d` | `random_init` - `ckpt:micro4_nextlatent` | 0.013 | [-0.001, 0.032] | 0.060 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:micro4_nextlatent` | -0.004 | [-0.011, 0.002] | 0.220 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:micro4_nextlatent` | -0.004 | [-0.014, 0.007] | 0.380 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:micro4_nextlatent` | -0.001 | [-0.010, 0.008] | 0.870 |
| `new_dx_365d/copd` | `random_init` - `ckpt:micro4_nextlatent` | -0.001 | [-0.010, 0.007] | 0.740 |
