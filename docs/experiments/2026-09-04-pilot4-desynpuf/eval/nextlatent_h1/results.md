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
| created | 2026-09-04T07:59:48+00:00 |
| runtime (s) | 340.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot4-desynpuf/nextlatent_h1/final.pt` | last@final |
| `ckpt:nextlatent_h1` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot4-desynpuf/nextlatent_h1/final.pt` | last@final |

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

| task | random_init | ckpt:nextlatent_h1 |
|---|---|---|
| `mortality_365d` | 0.534 [0.474, 0.603] | 0.611 [0.559, 0.668] |
| `inpatient_365d` | 0.647 [0.625, 0.669] | 0.660 [0.636, 0.683] |
| `readmission_30d` | 0.578 [0.531, 0.628] | 0.623 [0.581, 0.664] |
| `new_dx_365d/diabetes` | 0.710 [0.690, 0.732] | 0.720 [0.700, 0.742] |
| `new_dx_365d/heart_failure` | 0.666 [0.641, 0.692] | 0.687 [0.663, 0.712] |
| `new_dx_365d/ckd` | 0.658 [0.627, 0.685] | 0.683 [0.659, 0.712] |
| `new_dx_365d/copd` | 0.662 [0.634, 0.686] | 0.685 [0.662, 0.706] |

## AUPRC

| task | random_init | ckpt:nextlatent_h1 |
|---|---|---|
| `mortality_365d` | 0.027 [0.020, 0.043] | 0.029 [0.023, 0.040] |
| `inpatient_365d` | 0.292 [0.264, 0.326] | 0.308 [0.279, 0.342] |
| `readmission_30d` | 0.074 [0.054, 0.109] | 0.080 [0.061, 0.106] |
| `new_dx_365d/diabetes` | 0.384 [0.356, 0.420] | 0.391 [0.357, 0.431] |
| `new_dx_365d/heart_failure` | 0.224 [0.195, 0.254] | 0.225 [0.199, 0.252] |
| `new_dx_365d/ckd` | 0.154 [0.129, 0.184] | 0.164 [0.142, 0.192] |
| `new_dx_365d/copd` | 0.204 [0.179, 0.232] | 0.208 [0.187, 0.238] |

## BRIER

| task | random_init | ckpt:nextlatent_h1 |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1541 [0.1462, 0.1625] | 0.1524 [0.1451, 0.1601] |
| `readmission_30d` | 0.0468 [0.0408, 0.0532] | 0.0466 [0.0408, 0.0525] |
| `new_dx_365d/diabetes` | 0.1581 [0.1506, 0.1661] | 0.1564 [0.1496, 0.1641] |
| `new_dx_365d/heart_failure` | 0.1137 [0.1046, 0.1214] | 0.1129 [0.1032, 0.1202] |
| `new_dx_365d/ckd` | 0.0856 [0.0773, 0.0931] | 0.0848 [0.0764, 0.0921] |
| `new_dx_365d/copd` | 0.1096 [0.1007, 0.1185] | 0.1086 [0.1003, 0.1177] |

## CALIBRATION SLOPE

| task | random_init | ckpt:nextlatent_h1 |
|---|---|---|
| `mortality_365d` | 0.365 [-0.396, 1.258] | 0.738 [0.406, 1.149] |
| `inpatient_365d` | 0.929 [0.777, 1.088] | 0.953 [0.803, 1.117] |
| `readmission_30d` | 0.668 [0.306, 1.101] | 0.848 [0.530, 1.170] |
| `new_dx_365d/diabetes` | 1.116 [0.989, 1.241] | 0.967 [0.859, 1.092] |
| `new_dx_365d/heart_failure` | 0.928 [0.773, 1.121] | 0.908 [0.788, 1.057] |
| `new_dx_365d/ckd` | 0.857 [0.679, 1.024] | 0.888 [0.746, 1.064] |
| `new_dx_365d/copd` | 0.914 [0.743, 1.085] | 0.917 [0.757, 1.043] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:nextlatent_h1` | -0.077 | [-0.156, 0.004] | 0.090 |
| `inpatient_365d` | `random_init` - `ckpt:nextlatent_h1` | -0.013 | [-0.033, 0.003] | 0.140 |
| `readmission_30d` | `random_init` - `ckpt:nextlatent_h1` | -0.044 | [-0.090, 0.006] | 0.100 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:nextlatent_h1` | -0.010 | [-0.025, 0.007] | 0.210 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:nextlatent_h1` | -0.021 | [-0.042, 0.001] | 0.080 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:nextlatent_h1` | -0.025 | [-0.056, -0.002] | 0.030 |
| `new_dx_365d/copd` | `random_init` - `ckpt:nextlatent_h1` | -0.023 | [-0.044, -0.003] | 0.020 |
