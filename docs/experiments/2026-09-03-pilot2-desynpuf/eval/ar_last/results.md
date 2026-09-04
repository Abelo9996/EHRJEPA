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
| created | 2026-09-04T04:55:13+00:00 |
| runtime (s) | 197.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/ar/final.pt` | last@final |
| `ckpt:ar` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/ar/final.pt` | last@final |

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
| `mortality_365d` | 0.534 [0.474, 0.603] | 0.595 [0.521, 0.677] |
| `inpatient_365d` | 0.647 [0.625, 0.669] | 0.740 [0.718, 0.760] |
| `readmission_30d` | 0.578 [0.531, 0.628] | 0.674 [0.631, 0.718] |
| `new_dx_365d/diabetes` | 0.710 [0.690, 0.732] | 0.763 [0.745, 0.784] |
| `new_dx_365d/heart_failure` | 0.666 [0.641, 0.692] | 0.771 [0.749, 0.794] |
| `new_dx_365d/ckd` | 0.658 [0.627, 0.685] | 0.751 [0.726, 0.779] |
| `new_dx_365d/copd` | 0.662 [0.634, 0.686] | 0.761 [0.738, 0.785] |

## AUPRC

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.027 [0.020, 0.043] | 0.034 [0.024, 0.051] |
| `inpatient_365d` | 0.292 [0.264, 0.326] | 0.402 [0.361, 0.442] |
| `readmission_30d` | 0.074 [0.054, 0.109] | 0.106 [0.076, 0.150] |
| `new_dx_365d/diabetes` | 0.384 [0.356, 0.420] | 0.469 [0.434, 0.510] |
| `new_dx_365d/heart_failure` | 0.224 [0.195, 0.254] | 0.308 [0.266, 0.358] |
| `new_dx_365d/ckd` | 0.154 [0.129, 0.184] | 0.244 [0.207, 0.286] |
| `new_dx_365d/copd` | 0.204 [0.179, 0.232] | 0.298 [0.254, 0.347] |

## BRIER

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1541 [0.1462, 0.1625] | 0.1419 [0.1350, 0.1484] |
| `readmission_30d` | 0.0468 [0.0408, 0.0532] | 0.0460 [0.0402, 0.0517] |
| `new_dx_365d/diabetes` | 0.1581 [0.1506, 0.1661] | 0.1472 [0.1403, 0.1540] |
| `new_dx_365d/heart_failure` | 0.1137 [0.1046, 0.1214] | 0.1056 [0.0981, 0.1121] |
| `new_dx_365d/ckd` | 0.0856 [0.0773, 0.0931] | 0.0808 [0.0730, 0.0871] |
| `new_dx_365d/copd` | 0.1096 [0.1007, 0.1185] | 0.1020 [0.0948, 0.1100] |

## CALIBRATION SLOPE

| task | random_init | ckpt:ar |
|---|---|---|
| `mortality_365d` | 0.365 [-0.396, 1.258] | 0.789 [0.180, 1.465] |
| `inpatient_365d` | 0.929 [0.777, 1.088] | 1.110 [0.973, 1.253] |
| `readmission_30d` | 0.668 [0.306, 1.101] | 0.935 [0.685, 1.191] |
| `new_dx_365d/diabetes` | 1.116 [0.989, 1.241] | 1.041 [0.937, 1.173] |
| `new_dx_365d/heart_failure` | 0.928 [0.773, 1.121] | 1.062 [0.951, 1.212] |
| `new_dx_365d/ckd` | 0.857 [0.679, 1.024] | 0.948 [0.822, 1.102] |
| `new_dx_365d/copd` | 0.914 [0.743, 1.085] | 1.047 [0.913, 1.203] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:ar` | -0.061 | [-0.153, 0.035] | 0.190 |
| `inpatient_365d` | `random_init` - `ckpt:ar` | -0.094 | [-0.114, -0.067] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar` | -0.096 | [-0.155, -0.040] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar` | -0.053 | [-0.074, -0.035] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar` | -0.105 | [-0.129, -0.080] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar` | -0.093 | [-0.124, -0.068] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar` | -0.099 | [-0.123, -0.077] | 0.000 |
