# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `8721a62` |
| created | 2026-09-06T10:33:21+00:00 |
| runtime (s) | 372.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-desynpuf/hybrid/final.pt` | last@final |
| `ckpt:hybrid` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-desynpuf/hybrid/final.pt` | last@final |

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

| task | random_init | ckpt:hybrid |
|---|---|---|
| `mortality_365d` | 0.547 [0.490, 0.614] | 0.577 [0.508, 0.647] |
| `inpatient_365d` | 0.669 [0.643, 0.691] | 0.755 [0.733, 0.774] |
| `readmission_30d` | 0.614 [0.568, 0.653] | 0.687 [0.643, 0.729] |
| `new_dx_365d/diabetes` | 0.716 [0.693, 0.737] | 0.778 [0.760, 0.799] |
| `new_dx_365d/heart_failure` | 0.676 [0.650, 0.704] | 0.800 [0.783, 0.818] |
| `new_dx_365d/ckd` | 0.680 [0.654, 0.706] | 0.774 [0.747, 0.800] |
| `new_dx_365d/copd` | 0.681 [0.653, 0.702] | 0.778 [0.753, 0.802] |

## AUPRC

| task | random_init | ckpt:hybrid |
|---|---|---|
| `mortality_365d` | 0.024 [0.019, 0.033] | 0.035 [0.023, 0.073] |
| `inpatient_365d` | 0.315 [0.286, 0.352] | 0.430 [0.384, 0.472] |
| `readmission_30d` | 0.073 [0.056, 0.093] | 0.105 [0.081, 0.141] |
| `new_dx_365d/diabetes` | 0.389 [0.357, 0.430] | 0.489 [0.450, 0.530] |
| `new_dx_365d/heart_failure` | 0.239 [0.209, 0.272] | 0.371 [0.323, 0.423] |
| `new_dx_365d/ckd` | 0.159 [0.138, 0.184] | 0.275 [0.231, 0.323] |
| `new_dx_365d/copd` | 0.214 [0.189, 0.251] | 0.344 [0.297, 0.391] |

## BRIER

| task | random_init | ckpt:hybrid |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0267] | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1516 [0.1440, 0.1592] | 0.1385 [0.1320, 0.1454] |
| `readmission_30d` | 0.0468 [0.0409, 0.0529] | 0.0462 [0.0402, 0.0520] |
| `new_dx_365d/diabetes` | 0.1569 [0.1501, 0.1651] | 0.1443 [0.1376, 0.1512] |
| `new_dx_365d/heart_failure` | 0.1129 [0.1033, 0.1195] | 0.1014 [0.0938, 0.1075] |
| `new_dx_365d/ckd` | 0.0850 [0.0767, 0.0924] | 0.0790 [0.0716, 0.0853] |
| `new_dx_365d/copd` | 0.1085 [0.1001, 0.1175] | 0.0987 [0.0912, 0.1058] |

## CALIBRATION SLOPE

| task | random_init | ckpt:hybrid |
|---|---|---|
| `mortality_365d` | 0.485 [-0.072, 1.187] | 0.471 [0.042, 0.905] |
| `inpatient_365d` | 1.034 [0.864, 1.207] | 1.039 [0.909, 1.162] |
| `readmission_30d` | 0.725 [0.415, 1.002] | 0.791 [0.607, 0.995] |
| `new_dx_365d/diabetes` | 1.046 [0.910, 1.187] | 0.953 [0.863, 1.074] |
| `new_dx_365d/heart_failure` | 0.886 [0.751, 1.060] | 1.053 [0.965, 1.181] |
| `new_dx_365d/ckd` | 0.920 [0.768, 1.086] | 0.902 [0.790, 1.023] |
| `new_dx_365d/copd` | 0.977 [0.807, 1.132] | 0.958 [0.842, 1.064] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:hybrid` | -0.030 | [-0.118, 0.052] | 0.470 |
| `inpatient_365d` | `random_init` - `ckpt:hybrid` | -0.087 | [-0.110, -0.064] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:hybrid` | -0.074 | [-0.125, -0.013] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:hybrid` | -0.062 | [-0.080, -0.045] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:hybrid` | -0.123 | [-0.147, -0.096] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:hybrid` | -0.094 | [-0.127, -0.067] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:hybrid` | -0.097 | [-0.122, -0.072] | 0.000 |
