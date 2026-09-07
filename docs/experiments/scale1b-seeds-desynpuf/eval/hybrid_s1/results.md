# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `fdc4f65` |
| created | 2026-09-07T07:07:39+00:00 |
| runtime (s) | 386.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/hybrid_s1/final.pt` | last@final |
| `ckpt:hybrid_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/hybrid_s1/final.pt` | last@final |

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

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.547 [0.490, 0.614] | 0.586 [0.511, 0.663] |
| `inpatient_365d` | 0.669 [0.643, 0.691] | 0.749 [0.725, 0.769] |
| `readmission_30d` | 0.614 [0.568, 0.653] | 0.685 [0.642, 0.724] |
| `new_dx_365d/diabetes` | 0.716 [0.693, 0.737] | 0.779 [0.762, 0.799] |
| `new_dx_365d/heart_failure` | 0.676 [0.650, 0.704] | 0.801 [0.785, 0.822] |
| `new_dx_365d/ckd` | 0.680 [0.654, 0.706] | 0.772 [0.746, 0.797] |
| `new_dx_365d/copd` | 0.681 [0.653, 0.702] | 0.780 [0.753, 0.804] |

## AUPRC

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.024 [0.019, 0.033] | 0.030 [0.022, 0.051] |
| `inpatient_365d` | 0.315 [0.286, 0.352] | 0.423 [0.374, 0.468] |
| `readmission_30d` | 0.073 [0.056, 0.093] | 0.100 [0.077, 0.141] |
| `new_dx_365d/diabetes` | 0.389 [0.357, 0.430] | 0.485 [0.448, 0.524] |
| `new_dx_365d/heart_failure` | 0.239 [0.209, 0.272] | 0.361 [0.317, 0.423] |
| `new_dx_365d/ckd` | 0.159 [0.138, 0.184] | 0.270 [0.226, 0.315] |
| `new_dx_365d/copd` | 0.214 [0.189, 0.251] | 0.352 [0.304, 0.405] |

## BRIER

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0267] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1516 [0.1440, 0.1592] | 0.1397 [0.1328, 0.1465] |
| `readmission_30d` | 0.0468 [0.0409, 0.0529] | 0.0463 [0.0407, 0.0524] |
| `new_dx_365d/diabetes` | 0.1569 [0.1501, 0.1651] | 0.1442 [0.1378, 0.1509] |
| `new_dx_365d/heart_failure` | 0.1129 [0.1033, 0.1195] | 0.1014 [0.0942, 0.1075] |
| `new_dx_365d/ckd` | 0.0850 [0.0767, 0.0924] | 0.0793 [0.0721, 0.0857] |
| `new_dx_365d/copd` | 0.1085 [0.1001, 0.1175] | 0.0982 [0.0906, 0.1057] |

## CALIBRATION SLOPE

| task | random_init | ckpt:hybrid_s1 |
|---|---|---|
| `mortality_365d` | 0.485 [-0.072, 1.187] | 0.527 [0.042, 1.004] |
| `inpatient_365d` | 1.034 [0.864, 1.207] | 0.997 [0.868, 1.126] |
| `readmission_30d` | 0.725 [0.415, 1.002] | 0.769 [0.584, 0.976] |
| `new_dx_365d/diabetes` | 1.046 [0.910, 1.187] | 0.961 [0.874, 1.086] |
| `new_dx_365d/heart_failure` | 0.886 [0.751, 1.060] | 1.008 [0.916, 1.126] |
| `new_dx_365d/ckd` | 0.920 [0.768, 1.086] | 0.973 [0.847, 1.084] |
| `new_dx_365d/copd` | 0.977 [0.807, 1.132] | 0.978 [0.859, 1.092] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:hybrid_s1` | -0.039 | [-0.126, 0.057] | 0.330 |
| `inpatient_365d` | `random_init` - `ckpt:hybrid_s1` | -0.080 | [-0.103, -0.058] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:hybrid_s1` | -0.071 | [-0.133, -0.013] | 0.010 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:hybrid_s1` | -0.063 | [-0.081, -0.048] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:hybrid_s1` | -0.125 | [-0.147, -0.099] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:hybrid_s1` | -0.092 | [-0.122, -0.068] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:hybrid_s1` | -0.098 | [-0.124, -0.074] | 0.000 |
