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
| created | 2026-09-07T13:26:36+00:00 |
| runtime (s) | 278.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_small_s1/final.pt` | last@final |
| `ckpt:hybrid_small_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_small_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 7358 (0.0190) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 7358 (0.2098) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3116 (0.0501) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 4939 (0.2264) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 6164 (0.1308) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 6487 (0.0999) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 6009 (0.1340) |  |

## AUROC

| task | random_init | ckpt:hybrid_small_s1 |
|---|---|---|
| `mortality_365d` | 0.534 [0.487, 0.570] | 0.639 [0.596, 0.677] |
| `inpatient_365d` | 0.657 [0.644, 0.669] | 0.754 [0.740, 0.767] |
| `readmission_30d` | 0.584 [0.536, 0.624] | 0.696 [0.663, 0.739] |
| `new_dx_365d/diabetes` | 0.706 [0.688, 0.721] | 0.770 [0.756, 0.785] |
| `new_dx_365d/heart_failure` | 0.673 [0.656, 0.691] | 0.773 [0.757, 0.788] |
| `new_dx_365d/ckd` | 0.673 [0.655, 0.693] | 0.772 [0.752, 0.791] |
| `new_dx_365d/copd` | 0.677 [0.660, 0.695] | 0.761 [0.743, 0.778] |

## AUPRC

| task | random_init | ckpt:hybrid_small_s1 |
|---|---|---|
| `mortality_365d` | 0.021 [0.017, 0.026] | 0.030 [0.024, 0.039] |
| `inpatient_365d` | 0.308 [0.287, 0.329] | 0.445 [0.416, 0.468] |
| `readmission_30d` | 0.077 [0.057, 0.104] | 0.106 [0.079, 0.140] |
| `new_dx_365d/diabetes` | 0.371 [0.344, 0.398] | 0.466 [0.437, 0.497] |
| `new_dx_365d/heart_failure` | 0.208 [0.190, 0.233] | 0.313 [0.283, 0.342] |
| `new_dx_365d/ckd` | 0.165 [0.150, 0.188] | 0.263 [0.236, 0.296] |
| `new_dx_365d/copd` | 0.220 [0.202, 0.244] | 0.337 [0.303, 0.372] |

## BRIER

| task | random_init | ckpt:hybrid_small_s1 |
|---|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0220] | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1578 [0.1532, 0.1629] | 0.1424 [0.1379, 0.1461] |
| `readmission_30d` | 0.0474 [0.0409, 0.0534] | 0.0469 [0.0416, 0.0526] |
| `new_dx_365d/diabetes` | 0.1602 [0.1540, 0.1661] | 0.1473 [0.1418, 0.1531] |
| `new_dx_365d/heart_failure` | 0.1094 [0.1032, 0.1147] | 0.1010 [0.0953, 0.1055] |
| `new_dx_365d/ckd` | 0.0873 [0.0815, 0.0922] | 0.0815 [0.0771, 0.0862] |
| `new_dx_365d/copd` | 0.1111 [0.1038, 0.1171] | 0.1027 [0.0968, 0.1080] |

## CALIBRATION SLOPE

| task | random_init | ckpt:hybrid_small_s1 |
|---|---|---|
| `mortality_365d` | 0.256 [-0.030, 0.505] | 0.862 [0.564, 1.098] |
| `inpatient_365d` | 0.983 [0.892, 1.069] | 1.049 [0.963, 1.128] |
| `readmission_30d` | 0.721 [0.291, 1.037] | 0.772 [0.623, 0.954] |
| `new_dx_365d/diabetes` | 1.070 [0.963, 1.172] | 0.946 [0.875, 1.024] |
| `new_dx_365d/heart_failure` | 0.940 [0.831, 1.045] | 0.980 [0.892, 1.076] |
| `new_dx_365d/ckd` | 0.927 [0.825, 1.050] | 0.958 [0.856, 1.059] |
| `new_dx_365d/copd` | 0.989 [0.878, 1.099] | 1.015 [0.923, 1.115] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:hybrid_small_s1` | -0.105 | [-0.156, -0.050] | 0.000 |
| `inpatient_365d` | `random_init` - `ckpt:hybrid_small_s1` | -0.097 | [-0.111, -0.081] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:hybrid_small_s1` | -0.112 | [-0.164, -0.065] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:hybrid_small_s1` | -0.063 | [-0.079, -0.051] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:hybrid_small_s1` | -0.100 | [-0.116, -0.084] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:hybrid_small_s1` | -0.099 | [-0.119, -0.079] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:hybrid_small_s1` | -0.084 | [-0.102, -0.064] | 0.000 |
