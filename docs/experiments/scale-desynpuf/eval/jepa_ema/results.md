# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `12478aa` |
| created | 2026-09-06T05:13:20+00:00 |
| runtime (s) | 461.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/jepa_ema/final.pt` | mean@final |
| `ckpt:jepa_ema` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/jepa_ema/final.pt` | mean@final |

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

| task | random_init | ckpt:jepa_ema |
|---|---|---|
| `mortality_365d` | 0.597 [0.526, 0.683] | 0.641 [0.562, 0.710] |
| `inpatient_365d` | 0.667 [0.646, 0.688] | 0.695 [0.673, 0.715] |
| `readmission_30d` | 0.649 [0.605, 0.688] | 0.658 [0.609, 0.697] |
| `new_dx_365d/diabetes` | 0.728 [0.709, 0.750] | 0.750 [0.732, 0.772] |
| `new_dx_365d/heart_failure` | 0.694 [0.670, 0.721] | 0.716 [0.690, 0.743] |
| `new_dx_365d/ckd` | 0.685 [0.656, 0.715] | 0.711 [0.684, 0.739] |
| `new_dx_365d/copd` | 0.699 [0.672, 0.725] | 0.719 [0.698, 0.743] |

## AUPRC

| task | random_init | ckpt:jepa_ema |
|---|---|---|
| `mortality_365d` | 0.038 [0.027, 0.069] | 0.036 [0.025, 0.069] |
| `inpatient_365d` | 0.311 [0.282, 0.340] | 0.337 [0.304, 0.366] |
| `readmission_30d` | 0.083 [0.066, 0.109] | 0.093 [0.073, 0.128] |
| `new_dx_365d/diabetes` | 0.400 [0.370, 0.440] | 0.452 [0.419, 0.492] |
| `new_dx_365d/heart_failure` | 0.258 [0.224, 0.298] | 0.273 [0.238, 0.313] |
| `new_dx_365d/ckd` | 0.161 [0.136, 0.188] | 0.192 [0.165, 0.228] |
| `new_dx_365d/copd` | 0.239 [0.210, 0.271] | 0.246 [0.219, 0.283] |

## BRIER

| task | random_init | ckpt:jepa_ema |
|---|---|---|
| `mortality_365d` | 0.0214 [0.0173, 0.0265] | 0.0215 [0.0172, 0.0265] |
| `inpatient_365d` | 0.1519 [0.1451, 0.1597] | 0.1489 [0.1422, 0.1567] |
| `readmission_30d` | 0.0467 [0.0406, 0.0526] | 0.0465 [0.0410, 0.0521] |
| `new_dx_365d/diabetes` | 0.1548 [0.1481, 0.1618] | 0.1497 [0.1424, 0.1569] |
| `new_dx_365d/heart_failure` | 0.1115 [0.1026, 0.1187] | 0.1098 [0.1013, 0.1167] |
| `new_dx_365d/ckd` | 0.0847 [0.0766, 0.0916] | 0.0835 [0.0755, 0.0907] |
| `new_dx_365d/copd` | 0.1071 [0.0986, 0.1154] | 0.1060 [0.0973, 0.1148] |

## CALIBRATION SLOPE

| task | random_init | ckpt:jepa_ema |
|---|---|---|
| `mortality_365d` | 0.761 [0.313, 1.252] | 0.860 [0.385, 1.317] |
| `inpatient_365d` | 0.928 [0.796, 1.070] | 1.018 [0.865, 1.169] |
| `readmission_30d` | 0.752 [0.528, 0.992] | 0.785 [0.539, 0.974] |
| `new_dx_365d/diabetes` | 0.927 [0.823, 1.048] | 0.994 [0.897, 1.129] |
| `new_dx_365d/heart_failure` | 0.924 [0.801, 1.102] | 1.015 [0.889, 1.183] |
| `new_dx_365d/ckd` | 0.912 [0.748, 1.094] | 0.944 [0.805, 1.114] |
| `new_dx_365d/copd` | 0.985 [0.836, 1.139] | 0.998 [0.867, 1.143] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:jepa_ema` | -0.044 | [-0.094, 0.010] | 0.130 |
| `inpatient_365d` | `random_init` - `ckpt:jepa_ema` | -0.028 | [-0.041, -0.013] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:jepa_ema` | -0.009 | [-0.046, 0.032] | 0.660 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:jepa_ema` | -0.022 | [-0.032, -0.011] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:jepa_ema` | -0.022 | [-0.036, -0.009] | 0.010 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:jepa_ema` | -0.026 | [-0.041, -0.010] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:jepa_ema` | -0.020 | [-0.032, -0.005] | 0.000 |
