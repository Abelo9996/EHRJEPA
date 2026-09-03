# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `1f2004a` |
| created | 2026-09-03T16:35:37+00:00 |
| runtime (s) | 363.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid/micro_jepa_ema/final.pt` | cls_mean@final |
| `ckpt:micro_jepa_ema` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid/micro_jepa_ema/final.pt` | cls_mean@final |

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

| task | random_init | ckpt:micro_jepa_ema |
|---|---|---|
| `mortality_365d` | 0.591 [0.516, 0.664] | 0.616 [0.555, 0.684] |
| `inpatient_365d` | 0.672 [0.646, 0.693] | 0.670 [0.645, 0.688] |
| `readmission_30d` | 0.652 [0.604, 0.686] | 0.627 [0.581, 0.674] |
| `new_dx_365d/diabetes` | 0.737 [0.716, 0.758] | 0.715 [0.695, 0.736] |
| `new_dx_365d/heart_failure` | 0.699 [0.678, 0.725] | 0.690 [0.664, 0.717] |
| `new_dx_365d/ckd` | 0.700 [0.675, 0.728] | 0.692 [0.664, 0.719] |
| `new_dx_365d/copd` | 0.700 [0.673, 0.726] | 0.691 [0.668, 0.711] |

## AUPRC

| task | random_init | ckpt:micro_jepa_ema |
|---|---|---|
| `mortality_365d` | 0.033 [0.023, 0.057] | 0.038 [0.026, 0.071] |
| `inpatient_365d` | 0.318 [0.284, 0.350] | 0.309 [0.280, 0.339] |
| `readmission_30d` | 0.091 [0.069, 0.121] | 0.080 [0.061, 0.106] |
| `new_dx_365d/diabetes` | 0.417 [0.383, 0.459] | 0.385 [0.354, 0.427] |
| `new_dx_365d/heart_failure` | 0.259 [0.229, 0.302] | 0.242 [0.210, 0.279] |
| `new_dx_365d/ckd` | 0.174 [0.152, 0.210] | 0.172 [0.148, 0.205] |
| `new_dx_365d/copd` | 0.235 [0.210, 0.269] | 0.223 [0.195, 0.253] |

## BRIER

| task | random_init | ckpt:micro_jepa_ema |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] | 0.0214 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1512 [0.1441, 0.1588] | 0.1516 [0.1449, 0.1590] |
| `readmission_30d` | 0.0464 [0.0403, 0.0522] | 0.0472 [0.0413, 0.0531] |
| `new_dx_365d/diabetes` | 0.1532 [0.1462, 0.1608] | 0.1577 [0.1512, 0.1648] |
| `new_dx_365d/heart_failure` | 0.1112 [0.1023, 0.1185] | 0.1124 [0.1035, 0.1196] |
| `new_dx_365d/ckd` | 0.0841 [0.0756, 0.0907] | 0.0843 [0.0764, 0.0912] |
| `new_dx_365d/copd` | 0.1071 [0.0986, 0.1154] | 0.1078 [0.0992, 0.1165] |

## CALIBRATION SLOPE

| task | random_init | ckpt:micro_jepa_ema |
|---|---|---|
| `mortality_365d` | 0.654 [0.242, 1.142] | 0.768 [0.347, 1.204] |
| `inpatient_365d` | 0.948 [0.801, 1.091] | 0.999 [0.852, 1.150] |
| `readmission_30d` | 0.856 [0.581, 1.083] | 0.517 [0.292, 0.761] |
| `new_dx_365d/diabetes` | 0.924 [0.825, 1.043] | 0.826 [0.724, 0.927] |
| `new_dx_365d/heart_failure` | 0.958 [0.840, 1.139] | 0.823 [0.724, 0.969] |
| `new_dx_365d/ckd` | 0.882 [0.737, 1.041] | 0.888 [0.750, 1.064] |
| `new_dx_365d/copd` | 0.972 [0.820, 1.106] | 1.026 [0.850, 1.187] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:micro_jepa_ema` | -0.025 | [-0.088, 0.037] | 0.390 |
| `inpatient_365d` | `random_init` - `ckpt:micro_jepa_ema` | 0.002 | [-0.009, 0.014] | 0.660 |
| `readmission_30d` | `random_init` - `ckpt:micro_jepa_ema` | 0.026 | [-0.010, 0.054] | 0.130 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:micro_jepa_ema` | 0.022 | [0.012, 0.032] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:micro_jepa_ema` | 0.010 | [-0.002, 0.021] | 0.100 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:micro_jepa_ema` | 0.009 | [-0.007, 0.025] | 0.290 |
| `new_dx_365d/copd` | `random_init` - `ckpt:micro_jepa_ema` | 0.010 | [0.000, 0.020] | 0.040 |
