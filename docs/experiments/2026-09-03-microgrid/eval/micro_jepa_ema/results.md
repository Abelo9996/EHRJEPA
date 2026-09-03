# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `4deb9c4` |
| created | 2026-09-03T16:55:40+00:00 |
| runtime (s) | 374.3 |

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
| `mortality_365d` | 0.602 [0.539, 0.670] | 0.616 [0.555, 0.684] |
| `inpatient_365d` | 0.685 [0.661, 0.707] | 0.670 [0.645, 0.688] |
| `readmission_30d` | 0.667 [0.623, 0.709] | 0.627 [0.581, 0.674] |
| `new_dx_365d/diabetes` | 0.726 [0.707, 0.748] | 0.715 [0.695, 0.736] |
| `new_dx_365d/heart_failure` | 0.700 [0.678, 0.729] | 0.690 [0.664, 0.717] |
| `new_dx_365d/ckd` | 0.701 [0.676, 0.727] | 0.692 [0.664, 0.719] |
| `new_dx_365d/copd` | 0.693 [0.669, 0.714] | 0.691 [0.668, 0.711] |

## AUPRC

| task | random_init | ckpt:micro_jepa_ema |
|---|---|---|
| `mortality_365d` | 0.032 [0.023, 0.049] | 0.038 [0.026, 0.071] |
| `inpatient_365d` | 0.330 [0.294, 0.365] | 0.309 [0.280, 0.339] |
| `readmission_30d` | 0.093 [0.072, 0.121] | 0.080 [0.061, 0.106] |
| `new_dx_365d/diabetes` | 0.393 [0.363, 0.431] | 0.385 [0.354, 0.427] |
| `new_dx_365d/heart_failure` | 0.260 [0.230, 0.306] | 0.242 [0.210, 0.279] |
| `new_dx_365d/ckd` | 0.176 [0.151, 0.212] | 0.172 [0.148, 0.205] |
| `new_dx_365d/copd` | 0.222 [0.192, 0.254] | 0.223 [0.195, 0.253] |

## BRIER

| task | random_init | ckpt:micro_jepa_ema |
|---|---|---|
| `mortality_365d` | 0.0216 [0.0174, 0.0266] | 0.0214 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1498 [0.1425, 0.1574] | 0.1516 [0.1449, 0.1590] |
| `readmission_30d` | 0.0465 [0.0406, 0.0524] | 0.0472 [0.0413, 0.0531] |
| `new_dx_365d/diabetes` | 0.1555 [0.1485, 0.1630] | 0.1577 [0.1512, 0.1648] |
| `new_dx_365d/heart_failure` | 0.1112 [0.1018, 0.1181] | 0.1124 [0.1035, 0.1196] |
| `new_dx_365d/ckd` | 0.0840 [0.0760, 0.0909] | 0.0843 [0.0764, 0.0912] |
| `new_dx_365d/copd` | 0.1083 [0.1000, 0.1168] | 0.1078 [0.0992, 0.1165] |

## CALIBRATION SLOPE

| task | random_init | ckpt:micro_jepa_ema |
|---|---|---|
| `mortality_365d` | 0.509 [0.185, 0.866] | 0.768 [0.347, 1.204] |
| `inpatient_365d` | 0.989 [0.853, 1.127] | 0.999 [0.852, 1.150] |
| `readmission_30d` | 0.776 [0.543, 0.994] | 0.517 [0.292, 0.761] |
| `new_dx_365d/diabetes` | 0.893 [0.784, 1.018] | 0.826 [0.724, 0.927] |
| `new_dx_365d/heart_failure` | 0.840 [0.730, 1.001] | 0.823 [0.724, 0.969] |
| `new_dx_365d/ckd` | 0.932 [0.790, 1.101] | 0.888 [0.750, 1.064] |
| `new_dx_365d/copd` | 0.843 [0.719, 0.975] | 1.026 [0.850, 1.187] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:micro_jepa_ema` | -0.014 | [-0.059, 0.023] | 0.580 |
| `inpatient_365d` | `random_init` - `ckpt:micro_jepa_ema` | 0.015 | [0.004, 0.026] | 0.010 |
| `readmission_30d` | `random_init` - `ckpt:micro_jepa_ema` | 0.041 | [0.012, 0.069] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:micro_jepa_ema` | 0.012 | [0.005, 0.017] | 0.010 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:micro_jepa_ema` | 0.010 | [-0.001, 0.023] | 0.120 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:micro_jepa_ema` | 0.009 | [-0.004, 0.022] | 0.130 |
| `new_dx_365d/copd` | `random_init` - `ckpt:micro_jepa_ema` | 0.003 | [-0.011, 0.015] | 0.750 |
