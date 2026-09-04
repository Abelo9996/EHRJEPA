# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `4828dcb` |
| created | 2026-09-03T19:28:38+00:00 |
| runtime (s) | 386.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/jepa_ema/final.pt` | cls_mean@final |
| `ckpt:jepa_ema` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/jepa_ema/final.pt` | cls_mean@final |

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
| `mortality_365d` | 0.602 [0.539, 0.670] | 0.610 [0.552, 0.681] |
| `inpatient_365d` | 0.685 [0.661, 0.707] | 0.680 [0.658, 0.700] |
| `readmission_30d` | 0.667 [0.623, 0.709] | 0.646 [0.601, 0.691] |
| `new_dx_365d/diabetes` | 0.726 [0.707, 0.748] | 0.742 [0.724, 0.764] |
| `new_dx_365d/heart_failure` | 0.700 [0.678, 0.729] | 0.703 [0.680, 0.727] |
| `new_dx_365d/ckd` | 0.701 [0.676, 0.727] | 0.694 [0.667, 0.726] |
| `new_dx_365d/copd` | 0.693 [0.669, 0.714] | 0.706 [0.680, 0.728] |

## AUPRC

| task | random_init | ckpt:jepa_ema |
|---|---|---|
| `mortality_365d` | 0.032 [0.023, 0.049] | 0.030 [0.023, 0.043] |
| `inpatient_365d` | 0.330 [0.294, 0.365] | 0.338 [0.305, 0.378] |
| `readmission_30d` | 0.093 [0.072, 0.121] | 0.087 [0.068, 0.117] |
| `new_dx_365d/diabetes` | 0.393 [0.363, 0.431] | 0.427 [0.395, 0.470] |
| `new_dx_365d/heart_failure` | 0.260 [0.230, 0.306] | 0.250 [0.217, 0.288] |
| `new_dx_365d/ckd` | 0.176 [0.151, 0.212] | 0.196 [0.166, 0.240] |
| `new_dx_365d/copd` | 0.222 [0.192, 0.254] | 0.239 [0.209, 0.273] |

## BRIER

| task | random_init | ckpt:jepa_ema |
|---|---|---|
| `mortality_365d` | 0.0216 [0.0174, 0.0266] | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1498 [0.1425, 0.1574] | 0.1497 [0.1423, 0.1568] |
| `readmission_30d` | 0.0465 [0.0406, 0.0524] | 0.0465 [0.0407, 0.0522] |
| `new_dx_365d/diabetes` | 0.1555 [0.1485, 0.1630] | 0.1520 [0.1448, 0.1585] |
| `new_dx_365d/heart_failure` | 0.1112 [0.1018, 0.1181] | 0.1112 [0.1022, 0.1184] |
| `new_dx_365d/ckd` | 0.0840 [0.0760, 0.0909] | 0.0837 [0.0754, 0.0912] |
| `new_dx_365d/copd` | 0.1083 [0.1000, 0.1168] | 0.1068 [0.0982, 0.1151] |

## CALIBRATION SLOPE

| task | random_init | ckpt:jepa_ema |
|---|---|---|
| `mortality_365d` | 0.509 [0.185, 0.866] | 0.689 [0.321, 1.146] |
| `inpatient_365d` | 0.989 [0.853, 1.127] | 0.994 [0.849, 1.149] |
| `readmission_30d` | 0.776 [0.543, 0.994] | 0.861 [0.612, 1.127] |
| `new_dx_365d/diabetes` | 0.893 [0.784, 1.018] | 0.972 [0.867, 1.103] |
| `new_dx_365d/heart_failure` | 0.840 [0.730, 1.001] | 0.955 [0.847, 1.104] |
| `new_dx_365d/ckd` | 0.932 [0.790, 1.101] | 0.883 [0.743, 1.078] |
| `new_dx_365d/copd` | 0.843 [0.719, 0.975] | 0.972 [0.812, 1.128] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:jepa_ema` | -0.008 | [-0.064, 0.038] | 0.710 |
| `inpatient_365d` | `random_init` - `ckpt:jepa_ema` | 0.005 | [-0.009, 0.021] | 0.520 |
| `readmission_30d` | `random_init` - `ckpt:jepa_ema` | 0.021 | [-0.005, 0.047] | 0.150 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:jepa_ema` | -0.015 | [-0.027, -0.006] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:jepa_ema` | -0.003 | [-0.015, 0.011] | 0.800 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:jepa_ema` | 0.006 | [-0.015, 0.022] | 0.560 |
| `new_dx_365d/copd` | `random_init` - `ckpt:jepa_ema` | -0.013 | [-0.025, -0.000] | 0.050 |
