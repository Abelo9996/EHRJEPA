# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `0f2d3ed` |
| created | 2026-09-03T16:26:46+00:00 |
| runtime (s) | 346.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid/micro_ar/final.pt` | cls_mean@final |
| `ckpt:micro_ar` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid/micro_ar/final.pt` | cls_mean@final |

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

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.591 [0.516, 0.664] | 0.614 [0.545, 0.676] |
| `inpatient_365d` | 0.672 [0.646, 0.693] | 0.694 [0.672, 0.714] |
| `readmission_30d` | 0.652 [0.604, 0.686] | 0.638 [0.588, 0.688] |
| `new_dx_365d/diabetes` | 0.737 [0.716, 0.758] | 0.738 [0.718, 0.759] |
| `new_dx_365d/heart_failure` | 0.699 [0.678, 0.725] | 0.704 [0.683, 0.729] |
| `new_dx_365d/ckd` | 0.700 [0.675, 0.728] | 0.705 [0.679, 0.735] |
| `new_dx_365d/copd` | 0.700 [0.673, 0.726] | 0.699 [0.672, 0.721] |

## AUPRC

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.033 [0.023, 0.057] | 0.030 [0.022, 0.045] |
| `inpatient_365d` | 0.318 [0.284, 0.350] | 0.344 [0.312, 0.383] |
| `readmission_30d` | 0.091 [0.069, 0.121] | 0.099 [0.068, 0.133] |
| `new_dx_365d/diabetes` | 0.417 [0.383, 0.459] | 0.422 [0.383, 0.464] |
| `new_dx_365d/heart_failure` | 0.259 [0.229, 0.302] | 0.260 [0.225, 0.300] |
| `new_dx_365d/ckd` | 0.174 [0.152, 0.210] | 0.179 [0.152, 0.213] |
| `new_dx_365d/copd` | 0.235 [0.210, 0.269] | 0.230 [0.204, 0.266] |

## BRIER

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1512 [0.1441, 0.1588] | 0.1486 [0.1418, 0.1556] |
| `readmission_30d` | 0.0464 [0.0403, 0.0522] | 0.0464 [0.0404, 0.0522] |
| `new_dx_365d/diabetes` | 0.1532 [0.1462, 0.1608] | 0.1530 [0.1459, 0.1600] |
| `new_dx_365d/heart_failure` | 0.1112 [0.1023, 0.1185] | 0.1110 [0.1018, 0.1176] |
| `new_dx_365d/ckd` | 0.0841 [0.0756, 0.0907] | 0.0838 [0.0757, 0.0906] |
| `new_dx_365d/copd` | 0.1071 [0.0986, 0.1154] | 0.1073 [0.0988, 0.1160] |

## CALIBRATION SLOPE

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.654 [0.242, 1.142] | 1.113 [0.419, 1.810] |
| `inpatient_365d` | 0.948 [0.801, 1.091] | 1.032 [0.884, 1.193] |
| `readmission_30d` | 0.856 [0.581, 1.083] | 1.203 [0.741, 1.692] |
| `new_dx_365d/diabetes` | 0.924 [0.825, 1.043] | 0.949 [0.832, 1.086] |
| `new_dx_365d/heart_failure` | 0.958 [0.840, 1.139] | 0.902 [0.799, 1.056] |
| `new_dx_365d/ckd` | 0.882 [0.737, 1.041] | 0.954 [0.806, 1.172] |
| `new_dx_365d/copd` | 0.972 [0.820, 1.106] | 0.922 [0.768, 1.069] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:micro_ar` | -0.022 | [-0.084, 0.045] | 0.550 |
| `inpatient_365d` | `random_init` - `ckpt:micro_ar` | -0.021 | [-0.035, -0.008] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:micro_ar` | 0.014 | [-0.021, 0.043] | 0.520 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:micro_ar` | -0.000 | [-0.012, 0.010] | 0.870 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:micro_ar` | -0.005 | [-0.016, 0.008] | 0.720 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:micro_ar` | -0.005 | [-0.020, 0.012] | 0.540 |
| `new_dx_365d/copd` | `random_init` - `ckpt:micro_ar` | 0.001 | [-0.011, 0.015] | 0.800 |
