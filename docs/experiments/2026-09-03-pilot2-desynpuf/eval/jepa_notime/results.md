# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `1bf4bc4` |
| created | 2026-09-04T02:31:26+00:00 |
| runtime (s) | 365.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot2-desynpuf/jepa_notime/final.pt` | mean@final |
| `ckpt:jepa_notime` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot2-desynpuf/jepa_notime/final.pt` | mean@final |

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

| task | random_init | ckpt:jepa_notime |
|---|---|---|
| `mortality_365d` | 0.570 [0.493, 0.635] | 0.633 [0.560, 0.697] |
| `inpatient_365d` | 0.677 [0.652, 0.698] | 0.692 [0.668, 0.711] |
| `readmission_30d` | 0.662 [0.608, 0.706] | 0.659 [0.608, 0.702] |
| `new_dx_365d/diabetes` | 0.725 [0.705, 0.746] | 0.748 [0.729, 0.772] |
| `new_dx_365d/heart_failure` | 0.688 [0.664, 0.718] | 0.709 [0.686, 0.732] |
| `new_dx_365d/ckd` | 0.696 [0.667, 0.724] | 0.712 [0.686, 0.738] |
| `new_dx_365d/copd` | 0.696 [0.672, 0.719] | 0.717 [0.693, 0.738] |

## AUPRC

| task | random_init | ckpt:jepa_notime |
|---|---|---|
| `mortality_365d` | 0.027 [0.019, 0.039] | 0.035 [0.025, 0.058] |
| `inpatient_365d` | 0.330 [0.297, 0.367] | 0.345 [0.313, 0.377] |
| `readmission_30d` | 0.085 [0.066, 0.110] | 0.098 [0.073, 0.137] |
| `new_dx_365d/diabetes` | 0.392 [0.363, 0.430] | 0.447 [0.409, 0.491] |
| `new_dx_365d/heart_failure` | 0.241 [0.213, 0.280] | 0.258 [0.224, 0.290] |
| `new_dx_365d/ckd` | 0.180 [0.153, 0.221] | 0.183 [0.159, 0.215] |
| `new_dx_365d/copd` | 0.229 [0.204, 0.265] | 0.246 [0.217, 0.291] |

## BRIER

| task | random_init | ckpt:jepa_notime |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1505 [0.1438, 0.1588] | 0.1487 [0.1420, 0.1563] |
| `readmission_30d` | 0.0468 [0.0409, 0.0524] | 0.0461 [0.0405, 0.0519] |
| `new_dx_365d/diabetes` | 0.1556 [0.1490, 0.1628] | 0.1506 [0.1439, 0.1583] |
| `new_dx_365d/heart_failure` | 0.1123 [0.1029, 0.1194] | 0.1107 [0.1016, 0.1178] |
| `new_dx_365d/ckd` | 0.0840 [0.0757, 0.0912] | 0.0836 [0.0757, 0.0905] |
| `new_dx_365d/copd` | 0.1077 [0.0993, 0.1164] | 0.1061 [0.0979, 0.1147] |

## CALIBRATION SLOPE

| task | random_init | ckpt:jepa_notime |
|---|---|---|
| `mortality_365d` | 0.537 [0.025, 1.036] | 0.940 [0.436, 1.459] |
| `inpatient_365d` | 1.045 [0.883, 1.213] | 1.045 [0.893, 1.189] |
| `readmission_30d` | 0.773 [0.508, 1.016] | 1.073 [0.717, 1.353] |
| `new_dx_365d/diabetes` | 0.932 [0.816, 1.063] | 0.940 [0.845, 1.071] |
| `new_dx_365d/heart_failure` | 0.869 [0.756, 1.032] | 0.941 [0.829, 1.088] |
| `new_dx_365d/ckd` | 0.988 [0.807, 1.168] | 0.966 [0.807, 1.142] |
| `new_dx_365d/copd` | 0.924 [0.777, 1.075] | 1.032 [0.873, 1.196] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:jepa_notime` | -0.063 | [-0.105, -0.017] | 0.030 |
| `inpatient_365d` | `random_init` - `ckpt:jepa_notime` | -0.015 | [-0.028, -0.002] | 0.030 |
| `readmission_30d` | `random_init` - `ckpt:jepa_notime` | 0.003 | [-0.034, 0.034] | 0.850 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:jepa_notime` | -0.022 | [-0.034, -0.013] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:jepa_notime` | -0.021 | [-0.035, -0.003] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:jepa_notime` | -0.016 | [-0.029, -0.002] | 0.030 |
| `new_dx_365d/copd` | `random_init` - `ckpt:jepa_notime` | -0.021 | [-0.035, -0.006] | 0.000 |
