# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `0019064` |
| created | 2026-09-03T20:52:01+00:00 |
| runtime (s) | 921.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid2/micro2_all_flags/final.pt` | mean@final |
| `ckpt:micro2_all_flags` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid2/micro2_all_flags/final.pt` | mean@final |

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

| task | random_init | ckpt:micro2_all_flags |
|---|---|---|
| `mortality_365d` | 0.570 [0.493, 0.635] | 0.557 [0.493, 0.617] |
| `inpatient_365d` | 0.677 [0.652, 0.698] | 0.676 [0.650, 0.697] |
| `readmission_30d` | 0.662 [0.608, 0.706] | 0.661 [0.614, 0.703] |
| `new_dx_365d/diabetes` | 0.725 [0.705, 0.746] | 0.723 [0.703, 0.743] |
| `new_dx_365d/heart_failure` | 0.688 [0.664, 0.718] | 0.688 [0.665, 0.716] |
| `new_dx_365d/ckd` | 0.696 [0.667, 0.724] | 0.694 [0.667, 0.723] |
| `new_dx_365d/copd` | 0.696 [0.672, 0.719] | 0.695 [0.669, 0.715] |

## AUPRC

| task | random_init | ckpt:micro2_all_flags |
|---|---|---|
| `mortality_365d` | 0.027 [0.019, 0.039] | 0.025 [0.019, 0.036] |
| `inpatient_365d` | 0.330 [0.297, 0.367] | 0.330 [0.296, 0.367] |
| `readmission_30d` | 0.085 [0.066, 0.110] | 0.087 [0.068, 0.116] |
| `new_dx_365d/diabetes` | 0.392 [0.363, 0.430] | 0.394 [0.361, 0.432] |
| `new_dx_365d/heart_failure` | 0.241 [0.213, 0.280] | 0.239 [0.210, 0.275] |
| `new_dx_365d/ckd` | 0.180 [0.153, 0.221] | 0.176 [0.152, 0.210] |
| `new_dx_365d/copd` | 0.229 [0.204, 0.265] | 0.217 [0.192, 0.248] |

## BRIER

| task | random_init | ckpt:micro2_all_flags |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1505 [0.1438, 0.1588] | 0.1507 [0.1436, 0.1589] |
| `readmission_30d` | 0.0468 [0.0409, 0.0524] | 0.0465 [0.0407, 0.0525] |
| `new_dx_365d/diabetes` | 0.1556 [0.1490, 0.1628] | 0.1558 [0.1493, 0.1629] |
| `new_dx_365d/heart_failure` | 0.1123 [0.1029, 0.1194] | 0.1123 [0.1031, 0.1195] |
| `new_dx_365d/ckd` | 0.0840 [0.0757, 0.0912] | 0.0841 [0.0760, 0.0913] |
| `new_dx_365d/copd` | 0.1077 [0.0993, 0.1164] | 0.1080 [0.0999, 0.1166] |

## CALIBRATION SLOPE

| task | random_init | ckpt:micro2_all_flags |
|---|---|---|
| `mortality_365d` | 0.537 [0.025, 1.036] | 0.473 [0.015, 0.957] |
| `inpatient_365d` | 1.045 [0.883, 1.213] | 1.032 [0.861, 1.201] |
| `readmission_30d` | 0.773 [0.508, 1.016] | 0.820 [0.583, 1.080] |
| `new_dx_365d/diabetes` | 0.932 [0.816, 1.063] | 0.920 [0.814, 1.041] |
| `new_dx_365d/heart_failure` | 0.869 [0.756, 1.032] | 0.845 [0.726, 0.997] |
| `new_dx_365d/ckd` | 0.988 [0.807, 1.168] | 1.068 [0.885, 1.280] |
| `new_dx_365d/copd` | 0.924 [0.777, 1.075] | 0.900 [0.765, 1.036] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:micro2_all_flags` | 0.013 | [-0.005, 0.034] | 0.180 |
| `inpatient_365d` | `random_init` - `ckpt:micro2_all_flags` | 0.001 | [-0.006, 0.007] | 0.750 |
| `readmission_30d` | `random_init` - `ckpt:micro2_all_flags` | 0.001 | [-0.015, 0.016] | 0.910 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:micro2_all_flags` | 0.002 | [-0.004, 0.009] | 0.430 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:micro2_all_flags` | 0.000 | [-0.007, 0.007] | 0.930 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:micro2_all_flags` | 0.002 | [-0.010, 0.012] | 0.740 |
| `new_dx_365d/copd` | `random_init` - `ckpt:micro2_all_flags` | 0.001 | [-0.006, 0.008] | 0.720 |
