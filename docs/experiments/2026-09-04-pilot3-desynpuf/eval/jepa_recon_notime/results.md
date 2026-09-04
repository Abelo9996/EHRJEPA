# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `b7aca44` |
| created | 2026-09-04T05:39:37+00:00 |
| runtime (s) | 371.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot3-desynpuf/jepa_recon_notime/final.pt` | mean@final |
| `ckpt:jepa_recon_notime` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot3-desynpuf/jepa_recon_notime/final.pt` | mean@final |

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

| task | random_init | ckpt:jepa_recon_notime |
|---|---|---|
| `mortality_365d` | 0.570 [0.493, 0.635] | 0.649 [0.572, 0.718] |
| `inpatient_365d` | 0.677 [0.652, 0.698] | 0.704 [0.682, 0.724] |
| `readmission_30d` | 0.662 [0.608, 0.706] | 0.671 [0.620, 0.717] |
| `new_dx_365d/diabetes` | 0.725 [0.705, 0.746] | 0.754 [0.735, 0.775] |
| `new_dx_365d/heart_failure` | 0.688 [0.664, 0.718] | 0.714 [0.690, 0.738] |
| `new_dx_365d/ckd` | 0.696 [0.667, 0.724] | 0.712 [0.682, 0.736] |
| `new_dx_365d/copd` | 0.696 [0.672, 0.719] | 0.717 [0.691, 0.739] |

## AUPRC

| task | random_init | ckpt:jepa_recon_notime |
|---|---|---|
| `mortality_365d` | 0.027 [0.019, 0.039] | 0.035 [0.026, 0.052] |
| `inpatient_365d` | 0.330 [0.297, 0.367] | 0.357 [0.325, 0.395] |
| `readmission_30d` | 0.085 [0.066, 0.110] | 0.104 [0.077, 0.139] |
| `new_dx_365d/diabetes` | 0.392 [0.363, 0.430] | 0.458 [0.424, 0.501] |
| `new_dx_365d/heart_failure` | 0.241 [0.213, 0.280] | 0.272 [0.241, 0.313] |
| `new_dx_365d/ckd` | 0.180 [0.153, 0.221] | 0.194 [0.166, 0.231] |
| `new_dx_365d/copd` | 0.229 [0.204, 0.265] | 0.248 [0.217, 0.290] |

## BRIER

| task | random_init | ckpt:jepa_recon_notime |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] | 0.0214 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1505 [0.1438, 0.1588] | 0.1472 [0.1402, 0.1548] |
| `readmission_30d` | 0.0468 [0.0409, 0.0524] | 0.0460 [0.0405, 0.0517] |
| `new_dx_365d/diabetes` | 0.1556 [0.1490, 0.1628] | 0.1490 [0.1421, 0.1562] |
| `new_dx_365d/heart_failure` | 0.1123 [0.1029, 0.1194] | 0.1100 [0.1010, 0.1170] |
| `new_dx_365d/ckd` | 0.0840 [0.0757, 0.0912] | 0.0834 [0.0752, 0.0907] |
| `new_dx_365d/copd` | 0.1077 [0.0993, 0.1164] | 0.1060 [0.0976, 0.1143] |

## CALIBRATION SLOPE

| task | random_init | ckpt:jepa_recon_notime |
|---|---|---|
| `mortality_365d` | 0.537 [0.025, 1.036] | 0.970 [0.489, 1.465] |
| `inpatient_365d` | 1.045 [0.883, 1.213] | 1.087 [0.949, 1.241] |
| `readmission_30d` | 0.773 [0.508, 1.016] | 1.033 [0.713, 1.321] |
| `new_dx_365d/diabetes` | 0.932 [0.816, 1.063] | 1.045 [0.930, 1.198] |
| `new_dx_365d/heart_failure` | 0.869 [0.756, 1.032] | 0.997 [0.883, 1.146] |
| `new_dx_365d/ckd` | 0.988 [0.807, 1.168] | 0.938 [0.792, 1.112] |
| `new_dx_365d/copd` | 0.924 [0.777, 1.075] | 1.030 [0.857, 1.193] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:jepa_recon_notime` | -0.079 | [-0.122, -0.029] | 0.000 |
| `inpatient_365d` | `random_init` - `ckpt:jepa_recon_notime` | -0.027 | [-0.044, -0.013] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:jepa_recon_notime` | -0.010 | [-0.048, 0.026] | 0.560 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:jepa_recon_notime` | -0.029 | [-0.040, -0.017] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:jepa_recon_notime` | -0.026 | [-0.038, -0.010] | 0.010 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:jepa_recon_notime` | -0.015 | [-0.030, -0.000] | 0.050 |
| `new_dx_365d/copd` | `random_init` - `ckpt:jepa_recon_notime` | -0.021 | [-0.036, -0.007] | 0.010 |
