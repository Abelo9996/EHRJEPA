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
| created | 2026-09-04T06:15:13+00:00 |
| runtime (s) | 157.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:recon_only` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot3-desynpuf/recon_only/final.pt` | mean@final |

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

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.607 [0.526, 0.674] |
| `inpatient_365d` | 0.708 [0.687, 0.728] |
| `readmission_30d` | 0.685 [0.644, 0.728] |
| `new_dx_365d/diabetes` | 0.744 [0.724, 0.764] |
| `new_dx_365d/heart_failure` | 0.714 [0.691, 0.738] |
| `new_dx_365d/ckd` | 0.719 [0.691, 0.745] |
| `new_dx_365d/copd` | 0.715 [0.690, 0.740] |

## AUPRC

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.032 [0.024, 0.048] |
| `inpatient_365d` | 0.377 [0.342, 0.412] |
| `readmission_30d` | 0.104 [0.083, 0.140] |
| `new_dx_365d/diabetes` | 0.432 [0.392, 0.475] |
| `new_dx_365d/heart_failure` | 0.260 [0.224, 0.299] |
| `new_dx_365d/ckd` | 0.202 [0.170, 0.241] |
| `new_dx_365d/copd` | 0.248 [0.220, 0.285] |

## BRIER

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1458 [0.1387, 0.1531] |
| `readmission_30d` | 0.0460 [0.0401, 0.0515] |
| `new_dx_365d/diabetes` | 0.1514 [0.1447, 0.1586] |
| `new_dx_365d/heart_failure` | 0.1105 [0.1014, 0.1175] |
| `new_dx_365d/ckd` | 0.0831 [0.0755, 0.0898] |
| `new_dx_365d/copd` | 0.1061 [0.0979, 0.1142] |

## CALIBRATION SLOPE

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.781 [0.231, 1.273] |
| `inpatient_365d` | 1.061 [0.939, 1.194] |
| `readmission_30d` | 1.030 [0.791, 1.255] |
| `new_dx_365d/diabetes` | 0.952 [0.847, 1.074] |
| `new_dx_365d/heart_failure` | 0.953 [0.860, 1.090] |
| `new_dx_365d/ckd` | 0.901 [0.765, 1.067] |
| `new_dx_365d/copd` | 0.964 [0.804, 1.117] |
