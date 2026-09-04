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
| created | 2026-09-04T04:18:30+00:00 |
| runtime (s) | 178.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_recon` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot2-desynpuf/jepa_recon/final.pt` | mean@final |

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

| task | ckpt:jepa_recon |
|---|---|
| `mortality_365d` | 0.595 [0.531, 0.663] |
| `inpatient_365d` | 0.708 [0.686, 0.726] |
| `readmission_30d` | 0.661 [0.616, 0.708] |
| `new_dx_365d/diabetes` | 0.741 [0.722, 0.763] |
| `new_dx_365d/heart_failure` | 0.712 [0.689, 0.735] |
| `new_dx_365d/ckd` | 0.717 [0.694, 0.744] |
| `new_dx_365d/copd` | 0.711 [0.689, 0.736] |

## AUPRC

| task | ckpt:jepa_recon |
|---|---|
| `mortality_365d` | 0.027 [0.021, 0.040] |
| `inpatient_365d` | 0.353 [0.320, 0.386] |
| `readmission_30d` | 0.095 [0.075, 0.124] |
| `new_dx_365d/diabetes` | 0.435 [0.401, 0.477] |
| `new_dx_365d/heart_failure` | 0.255 [0.229, 0.297] |
| `new_dx_365d/ckd` | 0.198 [0.170, 0.240] |
| `new_dx_365d/copd` | 0.249 [0.217, 0.288] |

## BRIER

| task | ckpt:jepa_recon |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1470 [0.1401, 0.1539] |
| `readmission_30d` | 0.0462 [0.0405, 0.0518] |
| `new_dx_365d/diabetes` | 0.1518 [0.1442, 0.1588] |
| `new_dx_365d/heart_failure` | 0.1105 [0.1012, 0.1176] |
| `new_dx_365d/ckd` | 0.0831 [0.0752, 0.0903] |
| `new_dx_365d/copd` | 0.1063 [0.0980, 0.1146] |

## CALIBRATION SLOPE

| task | ckpt:jepa_recon |
|---|---|
| `mortality_365d` | 0.732 [0.293, 1.194] |
| `inpatient_365d` | 1.094 [0.936, 1.273] |
| `readmission_30d` | 0.916 [0.644, 1.188] |
| `new_dx_365d/diabetes` | 0.990 [0.873, 1.141] |
| `new_dx_365d/heart_failure` | 1.032 [0.917, 1.185] |
| `new_dx_365d/ckd` | 0.952 [0.812, 1.147] |
| `new_dx_365d/copd` | 0.949 [0.807, 1.117] |
