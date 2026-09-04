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
| created | 2026-09-04T03:04:39+00:00 |
| runtime (s) | 175.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_content` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot2-desynpuf/jepa_content/final.pt` | mean@final |

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

| task | ckpt:jepa_content |
|---|---|
| `mortality_365d` | 0.600 [0.522, 0.673] |
| `inpatient_365d` | 0.691 [0.667, 0.713] |
| `readmission_30d` | 0.649 [0.595, 0.697] |
| `new_dx_365d/diabetes` | 0.731 [0.712, 0.751] |
| `new_dx_365d/heart_failure` | 0.702 [0.677, 0.729] |
| `new_dx_365d/ckd` | 0.698 [0.671, 0.729] |
| `new_dx_365d/copd` | 0.701 [0.676, 0.722] |

## AUPRC

| task | ckpt:jepa_content |
|---|---|
| `mortality_365d` | 0.029 [0.022, 0.039] |
| `inpatient_365d` | 0.359 [0.324, 0.396] |
| `readmission_30d` | 0.094 [0.068, 0.123] |
| `new_dx_365d/diabetes` | 0.421 [0.388, 0.461] |
| `new_dx_365d/heart_failure` | 0.264 [0.232, 0.305] |
| `new_dx_365d/ckd` | 0.198 [0.168, 0.236] |
| `new_dx_365d/copd` | 0.232 [0.203, 0.269] |

## BRIER

| task | ckpt:jepa_content |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1483 [0.1410, 0.1563] |
| `readmission_30d` | 0.0463 [0.0407, 0.0521] |
| `new_dx_365d/diabetes` | 0.1538 [0.1468, 0.1616] |
| `new_dx_365d/heart_failure` | 0.1108 [0.1016, 0.1175] |
| `new_dx_365d/ckd` | 0.0835 [0.0755, 0.0905] |
| `new_dx_365d/copd` | 0.1074 [0.0991, 0.1162] |

## CALIBRATION SLOPE

| task | ckpt:jepa_content |
|---|---|
| `mortality_365d` | 0.686 [0.185, 1.197] |
| `inpatient_365d` | 1.133 [0.979, 1.290] |
| `readmission_30d` | 1.023 [0.659, 1.348] |
| `new_dx_365d/diabetes` | 0.928 [0.839, 1.043] |
| `new_dx_365d/heart_failure` | 0.978 [0.850, 1.146] |
| `new_dx_365d/ckd` | 0.929 [0.779, 1.109] |
| `new_dx_365d/copd` | 0.941 [0.801, 1.080] |
