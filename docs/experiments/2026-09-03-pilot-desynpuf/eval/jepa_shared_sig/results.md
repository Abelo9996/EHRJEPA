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
| created | 2026-09-03T20:45:31+00:00 |
| runtime (s) | 390.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_shared_sig` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/jepa_shared_sig/final.pt` | cls_mean@final |

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

| task | ckpt:jepa_shared_sig |
|---|---|
| `mortality_365d` | 0.657 [0.599, 0.715] |
| `inpatient_365d` | 0.678 [0.657, 0.699] |
| `readmission_30d` | 0.629 [0.587, 0.679] |
| `new_dx_365d/diabetes` | 0.729 [0.711, 0.751] |
| `new_dx_365d/heart_failure` | 0.696 [0.671, 0.721] |
| `new_dx_365d/ckd` | 0.686 [0.661, 0.715] |
| `new_dx_365d/copd` | 0.695 [0.671, 0.717] |

## AUPRC

| task | ckpt:jepa_shared_sig |
|---|---|
| `mortality_365d` | 0.041 [0.029, 0.075] |
| `inpatient_365d` | 0.323 [0.294, 0.353] |
| `readmission_30d` | 0.087 [0.064, 0.117] |
| `new_dx_365d/diabetes` | 0.392 [0.362, 0.431] |
| `new_dx_365d/heart_failure` | 0.237 [0.210, 0.272] |
| `new_dx_365d/ckd` | 0.174 [0.151, 0.211] |
| `new_dx_365d/copd` | 0.232 [0.206, 0.269] |

## BRIER

| task | ckpt:jepa_shared_sig |
|---|---|
| `mortality_365d` | 0.0214 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1503 [0.1432, 0.1573] |
| `readmission_30d` | 0.0465 [0.0405, 0.0527] |
| `new_dx_365d/diabetes` | 0.1549 [0.1477, 0.1617] |
| `new_dx_365d/heart_failure` | 0.1117 [0.1028, 0.1185] |
| `new_dx_365d/ckd` | 0.0843 [0.0764, 0.0916] |
| `new_dx_365d/copd` | 0.1073 [0.0989, 0.1161] |

## CALIBRATION SLOPE

| task | ckpt:jepa_shared_sig |
|---|---|
| `mortality_365d` | 1.691 [1.029, 2.550] |
| `inpatient_365d` | 0.988 [0.836, 1.135] |
| `readmission_30d` | 0.807 [0.510, 1.174] |
| `new_dx_365d/diabetes` | 0.927 [0.822, 1.070] |
| `new_dx_365d/heart_failure` | 0.921 [0.804, 1.080] |
| `new_dx_365d/ckd` | 0.914 [0.757, 1.117] |
| `new_dx_365d/copd` | 0.985 [0.829, 1.152] |
