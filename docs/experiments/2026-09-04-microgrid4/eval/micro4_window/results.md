# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `844e1c1` |
| created | 2026-09-04T07:02:06+00:00 |
| runtime (s) | 309.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:micro4_window` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-microgrid4/micro4_window/final.pt` | last@final |

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

| task | ckpt:micro4_window |
|---|---|
| `mortality_365d` | 0.540 [0.479, 0.612] |
| `inpatient_365d` | 0.642 [0.620, 0.666] |
| `readmission_30d` | 0.574 [0.520, 0.624] |
| `new_dx_365d/diabetes` | 0.704 [0.683, 0.725] |
| `new_dx_365d/heart_failure` | 0.661 [0.634, 0.690] |
| `new_dx_365d/ckd` | 0.649 [0.613, 0.677] |
| `new_dx_365d/copd` | 0.665 [0.637, 0.688] |

## AUPRC

| task | ckpt:micro4_window |
|---|---|
| `mortality_365d` | 0.027 [0.021, 0.042] |
| `inpatient_365d` | 0.287 [0.259, 0.315] |
| `readmission_30d` | 0.068 [0.051, 0.095] |
| `new_dx_365d/diabetes` | 0.370 [0.341, 0.404] |
| `new_dx_365d/heart_failure` | 0.224 [0.196, 0.256] |
| `new_dx_365d/ckd` | 0.153 [0.128, 0.181] |
| `new_dx_365d/copd` | 0.205 [0.180, 0.234] |

## BRIER

| task | ckpt:micro4_window |
|---|---|
| `mortality_365d` | 0.0215 [0.0174, 0.0266] |
| `inpatient_365d` | 0.1547 [0.1469, 0.1633] |
| `readmission_30d` | 0.0469 [0.0409, 0.0533] |
| `new_dx_365d/diabetes` | 0.1595 [0.1522, 0.1673] |
| `new_dx_365d/heart_failure` | 0.1139 [0.1050, 0.1215] |
| `new_dx_365d/ckd` | 0.0860 [0.0776, 0.0938] |
| `new_dx_365d/copd` | 0.1096 [0.1009, 0.1184] |

## CALIBRATION SLOPE

| task | ckpt:micro4_window |
|---|---|
| `mortality_365d` | 0.392 [-0.392, 1.361] |
| `inpatient_365d` | 0.914 [0.758, 1.092] |
| `readmission_30d` | 0.605 [0.161, 1.046] |
| `new_dx_365d/diabetes` | 1.134 [1.008, 1.264] |
| `new_dx_365d/heart_failure` | 0.948 [0.782, 1.141] |
| `new_dx_365d/ckd` | 0.788 [0.589, 0.949] |
| `new_dx_365d/copd` | 0.944 [0.775, 1.100] |
