# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `e4a0e2d` |
| created | 2026-09-04T01:48:11+00:00 |
| runtime (s) | 235.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_ema_block` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot-desynpuf/jepa_ema_block/final.pt` | cls_mean@final |

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

| task | ckpt:jepa_ema_block |
|---|---|
| `mortality_365d` | 0.586 [0.512, 0.659] |
| `inpatient_365d` | 0.676 [0.655, 0.697] |
| `readmission_30d` | 0.644 [0.594, 0.690] |
| `new_dx_365d/diabetes` | 0.730 [0.710, 0.750] |
| `new_dx_365d/heart_failure` | 0.710 [0.685, 0.736] |
| `new_dx_365d/ckd` | 0.694 [0.667, 0.723] |
| `new_dx_365d/copd` | 0.704 [0.681, 0.727] |

## AUPRC

| task | ckpt:jepa_ema_block |
|---|---|
| `mortality_365d` | 0.028 [0.022, 0.042] |
| `inpatient_365d` | 0.328 [0.293, 0.361] |
| `readmission_30d` | 0.092 [0.070, 0.130] |
| `new_dx_365d/diabetes` | 0.416 [0.386, 0.453] |
| `new_dx_365d/heart_failure` | 0.269 [0.234, 0.309] |
| `new_dx_365d/ckd` | 0.182 [0.155, 0.219] |
| `new_dx_365d/copd` | 0.232 [0.205, 0.269] |

## BRIER

| task | ckpt:jepa_ema_block |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] |
| `inpatient_365d` | 0.1501 [0.1427, 0.1579] |
| `readmission_30d` | 0.0463 [0.0404, 0.0521] |
| `new_dx_365d/diabetes` | 0.1539 [0.1470, 0.1614] |
| `new_dx_365d/heart_failure` | 0.1102 [0.1013, 0.1168] |
| `new_dx_365d/ckd` | 0.0840 [0.0758, 0.0913] |
| `new_dx_365d/copd` | 0.1069 [0.0985, 0.1153] |

## CALIBRATION SLOPE

| task | ckpt:jepa_ema_block |
|---|---|
| `mortality_365d` | 0.614 [0.098, 1.110] |
| `inpatient_365d` | 0.976 [0.826, 1.126] |
| `readmission_30d` | 0.921 [0.598, 1.202] |
| `new_dx_365d/diabetes` | 0.954 [0.845, 1.085] |
| `new_dx_365d/heart_failure` | 1.063 [0.928, 1.224] |
| `new_dx_365d/ckd` | 0.885 [0.730, 1.094] |
| `new_dx_365d/copd` | 0.986 [0.833, 1.147] |
