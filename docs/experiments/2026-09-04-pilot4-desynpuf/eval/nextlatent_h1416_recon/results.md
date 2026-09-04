# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `1c43622` |
| created | 2026-09-04T09:47:32+00:00 |
| runtime (s) | 163.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:nextlatent_h1416_recon` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot4-desynpuf/nextlatent_h1416_recon/final.pt` | last@final |

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

| task | ckpt:nextlatent_h1416_recon |
|---|---|
| `mortality_365d` | 0.633 [0.557, 0.713] |
| `inpatient_365d` | 0.742 [0.718, 0.761] |
| `readmission_30d` | 0.699 [0.661, 0.736] |
| `new_dx_365d/diabetes` | 0.762 [0.746, 0.783] |
| `new_dx_365d/heart_failure` | 0.764 [0.746, 0.786] |
| `new_dx_365d/ckd` | 0.753 [0.731, 0.777] |
| `new_dx_365d/copd` | 0.749 [0.726, 0.771] |

## AUPRC

| task | ckpt:nextlatent_h1416_recon |
|---|---|
| `mortality_365d` | 0.035 [0.025, 0.055] |
| `inpatient_365d` | 0.394 [0.356, 0.435] |
| `readmission_30d` | 0.120 [0.089, 0.177] |
| `new_dx_365d/diabetes` | 0.454 [0.413, 0.494] |
| `new_dx_365d/heart_failure` | 0.306 [0.266, 0.346] |
| `new_dx_365d/ckd` | 0.234 [0.198, 0.283] |
| `new_dx_365d/copd` | 0.293 [0.253, 0.338] |

## BRIER

| task | ckpt:nextlatent_h1416_recon |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1421 [0.1349, 0.1486] |
| `readmission_30d` | 0.0457 [0.0397, 0.0515] |
| `new_dx_365d/diabetes` | 0.1481 [0.1415, 0.1542] |
| `new_dx_365d/heart_failure` | 0.1061 [0.0984, 0.1124] |
| `new_dx_365d/ckd` | 0.0811 [0.0733, 0.0876] |
| `new_dx_365d/copd` | 0.1030 [0.0953, 0.1106] |

## CALIBRATION SLOPE

| task | ckpt:nextlatent_h1416_recon |
|---|---|
| `mortality_365d` | 0.782 [0.310, 1.275] |
| `inpatient_365d` | 1.076 [0.937, 1.228] |
| `readmission_30d` | 1.000 [0.808, 1.225] |
| `new_dx_365d/diabetes` | 0.958 [0.868, 1.083] |
| `new_dx_365d/heart_failure` | 1.024 [0.923, 1.157] |
| `new_dx_365d/ckd` | 0.960 [0.843, 1.108] |
| `new_dx_365d/copd` | 0.994 [0.855, 1.139] |
