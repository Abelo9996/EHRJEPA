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
| created | 2026-09-04T06:50:28+00:00 |
| runtime (s) | 273.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:recon_only_notime` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-04-pilot3-desynpuf/recon_only_notime/final.pt` | mean@final |

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

| task | ckpt:recon_only_notime |
|---|---|
| `mortality_365d` | 0.617 [0.542, 0.693] |
| `inpatient_365d` | 0.701 [0.679, 0.720] |
| `readmission_30d` | 0.680 [0.637, 0.716] |
| `new_dx_365d/diabetes` | 0.745 [0.727, 0.765] |
| `new_dx_365d/heart_failure` | 0.707 [0.684, 0.733] |
| `new_dx_365d/ckd` | 0.722 [0.697, 0.750] |
| `new_dx_365d/copd` | 0.712 [0.687, 0.738] |

## AUPRC

| task | ckpt:recon_only_notime |
|---|---|
| `mortality_365d` | 0.032 [0.024, 0.047] |
| `inpatient_365d` | 0.356 [0.323, 0.391] |
| `readmission_30d` | 0.096 [0.078, 0.131] |
| `new_dx_365d/diabetes` | 0.440 [0.408, 0.478] |
| `new_dx_365d/heart_failure` | 0.268 [0.233, 0.310] |
| `new_dx_365d/ckd` | 0.202 [0.173, 0.245] |
| `new_dx_365d/copd` | 0.253 [0.220, 0.298] |

## BRIER

| task | ckpt:recon_only_notime |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1476 [0.1407, 0.1547] |
| `readmission_30d` | 0.0461 [0.0403, 0.0517] |
| `new_dx_365d/diabetes` | 0.1514 [0.1440, 0.1590] |
| `new_dx_365d/heart_failure` | 0.1106 [0.1014, 0.1175] |
| `new_dx_365d/ckd` | 0.0831 [0.0753, 0.0899] |
| `new_dx_365d/copd` | 0.1063 [0.0978, 0.1149] |

## CALIBRATION SLOPE

| task | ckpt:recon_only_notime |
|---|---|
| `mortality_365d` | 0.762 [0.264, 1.292] |
| `inpatient_365d` | 0.998 [0.882, 1.125] |
| `readmission_30d` | 0.963 [0.748, 1.215] |
| `new_dx_365d/diabetes` | 0.961 [0.870, 1.083] |
| `new_dx_365d/heart_failure` | 0.893 [0.778, 1.033] |
| `new_dx_365d/ckd` | 0.921 [0.789, 1.087] |
| `new_dx_365d/copd` | 0.945 [0.801, 1.092] |
