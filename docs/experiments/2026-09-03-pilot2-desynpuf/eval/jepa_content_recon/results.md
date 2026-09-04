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
| created | 2026-09-04T03:39:43+00:00 |
| runtime (s) | 182.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:jepa_content_recon` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-pilot2-desynpuf/jepa_content_recon/final.pt` | mean@final |

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

| task | ckpt:jepa_content_recon |
|---|---|
| `mortality_365d` | 0.627 [0.551, 0.695] |
| `inpatient_365d` | 0.709 [0.687, 0.728] |
| `readmission_30d` | 0.675 [0.625, 0.723] |
| `new_dx_365d/diabetes` | 0.743 [0.724, 0.766] |
| `new_dx_365d/heart_failure` | 0.715 [0.691, 0.740] |
| `new_dx_365d/ckd` | 0.717 [0.692, 0.747] |
| `new_dx_365d/copd` | 0.714 [0.690, 0.738] |

## AUPRC

| task | ckpt:jepa_content_recon |
|---|---|
| `mortality_365d` | 0.034 [0.024, 0.054] |
| `inpatient_365d` | 0.365 [0.331, 0.399] |
| `readmission_30d` | 0.111 [0.077, 0.154] |
| `new_dx_365d/diabetes` | 0.437 [0.403, 0.480] |
| `new_dx_365d/heart_failure` | 0.258 [0.228, 0.297] |
| `new_dx_365d/ckd` | 0.191 [0.163, 0.226] |
| `new_dx_365d/copd` | 0.245 [0.215, 0.288] |

## BRIER

| task | ckpt:jepa_content_recon |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1464 [0.1395, 0.1536] |
| `readmission_30d` | 0.0459 [0.0403, 0.0516] |
| `new_dx_365d/diabetes` | 0.1515 [0.1443, 0.1588] |
| `new_dx_365d/heart_failure` | 0.1105 [0.1014, 0.1174] |
| `new_dx_365d/ckd` | 0.0834 [0.0755, 0.0905] |
| `new_dx_365d/copd` | 0.1062 [0.0982, 0.1147] |

## CALIBRATION SLOPE

| task | ckpt:jepa_content_recon |
|---|---|
| `mortality_365d` | 1.052 [0.463, 1.663] |
| `inpatient_365d` | 1.073 [0.936, 1.215] |
| `readmission_30d` | 1.208 [0.836, 1.555] |
| `new_dx_365d/diabetes` | 0.923 [0.830, 1.050] |
| `new_dx_365d/heart_failure` | 0.948 [0.833, 1.099] |
| `new_dx_365d/ckd` | 0.927 [0.789, 1.111] |
| `new_dx_365d/copd` | 0.985 [0.841, 1.142] |
