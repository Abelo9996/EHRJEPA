# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `5353d33` |
| created | 2026-09-10T15:02:29+00:00 |
| runtime (s) | 60.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/hybrid_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 7358 (0.0190) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 7358 (0.2098) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3116 (0.0501) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 4939 (0.2264) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 6164 (0.1308) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 6487 (0.0999) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 6009 (0.1340) |  |

## AUROC

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.601 [0.550, 0.645] |
| `inpatient_365d` | 0.756 [0.741, 0.767] |
| `readmission_30d` | 0.685 [0.646, 0.725] |
| `new_dx_365d/diabetes` | 0.779 [0.766, 0.794] |
| `new_dx_365d/heart_failure` | 0.791 [0.777, 0.805] |
| `new_dx_365d/ckd` | 0.784 [0.764, 0.802] |
| `new_dx_365d/copd` | 0.769 [0.750, 0.786] |

## AUPRC

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.030 [0.023, 0.045] |
| `inpatient_365d` | 0.446 [0.420, 0.472] |
| `readmission_30d` | 0.102 [0.083, 0.133] |
| `new_dx_365d/diabetes` | 0.485 [0.455, 0.517] |
| `new_dx_365d/heart_failure` | 0.336 [0.305, 0.368] |
| `new_dx_365d/ckd` | 0.298 [0.268, 0.331] |
| `new_dx_365d/copd` | 0.357 [0.319, 0.396] |

## BRIER

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1422 [0.1377, 0.1462] |
| `readmission_30d` | 0.0469 [0.0410, 0.0527] |
| `new_dx_365d/diabetes` | 0.1452 [0.1396, 0.1504] |
| `new_dx_365d/heart_failure` | 0.0989 [0.0940, 0.1033] |
| `new_dx_365d/ckd` | 0.0798 [0.0753, 0.0840] |
| `new_dx_365d/copd` | 0.1007 [0.0948, 0.1063] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.643 [0.358, 0.921] |
| `inpatient_365d` | 0.987 [0.905, 1.056] |
| `readmission_30d` | 0.774 [0.598, 0.960] |
| `new_dx_365d/diabetes` | 0.952 [0.885, 1.034] |
| `new_dx_365d/heart_failure` | 0.996 [0.923, 1.086] |
| `new_dx_365d/ckd` | 0.946 [0.858, 1.034] |
| `new_dx_365d/copd` | 0.950 [0.858, 1.038] |
