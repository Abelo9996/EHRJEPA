# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `7bfd19f` |
| created | 2026-09-07T23:49:18+00:00 |
| runtime (s) | 230.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_h1_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_h1_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_h1_s1 |
|---|---|
| `mortality_365d` | 0.619 [0.574, 0.661] |
| `inpatient_365d` | 0.745 [0.731, 0.757] |
| `readmission_30d` | 0.700 [0.661, 0.741] |
| `new_dx_365d/diabetes` | 0.768 [0.754, 0.783] |
| `new_dx_365d/heart_failure` | 0.769 [0.754, 0.784] |
| `new_dx_365d/ckd` | 0.767 [0.748, 0.787] |
| `new_dx_365d/copd` | 0.757 [0.741, 0.775] |

## AUPRC

| task | ckpt:hybrid_h1_s1 |
|---|---|
| `mortality_365d` | 0.029 [0.023, 0.038] |
| `inpatient_365d` | 0.427 [0.397, 0.454] |
| `readmission_30d` | 0.108 [0.086, 0.151] |
| `new_dx_365d/diabetes` | 0.474 [0.442, 0.506] |
| `new_dx_365d/heart_failure` | 0.314 [0.285, 0.343] |
| `new_dx_365d/ckd` | 0.270 [0.244, 0.304] |
| `new_dx_365d/copd` | 0.327 [0.296, 0.369] |

## BRIER

| task | ckpt:hybrid_h1_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1444 [0.1403, 0.1485] |
| `readmission_30d` | 0.0465 [0.0411, 0.0522] |
| `new_dx_365d/diabetes` | 0.1471 [0.1413, 0.1526] |
| `new_dx_365d/heart_failure` | 0.1013 [0.0960, 0.1061] |
| `new_dx_365d/ckd` | 0.0815 [0.0766, 0.0861] |
| `new_dx_365d/copd` | 0.1033 [0.0972, 0.1087] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_h1_s1 |
|---|---|
| `mortality_365d` | 0.740 [0.422, 0.990] |
| `inpatient_365d` | 1.012 [0.926, 1.084] |
| `readmission_30d` | 0.918 [0.725, 1.154] |
| `new_dx_365d/diabetes` | 0.945 [0.876, 1.027] |
| `new_dx_365d/heart_failure` | 0.971 [0.888, 1.070] |
| `new_dx_365d/ckd` | 0.992 [0.903, 1.085] |
| `new_dx_365d/copd` | 0.977 [0.892, 1.076] |
