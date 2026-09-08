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
| created | 2026-09-08T01:26:34+00:00 |
| runtime (s) | 166.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_small_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_small_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_small_s2 |
|---|---|
| `mortality_365d` | 0.624 [0.579, 0.661] |
| `inpatient_365d` | 0.753 [0.737, 0.766] |
| `readmission_30d` | 0.697 [0.657, 0.737] |
| `new_dx_365d/diabetes` | 0.772 [0.759, 0.787] |
| `new_dx_365d/heart_failure` | 0.771 [0.755, 0.785] |
| `new_dx_365d/ckd` | 0.772 [0.752, 0.789] |
| `new_dx_365d/copd` | 0.756 [0.739, 0.773] |

## AUPRC

| task | ckpt:hybrid_small_s2 |
|---|---|
| `mortality_365d` | 0.032 [0.025, 0.053] |
| `inpatient_365d` | 0.439 [0.414, 0.463] |
| `readmission_30d` | 0.108 [0.082, 0.143] |
| `new_dx_365d/diabetes` | 0.475 [0.445, 0.506] |
| `new_dx_365d/heart_failure` | 0.316 [0.290, 0.343] |
| `new_dx_365d/ckd` | 0.276 [0.249, 0.314] |
| `new_dx_365d/copd` | 0.324 [0.292, 0.361] |

## BRIER

| task | ckpt:hybrid_small_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1429 [0.1383, 0.1469] |
| `readmission_30d` | 0.0466 [0.0410, 0.0522] |
| `new_dx_365d/diabetes` | 0.1466 [0.1409, 0.1516] |
| `new_dx_365d/heart_failure` | 0.1012 [0.0958, 0.1062] |
| `new_dx_365d/ckd` | 0.0811 [0.0763, 0.0857] |
| `new_dx_365d/copd` | 0.1036 [0.0976, 0.1092] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_small_s2 |
|---|---|
| `mortality_365d` | 0.797 [0.508, 1.060] |
| `inpatient_365d` | 1.046 [0.963, 1.122] |
| `readmission_30d` | 0.918 [0.729, 1.143] |
| `new_dx_365d/diabetes` | 0.980 [0.908, 1.062] |
| `new_dx_365d/heart_failure` | 0.948 [0.872, 1.034] |
| `new_dx_365d/ckd` | 0.984 [0.887, 1.084] |
| `new_dx_365d/copd` | 0.992 [0.906, 1.096] |
