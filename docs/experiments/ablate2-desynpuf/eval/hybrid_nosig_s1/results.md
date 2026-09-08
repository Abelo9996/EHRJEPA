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
| created | 2026-09-07T23:06:58+00:00 |
| runtime (s) | 250.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_nosig_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_nosig_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_nosig_s1 |
|---|---|
| `mortality_365d` | 0.601 [0.557, 0.641] |
| `inpatient_365d` | 0.758 [0.742, 0.770] |
| `readmission_30d` | 0.705 [0.672, 0.742] |
| `new_dx_365d/diabetes` | 0.777 [0.765, 0.793] |
| `new_dx_365d/heart_failure` | 0.784 [0.768, 0.798] |
| `new_dx_365d/ckd` | 0.781 [0.762, 0.800] |
| `new_dx_365d/copd` | 0.768 [0.749, 0.786] |

## AUPRC

| task | ckpt:hybrid_nosig_s1 |
|---|---|
| `mortality_365d` | 0.030 [0.023, 0.047] |
| `inpatient_365d` | 0.450 [0.423, 0.479] |
| `readmission_30d` | 0.116 [0.083, 0.150] |
| `new_dx_365d/diabetes` | 0.484 [0.453, 0.511] |
| `new_dx_365d/heart_failure` | 0.328 [0.298, 0.360] |
| `new_dx_365d/ckd` | 0.273 [0.246, 0.303] |
| `new_dx_365d/copd` | 0.357 [0.322, 0.397] |

## BRIER

| task | ckpt:hybrid_nosig_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1418 [0.1372, 0.1459] |
| `readmission_30d` | 0.0464 [0.0409, 0.0520] |
| `new_dx_365d/diabetes` | 0.1454 [0.1400, 0.1513] |
| `new_dx_365d/heart_failure` | 0.0997 [0.0947, 0.1042] |
| `new_dx_365d/ckd` | 0.0809 [0.0756, 0.0852] |
| `new_dx_365d/copd` | 0.1010 [0.0951, 0.1060] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_nosig_s1 |
|---|---|
| `mortality_365d` | 0.656 [0.344, 0.925] |
| `inpatient_365d` | 1.032 [0.954, 1.107] |
| `readmission_30d` | 0.921 [0.756, 1.130] |
| `new_dx_365d/diabetes` | 0.969 [0.896, 1.045] |
| `new_dx_365d/heart_failure` | 1.003 [0.919, 1.099] |
| `new_dx_365d/ckd` | 0.943 [0.861, 1.033] |
| `new_dx_365d/copd` | 0.980 [0.892, 1.074] |
