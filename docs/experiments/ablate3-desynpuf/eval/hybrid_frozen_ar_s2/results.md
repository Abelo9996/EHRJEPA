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
| created | 2026-09-08T10:35:04+00:00 |
| runtime (s) | 205.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_frozen_ar_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_frozen_ar_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_frozen_ar_s2 |
|---|---|
| `mortality_365d` | 0.610 [0.565, 0.645] |
| `inpatient_365d` | 0.758 [0.743, 0.771] |
| `readmission_30d` | 0.680 [0.634, 0.719] |
| `new_dx_365d/diabetes` | 0.776 [0.763, 0.791] |
| `new_dx_365d/heart_failure` | 0.794 [0.779, 0.808] |
| `new_dx_365d/ckd` | 0.783 [0.764, 0.800] |
| `new_dx_365d/copd` | 0.770 [0.751, 0.788] |

## AUPRC

| task | ckpt:hybrid_frozen_ar_s2 |
|---|---|
| `mortality_365d` | 0.028 [0.022, 0.048] |
| `inpatient_365d` | 0.452 [0.425, 0.479] |
| `readmission_30d` | 0.103 [0.080, 0.144] |
| `new_dx_365d/diabetes` | 0.478 [0.448, 0.508] |
| `new_dx_365d/heart_failure` | 0.347 [0.310, 0.380] |
| `new_dx_365d/ckd` | 0.294 [0.263, 0.325] |
| `new_dx_365d/copd` | 0.345 [0.310, 0.385] |

## BRIER

| task | ckpt:hybrid_frozen_ar_s2 |
|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0220] |
| `inpatient_365d` | 0.1417 [0.1375, 0.1460] |
| `readmission_30d` | 0.0468 [0.0411, 0.0523] |
| `new_dx_365d/diabetes` | 0.1461 [0.1400, 0.1515] |
| `new_dx_365d/heart_failure` | 0.0983 [0.0932, 0.1031] |
| `new_dx_365d/ckd` | 0.0800 [0.0754, 0.0843] |
| `new_dx_365d/copd` | 0.1013 [0.0954, 0.1062] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_frozen_ar_s2 |
|---|---|
| `mortality_365d` | 0.643 [0.364, 0.878] |
| `inpatient_365d` | 1.049 [0.967, 1.124] |
| `readmission_30d` | 0.809 [0.610, 1.033] |
| `new_dx_365d/diabetes` | 0.934 [0.869, 1.005] |
| `new_dx_365d/heart_failure` | 0.977 [0.895, 1.066] |
| `new_dx_365d/ckd` | 1.003 [0.920, 1.105] |
| `new_dx_365d/copd` | 0.963 [0.875, 1.056] |
