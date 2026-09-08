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
| created | 2026-09-08T08:10:10+00:00 |
| runtime (s) | 176.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_frozen_ar_init_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_frozen_ar_init_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_frozen_ar_init_s1 |
|---|---|
| `mortality_365d` | 0.615 [0.566, 0.650] |
| `inpatient_365d` | 0.752 [0.738, 0.765] |
| `readmission_30d` | 0.673 [0.626, 0.717] |
| `new_dx_365d/diabetes` | 0.773 [0.759, 0.790] |
| `new_dx_365d/heart_failure` | 0.787 [0.770, 0.800] |
| `new_dx_365d/ckd` | 0.779 [0.761, 0.795] |
| `new_dx_365d/copd` | 0.767 [0.749, 0.786] |

## AUPRC

| task | ckpt:hybrid_frozen_ar_init_s1 |
|---|---|
| `mortality_365d` | 0.029 [0.023, 0.038] |
| `inpatient_365d` | 0.442 [0.412, 0.470] |
| `readmission_30d` | 0.094 [0.069, 0.124] |
| `new_dx_365d/diabetes` | 0.469 [0.440, 0.501] |
| `new_dx_365d/heart_failure` | 0.341 [0.307, 0.374] |
| `new_dx_365d/ckd` | 0.280 [0.253, 0.313] |
| `new_dx_365d/copd` | 0.344 [0.310, 0.380] |

## BRIER

| task | ckpt:hybrid_frozen_ar_init_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1429 [0.1384, 0.1470] |
| `readmission_30d` | 0.0470 [0.0414, 0.0527] |
| `new_dx_365d/diabetes` | 0.1469 [0.1410, 0.1522] |
| `new_dx_365d/heart_failure` | 0.0991 [0.0941, 0.1039] |
| `new_dx_365d/ckd` | 0.0805 [0.0759, 0.0850] |
| `new_dx_365d/copd` | 0.1016 [0.0954, 0.1068] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_frozen_ar_init_s1 |
|---|---|
| `mortality_365d` | 0.687 [0.389, 0.917] |
| `inpatient_365d` | 1.012 [0.936, 1.094] |
| `readmission_30d` | 0.780 [0.567, 0.973] |
| `new_dx_365d/diabetes` | 0.912 [0.842, 0.992] |
| `new_dx_365d/heart_failure` | 0.997 [0.912, 1.079] |
| `new_dx_365d/ckd` | 1.006 [0.925, 1.101] |
| `new_dx_365d/copd` | 0.950 [0.854, 1.039] |
