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
| created | 2026-09-08T11:13:27+00:00 |
| runtime (s) | 177.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_frozen_ar_init_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_frozen_ar_init_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_frozen_ar_init_s2 |
|---|---|
| `mortality_365d` | 0.616 [0.565, 0.655] |
| `inpatient_365d` | 0.753 [0.739, 0.766] |
| `readmission_30d` | 0.677 [0.627, 0.721] |
| `new_dx_365d/diabetes` | 0.773 [0.759, 0.789] |
| `new_dx_365d/heart_failure` | 0.787 [0.770, 0.800] |
| `new_dx_365d/ckd` | 0.780 [0.762, 0.798] |
| `new_dx_365d/copd` | 0.768 [0.750, 0.786] |

## AUPRC

| task | ckpt:hybrid_frozen_ar_init_s2 |
|---|---|
| `mortality_365d` | 0.028 [0.023, 0.036] |
| `inpatient_365d` | 0.442 [0.411, 0.471] |
| `readmission_30d` | 0.098 [0.073, 0.131] |
| `new_dx_365d/diabetes` | 0.470 [0.441, 0.503] |
| `new_dx_365d/heart_failure` | 0.345 [0.311, 0.378] |
| `new_dx_365d/ckd` | 0.284 [0.256, 0.315] |
| `new_dx_365d/copd` | 0.345 [0.313, 0.384] |

## BRIER

| task | ckpt:hybrid_frozen_ar_init_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1427 [0.1384, 0.1469] |
| `readmission_30d` | 0.0469 [0.0413, 0.0525] |
| `new_dx_365d/diabetes` | 0.1469 [0.1411, 0.1525] |
| `new_dx_365d/heart_failure` | 0.0990 [0.0939, 0.1038] |
| `new_dx_365d/ckd` | 0.0803 [0.0755, 0.0848] |
| `new_dx_365d/copd` | 0.1016 [0.0953, 0.1067] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_frozen_ar_init_s2 |
|---|---|
| `mortality_365d` | 0.682 [0.378, 0.916] |
| `inpatient_365d` | 1.017 [0.938, 1.101] |
| `readmission_30d` | 0.798 [0.580, 1.005] |
| `new_dx_365d/diabetes` | 0.933 [0.866, 1.015] |
| `new_dx_365d/heart_failure` | 0.997 [0.909, 1.075] |
| `new_dx_365d/ckd` | 1.017 [0.937, 1.113] |
| `new_dx_365d/copd` | 0.953 [0.861, 1.041] |
