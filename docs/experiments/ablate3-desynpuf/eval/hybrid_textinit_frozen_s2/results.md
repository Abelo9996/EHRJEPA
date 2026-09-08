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
| created | 2026-09-08T12:29:44+00:00 |
| runtime (s) | 249.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_textinit_frozen_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_textinit_frozen_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_textinit_frozen_s2 |
|---|---|
| `mortality_365d` | 0.609 [0.565, 0.652] |
| `inpatient_365d` | 0.754 [0.740, 0.767] |
| `readmission_30d` | 0.688 [0.649, 0.730] |
| `new_dx_365d/diabetes` | 0.773 [0.759, 0.787] |
| `new_dx_365d/heart_failure` | 0.771 [0.757, 0.785] |
| `new_dx_365d/ckd` | 0.782 [0.764, 0.802] |
| `new_dx_365d/copd` | 0.757 [0.742, 0.775] |

## AUPRC

| task | ckpt:hybrid_textinit_frozen_s2 |
|---|---|
| `mortality_365d` | 0.031 [0.023, 0.055] |
| `inpatient_365d` | 0.440 [0.414, 0.467] |
| `readmission_30d` | 0.106 [0.082, 0.143] |
| `new_dx_365d/diabetes` | 0.476 [0.447, 0.506] |
| `new_dx_365d/heart_failure` | 0.313 [0.283, 0.343] |
| `new_dx_365d/ckd` | 0.281 [0.253, 0.316] |
| `new_dx_365d/copd` | 0.347 [0.312, 0.387] |

## BRIER

| task | ckpt:hybrid_textinit_frozen_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1426 [0.1382, 0.1469] |
| `readmission_30d` | 0.0466 [0.0409, 0.0524] |
| `new_dx_365d/diabetes` | 0.1465 [0.1412, 0.1514] |
| `new_dx_365d/heart_failure` | 0.1014 [0.0961, 0.1060] |
| `new_dx_365d/ckd` | 0.0804 [0.0754, 0.0851] |
| `new_dx_365d/copd` | 0.1027 [0.0970, 0.1079] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_textinit_frozen_s2 |
|---|---|
| `mortality_365d` | 0.717 [0.428, 1.000] |
| `inpatient_365d` | 1.029 [0.946, 1.103] |
| `readmission_30d` | 0.880 [0.697, 1.113] |
| `new_dx_365d/diabetes` | 0.950 [0.877, 1.021] |
| `new_dx_365d/heart_failure` | 0.925 [0.854, 1.006] |
| `new_dx_365d/ckd` | 1.011 [0.922, 1.130] |
| `new_dx_365d/copd` | 0.964 [0.879, 1.066] |
