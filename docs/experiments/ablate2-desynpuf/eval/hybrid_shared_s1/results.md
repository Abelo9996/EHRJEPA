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
| created | 2026-09-07T22:24:39+00:00 |
| runtime (s) | 263.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_shared_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_shared_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_shared_s1 |
|---|---|
| `mortality_365d` | 0.608 [0.563, 0.646] |
| `inpatient_365d` | 0.751 [0.737, 0.763] |
| `readmission_30d` | 0.685 [0.645, 0.723] |
| `new_dx_365d/diabetes` | 0.773 [0.760, 0.788] |
| `new_dx_365d/heart_failure` | 0.774 [0.760, 0.788] |
| `new_dx_365d/ckd` | 0.772 [0.754, 0.790] |
| `new_dx_365d/copd` | 0.758 [0.740, 0.778] |

## AUPRC

| task | ckpt:hybrid_shared_s1 |
|---|---|
| `mortality_365d` | 0.027 [0.022, 0.035] |
| `inpatient_365d` | 0.435 [0.410, 0.462] |
| `readmission_30d` | 0.105 [0.077, 0.138] |
| `new_dx_365d/diabetes` | 0.474 [0.443, 0.502] |
| `new_dx_365d/heart_failure` | 0.314 [0.285, 0.346] |
| `new_dx_365d/ckd` | 0.265 [0.241, 0.302] |
| `new_dx_365d/copd` | 0.344 [0.310, 0.381] |

## BRIER

| task | ckpt:hybrid_shared_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1432 [0.1389, 0.1472] |
| `readmission_30d` | 0.0469 [0.0413, 0.0524] |
| `new_dx_365d/diabetes` | 0.1465 [0.1407, 0.1517] |
| `new_dx_365d/heart_failure` | 0.1010 [0.0957, 0.1055] |
| `new_dx_365d/ckd` | 0.0814 [0.0762, 0.0861] |
| `new_dx_365d/copd` | 0.1026 [0.0962, 0.1080] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_shared_s1 |
|---|---|
| `mortality_365d` | 0.735 [0.408, 1.019] |
| `inpatient_365d` | 1.072 [0.990, 1.144] |
| `readmission_30d` | 0.784 [0.619, 0.974] |
| `new_dx_365d/diabetes` | 0.972 [0.896, 1.052] |
| `new_dx_365d/heart_failure` | 0.984 [0.900, 1.071] |
| `new_dx_365d/ckd` | 0.969 [0.876, 1.078] |
| `new_dx_365d/copd` | 0.956 [0.885, 1.060] |
