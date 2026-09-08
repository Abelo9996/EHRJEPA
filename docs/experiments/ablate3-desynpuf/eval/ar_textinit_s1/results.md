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
| created | 2026-09-08T09:56:46+00:00 |
| runtime (s) | 200.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_textinit_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/ar_textinit_s1/final.pt` | last@final |

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

| task | ckpt:ar_textinit_s1 |
|---|---|
| `mortality_365d` | 0.599 [0.555, 0.639] |
| `inpatient_365d` | 0.752 [0.736, 0.764] |
| `readmission_30d` | 0.675 [0.632, 0.719] |
| `new_dx_365d/diabetes` | 0.769 [0.755, 0.782] |
| `new_dx_365d/heart_failure` | 0.773 [0.757, 0.787] |
| `new_dx_365d/ckd` | 0.772 [0.754, 0.791] |
| `new_dx_365d/copd` | 0.761 [0.741, 0.779] |

## AUPRC

| task | ckpt:ar_textinit_s1 |
|---|---|
| `mortality_365d` | 0.033 [0.023, 0.051] |
| `inpatient_365d` | 0.429 [0.401, 0.454] |
| `readmission_30d` | 0.106 [0.078, 0.145] |
| `new_dx_365d/diabetes` | 0.484 [0.454, 0.511] |
| `new_dx_365d/heart_failure` | 0.311 [0.281, 0.343] |
| `new_dx_365d/ckd` | 0.264 [0.240, 0.291] |
| `new_dx_365d/copd` | 0.340 [0.305, 0.377] |

## BRIER

| task | ckpt:ar_textinit_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1436 [0.1394, 0.1479] |
| `readmission_30d` | 0.0468 [0.0411, 0.0525] |
| `new_dx_365d/diabetes` | 0.1469 [0.1420, 0.1519] |
| `new_dx_365d/heart_failure` | 0.1012 [0.0955, 0.1058] |
| `new_dx_365d/ckd` | 0.0814 [0.0764, 0.0860] |
| `new_dx_365d/copd` | 0.1024 [0.0965, 0.1077] |

## CALIBRATION SLOPE

| task | ckpt:ar_textinit_s1 |
|---|---|
| `mortality_365d` | 0.643 [0.379, 0.921] |
| `inpatient_365d` | 1.056 [0.970, 1.131] |
| `readmission_30d` | 0.925 [0.696, 1.180] |
| `new_dx_365d/diabetes` | 0.987 [0.916, 1.066] |
| `new_dx_365d/heart_failure` | 0.998 [0.910, 1.096] |
| `new_dx_365d/ckd` | 0.982 [0.890, 1.090] |
| `new_dx_365d/copd` | 0.950 [0.856, 1.058] |
