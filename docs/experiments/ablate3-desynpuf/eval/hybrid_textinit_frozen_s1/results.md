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
| created | 2026-09-08T09:26:17+00:00 |
| runtime (s) | 249.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_textinit_frozen_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_textinit_frozen_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_textinit_frozen_s1 |
|---|---|
| `mortality_365d` | 0.617 [0.573, 0.653] |
| `inpatient_365d` | 0.753 [0.738, 0.765] |
| `readmission_30d` | 0.693 [0.655, 0.738] |
| `new_dx_365d/diabetes` | 0.778 [0.765, 0.792] |
| `new_dx_365d/heart_failure` | 0.769 [0.755, 0.785] |
| `new_dx_365d/ckd` | 0.772 [0.753, 0.791] |
| `new_dx_365d/copd` | 0.766 [0.749, 0.787] |

## AUPRC

| task | ckpt:hybrid_textinit_frozen_s1 |
|---|---|
| `mortality_365d` | 0.031 [0.022, 0.052] |
| `inpatient_365d` | 0.436 [0.410, 0.462] |
| `readmission_30d` | 0.113 [0.086, 0.154] |
| `new_dx_365d/diabetes` | 0.483 [0.454, 0.514] |
| `new_dx_365d/heart_failure` | 0.315 [0.286, 0.348] |
| `new_dx_365d/ckd` | 0.271 [0.244, 0.304] |
| `new_dx_365d/copd` | 0.353 [0.317, 0.393] |

## BRIER

| task | ckpt:hybrid_textinit_frozen_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1430 [0.1386, 0.1470] |
| `readmission_30d` | 0.0465 [0.0410, 0.0523] |
| `new_dx_365d/diabetes` | 0.1453 [0.1400, 0.1501] |
| `new_dx_365d/heart_failure` | 0.1013 [0.0961, 0.1057] |
| `new_dx_365d/ckd` | 0.0813 [0.0762, 0.0861] |
| `new_dx_365d/copd` | 0.1014 [0.0955, 0.1066] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_textinit_frozen_s1 |
|---|---|
| `mortality_365d` | 0.809 [0.504, 1.063] |
| `inpatient_365d` | 1.042 [0.959, 1.116] |
| `readmission_30d` | 0.848 [0.681, 1.063] |
| `new_dx_365d/diabetes` | 0.971 [0.900, 1.054] |
| `new_dx_365d/heart_failure` | 0.958 [0.882, 1.047] |
| `new_dx_365d/ckd` | 0.946 [0.851, 1.048] |
| `new_dx_365d/copd` | 0.980 [0.892, 1.086] |
