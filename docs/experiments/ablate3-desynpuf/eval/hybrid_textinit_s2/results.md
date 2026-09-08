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
| created | 2026-09-08T11:51:43+00:00 |
| runtime (s) | 201.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_textinit_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_textinit_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_textinit_s2 |
|---|---|
| `mortality_365d` | 0.617 [0.572, 0.655] |
| `inpatient_365d` | 0.752 [0.737, 0.764] |
| `readmission_30d` | 0.697 [0.657, 0.737] |
| `new_dx_365d/diabetes` | 0.770 [0.758, 0.785] |
| `new_dx_365d/heart_failure` | 0.773 [0.758, 0.787] |
| `new_dx_365d/ckd` | 0.781 [0.764, 0.799] |
| `new_dx_365d/copd` | 0.765 [0.748, 0.784] |

## AUPRC

| task | ckpt:hybrid_textinit_s2 |
|---|---|
| `mortality_365d` | 0.031 [0.023, 0.048] |
| `inpatient_365d` | 0.443 [0.414, 0.467] |
| `readmission_30d` | 0.110 [0.083, 0.142] |
| `new_dx_365d/diabetes` | 0.482 [0.452, 0.510] |
| `new_dx_365d/heart_failure` | 0.316 [0.286, 0.341] |
| `new_dx_365d/ckd` | 0.281 [0.255, 0.314] |
| `new_dx_365d/copd` | 0.348 [0.311, 0.384] |

## BRIER

| task | ckpt:hybrid_textinit_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1428 [0.1384, 0.1469] |
| `readmission_30d` | 0.0465 [0.0410, 0.0523] |
| `new_dx_365d/diabetes` | 0.1466 [0.1413, 0.1516] |
| `new_dx_365d/heart_failure` | 0.1009 [0.0958, 0.1053] |
| `new_dx_365d/ckd` | 0.0805 [0.0758, 0.0850] |
| `new_dx_365d/copd` | 0.1018 [0.0960, 0.1068] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_textinit_s2 |
|---|---|
| `mortality_365d` | 0.727 [0.419, 0.979] |
| `inpatient_365d` | 1.040 [0.962, 1.113] |
| `readmission_30d` | 0.884 [0.696, 1.093] |
| `new_dx_365d/diabetes` | 0.974 [0.906, 1.048] |
| `new_dx_365d/heart_failure` | 1.006 [0.920, 1.095] |
| `new_dx_365d/ckd` | 0.998 [0.900, 1.095] |
| `new_dx_365d/copd` | 0.976 [0.890, 1.081] |
