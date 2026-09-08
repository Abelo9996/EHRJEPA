# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `39490e9` |
| created | 2026-09-08T14:37:00+00:00 |
| runtime (s) | 204.3 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_nosig_frozen_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate4-desynpuf/hybrid_nosig_frozen_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_nosig_frozen_s2 |
|---|---|
| `mortality_365d` | 0.613 [0.570, 0.649] |
| `inpatient_365d` | 0.759 [0.744, 0.770] |
| `readmission_30d` | 0.694 [0.652, 0.733] |
| `new_dx_365d/diabetes` | 0.776 [0.763, 0.791] |
| `new_dx_365d/heart_failure` | 0.796 [0.781, 0.810] |
| `new_dx_365d/ckd` | 0.780 [0.761, 0.798] |
| `new_dx_365d/copd` | 0.771 [0.751, 0.790] |

## AUPRC

| task | ckpt:hybrid_nosig_frozen_s2 |
|---|---|
| `mortality_365d` | 0.026 [0.022, 0.034] |
| `inpatient_365d` | 0.441 [0.414, 0.466] |
| `readmission_30d` | 0.108 [0.083, 0.146] |
| `new_dx_365d/diabetes` | 0.483 [0.453, 0.512] |
| `new_dx_365d/heart_failure` | 0.353 [0.321, 0.388] |
| `new_dx_365d/ckd` | 0.290 [0.260, 0.322] |
| `new_dx_365d/copd` | 0.341 [0.306, 0.380] |

## BRIER

| task | ckpt:hybrid_nosig_frozen_s2 |
|---|---|
| `mortality_365d` | 0.0187 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1420 [0.1378, 0.1463] |
| `readmission_30d` | 0.0466 [0.0412, 0.0523] |
| `new_dx_365d/diabetes` | 0.1457 [0.1400, 0.1506] |
| `new_dx_365d/heart_failure` | 0.0979 [0.0928, 0.1027] |
| `new_dx_365d/ckd` | 0.0802 [0.0753, 0.0850] |
| `new_dx_365d/copd` | 0.1013 [0.0955, 0.1063] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_nosig_frozen_s2 |
|---|---|
| `mortality_365d` | 0.659 [0.399, 0.871] |
| `inpatient_365d` | 1.023 [0.943, 1.093] |
| `readmission_30d` | 0.883 [0.692, 1.094] |
| `new_dx_365d/diabetes` | 0.941 [0.872, 1.011] |
| `new_dx_365d/heart_failure` | 1.001 [0.925, 1.089] |
| `new_dx_365d/ckd` | 1.009 [0.920, 1.109] |
| `new_dx_365d/copd` | 0.972 [0.879, 1.064] |
