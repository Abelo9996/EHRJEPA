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
| created | 2026-09-08T03:37:51+00:00 |
| runtime (s) | 401.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_large_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_large_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_large_s2 |
|---|---|
| `mortality_365d` | 0.614 [0.567, 0.650] |
| `inpatient_365d` | 0.758 [0.744, 0.770] |
| `readmission_30d` | 0.678 [0.636, 0.719] |
| `new_dx_365d/diabetes` | 0.779 [0.766, 0.793] |
| `new_dx_365d/heart_failure` | 0.787 [0.772, 0.801] |
| `new_dx_365d/ckd` | 0.782 [0.766, 0.801] |
| `new_dx_365d/copd` | 0.767 [0.750, 0.786] |

## AUPRC

| task | ckpt:hybrid_large_s2 |
|---|---|
| `mortality_365d` | 0.029 [0.022, 0.046] |
| `inpatient_365d` | 0.449 [0.423, 0.474] |
| `readmission_30d` | 0.101 [0.079, 0.134] |
| `new_dx_365d/diabetes` | 0.483 [0.455, 0.512] |
| `new_dx_365d/heart_failure` | 0.340 [0.308, 0.371] |
| `new_dx_365d/ckd` | 0.285 [0.254, 0.322] |
| `new_dx_365d/copd` | 0.359 [0.325, 0.395] |

## BRIER

| task | ckpt:hybrid_large_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1418 [0.1372, 0.1456] |
| `readmission_30d` | 0.0470 [0.0411, 0.0529] |
| `new_dx_365d/diabetes` | 0.1450 [0.1393, 0.1499] |
| `new_dx_365d/heart_failure` | 0.0993 [0.0941, 0.1038] |
| `new_dx_365d/ckd` | 0.0802 [0.0752, 0.0849] |
| `new_dx_365d/copd` | 0.1006 [0.0948, 0.1056] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_large_s2 |
|---|---|
| `mortality_365d` | 0.650 [0.355, 0.893] |
| `inpatient_365d` | 1.032 [0.946, 1.111] |
| `readmission_30d` | 0.716 [0.550, 0.896] |
| `new_dx_365d/diabetes` | 0.960 [0.890, 1.036] |
| `new_dx_365d/heart_failure` | 0.945 [0.862, 1.024] |
| `new_dx_365d/ckd` | 0.987 [0.895, 1.089] |
| `new_dx_365d/copd` | 0.964 [0.885, 1.054] |
