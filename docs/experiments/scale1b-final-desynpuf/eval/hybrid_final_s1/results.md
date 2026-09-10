# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `5353d33` |
| created | 2026-09-09T19:46:29+00:00 |
| runtime (s) | 182.8 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_final_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-final-desynpuf/hybrid_final_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_final_s1 |
|---|---|
| `mortality_365d` | 0.603 [0.552, 0.640] |
| `inpatient_365d` | 0.757 [0.742, 0.770] |
| `readmission_30d` | 0.695 [0.655, 0.732] |
| `new_dx_365d/diabetes` | 0.778 [0.764, 0.793] |
| `new_dx_365d/heart_failure` | 0.797 [0.782, 0.812] |
| `new_dx_365d/ckd` | 0.784 [0.766, 0.802] |
| `new_dx_365d/copd` | 0.772 [0.753, 0.791] |

## AUPRC

| task | ckpt:hybrid_final_s1 |
|---|---|
| `mortality_365d` | 0.029 [0.022, 0.040] |
| `inpatient_365d` | 0.452 [0.427, 0.477] |
| `readmission_30d` | 0.108 [0.086, 0.148] |
| `new_dx_365d/diabetes` | 0.482 [0.451, 0.511] |
| `new_dx_365d/heart_failure` | 0.342 [0.310, 0.372] |
| `new_dx_365d/ckd` | 0.297 [0.267, 0.334] |
| `new_dx_365d/copd` | 0.370 [0.333, 0.414] |

## BRIER

| task | ckpt:hybrid_final_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1417 [0.1370, 0.1458] |
| `readmission_30d` | 0.0468 [0.0411, 0.0525] |
| `new_dx_365d/diabetes` | 0.1455 [0.1398, 0.1503] |
| `new_dx_365d/heart_failure` | 0.0985 [0.0934, 0.1029] |
| `new_dx_365d/ckd` | 0.0797 [0.0747, 0.0843] |
| `new_dx_365d/copd` | 0.0998 [0.0937, 0.1046] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_final_s1 |
|---|---|
| `mortality_365d` | 0.626 [0.345, 0.890] |
| `inpatient_365d` | 0.997 [0.917, 1.061] |
| `readmission_30d` | 0.815 [0.636, 0.997] |
| `new_dx_365d/diabetes` | 0.945 [0.875, 1.023] |
| `new_dx_365d/heart_failure` | 0.955 [0.885, 1.042] |
| `new_dx_365d/ckd` | 1.016 [0.924, 1.110] |
| `new_dx_365d/copd` | 0.965 [0.879, 1.053] |
