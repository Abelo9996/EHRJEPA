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
| created | 2026-09-08T13:00:13+00:00 |
| runtime (s) | 203.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_textinit_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/ar_textinit_s2/final.pt` | last@final |

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

| task | ckpt:ar_textinit_s2 |
|---|---|
| `mortality_365d` | 0.611 [0.560, 0.645] |
| `inpatient_365d` | 0.748 [0.733, 0.759] |
| `readmission_30d` | 0.658 [0.619, 0.706] |
| `new_dx_365d/diabetes` | 0.773 [0.760, 0.787] |
| `new_dx_365d/heart_failure` | 0.775 [0.761, 0.790] |
| `new_dx_365d/ckd` | 0.770 [0.754, 0.789] |
| `new_dx_365d/copd` | 0.762 [0.743, 0.782] |

## AUPRC

| task | ckpt:ar_textinit_s2 |
|---|---|
| `mortality_365d` | 0.027 [0.022, 0.039] |
| `inpatient_365d` | 0.430 [0.401, 0.454] |
| `readmission_30d` | 0.091 [0.068, 0.121] |
| `new_dx_365d/diabetes` | 0.473 [0.445, 0.500] |
| `new_dx_365d/heart_failure` | 0.326 [0.295, 0.357] |
| `new_dx_365d/ckd` | 0.262 [0.238, 0.294] |
| `new_dx_365d/copd` | 0.343 [0.310, 0.379] |

## BRIER

| task | ckpt:ar_textinit_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1439 [0.1397, 0.1483] |
| `readmission_30d` | 0.0470 [0.0411, 0.0529] |
| `new_dx_365d/diabetes` | 0.1466 [0.1415, 0.1514] |
| `new_dx_365d/heart_failure` | 0.1004 [0.0952, 0.1049] |
| `new_dx_365d/ckd` | 0.0817 [0.0768, 0.0862] |
| `new_dx_365d/copd` | 0.1023 [0.0961, 0.1076] |

## CALIBRATION SLOPE

| task | ckpt:ar_textinit_s2 |
|---|---|
| `mortality_365d` | 0.670 [0.343, 0.917] |
| `inpatient_365d` | 1.011 [0.929, 1.082] |
| `readmission_30d` | 0.867 [0.647, 1.121] |
| `new_dx_365d/diabetes` | 0.934 [0.866, 1.009] |
| `new_dx_365d/heart_failure` | 0.984 [0.896, 1.068] |
| `new_dx_365d/ckd` | 0.923 [0.845, 1.021] |
| `new_dx_365d/copd` | 0.967 [0.883, 1.070] |
