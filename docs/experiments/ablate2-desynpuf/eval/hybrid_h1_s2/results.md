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
| created | 2026-09-08T05:41:33+00:00 |
| runtime (s) | 210.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_h1_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_h1_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_h1_s2 |
|---|---|
| `mortality_365d` | 0.610 [0.563, 0.649] |
| `inpatient_365d` | 0.749 [0.734, 0.761] |
| `readmission_30d` | 0.699 [0.668, 0.737] |
| `new_dx_365d/diabetes` | 0.771 [0.757, 0.787] |
| `new_dx_365d/heart_failure` | 0.768 [0.753, 0.782] |
| `new_dx_365d/ckd` | 0.767 [0.747, 0.787] |
| `new_dx_365d/copd` | 0.753 [0.736, 0.773] |

## AUPRC

| task | ckpt:hybrid_h1_s2 |
|---|---|
| `mortality_365d` | 0.029 [0.023, 0.039] |
| `inpatient_365d` | 0.433 [0.406, 0.457] |
| `readmission_30d` | 0.115 [0.082, 0.155] |
| `new_dx_365d/diabetes` | 0.479 [0.450, 0.511] |
| `new_dx_365d/heart_failure` | 0.324 [0.292, 0.354] |
| `new_dx_365d/ckd` | 0.267 [0.242, 0.303] |
| `new_dx_365d/copd` | 0.337 [0.302, 0.374] |

## BRIER

| task | ckpt:hybrid_h1_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1437 [0.1392, 0.1478] |
| `readmission_30d` | 0.0466 [0.0409, 0.0523] |
| `new_dx_365d/diabetes` | 0.1465 [0.1411, 0.1518] |
| `new_dx_365d/heart_failure` | 0.1010 [0.0957, 0.1056] |
| `new_dx_365d/ckd` | 0.0815 [0.0764, 0.0863] |
| `new_dx_365d/copd` | 0.1033 [0.0971, 0.1089] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_h1_s2 |
|---|---|
| `mortality_365d` | 0.733 [0.434, 0.992] |
| `inpatient_365d` | 1.045 [0.958, 1.129] |
| `readmission_30d` | 0.898 [0.730, 1.114] |
| `new_dx_365d/diabetes` | 0.959 [0.886, 1.032] |
| `new_dx_365d/heart_failure` | 0.998 [0.909, 1.077] |
| `new_dx_365d/ckd` | 1.041 [0.936, 1.156] |
| `new_dx_365d/copd` | 0.945 [0.859, 1.052] |
