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
| created | 2026-09-08T04:19:56+00:00 |
| runtime (s) | 240.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_shared_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_shared_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_shared_s2 |
|---|---|
| `mortality_365d` | 0.600 [0.555, 0.638] |
| `inpatient_365d` | 0.751 [0.736, 0.763] |
| `readmission_30d` | 0.683 [0.644, 0.725] |
| `new_dx_365d/diabetes` | 0.767 [0.754, 0.783] |
| `new_dx_365d/heart_failure` | 0.771 [0.756, 0.786] |
| `new_dx_365d/ckd` | 0.774 [0.755, 0.793] |
| `new_dx_365d/copd` | 0.754 [0.736, 0.774] |

## AUPRC

| task | ckpt:hybrid_shared_s2 |
|---|---|
| `mortality_365d` | 0.027 [0.022, 0.036] |
| `inpatient_365d` | 0.440 [0.412, 0.464] |
| `readmission_30d` | 0.106 [0.082, 0.142] |
| `new_dx_365d/diabetes` | 0.462 [0.432, 0.495] |
| `new_dx_365d/heart_failure` | 0.317 [0.286, 0.349] |
| `new_dx_365d/ckd` | 0.277 [0.249, 0.313] |
| `new_dx_365d/copd` | 0.337 [0.304, 0.375] |

## BRIER

| task | ckpt:hybrid_shared_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1431 [0.1385, 0.1472] |
| `readmission_30d` | 0.0467 [0.0412, 0.0523] |
| `new_dx_365d/diabetes` | 0.1480 [0.1423, 0.1532] |
| `new_dx_365d/heart_failure` | 0.1011 [0.0957, 0.1057] |
| `new_dx_365d/ckd` | 0.0809 [0.0757, 0.0853] |
| `new_dx_365d/copd` | 0.1031 [0.0969, 0.1086] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_shared_s2 |
|---|---|
| `mortality_365d` | 0.678 [0.351, 0.960] |
| `inpatient_365d` | 1.062 [0.982, 1.147] |
| `readmission_30d` | 0.802 [0.624, 1.002] |
| `new_dx_365d/diabetes` | 0.939 [0.870, 1.014] |
| `new_dx_365d/heart_failure` | 0.973 [0.893, 1.057] |
| `new_dx_365d/ckd` | 1.014 [0.916, 1.123] |
| `new_dx_365d/copd` | 0.949 [0.863, 1.057] |
