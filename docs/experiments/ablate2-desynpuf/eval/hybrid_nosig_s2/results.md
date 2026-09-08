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
| created | 2026-09-08T05:03:01+00:00 |
| runtime (s) | 219.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_nosig_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_nosig_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_nosig_s2 |
|---|---|
| `mortality_365d` | 0.596 [0.548, 0.634] |
| `inpatient_365d` | 0.757 [0.742, 0.769] |
| `readmission_30d` | 0.691 [0.654, 0.731] |
| `new_dx_365d/diabetes` | 0.780 [0.767, 0.794] |
| `new_dx_365d/heart_failure` | 0.784 [0.768, 0.799] |
| `new_dx_365d/ckd` | 0.783 [0.764, 0.799] |
| `new_dx_365d/copd` | 0.772 [0.754, 0.793] |

## AUPRC

| task | ckpt:hybrid_nosig_s2 |
|---|---|
| `mortality_365d` | 0.027 [0.021, 0.039] |
| `inpatient_365d` | 0.441 [0.415, 0.466] |
| `readmission_30d` | 0.105 [0.081, 0.141] |
| `new_dx_365d/diabetes` | 0.490 [0.461, 0.520] |
| `new_dx_365d/heart_failure` | 0.324 [0.297, 0.357] |
| `new_dx_365d/ckd` | 0.295 [0.267, 0.326] |
| `new_dx_365d/copd` | 0.360 [0.327, 0.396] |

## BRIER

| task | ckpt:hybrid_nosig_s2 |
|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1423 [0.1378, 0.1466] |
| `readmission_30d` | 0.0467 [0.0412, 0.0522] |
| `new_dx_365d/diabetes` | 0.1446 [0.1396, 0.1494] |
| `new_dx_365d/heart_failure` | 0.1000 [0.0947, 0.1046] |
| `new_dx_365d/ckd` | 0.0798 [0.0747, 0.0845] |
| `new_dx_365d/copd` | 0.1003 [0.0943, 0.1054] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_nosig_s2 |
|---|---|
| `mortality_365d` | 0.597 [0.293, 0.839] |
| `inpatient_365d` | 1.045 [0.965, 1.119] |
| `readmission_30d` | 0.853 [0.677, 1.061] |
| `new_dx_365d/diabetes` | 0.981 [0.913, 1.061] |
| `new_dx_365d/heart_failure` | 0.968 [0.886, 1.061] |
| `new_dx_365d/ckd` | 1.002 [0.906, 1.092] |
| `new_dx_365d/copd` | 0.979 [0.891, 1.080] |
