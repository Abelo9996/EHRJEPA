# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `12478aa` |
| created | 2026-09-06T04:29:18+00:00 |
| runtime (s) | 315.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:recon_only` | probe | `/home/gaming_pc/EHRJEPA/runs/scale-desynpuf/recon_only/final.pt` | mean@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 3000 (0.0220) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 3000 (0.2010) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3000 (0.0493) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 3000 (0.2240) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 3000 (0.1370) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 3000 (0.0970) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 3000 (0.1303) |  |

## AUROC

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.651 [0.578, 0.724] |
| `inpatient_365d` | 0.733 [0.710, 0.752] |
| `readmission_30d` | 0.689 [0.652, 0.732] |
| `new_dx_365d/diabetes` | 0.757 [0.739, 0.778] |
| `new_dx_365d/heart_failure` | 0.745 [0.725, 0.768] |
| `new_dx_365d/ckd` | 0.734 [0.711, 0.761] |
| `new_dx_365d/copd` | 0.724 [0.702, 0.749] |

## AUPRC

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.038 [0.028, 0.064] |
| `inpatient_365d` | 0.392 [0.353, 0.433] |
| `readmission_30d` | 0.108 [0.083, 0.149] |
| `new_dx_365d/diabetes` | 0.453 [0.415, 0.493] |
| `new_dx_365d/heart_failure` | 0.300 [0.260, 0.347] |
| `new_dx_365d/ckd` | 0.222 [0.185, 0.273] |
| `new_dx_365d/copd` | 0.244 [0.211, 0.282] |

## BRIER

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.0214 [0.0172, 0.0264] |
| `inpatient_365d` | 0.1430 [0.1362, 0.1496] |
| `readmission_30d` | 0.0461 [0.0404, 0.0517] |
| `new_dx_365d/diabetes` | 0.1487 [0.1416, 0.1554] |
| `new_dx_365d/heart_failure` | 0.1075 [0.0989, 0.1142] |
| `new_dx_365d/ckd` | 0.0819 [0.0744, 0.0888] |
| `new_dx_365d/copd` | 0.1060 [0.0983, 0.1143] |

## CALIBRATION SLOPE

| task | ckpt:recon_only |
|---|---|
| `mortality_365d` | 0.958 [0.486, 1.485] |
| `inpatient_365d` | 1.055 [0.912, 1.210] |
| `readmission_30d` | 0.884 [0.715, 1.100] |
| `new_dx_365d/diabetes` | 0.968 [0.872, 1.097] |
| `new_dx_365d/heart_failure` | 1.018 [0.909, 1.151] |
| `new_dx_365d/ckd` | 0.999 [0.856, 1.175] |
| `new_dx_365d/copd` | 0.932 [0.787, 1.078] |
