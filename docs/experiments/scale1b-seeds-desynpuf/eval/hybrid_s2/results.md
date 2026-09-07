# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `fdc4f65` |
| created | 2026-09-07T12:26:18+00:00 |
| runtime (s) | 186.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/scale1b-seeds-desynpuf/hybrid_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.597 [0.532, 0.663] |
| `inpatient_365d` | 0.756 [0.731, 0.775] |
| `readmission_30d` | 0.684 [0.644, 0.726] |
| `new_dx_365d/diabetes` | 0.778 [0.763, 0.799] |
| `new_dx_365d/heart_failure` | 0.795 [0.776, 0.814] |
| `new_dx_365d/ckd` | 0.775 [0.754, 0.800] |
| `new_dx_365d/copd` | 0.776 [0.751, 0.799] |

## AUPRC

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.033 [0.023, 0.061] |
| `inpatient_365d` | 0.434 [0.387, 0.478] |
| `readmission_30d` | 0.101 [0.078, 0.134] |
| `new_dx_365d/diabetes` | 0.482 [0.444, 0.522] |
| `new_dx_365d/heart_failure` | 0.346 [0.302, 0.400] |
| `new_dx_365d/ckd` | 0.272 [0.226, 0.321] |
| `new_dx_365d/copd` | 0.344 [0.301, 0.397] |

## BRIER

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1384 [0.1318, 0.1454] |
| `readmission_30d` | 0.0463 [0.0404, 0.0521] |
| `new_dx_365d/diabetes` | 0.1444 [0.1372, 0.1513] |
| `new_dx_365d/heart_failure` | 0.1024 [0.0951, 0.1092] |
| `new_dx_365d/ckd` | 0.0792 [0.0723, 0.0854] |
| `new_dx_365d/copd` | 0.0989 [0.0909, 0.1063] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_s2 |
|---|---|
| `mortality_365d` | 0.560 [0.158, 1.058] |
| `inpatient_365d` | 1.016 [0.881, 1.143] |
| `readmission_30d` | 0.770 [0.590, 0.979] |
| `new_dx_365d/diabetes` | 0.951 [0.868, 1.053] |
| `new_dx_365d/heart_failure` | 1.031 [0.922, 1.164] |
| `new_dx_365d/ckd` | 0.912 [0.809, 1.038] |
| `new_dx_365d/copd` | 0.967 [0.850, 1.071] |
