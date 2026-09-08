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
| created | 2026-09-08T00:32:36+00:00 |
| runtime (s) | 261.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_recon03_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_recon03_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_recon03_s1 |
|---|---|
| `mortality_365d` | 0.618 [0.569, 0.657] |
| `inpatient_365d` | 0.759 [0.743, 0.770] |
| `readmission_30d` | 0.692 [0.655, 0.732] |
| `new_dx_365d/diabetes` | 0.774 [0.761, 0.790] |
| `new_dx_365d/heart_failure` | 0.784 [0.769, 0.799] |
| `new_dx_365d/ckd` | 0.772 [0.752, 0.789] |
| `new_dx_365d/copd` | 0.766 [0.749, 0.785] |

## AUPRC

| task | ckpt:hybrid_recon03_s1 |
|---|---|
| `mortality_365d` | 0.029 [0.024, 0.043] |
| `inpatient_365d` | 0.442 [0.412, 0.467] |
| `readmission_30d` | 0.104 [0.079, 0.142] |
| `new_dx_365d/diabetes` | 0.475 [0.448, 0.505] |
| `new_dx_365d/heart_failure` | 0.331 [0.303, 0.364] |
| `new_dx_365d/ckd` | 0.271 [0.238, 0.305] |
| `new_dx_365d/copd` | 0.355 [0.322, 0.393] |

## BRIER

| task | ckpt:hybrid_recon03_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1422 [0.1381, 0.1465] |
| `readmission_30d` | 0.0466 [0.0411, 0.0521] |
| `new_dx_365d/diabetes` | 0.1463 [0.1409, 0.1514] |
| `new_dx_365d/heart_failure` | 0.0997 [0.0948, 0.1044] |
| `new_dx_365d/ckd` | 0.0812 [0.0760, 0.0857] |
| `new_dx_365d/copd` | 0.1015 [0.0954, 0.1067] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_recon03_s1 |
|---|---|
| `mortality_365d` | 0.755 [0.460, 1.034] |
| `inpatient_365d` | 1.069 [0.978, 1.136] |
| `readmission_30d` | 0.873 [0.697, 1.096] |
| `new_dx_365d/diabetes` | 0.966 [0.892, 1.048] |
| `new_dx_365d/heart_failure` | 0.981 [0.908, 1.070] |
| `new_dx_365d/ckd` | 0.912 [0.825, 0.997] |
| `new_dx_365d/copd` | 0.983 [0.904, 1.087] |
