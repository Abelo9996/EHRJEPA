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
| created | 2026-09-08T06:21:08+00:00 |
| runtime (s) | 227.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_recon03_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate2-desynpuf/hybrid_recon03_s2/final.pt` | last@final |

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

| task | ckpt:hybrid_recon03_s2 |
|---|---|
| `mortality_365d` | 0.614 [0.568, 0.652] |
| `inpatient_365d` | 0.753 [0.738, 0.765] |
| `readmission_30d` | 0.695 [0.657, 0.735] |
| `new_dx_365d/diabetes` | 0.777 [0.763, 0.791] |
| `new_dx_365d/heart_failure` | 0.785 [0.770, 0.800] |
| `new_dx_365d/ckd` | 0.780 [0.762, 0.798] |
| `new_dx_365d/copd` | 0.774 [0.754, 0.792] |

## AUPRC

| task | ckpt:hybrid_recon03_s2 |
|---|---|
| `mortality_365d` | 0.027 [0.022, 0.037] |
| `inpatient_365d` | 0.439 [0.414, 0.463] |
| `readmission_30d` | 0.112 [0.079, 0.148] |
| `new_dx_365d/diabetes` | 0.484 [0.455, 0.515] |
| `new_dx_365d/heart_failure` | 0.334 [0.302, 0.368] |
| `new_dx_365d/ckd` | 0.292 [0.262, 0.331] |
| `new_dx_365d/copd` | 0.353 [0.318, 0.390] |

## BRIER

| task | ckpt:hybrid_recon03_s2 |
|---|---|
| `mortality_365d` | 0.0186 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1429 [0.1385, 0.1472] |
| `readmission_30d` | 0.0465 [0.0410, 0.0520] |
| `new_dx_365d/diabetes` | 0.1454 [0.1401, 0.1504] |
| `new_dx_365d/heart_failure` | 0.0994 [0.0943, 0.1036] |
| `new_dx_365d/ckd` | 0.0802 [0.0752, 0.0847] |
| `new_dx_365d/copd` | 0.1007 [0.0947, 0.1056] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_recon03_s2 |
|---|---|
| `mortality_365d` | 0.692 [0.405, 0.921] |
| `inpatient_365d` | 1.057 [0.969, 1.133] |
| `readmission_30d` | 0.914 [0.716, 1.135] |
| `new_dx_365d/diabetes` | 0.986 [0.911, 1.063] |
| `new_dx_365d/heart_failure` | 0.995 [0.912, 1.097] |
| `new_dx_365d/ckd` | 1.000 [0.908, 1.104] |
| `new_dx_365d/copd` | 1.011 [0.919, 1.113] |
