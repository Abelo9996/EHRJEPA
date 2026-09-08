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
| created | 2026-09-08T08:48:20+00:00 |
| runtime (s) | 200.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_textinit_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/ablate3-desynpuf/hybrid_textinit_s1/final.pt` | last@final |

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

| task | ckpt:hybrid_textinit_s1 |
|---|---|
| `mortality_365d` | 0.624 [0.581, 0.659] |
| `inpatient_365d` | 0.754 [0.738, 0.765] |
| `readmission_30d` | 0.694 [0.656, 0.736] |
| `new_dx_365d/diabetes` | 0.773 [0.761, 0.788] |
| `new_dx_365d/heart_failure` | 0.771 [0.758, 0.785] |
| `new_dx_365d/ckd` | 0.779 [0.761, 0.798] |
| `new_dx_365d/copd` | 0.763 [0.746, 0.781] |

## AUPRC

| task | ckpt:hybrid_textinit_s1 |
|---|---|
| `mortality_365d` | 0.032 [0.024, 0.049] |
| `inpatient_365d` | 0.441 [0.413, 0.469] |
| `readmission_30d` | 0.101 [0.079, 0.137] |
| `new_dx_365d/diabetes` | 0.480 [0.452, 0.509] |
| `new_dx_365d/heart_failure` | 0.312 [0.280, 0.340] |
| `new_dx_365d/ckd` | 0.277 [0.247, 0.310] |
| `new_dx_365d/copd` | 0.344 [0.312, 0.378] |

## BRIER

| task | ckpt:hybrid_textinit_s1 |
|---|---|
| `mortality_365d` | 0.0186 [0.0156, 0.0219] |
| `inpatient_365d` | 0.1427 [0.1381, 0.1470] |
| `readmission_30d` | 0.0467 [0.0413, 0.0524] |
| `new_dx_365d/diabetes` | 0.1462 [0.1407, 0.1515] |
| `new_dx_365d/heart_failure` | 0.1014 [0.0966, 0.1058] |
| `new_dx_365d/ckd` | 0.0808 [0.0758, 0.0855] |
| `new_dx_365d/copd` | 0.1023 [0.0960, 0.1074] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_textinit_s1 |
|---|---|
| `mortality_365d` | 0.758 [0.466, 1.014] |
| `inpatient_365d` | 1.042 [0.952, 1.123] |
| `readmission_30d` | 0.845 [0.678, 1.048] |
| `new_dx_365d/diabetes` | 0.986 [0.913, 1.062] |
| `new_dx_365d/heart_failure` | 0.963 [0.889, 1.049] |
| `new_dx_365d/ckd` | 1.001 [0.900, 1.102] |
| `new_dx_365d/copd` | 0.985 [0.900, 1.083] |
