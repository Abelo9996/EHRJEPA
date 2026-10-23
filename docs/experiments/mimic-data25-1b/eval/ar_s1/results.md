# Downstream evaluation -- mimic-iv-25

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `mimic-iv-25` |
| MEDS | `data/meds/mimic-iv-25` |
| cache | `data/cache/mimic-iv-25` |
| tasks | `data/tasks/mimic-iv-25` |
| anchor seed | 20260903 |
| commit | `b35a09f` |
| created | 2026-10-08T20:11:34+00:00 |
| runtime (s) | 182.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:ar_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 27644 (0.2133) | 3492 (0.2153) | 3331 (0.2017) |  |
| `inpatient_365d` | 27644 (0.3697) | 3492 (0.3691) | 3331 (0.3723) |  |
| `readmission_30d` | 25686 (0.1365) | 3201 (0.1340) | 3091 (0.1330) |  |
| `new_dx_365d/diabetes` | 24926 (0.1025) | 3170 (0.1000) | 3025 (0.1055) |  |
| `new_dx_365d/heart_failure` | 26017 (0.0852) | 3306 (0.0796) | 3132 (0.0843) |  |
| `new_dx_365d/ckd` | 25867 (0.0748) | 3297 (0.0673) | 3119 (0.0792) |  |
| `new_dx_365d/copd` | 26621 (0.0465) | 3374 (0.0492) | 3223 (0.0428) |  |

## AUROC

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.928 [0.919, 0.937] |
| `inpatient_365d` | 0.695 [0.680, 0.714] |
| `readmission_30d` | 0.660 [0.635, 0.685] |
| `new_dx_365d/diabetes` | 0.865 [0.845, 0.885] |
| `new_dx_365d/heart_failure` | 0.897 [0.875, 0.914] |
| `new_dx_365d/ckd` | 0.899 [0.883, 0.914] |
| `new_dx_365d/copd` | 0.819 [0.789, 0.848] |

## AUPRC

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.753 [0.726, 0.786] |
| `inpatient_365d` | 0.584 [0.556, 0.615] |
| `readmission_30d` | 0.253 [0.222, 0.287] |
| `new_dx_365d/diabetes` | 0.541 [0.489, 0.593] |
| `new_dx_365d/heart_failure` | 0.491 [0.430, 0.550] |
| `new_dx_365d/ckd` | 0.453 [0.391, 0.516] |
| `new_dx_365d/copd` | 0.147 [0.113, 0.191] |

## BRIER

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.0838 [0.0768, 0.0888] |
| `inpatient_365d` | 0.2072 [0.2015, 0.2137] |
| `readmission_30d` | 0.1098 [0.1023, 0.1170] |
| `new_dx_365d/diabetes` | 0.0678 [0.0603, 0.0749] |
| `new_dx_365d/heart_failure` | 0.0563 [0.0511, 0.0622] |
| `new_dx_365d/ckd` | 0.0553 [0.0489, 0.0609] |
| `new_dx_365d/copd` | 0.0386 [0.0339, 0.0439] |

## CALIBRATION SLOPE

| task | ckpt:ar_s1 |
|---|---|
| `mortality_365d` | 0.993 [0.926, 1.077] |
| `inpatient_365d` | 0.858 [0.775, 0.942] |
| `readmission_30d` | 0.866 [0.733, 1.000] |
| `new_dx_365d/diabetes` | 0.973 [0.887, 1.082] |
| `new_dx_365d/heart_failure` | 1.010 [0.900, 1.109] |
| `new_dx_365d/ckd` | 1.105 [0.993, 1.229] |
| `new_dx_365d/copd` | 1.022 [0.884, 1.173] |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
