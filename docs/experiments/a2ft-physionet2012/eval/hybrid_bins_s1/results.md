# Downstream evaluation -- physionet2012

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2012` |
| MEDS | `data/meds/physionet2012` |
| cache | `data/cache/physionet2012` |
| tasks | `data/tasks/physionet2012` |
| anchor seed | 20260903 |
| commit | `b35a09f` |
| created | 2026-09-15T15:01:09+00:00 |
| runtime (s) | 286.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:hybrid_bins_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/hybrid_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ft:hybrid_bins_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.797 [0.766, 0.829] |
| `mortality_inhospital/48h` | 0.830 [0.801, 0.860] |

## AUPRC

| task | ft:hybrid_bins_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.386 [0.309, 0.472] |
| `mortality_inhospital/48h` | 0.436 [0.359, 0.528] |

## BRIER

| task | ft:hybrid_bins_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.1028 [0.0896, 0.1141] |
| `mortality_inhospital/48h` | 0.0987 [0.0865, 0.1113] |

## CALIBRATION SLOPE

| task | ft:hybrid_bins_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.749 [0.637, 0.871] |
| `mortality_inhospital/48h` | 0.772 [0.667, 0.894] |

## Skipped

| task | reason |
|---|---|
| `mortality_365d` | death |
| `inpatient_365d` | inpatient_admission |
| `readmission_30d` | inpatient_admission |
| `new_dx_365d/diabetes` | dx_diabetes |
| `new_dx_365d/heart_failure` | dx_heart_failure |
| `new_dx_365d/ckd` | dx_ckd |
| `new_dx_365d/copd` | dx_copd |
| `sepsis_6h` | not defined on source family 'physionet2012' |
| `sepsis_stay` | not defined on source family 'physionet2012' |
