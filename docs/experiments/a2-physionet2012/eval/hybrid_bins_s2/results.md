# Downstream evaluation -- physionet2012

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2012` |
| MEDS | `data/meds/physionet2012` |
| cache | `data/cache/physionet2012` |
| tasks | `data/tasks/physionet2012` |
| anchor seed | 20260903 |
| commit | `039c63c` |
| created | 2026-09-14T16:43:10+00:00 |
| runtime (s) | 63.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ckpt:hybrid_bins_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/hybrid_bins_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.769 [0.736, 0.803] |
| `mortality_inhospital/48h` | 0.788 [0.757, 0.822] |

## AUPRC

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.346 [0.279, 0.422] |
| `mortality_inhospital/48h` | 0.394 [0.327, 0.479] |

## BRIER

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 0.1050 [0.0920, 0.1170] |
| `mortality_inhospital/48h` | 0.1020 [0.0911, 0.1148] |

## CALIBRATION SLOPE

| task | ckpt:hybrid_bins_s2 |
|---|---|
| `mortality_inhospital/24h` | 1.030 [0.874, 1.226] |
| `mortality_inhospital/48h` | 0.886 [0.771, 1.047] |

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
