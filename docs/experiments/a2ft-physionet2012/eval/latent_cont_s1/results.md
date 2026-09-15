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
| created | 2026-09-15T15:05:58+00:00 |
| runtime (s) | 259.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `ft:latent_cont_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/latent_cont_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | ft:latent_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.770 [0.732, 0.809] |
| `mortality_inhospital/48h` | 0.810 [0.778, 0.848] |

## AUPRC

| task | ft:latent_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.357 [0.282, 0.429] |
| `mortality_inhospital/48h` | 0.430 [0.361, 0.519] |

## BRIER

| task | ft:latent_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.1060 [0.0926, 0.1183] |
| `mortality_inhospital/48h` | 0.1000 [0.0877, 0.1140] |

## CALIBRATION SLOPE

| task | ft:latent_cont_s1 |
|---|---|
| `mortality_inhospital/24h` | 0.767 [0.645, 0.905] |
| `mortality_inhospital/48h` | 0.839 [0.723, 0.987] |

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
