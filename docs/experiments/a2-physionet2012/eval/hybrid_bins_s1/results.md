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
| created | 2026-09-14T16:09:26+00:00 |
| runtime (s) | 129.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/hybrid_bins_s1/final.pt` | last@final |
| `ckpt:hybrid_bins_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/hybrid_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `mortality_inhospital/24h` | 0.648 [0.599, 0.688] | 0.763 [0.732, 0.797] |
| `mortality_inhospital/48h` | 0.703 [0.662, 0.750] | 0.781 [0.745, 0.814] |

## AUPRC

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `mortality_inhospital/24h` | 0.242 [0.184, 0.312] | 0.352 [0.281, 0.433] |
| `mortality_inhospital/48h` | 0.283 [0.231, 0.358] | 0.411 [0.339, 0.492] |

## BRIER

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `mortality_inhospital/24h` | 0.1157 [0.1021, 0.1296] | 0.1057 [0.0931, 0.1193] |
| `mortality_inhospital/48h` | 0.1117 [0.0979, 0.1241] | 0.1036 [0.0923, 0.1180] |

## CALIBRATION SLOPE

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `mortality_inhospital/24h` | 0.703 [0.468, 0.910] | 0.951 [0.803, 1.126] |
| `mortality_inhospital/48h` | 0.924 [0.729, 1.171] | 0.785 [0.674, 0.930] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_inhospital/24h` | `random_init` - `ckpt:hybrid_bins_s1` | -0.116 | [-0.159, -0.081] | 0.000 |
| `mortality_inhospital/48h` | `random_init` - `ckpt:hybrid_bins_s1` | -0.078 | [-0.115, -0.035] | 0.000 |

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
