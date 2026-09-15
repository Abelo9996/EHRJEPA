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
| created | 2026-09-15T14:51:41+00:00 |
| runtime (s) | 564.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `ft_random` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/ar_bins_s1/final.pt` | last@final |
| `ft:ar_bins_s1` | finetune | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2012/ar_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 9626 (0.1415) | 1198 (0.1519) | 1154 (0.1386) |  |
| `mortality_inhospital/48h` | 9633 (0.1417) | 1198 (0.1519) | 1154 (0.1386) |  |

## AUROC

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 0.789 [0.756, 0.825] | 0.829 [0.797, 0.864] | 0.654 [0.604, 0.699] | 0.800 [0.768, 0.830] |
| `mortality_inhospital/48h` | 0.828 [0.797, 0.866] | 0.859 [0.829, 0.890] | 0.684 [0.640, 0.728] | 0.835 [0.807, 0.869] |

## AUPRC

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 0.394 [0.319, 0.472] | 0.465 [0.385, 0.550] | 0.265 [0.204, 0.331] | 0.387 [0.315, 0.471] |
| `mortality_inhospital/48h` | 0.457 [0.389, 0.535] | 0.516 [0.435, 0.597] | 0.283 [0.229, 0.358] | 0.460 [0.384, 0.547] |

## BRIER

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 0.1019 [0.0896, 0.1137] | 0.0948 [0.0835, 0.1062] | 0.1137 [0.0990, 0.1285] | 0.1032 [0.0913, 0.1161] |
| `mortality_inhospital/48h` | 0.0963 [0.0839, 0.1091] | 0.0907 [0.0794, 0.1030] | 0.1119 [0.0971, 0.1269] | 0.0955 [0.0837, 0.1092] |

## CALIBRATION SLOPE

| task | lr | gbm | ft_random | ft:ar_bins_s1 |
|---|---|---|---|---|
| `mortality_inhospital/24h` | 1.162 [0.995, 1.400] | 0.877 [0.749, 1.024] | 0.866 [0.614, 1.111] | 0.735 [0.630, 0.867] |
| `mortality_inhospital/48h` | 1.376 [1.207, 1.630] | 0.875 [0.758, 1.014] | 0.865 [0.651, 1.093] | 0.806 [0.702, 0.938] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_inhospital/24h` | `lr` - `gbm` | -0.040 | [-0.064, -0.016] | 0.000 |
| `mortality_inhospital/24h` | `lr` - `ft_random` | 0.135 | [0.081, 0.186] | 0.000 |
| `mortality_inhospital/24h` | `lr` - `ft:ar_bins_s1` | -0.011 | [-0.040, 0.011] | 0.380 |
| `mortality_inhospital/24h` | `gbm` - `ft_random` | 0.175 | [0.125, 0.221] | 0.000 |
| `mortality_inhospital/24h` | `gbm` - `ft:ar_bins_s1` | 0.029 | [0.007, 0.049] | 0.010 |
| `mortality_inhospital/24h` | `ft_random` - `ft:ar_bins_s1` | -0.146 | [-0.192, -0.106] | 0.000 |
| `mortality_inhospital/48h` | `lr` - `gbm` | -0.031 | [-0.050, -0.010] | 0.000 |
| `mortality_inhospital/48h` | `lr` - `ft_random` | 0.144 | [0.097, 0.184] | 0.000 |
| `mortality_inhospital/48h` | `lr` - `ft:ar_bins_s1` | -0.007 | [-0.029, 0.014] | 0.590 |
| `mortality_inhospital/48h` | `gbm` - `ft_random` | 0.175 | [0.125, 0.219] | 0.000 |
| `mortality_inhospital/48h` | `gbm` - `ft:ar_bins_s1` | 0.023 | [0.003, 0.040] | 0.020 |
| `mortality_inhospital/48h` | `ft_random` - `ft:ar_bins_s1` | -0.151 | [-0.191, -0.101] | 0.000 |

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
