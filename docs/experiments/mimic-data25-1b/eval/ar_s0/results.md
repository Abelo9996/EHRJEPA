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
| created | 2026-10-08T12:28:04+00:00 |
| runtime (s) | 1865.0 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s0/final.pt` | last@final |
| `ckpt:ar_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s0/final.pt` | last@final |

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

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.939 [0.931, 0.947] | 0.940 [0.932, 0.949] | 0.848 [0.833, 0.862] | 0.929 [0.918, 0.939] |
| `inpatient_365d` | 0.707 [0.690, 0.724] | 0.724 [0.706, 0.739] | 0.559 [0.539, 0.579] | 0.689 [0.673, 0.705] |
| `readmission_30d` | 0.689 [0.662, 0.712] | 0.697 [0.672, 0.724] | 0.574 [0.544, 0.605] | 0.664 [0.636, 0.688] |
| `new_dx_365d/diabetes` | 0.906 [0.884, 0.925] | 0.919 [0.900, 0.937] | 0.735 [0.711, 0.757] | 0.873 [0.853, 0.895] |
| `new_dx_365d/heart_failure` | 0.914 [0.897, 0.929] | 0.911 [0.892, 0.929] | 0.827 [0.807, 0.848] | 0.900 [0.882, 0.916] |
| `new_dx_365d/ckd` | 0.908 [0.890, 0.922] | 0.940 [0.926, 0.953] | 0.800 [0.773, 0.820] | 0.895 [0.878, 0.911] |
| `new_dx_365d/copd` | 0.854 [0.819, 0.885] | 0.856 [0.819, 0.888] | 0.743 [0.716, 0.781] | 0.818 [0.784, 0.847] |

## AUPRC

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.811 [0.787, 0.839] | 0.816 [0.792, 0.842] | 0.586 [0.547, 0.629] | 0.766 [0.737, 0.803] |
| `inpatient_365d` | 0.599 [0.568, 0.624] | 0.611 [0.582, 0.635] | 0.416 [0.391, 0.443] | 0.574 [0.544, 0.604] |
| `readmission_30d` | 0.288 [0.252, 0.324] | 0.290 [0.251, 0.331] | 0.178 [0.155, 0.205] | 0.267 [0.233, 0.307] |
| `new_dx_365d/diabetes` | 0.672 [0.620, 0.725] | 0.737 [0.691, 0.781] | 0.204 [0.177, 0.242] | 0.574 [0.523, 0.628] |
| `new_dx_365d/heart_failure` | 0.576 [0.508, 0.643] | 0.574 [0.514, 0.638] | 0.293 [0.244, 0.338] | 0.488 [0.427, 0.545] |
| `new_dx_365d/ckd` | 0.518 [0.458, 0.589] | 0.647 [0.582, 0.702] | 0.258 [0.201, 0.312] | 0.445 [0.374, 0.508] |
| `new_dx_365d/copd` | 0.304 [0.235, 0.399] | 0.355 [0.275, 0.443] | 0.086 [0.071, 0.110] | 0.143 [0.109, 0.190] |

## BRIER

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.0765 [0.0711, 0.0818] | 0.0745 [0.0681, 0.0806] | 0.1165 [0.1100, 0.1229] | 0.0828 [0.0758, 0.0881] |
| `inpatient_365d` | 0.2037 [0.1984, 0.2095] | 0.1995 [0.1943, 0.2056] | 0.2319 [0.2272, 0.2369] | 0.2087 [0.2038, 0.2142] |
| `readmission_30d` | 0.1072 [0.0999, 0.1146] | 0.1072 [0.0996, 0.1150] | 0.1142 [0.1064, 0.1218] | 0.1089 [0.1017, 0.1161] |
| `new_dx_365d/diabetes` | 0.0582 [0.0531, 0.0649] | 0.0496 [0.0440, 0.0560] | 0.0888 [0.0802, 0.0975] | 0.0655 [0.0585, 0.0723] |
| `new_dx_365d/heart_failure` | 0.0518 [0.0466, 0.0574] | 0.0516 [0.0459, 0.0570] | 0.0674 [0.0609, 0.0741] | 0.0565 [0.0511, 0.0626] |
| `new_dx_365d/ckd` | 0.0522 [0.0458, 0.0568] | 0.0431 [0.0376, 0.0479] | 0.0656 [0.0588, 0.0713] | 0.0556 [0.0488, 0.0607] |
| `new_dx_365d/copd` | 0.0353 [0.0309, 0.0400] | 0.0334 [0.0283, 0.0383] | 0.0405 [0.0354, 0.0458] | 0.0387 [0.0340, 0.0440] |

## CALIBRATION SLOPE

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 1.259 [1.178, 1.370] | 0.833 [0.781, 0.900] | 0.976 [0.900, 1.058] | 1.056 [0.979, 1.146] |
| `inpatient_365d` | 0.948 [0.843, 1.031] | 0.979 [0.892, 1.055] | 0.683 [0.447, 0.910] | 0.899 [0.811, 0.998] |
| `readmission_30d` | 1.016 [0.864, 1.145] | 0.938 [0.806, 1.073] | 0.902 [0.566, 1.292] | 0.906 [0.747, 1.053] |
| `new_dx_365d/diabetes` | 1.441 [1.290, 1.645] | 1.140 [1.037, 1.266] | 1.130 [0.981, 1.297] | 1.045 [0.938, 1.167] |
| `new_dx_365d/heart_failure` | 1.263 [1.139, 1.427] | 0.937 [0.851, 1.039] | 0.944 [0.831, 1.048] | 1.026 [0.923, 1.137] |
| `new_dx_365d/ckd` | 1.334 [1.203, 1.495] | 1.022 [0.924, 1.120] | 0.987 [0.845, 1.100] | 1.069 [0.966, 1.188] |
| `new_dx_365d/copd` | 1.312 [1.124, 1.537] | 0.936 [0.811, 1.065] | 0.753 [0.643, 0.914] | 1.023 [0.862, 1.204] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | -0.001 | [-0.006, 0.003] | 0.510 |
| `mortality_365d` | `lr` - `random_init` | 0.091 | [0.079, 0.105] | 0.000 |
| `mortality_365d` | `lr` - `ckpt:ar_s0` | 0.010 | [0.004, 0.019] | 0.000 |
| `mortality_365d` | `gbm` - `random_init` | 0.092 | [0.080, 0.106] | 0.000 |
| `mortality_365d` | `gbm` - `ckpt:ar_s0` | 0.012 | [0.005, 0.019] | 0.000 |
| `mortality_365d` | `random_init` - `ckpt:ar_s0` | -0.080 | [-0.095, -0.067] | 0.000 |
| `inpatient_365d` | `lr` - `gbm` | -0.016 | [-0.027, -0.006] | 0.000 |
| `inpatient_365d` | `lr` - `random_init` | 0.148 | [0.126, 0.174] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar_s0` | 0.018 | [-0.000, 0.035] | 0.060 |
| `inpatient_365d` | `gbm` - `random_init` | 0.165 | [0.141, 0.187] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s0` | 0.034 | [0.019, 0.052] | 0.000 |
| `inpatient_365d` | `random_init` - `ckpt:ar_s0` | -0.130 | [-0.153, -0.107] | 0.000 |
| `readmission_30d` | `lr` - `gbm` | -0.007 | [-0.026, 0.012] | 0.540 |
| `readmission_30d` | `lr` - `random_init` | 0.115 | [0.085, 0.148] | 0.000 |
| `readmission_30d` | `lr` - `ckpt:ar_s0` | 0.026 | [0.004, 0.046] | 0.010 |
| `readmission_30d` | `gbm` - `random_init` | 0.123 | [0.093, 0.153] | 0.000 |
| `readmission_30d` | `gbm` - `ckpt:ar_s0` | 0.033 | [0.009, 0.056] | 0.000 |
| `readmission_30d` | `random_init` - `ckpt:ar_s0` | -0.090 | [-0.123, -0.060] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.013 | [-0.024, -0.004] | 0.010 |
| `new_dx_365d/diabetes` | `lr` - `random_init` | 0.172 | [0.147, 0.198] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s0` | 0.033 | [0.018, 0.051] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `random_init` | 0.185 | [0.158, 0.212] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s0` | 0.046 | [0.030, 0.062] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar_s0` | -0.139 | [-0.165, -0.116] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | 0.004 | [-0.007, 0.013] | 0.500 |
| `new_dx_365d/heart_failure` | `lr` - `random_init` | 0.087 | [0.069, 0.108] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s0` | 0.014 | [0.004, 0.026] | 0.010 |
| `new_dx_365d/heart_failure` | `gbm` - `random_init` | 0.083 | [0.064, 0.102] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s0` | 0.011 | [-0.003, 0.025] | 0.150 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar_s0` | -0.073 | [-0.094, -0.054] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.032 | [-0.043, -0.021] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `random_init` | 0.108 | [0.086, 0.130] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s0` | 0.013 | [-0.002, 0.025] | 0.090 |
| `new_dx_365d/ckd` | `gbm` - `random_init` | 0.140 | [0.117, 0.163] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s0` | 0.045 | [0.030, 0.058] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar_s0` | -0.095 | [-0.122, -0.071] | 0.000 |
| `new_dx_365d/copd` | `lr` - `gbm` | -0.002 | [-0.022, 0.016] | 0.780 |
| `new_dx_365d/copd` | `lr` - `random_init` | 0.111 | [0.082, 0.145] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s0` | 0.036 | [0.013, 0.060] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `random_init` | 0.113 | [0.081, 0.149] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s0` | 0.038 | [0.014, 0.068] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar_s0` | -0.075 | [-0.105, -0.044] | 0.000 |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
