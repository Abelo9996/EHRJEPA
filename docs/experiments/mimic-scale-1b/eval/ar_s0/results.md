# Downstream evaluation -- mimic-iv

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `mimic-iv` |
| MEDS | `data/meds/mimic-iv` |
| cache | `data/cache/mimic-iv` |
| tasks | `data/tasks/mimic-iv` |
| anchor seed | 20260903 |
| commit | `b35a09f` |
| created | 2026-10-04T17:06:33+00:00 |
| runtime (s) | 864.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/ar_s0/final.pt` | last@final |
| `ckpt:ar_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-scale-1b/ar_s0/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 9044 (0.2074) | 1184 (0.2095) | 1120 (0.1875) |  |
| `inpatient_365d` | 9044 (0.3638) | 1184 (0.3910) | 1120 (0.3839) |  |
| `readmission_30d` | 8472 (0.1327) | 1063 (0.1251) | 1038 (0.1204) |  |
| `new_dx_365d/diabetes` | 8130 (0.1000) | 1076 (0.1115) | 1025 (0.1093) |  |
| `new_dx_365d/heart_failure` | 8509 (0.0885) | 1128 (0.0878) | 1044 (0.0805) |  |
| `new_dx_365d/ckd` | 8463 (0.0740) | 1124 (0.0783) | 1039 (0.0751) |  |
| `new_dx_365d/copd` | 8697 (0.0455) | 1146 (0.0489) | 1082 (0.0416) |  |

## AUROC

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.939 [0.925, 0.955] | 0.935 [0.923, 0.949] | 0.847 [0.818, 0.874] | 0.918 [0.904, 0.934] |
| `inpatient_365d` | 0.680 [0.653, 0.716] | 0.700 [0.670, 0.738] | 0.511 [0.478, 0.541] | 0.659 [0.623, 0.696] |
| `readmission_30d` | 0.626 [0.578, 0.680] | 0.617 [0.560, 0.673] | 0.557 [0.504, 0.610] | 0.599 [0.554, 0.643] |
| `new_dx_365d/diabetes` | 0.906 [0.878, 0.935] | 0.913 [0.879, 0.941] | 0.691 [0.644, 0.738] | 0.848 [0.807, 0.886] |
| `new_dx_365d/heart_failure` | 0.875 [0.842, 0.910] | 0.900 [0.868, 0.929] | 0.805 [0.764, 0.846] | 0.865 [0.827, 0.896] |
| `new_dx_365d/ckd` | 0.889 [0.863, 0.916] | 0.930 [0.913, 0.947] | 0.755 [0.708, 0.804] | 0.896 [0.869, 0.926] |
| `new_dx_365d/copd` | 0.796 [0.730, 0.864] | 0.789 [0.709, 0.866] | 0.679 [0.615, 0.762] | 0.762 [0.706, 0.830] |

## AUPRC

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.794 [0.742, 0.840] | 0.788 [0.747, 0.837] | 0.580 [0.506, 0.642] | 0.710 [0.656, 0.763] |
| `inpatient_365d` | 0.581 [0.530, 0.631] | 0.585 [0.531, 0.632] | 0.392 [0.351, 0.436] | 0.553 [0.506, 0.601] |
| `readmission_30d` | 0.192 [0.144, 0.255] | 0.178 [0.133, 0.241] | 0.145 [0.120, 0.183] | 0.183 [0.132, 0.244] |
| `new_dx_365d/diabetes` | 0.645 [0.558, 0.722] | 0.730 [0.658, 0.796] | 0.187 [0.143, 0.239] | 0.494 [0.397, 0.594] |
| `new_dx_365d/heart_failure` | 0.499 [0.388, 0.576] | 0.536 [0.431, 0.627] | 0.303 [0.225, 0.400] | 0.411 [0.300, 0.488] |
| `new_dx_365d/ckd` | 0.438 [0.322, 0.578] | 0.549 [0.443, 0.663] | 0.210 [0.156, 0.309] | 0.443 [0.331, 0.563] |
| `new_dx_365d/copd` | 0.204 [0.127, 0.315] | 0.360 [0.231, 0.491] | 0.070 [0.050, 0.105] | 0.117 [0.073, 0.197] |

## BRIER

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.0761 [0.0681, 0.0840] | 0.0759 [0.0654, 0.0856] | 0.1126 [0.1011, 0.1240] | 0.0879 [0.0790, 0.0972] |
| `inpatient_365d` | 0.2131 [0.2015, 0.2236] | 0.2106 [0.1981, 0.2233] | 0.2409 [0.2318, 0.2479] | 0.2183 [0.2066, 0.2277] |
| `readmission_30d` | 0.1038 [0.0897, 0.1178] | 0.1075 [0.0917, 0.1229] | 0.1060 [0.0913, 0.1222] | 0.1073 [0.0936, 0.1213] |
| `new_dx_365d/diabetes` | 0.0672 [0.0562, 0.0747] | 0.0583 [0.0456, 0.0678] | 0.0939 [0.0791, 0.1048] | 0.0755 [0.0638, 0.0847] |
| `new_dx_365d/heart_failure` | 0.0558 [0.0453, 0.0661] | 0.0528 [0.0427, 0.0642] | 0.0651 [0.0543, 0.0748] | 0.0590 [0.0491, 0.0690] |
| `new_dx_365d/ckd` | 0.0546 [0.0444, 0.0655] | 0.0475 [0.0377, 0.0584] | 0.0646 [0.0516, 0.0769] | 0.0533 [0.0432, 0.0635] |
| `new_dx_365d/copd` | 0.0369 [0.0279, 0.0470] | 0.0327 [0.0237, 0.0429] | 0.0420 [0.0328, 0.0524] | 0.0386 [0.0297, 0.0492] |

## CALIBRATION SLOPE

| task | lr | gbm | random_init | ckpt:ar_s0 |
|---|---|---|---|---|
| `mortality_365d` | 1.557 [1.372, 1.840] | 0.983 [0.903, 1.121] | 1.030 [0.901, 1.173] | 1.180 [1.048, 1.343] |
| `inpatient_365d` | 0.898 [0.747, 1.108] | 0.791 [0.667, 0.961] | 0.142 [-0.177, 0.464] | 0.879 [0.663, 1.080] |
| `readmission_30d` | 0.963 [0.604, 1.393] | 0.459 [0.216, 0.710] | 0.553 [0.045, 1.132] | 0.501 [0.296, 0.707] |
| `new_dx_365d/diabetes` | 1.922 [1.628, 2.310] | -- | 0.847 [0.598, 1.141] | 1.339 [1.085, 1.610] |
| `new_dx_365d/heart_failure` | 1.280 [1.083, 1.528] | 0.849 [0.733, 0.983] | 0.918 [0.741, 1.122] | 1.027 [0.852, 1.199] |
| `new_dx_365d/ckd` | 1.448 [1.206, 1.785] | 0.817 [0.705, 0.955] | 1.015 [0.787, 1.321] | 1.303 [1.093, 1.571] |
| `new_dx_365d/copd` | 1.277 [0.908, 1.778] | 0.793 [0.595, 1.041] | 0.390 [0.236, 0.619] | 0.935 [0.690, 1.298] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | 0.004 | [-0.004, 0.013] | 0.350 |
| `mortality_365d` | `lr` - `random_init` | 0.092 | [0.071, 0.118] | 0.000 |
| `mortality_365d` | `lr` - `ckpt:ar_s0` | 0.021 | [0.006, 0.033] | 0.000 |
| `mortality_365d` | `gbm` - `random_init` | 0.088 | [0.067, 0.115] | 0.000 |
| `mortality_365d` | `gbm` - `ckpt:ar_s0` | 0.017 | [0.005, 0.030] | 0.000 |
| `mortality_365d` | `random_init` - `ckpt:ar_s0` | -0.071 | [-0.096, -0.049] | 0.000 |
| `inpatient_365d` | `lr` - `gbm` | -0.020 | [-0.040, 0.001] | 0.080 |
| `inpatient_365d` | `lr` - `random_init` | 0.169 | [0.132, 0.214] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar_s0` | 0.021 | [-0.012, 0.053] | 0.250 |
| `inpatient_365d` | `gbm` - `random_init` | 0.190 | [0.151, 0.236] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s0` | 0.041 | [0.010, 0.078] | 0.020 |
| `inpatient_365d` | `random_init` - `ckpt:ar_s0` | -0.149 | [-0.189, -0.111] | 0.000 |
| `readmission_30d` | `lr` - `gbm` | 0.008 | [-0.027, 0.056] | 0.670 |
| `readmission_30d` | `lr` - `random_init` | 0.069 | [0.001, 0.137] | 0.050 |
| `readmission_30d` | `lr` - `ckpt:ar_s0` | 0.027 | [-0.019, 0.071] | 0.200 |
| `readmission_30d` | `gbm` - `random_init` | 0.060 | [-0.013, 0.131] | 0.150 |
| `readmission_30d` | `gbm` - `ckpt:ar_s0` | 0.018 | [-0.031, 0.069] | 0.430 |
| `readmission_30d` | `random_init` - `ckpt:ar_s0` | -0.042 | [-0.105, 0.024] | 0.230 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.007 | [-0.030, 0.013] | 0.570 |
| `new_dx_365d/diabetes` | `lr` - `random_init` | 0.215 | [0.160, 0.263] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s0` | 0.057 | [0.024, 0.091] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `random_init` | 0.222 | [0.164, 0.278] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s0` | 0.064 | [0.027, 0.108] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:ar_s0` | -0.158 | [-0.215, -0.097] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | -0.024 | [-0.047, -0.007] | 0.030 |
| `new_dx_365d/heart_failure` | `lr` - `random_init` | 0.070 | [0.034, 0.111] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s0` | 0.010 | [-0.012, 0.036] | 0.490 |
| `new_dx_365d/heart_failure` | `gbm` - `random_init` | 0.094 | [0.049, 0.136] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s0` | 0.035 | [0.009, 0.067] | 0.020 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:ar_s0` | -0.060 | [-0.105, -0.022] | 0.020 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.040 | [-0.067, -0.015] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `random_init` | 0.134 | [0.079, 0.186] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s0` | -0.007 | [-0.035, 0.023] | 0.720 |
| `new_dx_365d/ckd` | `gbm` - `random_init` | 0.174 | [0.127, 0.219] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s0` | 0.033 | [0.003, 0.063] | 0.010 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:ar_s0` | -0.141 | [-0.198, -0.082] | 0.000 |
| `new_dx_365d/copd` | `lr` - `gbm` | 0.007 | [-0.058, 0.062] | 0.770 |
| `new_dx_365d/copd` | `lr` - `random_init` | 0.117 | [0.050, 0.167] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s0` | 0.034 | [-0.026, 0.090] | 0.300 |
| `new_dx_365d/copd` | `gbm` - `random_init` | 0.110 | [0.034, 0.178] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s0` | 0.027 | [-0.047, 0.093] | 0.630 |
| `new_dx_365d/copd` | `random_init` - `ckpt:ar_s0` | -0.083 | [-0.145, -0.032] | 0.000 |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
