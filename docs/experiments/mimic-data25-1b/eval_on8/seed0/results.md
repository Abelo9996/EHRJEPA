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
| created | 2026-10-08T20:14:50+00:00 |
| runtime (s) | 2507.5 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `ckpt:ar_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s0/final.pt` | last@final |
| `ckpt:hybrid_s0` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s0/final.pt` | last@final |

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

| task | lr | gbm | ckpt:ar_s0 | ckpt:hybrid_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.939 [0.925, 0.955] | 0.935 [0.923, 0.949] | 0.933 [0.918, 0.948] | 0.937 [0.921, 0.951] |
| `inpatient_365d` | 0.680 [0.653, 0.716] | 0.700 [0.670, 0.738] | 0.667 [0.630, 0.706] | 0.727 [0.691, 0.760] |
| `readmission_30d` | 0.626 [0.578, 0.680] | 0.617 [0.560, 0.673] | 0.607 [0.552, 0.656] | 0.633 [0.578, 0.681] |
| `new_dx_365d/diabetes` | 0.906 [0.878, 0.935] | 0.913 [0.879, 0.941] | 0.879 [0.837, 0.914] | 0.890 [0.857, 0.922] |
| `new_dx_365d/heart_failure` | 0.875 [0.842, 0.910] | 0.900 [0.868, 0.929] | 0.879 [0.840, 0.906] | 0.879 [0.844, 0.903] |
| `new_dx_365d/ckd` | 0.889 [0.863, 0.916] | 0.930 [0.913, 0.947] | 0.916 [0.890, 0.939] | 0.908 [0.875, 0.933] |
| `new_dx_365d/copd` | 0.796 [0.730, 0.864] | 0.789 [0.709, 0.866] | 0.763 [0.706, 0.828] | 0.745 [0.691, 0.809] |

## AUPRC

| task | lr | gbm | ckpt:ar_s0 | ckpt:hybrid_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.794 [0.742, 0.840] | 0.788 [0.747, 0.837] | 0.742 [0.686, 0.793] | 0.774 [0.723, 0.822] |
| `inpatient_365d` | 0.581 [0.530, 0.631] | 0.585 [0.531, 0.632] | 0.564 [0.509, 0.611] | 0.638 [0.591, 0.684] |
| `readmission_30d` | 0.192 [0.144, 0.255] | 0.178 [0.133, 0.241] | 0.188 [0.136, 0.246] | 0.201 [0.147, 0.274] |
| `new_dx_365d/diabetes` | 0.645 [0.558, 0.722] | 0.730 [0.658, 0.796] | 0.612 [0.510, 0.689] | 0.648 [0.550, 0.728] |
| `new_dx_365d/heart_failure` | 0.499 [0.388, 0.576] | 0.536 [0.431, 0.627] | 0.459 [0.346, 0.546] | 0.441 [0.343, 0.516] |
| `new_dx_365d/ckd` | 0.438 [0.322, 0.578] | 0.549 [0.443, 0.663] | 0.472 [0.364, 0.612] | 0.491 [0.371, 0.616] |
| `new_dx_365d/copd` | 0.204 [0.127, 0.315] | 0.360 [0.231, 0.491] | 0.111 [0.074, 0.174] | 0.137 [0.073, 0.225] |

## BRIER

| task | lr | gbm | ckpt:ar_s0 | ckpt:hybrid_s0 |
|---|---|---|---|---|
| `mortality_365d` | 0.0761 [0.0681, 0.0840] | 0.0759 [0.0654, 0.0856] | 0.0798 [0.0715, 0.0879] | 0.0769 [0.0680, 0.0864] |
| `inpatient_365d` | 0.2131 [0.2015, 0.2236] | 0.2106 [0.1981, 0.2233] | 0.2167 [0.2042, 0.2270] | 0.2000 [0.1884, 0.2106] |
| `readmission_30d` | 0.1038 [0.0897, 0.1178] | 0.1075 [0.0917, 0.1229] | 0.1056 [0.0909, 0.1194] | 0.1050 [0.0912, 0.1190] |
| `new_dx_365d/diabetes` | 0.0672 [0.0562, 0.0747] | 0.0583 [0.0456, 0.0678] | 0.0669 [0.0565, 0.0763] | 0.0628 [0.0525, 0.0715] |
| `new_dx_365d/heart_failure` | 0.0558 [0.0453, 0.0661] | 0.0528 [0.0427, 0.0642] | 0.0568 [0.0481, 0.0670] | 0.0580 [0.0489, 0.0682] |
| `new_dx_365d/ckd` | 0.0546 [0.0444, 0.0655] | 0.0475 [0.0377, 0.0584] | 0.0513 [0.0413, 0.0609] | 0.0509 [0.0402, 0.0610] |
| `new_dx_365d/copd` | 0.0369 [0.0279, 0.0470] | 0.0327 [0.0237, 0.0429] | 0.0387 [0.0293, 0.0493] | 0.0388 [0.0292, 0.0489] |

## CALIBRATION SLOPE

| task | lr | gbm | ckpt:ar_s0 | ckpt:hybrid_s0 |
|---|---|---|---|---|
| `mortality_365d` | 1.557 [1.372, 1.840] | 0.983 [0.903, 1.121] | 1.215 [1.072, 1.389] | 1.344 [1.175, 1.556] |
| `inpatient_365d` | 0.898 [0.747, 1.108] | 0.791 [0.667, 0.961] | 0.845 [0.675, 1.069] | 0.928 [0.775, 1.110] |
| `readmission_30d` | 0.963 [0.604, 1.393] | 0.459 [0.216, 0.710] | 0.598 [0.275, 0.862] | 0.647 [0.416, 0.877] |
| `new_dx_365d/diabetes` | 1.922 [1.628, 2.310] | -- | 1.425 [1.159, 1.720] | 1.428 [1.227, 1.702] |
| `new_dx_365d/heart_failure` | 1.280 [1.083, 1.528] | 0.849 [0.733, 0.983] | 1.052 [0.863, 1.251] | 1.098 [0.905, 1.295] |
| `new_dx_365d/ckd` | 1.448 [1.206, 1.785] | 0.817 [0.705, 0.955] | 1.271 [1.087, 1.537] | 1.234 [1.009, 1.545] |
| `new_dx_365d/copd` | 1.277 [0.908, 1.778] | 0.793 [0.595, 1.041] | 0.909 [0.667, 1.233] | 0.855 [0.618, 1.180] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | 0.004 | [-0.004, 0.013] | 0.350 |
| `mortality_365d` | `lr` - `ckpt:ar_s0` | 0.006 | [-0.006, 0.018] | 0.360 |
| `mortality_365d` | `lr` - `ckpt:hybrid_s0` | 0.002 | [-0.009, 0.016] | 0.830 |
| `mortality_365d` | `gbm` - `ckpt:ar_s0` | 0.002 | [-0.010, 0.017] | 0.750 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_s0` | -0.002 | [-0.013, 0.012] | 0.770 |
| `mortality_365d` | `ckpt:ar_s0` - `ckpt:hybrid_s0` | -0.004 | [-0.013, 0.005] | 0.390 |
| `inpatient_365d` | `lr` - `gbm` | -0.020 | [-0.040, 0.001] | 0.080 |
| `inpatient_365d` | `lr` - `ckpt:ar_s0` | 0.013 | [-0.021, 0.043] | 0.360 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_s0` | -0.047 | [-0.077, -0.016] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s0` | 0.033 | [0.001, 0.072] | 0.050 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_s0` | -0.027 | [-0.057, 0.003] | 0.110 |
| `inpatient_365d` | `ckpt:ar_s0` - `ckpt:hybrid_s0` | -0.060 | [-0.087, -0.034] | 0.000 |
| `readmission_30d` | `lr` - `gbm` | 0.008 | [-0.027, 0.056] | 0.670 |
| `readmission_30d` | `lr` - `ckpt:ar_s0` | 0.018 | [-0.017, 0.063] | 0.260 |
| `readmission_30d` | `lr` - `ckpt:hybrid_s0` | -0.008 | [-0.043, 0.030] | 0.770 |
| `readmission_30d` | `gbm` - `ckpt:ar_s0` | 0.010 | [-0.029, 0.060] | 0.650 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_s0` | -0.016 | [-0.053, 0.028] | 0.550 |
| `readmission_30d` | `ckpt:ar_s0` - `ckpt:hybrid_s0` | -0.026 | [-0.058, 0.005] | 0.090 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.007 | [-0.030, 0.013] | 0.570 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s0` | 0.026 | [-0.008, 0.058] | 0.080 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_s0` | 0.016 | [-0.008, 0.040] | 0.180 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s0` | 0.034 | [0.000, 0.076] | 0.050 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_s0` | 0.023 | [-0.007, 0.054] | 0.150 |
| `new_dx_365d/diabetes` | `ckpt:ar_s0` - `ckpt:hybrid_s0` | -0.011 | [-0.035, 0.016] | 0.300 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | -0.024 | [-0.047, -0.007] | 0.030 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s0` | -0.004 | [-0.027, 0.026] | 0.820 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_s0` | -0.004 | [-0.025, 0.018] | 0.800 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s0` | 0.021 | [-0.008, 0.056] | 0.220 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_s0` | 0.020 | [-0.004, 0.047] | 0.090 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s0` - `ckpt:hybrid_s0` | -0.000 | [-0.019, 0.019] | 0.990 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.040 | [-0.067, -0.015] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s0` | -0.027 | [-0.045, -0.004] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_s0` | -0.019 | [-0.042, 0.005] | 0.150 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s0` | 0.013 | [-0.007, 0.036] | 0.230 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_s0` | 0.021 | [-0.002, 0.056] | 0.130 |
| `new_dx_365d/ckd` | `ckpt:ar_s0` - `ckpt:hybrid_s0` | 0.008 | [-0.011, 0.028] | 0.370 |
| `new_dx_365d/copd` | `lr` - `gbm` | 0.007 | [-0.058, 0.062] | 0.770 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s0` | 0.033 | [-0.016, 0.078] | 0.210 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_s0` | 0.051 | [0.005, 0.085] | 0.040 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s0` | 0.026 | [-0.042, 0.088] | 0.580 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_s0` | 0.044 | [-0.017, 0.098] | 0.210 |
| `new_dx_365d/copd` | `ckpt:ar_s0` - `ckpt:hybrid_s0` | 0.018 | [-0.009, 0.049] | 0.210 |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
