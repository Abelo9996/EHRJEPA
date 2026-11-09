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
| created | 2026-10-22T15:41:40+00:00 |
| runtime (s) | 2627.2 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `ckpt:ar_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s1/final.pt` | last@final |
| `ckpt:hybrid_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s1/final.pt` | last@final |
| `ckpt:ar_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/ar_s2/final.pt` | last@final |
| `ckpt:hybrid_s2` | probe | `/home/gaming_pc/EHRJEPA/runs/mimic-data25-1b/hybrid_s2/final.pt` | last@final |

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

| task | lr | gbm | ckpt:ar_s1 | ckpt:hybrid_s1 | ckpt:ar_s2 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|
| `mortality_365d` | 0.939 [0.925, 0.955] | 0.935 [0.923, 0.949] | 0.933 [0.917, 0.948] | 0.933 [0.916, 0.947] | 0.931 [0.915, 0.946] | 0.930 [0.914, 0.945] |
| `inpatient_365d` | 0.680 [0.653, 0.716] | 0.700 [0.670, 0.738] | 0.685 [0.656, 0.720] | 0.722 [0.691, 0.753] | 0.664 [0.634, 0.699] | 0.723 [0.689, 0.753] |
| `readmission_30d` | 0.626 [0.578, 0.680] | 0.617 [0.560, 0.673] | 0.619 [0.567, 0.665] | 0.623 [0.567, 0.672] | 0.612 [0.556, 0.657] | 0.647 [0.591, 0.697] |
| `new_dx_365d/diabetes` | 0.906 [0.878, 0.935] | 0.913 [0.879, 0.941] | 0.881 [0.841, 0.914] | 0.876 [0.841, 0.909] | 0.873 [0.834, 0.905] | 0.884 [0.848, 0.917] |
| `new_dx_365d/heart_failure` | 0.875 [0.842, 0.910] | 0.900 [0.868, 0.929] | 0.870 [0.826, 0.902] | 0.872 [0.835, 0.900] | 0.874 [0.835, 0.905] | 0.878 [0.843, 0.902] |
| `new_dx_365d/ckd` | 0.889 [0.863, 0.916] | 0.930 [0.913, 0.947] | 0.908 [0.881, 0.931] | 0.912 [0.885, 0.931] | 0.902 [0.871, 0.927] | 0.918 [0.895, 0.938] |
| `new_dx_365d/copd` | 0.796 [0.730, 0.864] | 0.789 [0.709, 0.866] | 0.752 [0.690, 0.826] | 0.751 [0.692, 0.818] | 0.763 [0.706, 0.831] | 0.734 [0.669, 0.803] |

## AUPRC

| task | lr | gbm | ckpt:ar_s1 | ckpt:hybrid_s1 | ckpt:ar_s2 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|
| `mortality_365d` | 0.794 [0.742, 0.840] | 0.788 [0.747, 0.837] | 0.760 [0.705, 0.810] | 0.750 [0.694, 0.797] | 0.745 [0.681, 0.796] | 0.738 [0.686, 0.791] |
| `inpatient_365d` | 0.581 [0.530, 0.631] | 0.585 [0.531, 0.632] | 0.592 [0.541, 0.637] | 0.638 [0.588, 0.678] | 0.555 [0.505, 0.606] | 0.637 [0.593, 0.683] |
| `readmission_30d` | 0.192 [0.144, 0.255] | 0.178 [0.133, 0.241] | 0.206 [0.153, 0.266] | 0.209 [0.152, 0.270] | 0.204 [0.149, 0.269] | 0.216 [0.156, 0.281] |
| `new_dx_365d/diabetes` | 0.645 [0.558, 0.722] | 0.730 [0.658, 0.796] | 0.592 [0.478, 0.682] | 0.597 [0.497, 0.677] | 0.579 [0.468, 0.660] | 0.623 [0.523, 0.715] |
| `new_dx_365d/heart_failure` | 0.499 [0.388, 0.576] | 0.536 [0.431, 0.627] | 0.427 [0.329, 0.511] | 0.440 [0.330, 0.524] | 0.468 [0.356, 0.551] | 0.413 [0.329, 0.512] |
| `new_dx_365d/ckd` | 0.438 [0.322, 0.578] | 0.549 [0.443, 0.663] | 0.443 [0.347, 0.570] | 0.488 [0.388, 0.604] | 0.430 [0.340, 0.551] | 0.501 [0.398, 0.612] |
| `new_dx_365d/copd` | 0.204 [0.127, 0.315] | 0.360 [0.231, 0.491] | 0.115 [0.076, 0.182] | 0.103 [0.071, 0.174] | 0.127 [0.080, 0.224] | 0.097 [0.065, 0.183] |

## BRIER

| task | lr | gbm | ckpt:ar_s1 | ckpt:hybrid_s1 | ckpt:ar_s2 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|
| `mortality_365d` | 0.0761 [0.0681, 0.0840] | 0.0759 [0.0654, 0.0856] | 0.0789 [0.0698, 0.0884] | 0.0795 [0.0705, 0.0891] | 0.0811 [0.0707, 0.0903] | 0.0812 [0.0724, 0.0904] |
| `inpatient_365d` | 0.2131 [0.2015, 0.2236] | 0.2106 [0.1981, 0.2233] | 0.2114 [0.1998, 0.2200] | 0.2008 [0.1887, 0.2107] | 0.2177 [0.2056, 0.2265] | 0.2006 [0.1891, 0.2118] |
| `readmission_30d` | 0.1038 [0.0897, 0.1178] | 0.1075 [0.0917, 0.1229] | 0.1047 [0.0908, 0.1190] | 0.1046 [0.0904, 0.1193] | 0.1051 [0.0909, 0.1197] | 0.1035 [0.0900, 0.1181] |
| `new_dx_365d/diabetes` | 0.0672 [0.0562, 0.0747] | 0.0583 [0.0456, 0.0678] | 0.0677 [0.0567, 0.0778] | 0.0672 [0.0558, 0.0766] | 0.0685 [0.0589, 0.0780] | 0.0657 [0.0543, 0.0749] |
| `new_dx_365d/heart_failure` | 0.0558 [0.0453, 0.0661] | 0.0528 [0.0427, 0.0642] | 0.0578 [0.0495, 0.0678] | 0.0588 [0.0495, 0.0690] | 0.0561 [0.0462, 0.0662] | 0.0588 [0.0493, 0.0698] |
| `new_dx_365d/ckd` | 0.0546 [0.0444, 0.0655] | 0.0475 [0.0377, 0.0584] | 0.0525 [0.0420, 0.0630] | 0.0510 [0.0408, 0.0606] | 0.0533 [0.0430, 0.0634] | 0.0503 [0.0402, 0.0603] |
| `new_dx_365d/copd` | 0.0369 [0.0279, 0.0470] | 0.0327 [0.0237, 0.0429] | 0.0387 [0.0292, 0.0490] | 0.0389 [0.0294, 0.0493] | 0.0385 [0.0292, 0.0484] | 0.0391 [0.0297, 0.0494] |

## CALIBRATION SLOPE

| task | lr | gbm | ckpt:ar_s1 | ckpt:hybrid_s1 | ckpt:ar_s2 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|
| `mortality_365d` | 1.557 [1.372, 1.840] | 0.983 [0.903, 1.121] | 1.241 [1.093, 1.425] | 1.244 [1.085, 1.418] | 1.203 [1.075, 1.367] | 1.238 [1.109, 1.429] |
| `inpatient_365d` | 0.898 [0.747, 1.108] | 0.791 [0.667, 0.961] | 0.943 [0.776, 1.130] | 0.938 [0.795, 1.091] | 0.812 [0.663, 1.014] | 0.932 [0.780, 1.108] |
| `readmission_30d` | 0.963 [0.604, 1.393] | 0.459 [0.216, 0.710] | 0.669 [0.377, 0.909] | 0.664 [0.370, 0.913] | 0.632 [0.341, 0.872] | 0.765 [0.486, 1.046] |
| `new_dx_365d/diabetes` | 1.922 [1.628, 2.310] | -- | 1.426 [1.184, 1.721] | 1.404 [1.178, 1.697] | 1.348 [1.121, 1.596] | 1.454 [1.225, 1.787] |
| `new_dx_365d/heart_failure` | 1.280 [1.083, 1.528] | 0.849 [0.733, 0.983] | 1.026 [0.845, 1.203] | 0.839 [0.695, 0.989] | 1.058 [0.866, 1.272] | 1.068 [0.899, 1.262] |
| `new_dx_365d/ckd` | 1.448 [1.206, 1.785] | 0.817 [0.705, 0.955] | 1.248 [1.038, 1.516] | 1.282 [1.087, 1.522] | 1.233 [1.035, 1.490] | 1.333 [1.106, 1.612] |
| `new_dx_365d/copd` | 1.277 [0.908, 1.778] | 0.793 [0.595, 1.041] | 0.871 [0.600, 1.278] | 0.854 [0.621, 1.178] | 0.903 [0.658, 1.295] | 0.823 [0.569, 1.171] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | 0.004 | [-0.004, 0.013] | 0.350 |
| `mortality_365d` | `lr` - `ckpt:ar_s1` | 0.007 | [-0.006, 0.021] | 0.350 |
| `mortality_365d` | `lr` - `ckpt:hybrid_s1` | 0.006 | [-0.005, 0.018] | 0.300 |
| `mortality_365d` | `lr` - `ckpt:ar_s2` | 0.008 | [-0.002, 0.019] | 0.250 |
| `mortality_365d` | `lr` - `ckpt:hybrid_s2` | 0.009 | [-0.003, 0.021] | 0.170 |
| `mortality_365d` | `gbm` - `ckpt:ar_s1` | 0.003 | [-0.009, 0.015] | 0.730 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_s1` | 0.002 | [-0.009, 0.016] | 0.740 |
| `mortality_365d` | `gbm` - `ckpt:ar_s2` | 0.004 | [-0.007, 0.018] | 0.550 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_s2` | 0.005 | [-0.007, 0.018] | 0.490 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.001 | [-0.009, 0.007] | 0.950 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.002 | [-0.007, 0.009] | 0.720 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | 0.002 | [-0.006, 0.010] | 0.550 |
| `mortality_365d` | `ckpt:hybrid_s1` - `ckpt:ar_s2` | 0.002 | [-0.008, 0.011] | 0.630 |
| `mortality_365d` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | 0.003 | [-0.005, 0.009] | 0.370 |
| `mortality_365d` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | 0.001 | [-0.008, 0.011] | 0.860 |
| `inpatient_365d` | `lr` - `gbm` | -0.020 | [-0.040, 0.001] | 0.080 |
| `inpatient_365d` | `lr` - `ckpt:ar_s1` | -0.005 | [-0.036, 0.021] | 0.730 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_s1` | -0.042 | [-0.075, -0.008] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar_s2` | 0.016 | [-0.018, 0.050] | 0.450 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_s2` | -0.043 | [-0.070, -0.011] | 0.010 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s1` | 0.015 | [-0.019, 0.044] | 0.450 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_s1` | -0.022 | [-0.054, 0.007] | 0.170 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s2` | 0.036 | [-0.001, 0.069] | 0.070 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_s2` | -0.023 | [-0.053, 0.008] | 0.150 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.037 | [-0.060, -0.010] | 0.000 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.021 | [-0.001, 0.042] | 0.080 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.038 | [-0.057, -0.013] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_s1` - `ckpt:ar_s2` | 0.058 | [0.033, 0.080] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.001 | [-0.015, 0.013] | 0.790 |
| `inpatient_365d` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.059 | [-0.084, -0.033] | 0.000 |
| `readmission_30d` | `lr` - `gbm` | 0.008 | [-0.027, 0.056] | 0.670 |
| `readmission_30d` | `lr` - `ckpt:ar_s1` | 0.006 | [-0.024, 0.044] | 0.650 |
| `readmission_30d` | `lr` - `ckpt:hybrid_s1` | 0.002 | [-0.026, 0.039] | 0.620 |
| `readmission_30d` | `lr` - `ckpt:ar_s2` | 0.014 | [-0.018, 0.057] | 0.400 |
| `readmission_30d` | `lr` - `ckpt:hybrid_s2` | -0.021 | [-0.052, 0.014] | 0.240 |
| `readmission_30d` | `gbm` - `ckpt:ar_s1` | -0.002 | [-0.042, 0.044] | 0.930 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_s1` | -0.006 | [-0.046, 0.039] | 0.950 |
| `readmission_30d` | `gbm` - `ckpt:ar_s2` | 0.005 | [-0.042, 0.057] | 0.730 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_s2` | -0.030 | [-0.073, 0.014] | 0.180 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.004 | [-0.027, 0.020] | 0.870 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.008 | [-0.021, 0.038] | 0.530 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.028 | [-0.053, -0.003] | 0.020 |
| `readmission_30d` | `ckpt:hybrid_s1` - `ckpt:ar_s2` | 0.011 | [-0.021, 0.038] | 0.480 |
| `readmission_30d` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.024 | [-0.042, -0.007] | 0.000 |
| `readmission_30d` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.035 | [-0.061, -0.003] | 0.040 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.007 | [-0.030, 0.013] | 0.570 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s1` | 0.025 | [-0.003, 0.051] | 0.070 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_s1` | 0.029 | [0.007, 0.052] | 0.020 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s2` | 0.033 | [0.006, 0.064] | 0.040 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_s2` | 0.021 | [-0.005, 0.049] | 0.090 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s1` | 0.032 | [-0.001, 0.064] | 0.070 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_s1` | 0.037 | [0.004, 0.073] | 0.020 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s2` | 0.040 | [0.008, 0.077] | 0.020 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_s2` | 0.029 | [0.001, 0.059] | 0.040 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | 0.005 | [-0.013, 0.026] | 0.730 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.009 | [-0.015, 0.031] | 0.430 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.003 | [-0.024, 0.017] | 0.800 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s1` - `ckpt:ar_s2` | 0.004 | [-0.017, 0.027] | 0.700 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.008 | [-0.026, 0.009] | 0.420 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.012 | [-0.038, 0.011] | 0.290 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | -0.024 | [-0.047, -0.007] | 0.030 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s1` | 0.005 | [-0.021, 0.035] | 0.770 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_s1` | 0.003 | [-0.023, 0.031] | 0.830 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s2` | 0.001 | [-0.022, 0.028] | 0.980 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_s2` | -0.003 | [-0.025, 0.019] | 0.960 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s1` | 0.029 | [-0.003, 0.068] | 0.070 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_s1` | 0.027 | [-0.001, 0.067] | 0.080 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s2` | 0.025 | [-0.009, 0.062] | 0.120 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_s2` | 0.022 | [0.001, 0.049] | 0.040 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.002 | [-0.030, 0.023] | 0.910 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.004 | [-0.024, 0.016] | 0.620 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.007 | [-0.032, 0.015] | 0.520 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s1` - `ckpt:ar_s2` | -0.002 | [-0.030, 0.020] | 0.750 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.006 | [-0.022, 0.013] | 0.680 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.003 | [-0.022, 0.015] | 0.840 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.040 | [-0.067, -0.015] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s1` | -0.019 | [-0.041, 0.006] | 0.140 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_s1` | -0.022 | [-0.046, 0.003] | 0.070 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s2` | -0.012 | [-0.031, 0.008] | 0.240 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_s2` | -0.029 | [-0.049, -0.006] | 0.020 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s1` | 0.022 | [-0.001, 0.048] | 0.070 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_s1` | 0.018 | [-0.003, 0.043] | 0.110 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s2` | 0.028 | [0.004, 0.053] | 0.010 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_s2` | 0.011 | [-0.012, 0.039] | 0.310 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.004 | [-0.022, 0.017] | 0.740 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.006 | [-0.010, 0.024] | 0.380 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.010 | [-0.027, 0.010] | 0.410 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s1` - `ckpt:ar_s2` | 0.010 | [-0.006, 0.024] | 0.290 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.007 | [-0.019, 0.006] | 0.300 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.017 | [-0.032, -0.000] | 0.050 |
| `new_dx_365d/copd` | `lr` - `gbm` | 0.007 | [-0.058, 0.062] | 0.770 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s1` | 0.044 | [-0.003, 0.087] | 0.090 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_s1` | 0.046 | [-0.000, 0.079] | 0.060 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s2` | 0.034 | [-0.015, 0.076] | 0.230 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_s2` | 0.062 | [0.016, 0.096] | 0.010 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s1` | 0.037 | [-0.035, 0.093] | 0.350 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_s1` | 0.038 | [-0.025, 0.100] | 0.400 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s2` | 0.026 | [-0.041, 0.086] | 0.560 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_s2` | 0.054 | [-0.013, 0.112] | 0.090 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | 0.001 | [-0.031, 0.031] | 0.890 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.011 | [-0.045, 0.026] | 0.510 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | 0.017 | [-0.025, 0.060] | 0.450 |
| `new_dx_365d/copd` | `ckpt:hybrid_s1` - `ckpt:ar_s2` | -0.012 | [-0.045, 0.029] | 0.570 |
| `new_dx_365d/copd` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | 0.016 | [-0.022, 0.049] | 0.390 |
| `new_dx_365d/copd` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | 0.028 | [-0.011, 0.063] | 0.190 |

## Skipped

| task | reason |
|---|---|
| `sepsis_6h` | not defined on source family 'mimic' |
| `sepsis_stay` | not defined on source family 'mimic' |
| `mortality_inhospital/24h` | not defined on source family 'mimic' |
| `mortality_inhospital/48h` | not defined on source family 'mimic' |
