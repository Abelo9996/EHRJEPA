# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `d64d7ab` |
| created | 2026-09-04T18:58:35+00:00 |
| runtime (s) | 422.6 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `ckpt:ar` | probe | `runs/2026-09-03-pilot-desynpuf/ar/final.pt` | last@final |
| `ckpt:ar_s1` | probe | `runs/2026-09-04-pilot5-seeds-desynpuf/ar_s1/final.pt` | last@final |
| `ckpt:ar_s2` | probe | `runs/2026-09-04-pilot5-seeds-desynpuf/ar_s2/final.pt` | last@final |
| `ckpt:nextlatent_h1416_recon` | probe | `runs/2026-09-04-pilot4-desynpuf/nextlatent_h1416_recon/final.pt` | last@final |
| `ckpt:hybrid_s1` | probe | `runs/2026-09-04-pilot5-seeds-desynpuf/hybrid_s1/final.pt` | last@final |
| `ckpt:hybrid_s2` | probe | `runs/2026-09-04-pilot5-seeds-desynpuf/hybrid_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 3000 (0.0220) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 3000 (0.2010) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3000 (0.0493) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 3000 (0.2240) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 3000 (0.1370) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 3000 (0.0970) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 3000 (0.1303) |  |

## AUROC

| task | lr | gbm | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 | ckpt:nextlatent_h1416_recon | ckpt:hybrid_s1 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 0.554 [0.483, 0.608] | 0.572 [0.481, 0.636] | 0.595 [0.521, 0.677] | 0.609 [0.545, 0.679] | 0.606 [0.532, 0.689] | 0.633 [0.557, 0.713] | 0.607 [0.545, 0.679] | 0.625 [0.558, 0.697] |
| `inpatient_365d` | 0.708 [0.683, 0.727] | 0.744 [0.719, 0.764] | 0.740 [0.718, 0.760] | 0.739 [0.717, 0.759] | 0.746 [0.724, 0.765] | 0.742 [0.718, 0.761] | 0.748 [0.726, 0.769] | 0.744 [0.723, 0.763] |
| `readmission_30d` | 0.658 [0.610, 0.708] | 0.672 [0.624, 0.717] | 0.674 [0.631, 0.718] | 0.689 [0.648, 0.735] | 0.684 [0.646, 0.727] | 0.699 [0.661, 0.736] | 0.700 [0.662, 0.741] | 0.693 [0.656, 0.735] |
| `new_dx_365d/diabetes` | 0.740 [0.721, 0.760] | 0.772 [0.755, 0.792] | 0.763 [0.745, 0.784] | 0.757 [0.741, 0.780] | 0.765 [0.748, 0.786] | 0.762 [0.746, 0.783] | 0.762 [0.745, 0.783] | 0.769 [0.753, 0.791] |
| `new_dx_365d/heart_failure` | 0.744 [0.722, 0.766] | 0.789 [0.773, 0.810] | 0.771 [0.749, 0.794] | 0.756 [0.735, 0.779] | 0.770 [0.748, 0.789] | 0.764 [0.746, 0.786] | 0.758 [0.738, 0.780] | 0.768 [0.750, 0.791] |
| `new_dx_365d/ckd` | 0.736 [0.715, 0.765] | 0.765 [0.741, 0.789] | 0.751 [0.726, 0.779] | 0.742 [0.716, 0.771] | 0.751 [0.725, 0.774] | 0.753 [0.731, 0.777] | 0.754 [0.731, 0.781] | 0.750 [0.726, 0.774] |
| `new_dx_365d/copd` | 0.726 [0.696, 0.751] | 0.767 [0.738, 0.791] | 0.761 [0.738, 0.785] | 0.759 [0.730, 0.780] | 0.761 [0.737, 0.781] | 0.749 [0.726, 0.771] | 0.754 [0.729, 0.777] | 0.751 [0.729, 0.775] |

## AUPRC

| task | lr | gbm | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 | ckpt:nextlatent_h1416_recon | ckpt:hybrid_s1 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 0.026 [0.019, 0.043] | 0.028 [0.021, 0.042] | 0.034 [0.024, 0.051] | 0.031 [0.023, 0.045] | 0.034 [0.024, 0.055] | 0.035 [0.025, 0.055] | 0.030 [0.023, 0.047] | 0.040 [0.024, 0.086] |
| `inpatient_365d` | 0.363 [0.329, 0.405] | 0.411 [0.369, 0.453] | 0.402 [0.361, 0.442] | 0.388 [0.355, 0.429] | 0.408 [0.370, 0.451] | 0.394 [0.356, 0.435] | 0.413 [0.376, 0.453] | 0.405 [0.369, 0.443] |
| `readmission_30d` | 0.098 [0.076, 0.140] | 0.107 [0.082, 0.145] | 0.106 [0.076, 0.150] | 0.113 [0.086, 0.160] | 0.121 [0.093, 0.174] | 0.120 [0.089, 0.177] | 0.109 [0.084, 0.149] | 0.126 [0.095, 0.188] |
| `new_dx_365d/diabetes` | 0.432 [0.393, 0.471] | 0.469 [0.432, 0.510] | 0.469 [0.434, 0.510] | 0.456 [0.418, 0.499] | 0.464 [0.430, 0.507] | 0.454 [0.413, 0.494] | 0.457 [0.420, 0.500] | 0.472 [0.436, 0.514] |
| `new_dx_365d/heart_failure` | 0.276 [0.243, 0.313] | 0.344 [0.303, 0.396] | 0.308 [0.266, 0.358] | 0.301 [0.261, 0.345] | 0.307 [0.271, 0.351] | 0.306 [0.266, 0.346] | 0.308 [0.268, 0.358] | 0.314 [0.269, 0.360] |
| `new_dx_365d/ckd` | 0.236 [0.200, 0.279] | 0.259 [0.223, 0.308] | 0.244 [0.207, 0.286] | 0.246 [0.207, 0.302] | 0.246 [0.204, 0.293] | 0.234 [0.198, 0.283] | 0.248 [0.209, 0.297] | 0.247 [0.209, 0.295] |
| `new_dx_365d/copd` | 0.317 [0.270, 0.363] | 0.344 [0.296, 0.393] | 0.298 [0.254, 0.347] | 0.306 [0.262, 0.352] | 0.314 [0.267, 0.359] | 0.293 [0.253, 0.338] | 0.296 [0.251, 0.337] | 0.275 [0.238, 0.315] |

## BRIER

| task | lr | gbm | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 | ckpt:nextlatent_h1416_recon | ckpt:hybrid_s1 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] | 0.0216 [0.0173, 0.0266] | 0.0215 [0.0173, 0.0265] | 0.0215 [0.0173, 0.0265] | 0.0215 [0.0172, 0.0265] | 0.0215 [0.0173, 0.0265] | 0.0215 [0.0173, 0.0266] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1474 [0.1406, 0.1554] | 0.1406 [0.1342, 0.1468] | 0.1419 [0.1350, 0.1484] | 0.1425 [0.1358, 0.1493] | 0.1410 [0.1336, 0.1477] | 0.1421 [0.1349, 0.1486] | 0.1405 [0.1336, 0.1473] | 0.1413 [0.1343, 0.1478] |
| `readmission_30d` | 0.0461 [0.0403, 0.0520] | 0.0461 [0.0399, 0.0522] | 0.0460 [0.0402, 0.0517] | 0.0458 [0.0400, 0.0514] | 0.0456 [0.0398, 0.0512] | 0.0457 [0.0397, 0.0515] | 0.0457 [0.0398, 0.0513] | 0.0456 [0.0398, 0.0511] |
| `new_dx_365d/diabetes` | 0.1529 [0.1459, 0.1615] | 0.1458 [0.1387, 0.1530] | 0.1472 [0.1403, 0.1540] | 0.1485 [0.1415, 0.1553] | 0.1471 [0.1405, 0.1535] | 0.1481 [0.1415, 0.1542] | 0.1479 [0.1411, 0.1544] | 0.1462 [0.1394, 0.1530] |
| `new_dx_365d/heart_failure` | 0.1101 [0.1016, 0.1168] | 0.1028 [0.0947, 0.1094] | 0.1056 [0.0981, 0.1121] | 0.1068 [0.0988, 0.1132] | 0.1058 [0.0979, 0.1122] | 0.1061 [0.0984, 0.1124] | 0.1064 [0.0992, 0.1130] | 0.1056 [0.0978, 0.1120] |
| `new_dx_365d/ckd` | 0.0818 [0.0739, 0.0888] | 0.0799 [0.0719, 0.0867] | 0.0808 [0.0730, 0.0871] | 0.0810 [0.0728, 0.0878] | 0.0807 [0.0730, 0.0873] | 0.0811 [0.0733, 0.0876] | 0.0806 [0.0727, 0.0871] | 0.0806 [0.0730, 0.0873] |
| `new_dx_365d/copd` | 0.1038 [0.0962, 0.1117] | 0.0988 [0.0910, 0.1068] | 0.1020 [0.0948, 0.1100] | 0.1018 [0.0936, 0.1094] | 0.1016 [0.0939, 0.1096] | 0.1030 [0.0953, 0.1106] | 0.1025 [0.0953, 0.1099] | 0.1036 [0.0958, 0.1111] |

## CALIBRATION SLOPE

| task | lr | gbm | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 | ckpt:nextlatent_h1416_recon | ckpt:hybrid_s1 | ckpt:hybrid_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 2.421 [-2.027, 5.716] | 0.425 [-0.285, 0.861] | 0.789 [0.180, 1.465] | 0.679 [0.273, 1.110] | 0.813 [0.243, 1.452] | 0.782 [0.310, 1.275] | 0.615 [0.262, 1.070] | 0.777 [0.361, 1.187] |
| `inpatient_365d` | 1.154 [1.006, 1.290] | 1.063 [0.931, 1.194] | 1.110 [0.973, 1.253] | 1.056 [0.920, 1.194] | 1.138 [1.012, 1.294] | 1.076 [0.937, 1.228] | 1.106 [0.971, 1.254] | 1.091 [0.956, 1.237] |
| `readmission_30d` | 1.062 [0.756, 1.393] | 0.741 [0.523, 0.941] | 0.935 [0.685, 1.191] | 1.036 [0.818, 1.314] | 1.096 [0.846, 1.384] | 1.000 [0.808, 1.225] | 1.033 [0.829, 1.276] | 1.046 [0.829, 1.318] |
| `new_dx_365d/diabetes` | 1.032 [0.924, 1.138] | 1.026 [0.933, 1.146] | 1.041 [0.937, 1.173] | 1.003 [0.898, 1.137] | 1.017 [0.910, 1.148] | 0.958 [0.868, 1.083] | 0.974 [0.872, 1.108] | 0.996 [0.901, 1.131] |
| `new_dx_365d/heart_failure` | 1.302 [1.176, 1.474] | 1.055 [0.961, 1.188] | 1.062 [0.951, 1.212] | 1.031 [0.913, 1.179] | 1.042 [0.930, 1.163] | 1.024 [0.923, 1.157] | 1.023 [0.923, 1.154] | 1.031 [0.921, 1.165] |
| `new_dx_365d/ckd` | 1.032 [0.914, 1.186] | 0.972 [0.875, 1.099] | 0.948 [0.822, 1.102] | 0.942 [0.809, 1.122] | 1.007 [0.869, 1.178] | 0.960 [0.843, 1.108] | 0.962 [0.848, 1.119] | 0.961 [0.839, 1.116] |
| `new_dx_365d/copd` | 1.444 [1.263, 1.611] | 1.011 [0.879, 1.124] | 1.047 [0.913, 1.203] | 0.998 [0.855, 1.115] | 1.099 [0.956, 1.259] | 0.994 [0.855, 1.139] | 1.015 [0.863, 1.155] | 1.013 [0.861, 1.171] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | -0.019 | [-0.085, 0.062] | 0.680 |
| `mortality_365d` | `lr` - `ckpt:ar` | -0.041 | [-0.141, 0.048] | 0.390 |
| `mortality_365d` | `lr` - `ckpt:ar_s1` | -0.056 | [-0.142, 0.022] | 0.170 |
| `mortality_365d` | `lr` - `ckpt:ar_s2` | -0.052 | [-0.153, 0.033] | 0.270 |
| `mortality_365d` | `lr` - `ckpt:nextlatent_h1416_recon` | -0.079 | [-0.177, -0.001] | 0.050 |
| `mortality_365d` | `lr` - `ckpt:hybrid_s1` | -0.053 | [-0.143, 0.025] | 0.190 |
| `mortality_365d` | `lr` - `ckpt:hybrid_s2` | -0.072 | [-0.158, 0.008] | 0.110 |
| `mortality_365d` | `gbm` - `ckpt:ar` | -0.023 | [-0.109, 0.049] | 0.570 |
| `mortality_365d` | `gbm` - `ckpt:ar_s1` | -0.037 | [-0.120, 0.044] | 0.320 |
| `mortality_365d` | `gbm` - `ckpt:ar_s2` | -0.033 | [-0.117, 0.029] | 0.320 |
| `mortality_365d` | `gbm` - `ckpt:nextlatent_h1416_recon` | -0.061 | [-0.142, 0.008] | 0.110 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_s1` | -0.035 | [-0.111, 0.031] | 0.290 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_s2` | -0.053 | [-0.147, 0.024] | 0.180 |
| `mortality_365d` | `ckpt:ar` - `ckpt:ar_s1` | -0.014 | [-0.054, 0.026] | 0.620 |
| `mortality_365d` | `ckpt:ar` - `ckpt:ar_s2` | -0.011 | [-0.045, 0.019] | 0.630 |
| `mortality_365d` | `ckpt:ar` - `ckpt:nextlatent_h1416_recon` | -0.038 | [-0.075, -0.003] | 0.040 |
| `mortality_365d` | `ckpt:ar` - `ckpt:hybrid_s1` | -0.012 | [-0.051, 0.030] | 0.560 |
| `mortality_365d` | `ckpt:ar` - `ckpt:hybrid_s2` | -0.030 | [-0.065, 0.011] | 0.140 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.004 | [-0.052, 0.050] | 0.940 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:nextlatent_h1416_recon` | -0.024 | [-0.063, 0.014] | 0.230 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | 0.002 | [-0.029, 0.034] | 0.910 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.016 | [-0.054, 0.022] | 0.280 |
| `mortality_365d` | `ckpt:ar_s2` - `ckpt:nextlatent_h1416_recon` | -0.027 | [-0.066, 0.010] | 0.170 |
| `mortality_365d` | `ckpt:ar_s2` - `ckpt:hybrid_s1` | -0.001 | [-0.036, 0.035] | 1.000 |
| `mortality_365d` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.020 | [-0.059, 0.022] | 0.260 |
| `mortality_365d` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s1` | 0.026 | [-0.009, 0.059] | 0.120 |
| `mortality_365d` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s2` | 0.008 | [-0.024, 0.045] | 0.680 |
| `mortality_365d` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.018 | [-0.047, 0.008] | 0.150 |
| `inpatient_365d` | `lr` - `gbm` | -0.036 | [-0.048, -0.023] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar` | -0.033 | [-0.046, -0.019] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar_s1` | -0.031 | [-0.047, -0.018] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar_s2` | -0.038 | [-0.053, -0.023] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:nextlatent_h1416_recon` | -0.034 | [-0.049, -0.020] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_s1` | -0.040 | [-0.054, -0.026] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_s2` | -0.036 | [-0.051, -0.023] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:ar` | 0.004 | [-0.008, 0.016] | 0.600 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s1` | 0.005 | [-0.009, 0.018] | 0.480 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s2` | -0.002 | [-0.017, 0.010] | 0.680 |
| `inpatient_365d` | `gbm` - `ckpt:nextlatent_h1416_recon` | 0.002 | [-0.011, 0.013] | 0.720 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_s1` | -0.004 | [-0.015, 0.007] | 0.450 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_s2` | 0.000 | [-0.011, 0.011] | 0.970 |
| `inpatient_365d` | `ckpt:ar` - `ckpt:ar_s1` | 0.001 | [-0.006, 0.008] | 0.890 |
| `inpatient_365d` | `ckpt:ar` - `ckpt:ar_s2` | -0.005 | [-0.014, 0.002] | 0.190 |
| `inpatient_365d` | `ckpt:ar` - `ckpt:nextlatent_h1416_recon` | -0.001 | [-0.009, 0.008] | 0.810 |
| `inpatient_365d` | `ckpt:ar` - `ckpt:hybrid_s1` | -0.007 | [-0.016, -0.000] | 0.050 |
| `inpatient_365d` | `ckpt:ar` - `ckpt:hybrid_s2` | -0.003 | [-0.013, 0.005] | 0.390 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.006 | [-0.015, 0.005] | 0.200 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:nextlatent_h1416_recon` | -0.003 | [-0.012, 0.007] | 0.680 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.009 | [-0.017, 0.001] | 0.070 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.005 | [-0.014, 0.004] | 0.350 |
| `inpatient_365d` | `ckpt:ar_s2` - `ckpt:nextlatent_h1416_recon` | 0.004 | [-0.006, 0.013] | 0.370 |
| `inpatient_365d` | `ckpt:ar_s2` - `ckpt:hybrid_s1` | -0.002 | [-0.011, 0.007] | 0.700 |
| `inpatient_365d` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | 0.002 | [-0.006, 0.010] | 0.620 |
| `inpatient_365d` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s1` | -0.006 | [-0.015, 0.001] | 0.130 |
| `inpatient_365d` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s2` | -0.002 | [-0.011, 0.007] | 0.630 |
| `inpatient_365d` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | 0.004 | [-0.003, 0.013] | 0.250 |
| `readmission_30d` | `lr` - `gbm` | -0.015 | [-0.047, 0.018] | 0.460 |
| `readmission_30d` | `lr` - `ckpt:ar` | -0.016 | [-0.051, 0.016] | 0.400 |
| `readmission_30d` | `lr` - `ckpt:ar_s1` | -0.031 | [-0.064, 0.004] | 0.090 |
| `readmission_30d` | `lr` - `ckpt:ar_s2` | -0.027 | [-0.070, 0.012] | 0.190 |
| `readmission_30d` | `lr` - `ckpt:nextlatent_h1416_recon` | -0.041 | [-0.074, -0.008] | 0.020 |
| `readmission_30d` | `lr` - `ckpt:hybrid_s1` | -0.042 | [-0.076, -0.011] | 0.020 |
| `readmission_30d` | `lr` - `ckpt:hybrid_s2` | -0.035 | [-0.072, -0.004] | 0.020 |
| `readmission_30d` | `gbm` - `ckpt:ar` | -0.001 | [-0.050, 0.037] | 0.930 |
| `readmission_30d` | `gbm` - `ckpt:ar_s1` | -0.017 | [-0.051, 0.021] | 0.410 |
| `readmission_30d` | `gbm` - `ckpt:ar_s2` | -0.012 | [-0.058, 0.027] | 0.550 |
| `readmission_30d` | `gbm` - `ckpt:nextlatent_h1416_recon` | -0.026 | [-0.065, 0.006] | 0.120 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_s1` | -0.027 | [-0.069, 0.006] | 0.110 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_s2` | -0.020 | [-0.055, 0.010] | 0.240 |
| `readmission_30d` | `ckpt:ar` - `ckpt:ar_s1` | -0.015 | [-0.041, 0.015] | 0.370 |
| `readmission_30d` | `ckpt:ar` - `ckpt:ar_s2` | -0.011 | [-0.040, 0.015] | 0.370 |
| `readmission_30d` | `ckpt:ar` - `ckpt:nextlatent_h1416_recon` | -0.025 | [-0.053, 0.001] | 0.060 |
| `readmission_30d` | `ckpt:ar` - `ckpt:hybrid_s1` | -0.026 | [-0.054, -0.000] | 0.050 |
| `readmission_30d` | `ckpt:ar` - `ckpt:hybrid_s2` | -0.019 | [-0.045, 0.008] | 0.160 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.005 | [-0.023, 0.026] | 0.960 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:nextlatent_h1416_recon` | -0.009 | [-0.038, 0.014] | 0.430 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.011 | [-0.036, 0.010] | 0.300 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.004 | [-0.030, 0.017] | 0.650 |
| `readmission_30d` | `ckpt:ar_s2` - `ckpt:nextlatent_h1416_recon` | -0.014 | [-0.040, 0.012] | 0.310 |
| `readmission_30d` | `ckpt:ar_s2` - `ckpt:hybrid_s1` | -0.015 | [-0.039, 0.016] | 0.230 |
| `readmission_30d` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.008 | [-0.029, 0.017] | 0.590 |
| `readmission_30d` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s1` | -0.001 | [-0.019, 0.017] | 0.890 |
| `readmission_30d` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s2` | 0.006 | [-0.012, 0.022] | 0.520 |
| `readmission_30d` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | 0.007 | [-0.010, 0.025] | 0.440 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.032 | [-0.046, -0.020] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar` | -0.023 | [-0.038, -0.010] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s1` | -0.018 | [-0.031, -0.005] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s2` | -0.025 | [-0.041, -0.011] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:nextlatent_h1416_recon` | -0.022 | [-0.038, -0.010] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_s1` | -0.022 | [-0.037, -0.010] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_s2` | -0.029 | [-0.044, -0.017] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar` | 0.009 | [-0.003, 0.023] | 0.210 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s1` | 0.015 | [0.000, 0.030] | 0.040 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s2` | 0.007 | [-0.007, 0.023] | 0.320 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:nextlatent_h1416_recon` | 0.010 | [-0.002, 0.024] | 0.140 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_s1` | 0.010 | [-0.003, 0.023] | 0.120 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_s2` | 0.003 | [-0.009, 0.017] | 0.780 |
| `new_dx_365d/diabetes` | `ckpt:ar` - `ckpt:ar_s1` | 0.005 | [-0.002, 0.014] | 0.170 |
| `new_dx_365d/diabetes` | `ckpt:ar` - `ckpt:ar_s2` | -0.002 | [-0.009, 0.005] | 0.630 |
| `new_dx_365d/diabetes` | `ckpt:ar` - `ckpt:nextlatent_h1416_recon` | 0.001 | [-0.006, 0.009] | 0.770 |
| `new_dx_365d/diabetes` | `ckpt:ar` - `ckpt:hybrid_s1` | 0.001 | [-0.007, 0.007] | 0.740 |
| `new_dx_365d/diabetes` | `ckpt:ar` - `ckpt:hybrid_s2` | -0.006 | [-0.014, 0.002] | 0.150 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.008 | [-0.015, 0.001] | 0.100 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:nextlatent_h1416_recon` | -0.004 | [-0.012, 0.004] | 0.360 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.005 | [-0.012, 0.003] | 0.270 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.011 | [-0.020, -0.003] | 0.010 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` - `ckpt:nextlatent_h1416_recon` | 0.003 | [-0.005, 0.012] | 0.470 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` - `ckpt:hybrid_s1` | 0.003 | [-0.006, 0.011] | 0.480 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | -0.004 | [-0.012, 0.004] | 0.360 |
| `new_dx_365d/diabetes` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s1` | -0.000 | [-0.009, 0.006] | 0.990 |
| `new_dx_365d/diabetes` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s2` | -0.007 | [-0.016, -0.001] | 0.040 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.007 | [-0.014, -0.000] | 0.040 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | -0.045 | [-0.060, -0.031] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar` | -0.028 | [-0.047, -0.009] | 0.010 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s1` | -0.012 | [-0.032, 0.008] | 0.330 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s2` | -0.026 | [-0.044, -0.007] | 0.020 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:nextlatent_h1416_recon` | -0.020 | [-0.040, -0.003] | 0.020 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_s1` | -0.015 | [-0.034, 0.005] | 0.220 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_s2` | -0.024 | [-0.044, -0.004] | 0.020 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar` | 0.018 | [0.003, 0.034] | 0.020 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s1` | 0.033 | [0.014, 0.052] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s2` | 0.019 | [0.003, 0.036] | 0.040 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:nextlatent_h1416_recon` | 0.025 | [0.010, 0.041] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_s1` | 0.031 | [0.015, 0.050] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_s2` | 0.021 | [0.006, 0.040] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar` - `ckpt:ar_s1` | 0.015 | [0.005, 0.026] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar` - `ckpt:ar_s2` | 0.001 | [-0.010, 0.013] | 0.880 |
| `new_dx_365d/heart_failure` | `ckpt:ar` - `ckpt:nextlatent_h1416_recon` | 0.007 | [-0.005, 0.017] | 0.270 |
| `new_dx_365d/heart_failure` | `ckpt:ar` - `ckpt:hybrid_s1` | 0.013 | [-0.000, 0.025] | 0.060 |
| `new_dx_365d/heart_failure` | `ckpt:ar` - `ckpt:hybrid_s2` | 0.003 | [-0.009, 0.016] | 0.670 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.014 | [-0.025, -0.002] | 0.020 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:nextlatent_h1416_recon` | -0.008 | [-0.020, 0.002] | 0.190 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.002 | [-0.013, 0.009] | 0.760 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.012 | [-0.024, 0.002] | 0.100 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` - `ckpt:nextlatent_h1416_recon` | 0.006 | [-0.004, 0.017] | 0.330 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` - `ckpt:hybrid_s1` | 0.012 | [-0.000, 0.023] | 0.060 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | 0.002 | [-0.009, 0.014] | 0.750 |
| `new_dx_365d/heart_failure` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s1` | 0.006 | [-0.004, 0.017] | 0.200 |
| `new_dx_365d/heart_failure` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s2` | -0.004 | [-0.012, 0.007] | 0.440 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | -0.010 | [-0.021, 0.000] | 0.050 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.029 | [-0.049, -0.012] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar` | -0.015 | [-0.038, 0.004] | 0.110 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s1` | -0.006 | [-0.027, 0.015] | 0.500 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s2` | -0.015 | [-0.037, 0.005] | 0.100 |
| `new_dx_365d/ckd` | `lr` - `ckpt:nextlatent_h1416_recon` | -0.017 | [-0.036, 0.003] | 0.140 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_s1` | -0.018 | [-0.040, -0.001] | 0.050 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_s2` | -0.014 | [-0.037, 0.009] | 0.190 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar` | 0.015 | [-0.003, 0.029] | 0.120 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s1` | 0.024 | [0.002, 0.041] | 0.030 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s2` | 0.014 | [-0.002, 0.033] | 0.080 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:nextlatent_h1416_recon` | 0.012 | [-0.005, 0.028] | 0.170 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_s1` | 0.011 | [-0.007, 0.028] | 0.190 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_s2` | 0.015 | [-0.003, 0.035] | 0.120 |
| `new_dx_365d/ckd` | `ckpt:ar` - `ckpt:ar_s1` | 0.009 | [-0.005, 0.024] | 0.240 |
| `new_dx_365d/ckd` | `ckpt:ar` - `ckpt:ar_s2` | -0.001 | [-0.013, 0.014] | 1.000 |
| `new_dx_365d/ckd` | `ckpt:ar` - `ckpt:nextlatent_h1416_recon` | -0.002 | [-0.016, 0.012] | 0.780 |
| `new_dx_365d/ckd` | `ckpt:ar` - `ckpt:hybrid_s1` | -0.003 | [-0.017, 0.012] | 0.560 |
| `new_dx_365d/ckd` | `ckpt:ar` - `ckpt:hybrid_s2` | 0.001 | [-0.012, 0.013] | 0.800 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.010 | [-0.024, 0.003] | 0.180 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:nextlatent_h1416_recon` | -0.011 | [-0.025, 0.004] | 0.120 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | -0.012 | [-0.026, 0.002] | 0.090 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | -0.008 | [-0.021, 0.009] | 0.390 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` - `ckpt:nextlatent_h1416_recon` | -0.002 | [-0.018, 0.014] | 0.810 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` - `ckpt:hybrid_s1` | -0.003 | [-0.018, 0.011] | 0.630 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | 0.001 | [-0.012, 0.016] | 0.830 |
| `new_dx_365d/ckd` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s1` | -0.001 | [-0.012, 0.011] | 0.800 |
| `new_dx_365d/ckd` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s2` | 0.003 | [-0.012, 0.019] | 0.660 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | 0.004 | [-0.009, 0.019] | 0.590 |
| `new_dx_365d/copd` | `lr` - `gbm` | -0.042 | [-0.057, -0.025] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar` | -0.035 | [-0.053, -0.016] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s1` | -0.033 | [-0.049, -0.014] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s2` | -0.035 | [-0.057, -0.013] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:nextlatent_h1416_recon` | -0.023 | [-0.041, -0.002] | 0.010 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_s1` | -0.029 | [-0.044, -0.010] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_s2` | -0.026 | [-0.043, -0.006] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar` | 0.006 | [-0.012, 0.024] | 0.440 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s1` | 0.008 | [-0.008, 0.026] | 0.310 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s2` | 0.007 | [-0.014, 0.026] | 0.440 |
| `new_dx_365d/copd` | `gbm` - `ckpt:nextlatent_h1416_recon` | 0.019 | [0.002, 0.037] | 0.030 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_s1` | 0.013 | [-0.003, 0.032] | 0.150 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_s2` | 0.016 | [-0.004, 0.033] | 0.100 |
| `new_dx_365d/copd` | `ckpt:ar` - `ckpt:ar_s1` | 0.002 | [-0.014, 0.015] | 0.780 |
| `new_dx_365d/copd` | `ckpt:ar` - `ckpt:ar_s2` | 0.001 | [-0.010, 0.011] | 0.910 |
| `new_dx_365d/copd` | `ckpt:ar` - `ckpt:nextlatent_h1416_recon` | 0.013 | [0.001, 0.030] | 0.050 |
| `new_dx_365d/copd` | `ckpt:ar` - `ckpt:hybrid_s1` | 0.007 | [-0.005, 0.020] | 0.330 |
| `new_dx_365d/copd` | `ckpt:ar` - `ckpt:hybrid_s2` | 0.010 | [-0.002, 0.023] | 0.160 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.001 | [-0.017, 0.012] | 0.880 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:nextlatent_h1416_recon` | 0.011 | [-0.005, 0.026] | 0.200 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:hybrid_s1` | 0.005 | [-0.010, 0.020] | 0.500 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:hybrid_s2` | 0.008 | [-0.009, 0.021] | 0.340 |
| `new_dx_365d/copd` | `ckpt:ar_s2` - `ckpt:nextlatent_h1416_recon` | 0.012 | [-0.002, 0.024] | 0.100 |
| `new_dx_365d/copd` | `ckpt:ar_s2` - `ckpt:hybrid_s1` | 0.006 | [-0.006, 0.018] | 0.340 |
| `new_dx_365d/copd` | `ckpt:ar_s2` - `ckpt:hybrid_s2` | 0.009 | [-0.003, 0.020] | 0.160 |
| `new_dx_365d/copd` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s1` | -0.006 | [-0.017, 0.008] | 0.300 |
| `new_dx_365d/copd` | `ckpt:nextlatent_h1416_recon` - `ckpt:hybrid_s2` | -0.003 | [-0.016, 0.010] | 0.620 |
| `new_dx_365d/copd` | `ckpt:hybrid_s1` - `ckpt:hybrid_s2` | 0.003 | [-0.009, 0.013] | 0.700 |

## Few-shot (k positives + k negatives from train, 5 seeds)

| task | model | k | n train | AUROC mean ± sd | AUPRC mean ± sd |
|---|---|---|---|---|---|
| `mortality_365d` | `lr` | 32 | 64 | 0.527 ± 0.044 | 0.026 ± 0.004 |
| `mortality_365d` | `lr` | 128 | 256 | 0.533 ± 0.048 | 0.027 ± 0.006 |
| `mortality_365d` | `lr` | 512 | 1024 | 0.556 ± 0.026 | 0.029 ± 0.002 |
| `mortality_365d` | `lr` | all | 58192 | 0.554 ± 0.000 | 0.026 ± 0.000 |
| `mortality_365d` | `ckpt:ar` | 32 | 64 | 0.549 ± 0.033 | 0.028 ± 0.003 |
| `mortality_365d` | `ckpt:ar` | 128 | 256 | 0.565 ± 0.018 | 0.029 ± 0.002 |
| `mortality_365d` | `ckpt:ar` | 512 | 1024 | 0.579 ± 0.013 | 0.032 ± 0.002 |
| `mortality_365d` | `ckpt:ar` | all | 58192 | 0.595 ± 0.000 | 0.034 ± 0.000 |
| `mortality_365d` | `ckpt:ar_s1` | 32 | 64 | 0.554 ± 0.031 | 0.031 ± 0.007 |
| `mortality_365d` | `ckpt:ar_s1` | 128 | 256 | 0.576 ± 0.014 | 0.032 ± 0.003 |
| `mortality_365d` | `ckpt:ar_s1` | 512 | 1024 | 0.592 ± 0.016 | 0.031 ± 0.002 |
| `mortality_365d` | `ckpt:ar_s1` | all | 58192 | 0.609 ± 0.000 | 0.031 ± 0.000 |
| `mortality_365d` | `ckpt:ar_s2` | 32 | 64 | 0.550 ± 0.040 | 0.028 ± 0.005 |
| `mortality_365d` | `ckpt:ar_s2` | 128 | 256 | 0.559 ± 0.014 | 0.029 ± 0.002 |
| `mortality_365d` | `ckpt:ar_s2` | 512 | 1024 | 0.589 ± 0.009 | 0.032 ± 0.002 |
| `mortality_365d` | `ckpt:ar_s2` | all | 58192 | 0.606 ± 0.000 | 0.034 ± 0.000 |
| `mortality_365d` | `ckpt:nextlatent_h1416_recon` | 32 | 64 | 0.568 ± 0.045 | 0.033 ± 0.006 |
| `mortality_365d` | `ckpt:nextlatent_h1416_recon` | 128 | 256 | 0.599 ± 0.024 | 0.034 ± 0.002 |
| `mortality_365d` | `ckpt:nextlatent_h1416_recon` | 512 | 1024 | 0.617 ± 0.013 | 0.034 ± 0.001 |
| `mortality_365d` | `ckpt:nextlatent_h1416_recon` | all | 58192 | 0.633 ± 0.000 | 0.035 ± 0.000 |
| `mortality_365d` | `ckpt:hybrid_s1` | 32 | 64 | 0.561 ± 0.046 | 0.033 ± 0.008 |
| `mortality_365d` | `ckpt:hybrid_s1` | 128 | 256 | 0.590 ± 0.022 | 0.036 ± 0.006 |
| `mortality_365d` | `ckpt:hybrid_s1` | 512 | 1024 | 0.600 ± 0.006 | 0.031 ± 0.001 |
| `mortality_365d` | `ckpt:hybrid_s1` | all | 58192 | 0.607 ± 0.000 | 0.030 ± 0.000 |
| `mortality_365d` | `ckpt:hybrid_s2` | 32 | 64 | 0.562 ± 0.048 | 0.032 ± 0.004 |
| `mortality_365d` | `ckpt:hybrid_s2` | 128 | 256 | 0.593 ± 0.030 | 0.038 ± 0.008 |
| `mortality_365d` | `ckpt:hybrid_s2` | 512 | 1024 | 0.613 ± 0.008 | 0.037 ± 0.005 |
| `mortality_365d` | `ckpt:hybrid_s2` | all | 58192 | 0.625 ± 0.000 | 0.040 ± 0.000 |
| `inpatient_365d` | `lr` | 32 | 64 | 0.591 ± 0.036 | 0.268 ± 0.017 |
| `inpatient_365d` | `lr` | 128 | 256 | 0.657 ± 0.009 | 0.308 ± 0.016 |
| `inpatient_365d` | `lr` | 512 | 1024 | 0.674 ± 0.004 | 0.319 ± 0.009 |
| `inpatient_365d` | `lr` | all | 58192 | 0.708 ± 0.000 | 0.363 ± 0.000 |
| `inpatient_365d` | `ckpt:ar` | 32 | 64 | 0.672 ± 0.020 | 0.323 ± 0.024 |
| `inpatient_365d` | `ckpt:ar` | 128 | 256 | 0.708 ± 0.010 | 0.357 ± 0.010 |
| `inpatient_365d` | `ckpt:ar` | 512 | 1024 | 0.723 ± 0.005 | 0.368 ± 0.014 |
| `inpatient_365d` | `ckpt:ar` | all | 58192 | 0.740 ± 0.000 | 0.402 ± 0.000 |
| `inpatient_365d` | `ckpt:ar_s1` | 32 | 64 | 0.669 ± 0.012 | 0.314 ± 0.012 |
| `inpatient_365d` | `ckpt:ar_s1` | 128 | 256 | 0.699 ± 0.013 | 0.345 ± 0.018 |
| `inpatient_365d` | `ckpt:ar_s1` | 512 | 1024 | 0.718 ± 0.004 | 0.359 ± 0.008 |
| `inpatient_365d` | `ckpt:ar_s1` | all | 58192 | 0.739 ± 0.000 | 0.388 ± 0.000 |
| `inpatient_365d` | `ckpt:ar_s2` | 32 | 64 | 0.671 ± 0.024 | 0.324 ± 0.026 |
| `inpatient_365d` | `ckpt:ar_s2` | 128 | 256 | 0.709 ± 0.008 | 0.361 ± 0.014 |
| `inpatient_365d` | `ckpt:ar_s2` | 512 | 1024 | 0.724 ± 0.004 | 0.363 ± 0.009 |
| `inpatient_365d` | `ckpt:ar_s2` | all | 58192 | 0.746 ± 0.000 | 0.408 ± 0.000 |
| `inpatient_365d` | `ckpt:nextlatent_h1416_recon` | 32 | 64 | 0.658 ± 0.025 | 0.294 ± 0.020 |
| `inpatient_365d` | `ckpt:nextlatent_h1416_recon` | 128 | 256 | 0.701 ± 0.007 | 0.339 ± 0.013 |
| `inpatient_365d` | `ckpt:nextlatent_h1416_recon` | 512 | 1024 | 0.715 ± 0.008 | 0.352 ± 0.020 |
| `inpatient_365d` | `ckpt:nextlatent_h1416_recon` | all | 58192 | 0.742 ± 0.000 | 0.394 ± 0.000 |
| `inpatient_365d` | `ckpt:hybrid_s1` | 32 | 64 | 0.661 ± 0.019 | 0.296 ± 0.017 |
| `inpatient_365d` | `ckpt:hybrid_s1` | 128 | 256 | 0.696 ± 0.012 | 0.334 ± 0.016 |
| `inpatient_365d` | `ckpt:hybrid_s1` | 512 | 1024 | 0.715 ± 0.005 | 0.357 ± 0.018 |
| `inpatient_365d` | `ckpt:hybrid_s1` | all | 58192 | 0.748 ± 0.000 | 0.413 ± 0.000 |
| `inpatient_365d` | `ckpt:hybrid_s2` | 32 | 64 | 0.658 ± 0.020 | 0.298 ± 0.016 |
| `inpatient_365d` | `ckpt:hybrid_s2` | 128 | 256 | 0.696 ± 0.009 | 0.337 ± 0.015 |
| `inpatient_365d` | `ckpt:hybrid_s2` | 512 | 1024 | 0.717 ± 0.006 | 0.356 ± 0.022 |
| `inpatient_365d` | `ckpt:hybrid_s2` | all | 58192 | 0.744 ± 0.000 | 0.405 ± 0.000 |
| `readmission_30d` | `lr` | 32 | 64 | 0.568 ± 0.031 | 0.065 ± 0.008 |
| `readmission_30d` | `lr` | 128 | 256 | 0.588 ± 0.018 | 0.077 ± 0.005 |
| `readmission_30d` | `lr` | 512 | 1024 | 0.628 ± 0.003 | 0.082 ± 0.002 |
| `readmission_30d` | `lr` | all | 24417 | 0.658 ± 0.000 | 0.098 ± 0.000 |
| `readmission_30d` | `ckpt:ar` | 32 | 64 | 0.577 ± 0.046 | 0.070 ± 0.013 |
| `readmission_30d` | `ckpt:ar` | 128 | 256 | 0.601 ± 0.018 | 0.078 ± 0.004 |
| `readmission_30d` | `ckpt:ar` | 512 | 1024 | 0.646 ± 0.009 | 0.100 ± 0.003 |
| `readmission_30d` | `ckpt:ar` | all | 24417 | 0.674 ± 0.000 | 0.106 ± 0.000 |
| `readmission_30d` | `ckpt:ar_s1` | 32 | 64 | 0.567 ± 0.054 | 0.070 ± 0.010 |
| `readmission_30d` | `ckpt:ar_s1` | 128 | 256 | 0.591 ± 0.030 | 0.078 ± 0.009 |
| `readmission_30d` | `ckpt:ar_s1` | 512 | 1024 | 0.650 ± 0.018 | 0.100 ± 0.004 |
| `readmission_30d` | `ckpt:ar_s1` | all | 24417 | 0.689 ± 0.000 | 0.113 ± 0.000 |
| `readmission_30d` | `ckpt:ar_s2` | 32 | 64 | 0.586 ± 0.048 | 0.076 ± 0.013 |
| `readmission_30d` | `ckpt:ar_s2` | 128 | 256 | 0.606 ± 0.033 | 0.080 ± 0.009 |
| `readmission_30d` | `ckpt:ar_s2` | 512 | 1024 | 0.657 ± 0.007 | 0.110 ± 0.010 |
| `readmission_30d` | `ckpt:ar_s2` | all | 24417 | 0.684 ± 0.000 | 0.121 ± 0.000 |
| `readmission_30d` | `ckpt:nextlatent_h1416_recon` | 32 | 64 | 0.615 ± 0.036 | 0.079 ± 0.010 |
| `readmission_30d` | `ckpt:nextlatent_h1416_recon` | 128 | 256 | 0.649 ± 0.021 | 0.094 ± 0.008 |
| `readmission_30d` | `ckpt:nextlatent_h1416_recon` | 512 | 1024 | 0.679 ± 0.013 | 0.113 ± 0.008 |
| `readmission_30d` | `ckpt:nextlatent_h1416_recon` | all | 24417 | 0.699 ± 0.000 | 0.120 ± 0.000 |
| `readmission_30d` | `ckpt:hybrid_s1` | 32 | 64 | 0.608 ± 0.041 | 0.079 ± 0.013 |
| `readmission_30d` | `ckpt:hybrid_s1` | 128 | 256 | 0.647 ± 0.025 | 0.089 ± 0.009 |
| `readmission_30d` | `ckpt:hybrid_s1` | 512 | 1024 | 0.685 ± 0.010 | 0.112 ± 0.003 |
| `readmission_30d` | `ckpt:hybrid_s1` | all | 24417 | 0.700 ± 0.000 | 0.109 ± 0.000 |
| `readmission_30d` | `ckpt:hybrid_s2` | 32 | 64 | 0.607 ± 0.039 | 0.080 ± 0.011 |
| `readmission_30d` | `ckpt:hybrid_s2` | 128 | 256 | 0.649 ± 0.026 | 0.088 ± 0.007 |
| `readmission_30d` | `ckpt:hybrid_s2` | 512 | 1024 | 0.680 ± 0.008 | 0.119 ± 0.004 |
| `readmission_30d` | `ckpt:hybrid_s2` | all | 24417 | 0.693 ± 0.000 | 0.126 ± 0.000 |
| `new_dx_365d/diabetes` | `lr` | 32 | 64 | 0.684 ± 0.020 | 0.344 ± 0.016 |
| `new_dx_365d/diabetes` | `lr` | 128 | 256 | 0.703 ± 0.001 | 0.360 ± 0.005 |
| `new_dx_365d/diabetes` | `lr` | 512 | 1024 | 0.711 ± 0.002 | 0.373 ± 0.006 |
| `new_dx_365d/diabetes` | `lr` | all | 39458 | 0.740 ± 0.000 | 0.432 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:ar` | 32 | 64 | 0.701 ± 0.035 | 0.377 ± 0.027 |
| `new_dx_365d/diabetes` | `ckpt:ar` | 128 | 256 | 0.737 ± 0.006 | 0.406 ± 0.015 |
| `new_dx_365d/diabetes` | `ckpt:ar` | 512 | 1024 | 0.750 ± 0.002 | 0.438 ± 0.003 |
| `new_dx_365d/diabetes` | `ckpt:ar` | all | 39458 | 0.763 ± 0.000 | 0.469 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | 32 | 64 | 0.689 ± 0.031 | 0.358 ± 0.031 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | 128 | 256 | 0.733 ± 0.002 | 0.401 ± 0.007 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | 512 | 1024 | 0.747 ± 0.002 | 0.438 ± 0.008 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | all | 39458 | 0.757 ± 0.000 | 0.456 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | 32 | 64 | 0.692 ± 0.029 | 0.363 ± 0.019 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | 128 | 256 | 0.736 ± 0.005 | 0.401 ± 0.014 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | 512 | 1024 | 0.749 ± 0.002 | 0.431 ± 0.005 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | all | 39458 | 0.765 ± 0.000 | 0.464 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:nextlatent_h1416_recon` | 32 | 64 | 0.698 ± 0.024 | 0.364 ± 0.020 |
| `new_dx_365d/diabetes` | `ckpt:nextlatent_h1416_recon` | 128 | 256 | 0.734 ± 0.008 | 0.397 ± 0.015 |
| `new_dx_365d/diabetes` | `ckpt:nextlatent_h1416_recon` | 512 | 1024 | 0.748 ± 0.002 | 0.422 ± 0.005 |
| `new_dx_365d/diabetes` | `ckpt:nextlatent_h1416_recon` | all | 39458 | 0.762 ± 0.000 | 0.454 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s1` | 32 | 64 | 0.694 ± 0.023 | 0.360 ± 0.017 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s1` | 128 | 256 | 0.727 ± 0.007 | 0.386 ± 0.009 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s1` | 512 | 1024 | 0.746 ± 0.002 | 0.420 ± 0.005 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s1` | all | 39458 | 0.762 ± 0.000 | 0.457 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s2` | 32 | 64 | 0.698 ± 0.026 | 0.371 ± 0.021 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s2` | 128 | 256 | 0.737 ± 0.009 | 0.401 ± 0.012 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s2` | 512 | 1024 | 0.752 ± 0.003 | 0.432 ± 0.005 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_s2` | all | 39458 | 0.769 ± 0.000 | 0.472 ± 0.000 |
| `new_dx_365d/heart_failure` | `lr` | 32 | 64 | 0.648 ± 0.027 | 0.208 ± 0.018 |
| `new_dx_365d/heart_failure` | `lr` | 128 | 256 | 0.666 ± 0.008 | 0.215 ± 0.015 |
| `new_dx_365d/heart_failure` | `lr` | 512 | 1024 | 0.692 ± 0.004 | 0.241 ± 0.007 |
| `new_dx_365d/heart_failure` | `lr` | all | 48770 | 0.744 ± 0.000 | 0.276 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | 32 | 64 | 0.695 ± 0.029 | 0.252 ± 0.019 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | 128 | 256 | 0.727 ± 0.011 | 0.265 ± 0.008 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | 512 | 1024 | 0.754 ± 0.006 | 0.292 ± 0.010 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | all | 48770 | 0.771 ± 0.000 | 0.308 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | 32 | 64 | 0.686 ± 0.027 | 0.245 ± 0.019 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | 128 | 256 | 0.716 ± 0.009 | 0.254 ± 0.007 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | 512 | 1024 | 0.740 ± 0.004 | 0.275 ± 0.007 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | all | 48770 | 0.756 ± 0.000 | 0.301 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | 32 | 64 | 0.699 ± 0.030 | 0.258 ± 0.024 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | 128 | 256 | 0.726 ± 0.015 | 0.266 ± 0.006 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | 512 | 1024 | 0.753 ± 0.005 | 0.289 ± 0.010 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | all | 48770 | 0.770 ± 0.000 | 0.307 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:nextlatent_h1416_recon` | 32 | 64 | 0.686 ± 0.024 | 0.236 ± 0.016 |
| `new_dx_365d/heart_failure` | `ckpt:nextlatent_h1416_recon` | 128 | 256 | 0.705 ± 0.011 | 0.240 ± 0.011 |
| `new_dx_365d/heart_failure` | `ckpt:nextlatent_h1416_recon` | 512 | 1024 | 0.738 ± 0.005 | 0.268 ± 0.005 |
| `new_dx_365d/heart_failure` | `ckpt:nextlatent_h1416_recon` | all | 48770 | 0.764 ± 0.000 | 0.306 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s1` | 32 | 64 | 0.687 ± 0.022 | 0.237 ± 0.015 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s1` | 128 | 256 | 0.701 ± 0.009 | 0.238 ± 0.009 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s1` | 512 | 1024 | 0.734 ± 0.004 | 0.276 ± 0.007 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s1` | all | 48770 | 0.758 ± 0.000 | 0.308 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s2` | 32 | 64 | 0.690 ± 0.025 | 0.236 ± 0.017 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s2` | 128 | 256 | 0.707 ± 0.006 | 0.243 ± 0.004 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s2` | 512 | 1024 | 0.743 ± 0.004 | 0.286 ± 0.010 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_s2` | all | 48770 | 0.768 ± 0.000 | 0.314 ± 0.000 |
| `new_dx_365d/ckd` | `lr` | 32 | 64 | 0.632 ± 0.053 | 0.151 ± 0.013 |
| `new_dx_365d/ckd` | `lr` | 128 | 256 | 0.670 ± 0.006 | 0.161 ± 0.008 |
| `new_dx_365d/ckd` | `lr` | 512 | 1024 | 0.689 ± 0.006 | 0.177 ± 0.003 |
| `new_dx_365d/ckd` | `lr` | all | 51106 | 0.736 ± 0.000 | 0.236 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:ar` | 32 | 64 | 0.673 ± 0.009 | 0.172 ± 0.011 |
| `new_dx_365d/ckd` | `ckpt:ar` | 128 | 256 | 0.711 ± 0.007 | 0.190 ± 0.002 |
| `new_dx_365d/ckd` | `ckpt:ar` | 512 | 1024 | 0.729 ± 0.003 | 0.218 ± 0.006 |
| `new_dx_365d/ckd` | `ckpt:ar` | all | 51106 | 0.751 ± 0.000 | 0.244 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | 32 | 64 | 0.673 ± 0.020 | 0.178 ± 0.017 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | 128 | 256 | 0.709 ± 0.008 | 0.183 ± 0.007 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | 512 | 1024 | 0.727 ± 0.006 | 0.213 ± 0.013 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | all | 51106 | 0.742 ± 0.000 | 0.246 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | 32 | 64 | 0.676 ± 0.018 | 0.175 ± 0.019 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | 128 | 256 | 0.720 ± 0.009 | 0.190 ± 0.009 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | 512 | 1024 | 0.730 ± 0.007 | 0.210 ± 0.007 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | all | 51106 | 0.751 ± 0.000 | 0.246 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:nextlatent_h1416_recon` | 32 | 64 | 0.678 ± 0.012 | 0.172 ± 0.010 |
| `new_dx_365d/ckd` | `ckpt:nextlatent_h1416_recon` | 128 | 256 | 0.707 ± 0.011 | 0.181 ± 0.012 |
| `new_dx_365d/ckd` | `ckpt:nextlatent_h1416_recon` | 512 | 1024 | 0.726 ± 0.006 | 0.204 ± 0.011 |
| `new_dx_365d/ckd` | `ckpt:nextlatent_h1416_recon` | all | 51106 | 0.753 ± 0.000 | 0.234 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s1` | 32 | 64 | 0.676 ± 0.014 | 0.174 ± 0.012 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s1` | 128 | 256 | 0.705 ± 0.008 | 0.184 ± 0.006 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s1` | 512 | 1024 | 0.727 ± 0.006 | 0.213 ± 0.017 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s1` | all | 51106 | 0.754 ± 0.000 | 0.248 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s2` | 32 | 64 | 0.673 ± 0.016 | 0.168 ± 0.017 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s2` | 128 | 256 | 0.698 ± 0.010 | 0.176 ± 0.007 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s2` | 512 | 1024 | 0.729 ± 0.005 | 0.214 ± 0.017 |
| `new_dx_365d/ckd` | `ckpt:hybrid_s2` | all | 51106 | 0.750 ± 0.000 | 0.247 ± 0.000 |
| `new_dx_365d/copd` | `lr` | 32 | 64 | 0.628 ± 0.065 | 0.193 ± 0.021 |
| `new_dx_365d/copd` | `lr` | 128 | 256 | 0.667 ± 0.003 | 0.215 ± 0.007 |
| `new_dx_365d/copd` | `lr` | 512 | 1024 | 0.682 ± 0.006 | 0.234 ± 0.009 |
| `new_dx_365d/copd` | `lr` | all | 48015 | 0.726 ± 0.000 | 0.317 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:ar` | 32 | 64 | 0.690 ± 0.040 | 0.235 ± 0.014 |
| `new_dx_365d/copd` | `ckpt:ar` | 128 | 256 | 0.719 ± 0.010 | 0.253 ± 0.015 |
| `new_dx_365d/copd` | `ckpt:ar` | 512 | 1024 | 0.742 ± 0.004 | 0.274 ± 0.008 |
| `new_dx_365d/copd` | `ckpt:ar` | all | 48015 | 0.761 ± 0.000 | 0.298 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | 32 | 64 | 0.679 ± 0.046 | 0.227 ± 0.016 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | 128 | 256 | 0.718 ± 0.009 | 0.254 ± 0.018 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | 512 | 1024 | 0.733 ± 0.003 | 0.259 ± 0.004 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | all | 48015 | 0.759 ± 0.000 | 0.306 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | 32 | 64 | 0.683 ± 0.042 | 0.231 ± 0.021 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | 128 | 256 | 0.716 ± 0.010 | 0.251 ± 0.012 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | 512 | 1024 | 0.736 ± 0.003 | 0.267 ± 0.007 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | all | 48015 | 0.761 ± 0.000 | 0.314 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:nextlatent_h1416_recon` | 32 | 64 | 0.666 ± 0.033 | 0.208 ± 0.011 |
| `new_dx_365d/copd` | `ckpt:nextlatent_h1416_recon` | 128 | 256 | 0.703 ± 0.011 | 0.229 ± 0.014 |
| `new_dx_365d/copd` | `ckpt:nextlatent_h1416_recon` | 512 | 1024 | 0.721 ± 0.002 | 0.250 ± 0.007 |
| `new_dx_365d/copd` | `ckpt:nextlatent_h1416_recon` | all | 48015 | 0.749 ± 0.000 | 0.293 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_s1` | 32 | 64 | 0.673 ± 0.034 | 0.213 ± 0.009 |
| `new_dx_365d/copd` | `ckpt:hybrid_s1` | 128 | 256 | 0.707 ± 0.011 | 0.232 ± 0.011 |
| `new_dx_365d/copd` | `ckpt:hybrid_s1` | 512 | 1024 | 0.727 ± 0.004 | 0.252 ± 0.004 |
| `new_dx_365d/copd` | `ckpt:hybrid_s1` | all | 48015 | 0.754 ± 0.000 | 0.296 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_s2` | 32 | 64 | 0.670 ± 0.042 | 0.207 ± 0.019 |
| `new_dx_365d/copd` | `ckpt:hybrid_s2` | 128 | 256 | 0.700 ± 0.019 | 0.226 ± 0.015 |
| `new_dx_365d/copd` | `ckpt:hybrid_s2` | 512 | 1024 | 0.727 ± 0.003 | 0.251 ± 0.005 |
| `new_dx_365d/copd` | `ckpt:hybrid_s2` | all | 48015 | 0.751 ± 0.000 | 0.275 ± 0.000 |
