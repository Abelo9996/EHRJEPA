# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `44fcab4` |
| created | 2026-09-10T15:04:30+00:00 |
| runtime (s) | 1440.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `ckpt:hybrid_final_s0` | probe | `runs/scale1b-final-desynpuf/hybrid_final_s0/final.pt` | last@final |
| `ckpt:hybrid_final_s1` | probe | `runs/scale1b-final-desynpuf/hybrid_final_s1/final.pt` | last@final |
| `ckpt:hybrid_final_s2` | probe | `runs/scale1b-final-desynpuf/hybrid_final_s2/final.pt` | last@final |
| `ckpt:ar` | probe | `runs/scale1b-desynpuf/ar/final.pt` | last@final |
| `ckpt:ar_s1` | probe | `runs/scale1b-seeds-desynpuf/ar_s1/final.pt` | last@final |
| `ckpt:ar_s2` | probe | `runs/scale1b-seeds-desynpuf/ar_s2/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `mortality_365d` | 58192 (0.0189) | 7285 (0.0177) | 7358 (0.0190) |  |
| `inpatient_365d` | 58192 (0.2065) | 7285 (0.2091) | 7358 (0.2098) |  |
| `readmission_30d` | 24417 (0.0572) | 3059 (0.0588) | 3116 (0.0501) |  |
| `new_dx_365d/diabetes` | 39458 (0.2262) | 4896 (0.2220) | 4939 (0.2264) |  |
| `new_dx_365d/heart_failure` | 48770 (0.1270) | 6090 (0.1251) | 6164 (0.1308) |  |
| `new_dx_365d/ckd` | 51106 (0.0933) | 6406 (0.0894) | 6487 (0.0999) |  |
| `new_dx_365d/copd` | 48015 (0.1323) | 5975 (0.1279) | 6009 (0.1340) |  |

## AUROC

| task | lr | gbm | ckpt:hybrid_final_s0 | ckpt:hybrid_final_s1 | ckpt:hybrid_final_s2 | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 0.566 [0.518, 0.616] | 0.556 [0.514, 0.603] | 0.601 [0.558, 0.641] | 0.603 [0.552, 0.640] | 0.600 [0.553, 0.636] | 0.604 [0.549, 0.647] | 0.612 [0.565, 0.653] | 0.601 [0.555, 0.640] |
| `inpatient_365d` | 0.712 [0.697, 0.724] | 0.746 [0.731, 0.755] | 0.760 [0.746, 0.772] | 0.757 [0.742, 0.770] | 0.756 [0.742, 0.768] | 0.743 [0.727, 0.756] | 0.740 [0.725, 0.752] | 0.743 [0.730, 0.756] |
| `readmission_30d` | 0.653 [0.611, 0.704] | 0.667 [0.625, 0.712] | 0.687 [0.643, 0.725] | 0.695 [0.655, 0.732] | 0.683 [0.643, 0.725] | 0.644 [0.598, 0.686] | 0.665 [0.619, 0.708] | 0.665 [0.617, 0.708] |
| `new_dx_365d/diabetes` | 0.737 [0.721, 0.753] | 0.771 [0.756, 0.784] | 0.779 [0.766, 0.795] | 0.778 [0.764, 0.793] | 0.776 [0.763, 0.792] | 0.763 [0.748, 0.780] | 0.764 [0.749, 0.780] | 0.761 [0.747, 0.775] |
| `new_dx_365d/heart_failure` | 0.741 [0.724, 0.758] | 0.784 [0.769, 0.800] | 0.798 [0.784, 0.811] | 0.797 [0.782, 0.812] | 0.800 [0.785, 0.813] | 0.769 [0.752, 0.783] | 0.771 [0.756, 0.787] | 0.774 [0.756, 0.788] |
| `new_dx_365d/ckd` | 0.739 [0.718, 0.762] | 0.771 [0.751, 0.790] | 0.784 [0.768, 0.803] | 0.784 [0.766, 0.802] | 0.782 [0.765, 0.802] | 0.770 [0.752, 0.790] | 0.766 [0.750, 0.786] | 0.766 [0.748, 0.787] |
| `new_dx_365d/copd` | 0.733 [0.715, 0.748] | 0.768 [0.748, 0.787] | 0.772 [0.753, 0.790] | 0.772 [0.753, 0.791] | 0.772 [0.754, 0.790] | 0.759 [0.741, 0.777] | 0.763 [0.746, 0.783] | 0.758 [0.738, 0.777] |

## AUPRC

| task | lr | gbm | ckpt:hybrid_final_s0 | ckpt:hybrid_final_s1 | ckpt:hybrid_final_s2 | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 0.024 [0.018, 0.036] | 0.023 [0.019, 0.030] | 0.028 [0.022, 0.040] | 0.029 [0.022, 0.040] | 0.029 [0.022, 0.042] | 0.027 [0.021, 0.036] | 0.028 [0.023, 0.037] | 0.026 [0.020, 0.035] |
| `inpatient_365d` | 0.377 [0.353, 0.400] | 0.425 [0.397, 0.452] | 0.454 [0.427, 0.479] | 0.452 [0.427, 0.477] | 0.446 [0.419, 0.473] | 0.423 [0.396, 0.451] | 0.409 [0.383, 0.434] | 0.425 [0.399, 0.449] |
| `readmission_30d` | 0.097 [0.076, 0.131] | 0.108 [0.080, 0.141] | 0.111 [0.084, 0.156] | 0.108 [0.086, 0.148] | 0.099 [0.077, 0.130] | 0.086 [0.064, 0.119] | 0.088 [0.067, 0.115] | 0.102 [0.075, 0.136] |
| `new_dx_365d/diabetes` | 0.421 [0.394, 0.451] | 0.469 [0.442, 0.497] | 0.486 [0.455, 0.517] | 0.482 [0.451, 0.511] | 0.481 [0.451, 0.514] | 0.454 [0.425, 0.485] | 0.454 [0.424, 0.484] | 0.446 [0.414, 0.471] |
| `new_dx_365d/heart_failure` | 0.263 [0.241, 0.290] | 0.335 [0.305, 0.368] | 0.358 [0.325, 0.391] | 0.342 [0.310, 0.372] | 0.350 [0.320, 0.381] | 0.317 [0.286, 0.346] | 0.302 [0.276, 0.328] | 0.319 [0.291, 0.351] |
| `new_dx_365d/ckd` | 0.246 [0.220, 0.278] | 0.265 [0.237, 0.293] | 0.299 [0.271, 0.333] | 0.297 [0.267, 0.334] | 0.297 [0.268, 0.330] | 0.275 [0.249, 0.303] | 0.259 [0.236, 0.287] | 0.271 [0.243, 0.302] |
| `new_dx_365d/copd` | 0.310 [0.278, 0.342] | 0.347 [0.313, 0.392] | 0.371 [0.334, 0.411] | 0.370 [0.333, 0.414] | 0.365 [0.332, 0.404] | 0.341 [0.306, 0.382] | 0.334 [0.303, 0.375] | 0.316 [0.281, 0.350] |

## BRIER

| task | lr | gbm | ckpt:hybrid_final_s0 | ckpt:hybrid_final_s1 | ckpt:hybrid_final_s2 | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 0.0187 [0.0157, 0.0219] | 0.0187 [0.0157, 0.0220] | 0.0186 [0.0157, 0.0219] | 0.0186 [0.0157, 0.0219] | 0.0186 [0.0156, 0.0219] | 0.0187 [0.0157, 0.0219] | 0.0186 [0.0157, 0.0219] | 0.0187 [0.0157, 0.0219] |
| `inpatient_365d` | 0.1515 [0.1470, 0.1559] | 0.1442 [0.1400, 0.1488] | 0.1412 [0.1371, 0.1455] | 0.1417 [0.1370, 0.1458] | 0.1422 [0.1378, 0.1466] | 0.1449 [0.1402, 0.1490] | 0.1460 [0.1417, 0.1508] | 0.1447 [0.1402, 0.1490] |
| `readmission_30d` | 0.0468 [0.0408, 0.0527] | 0.0468 [0.0409, 0.0527] | 0.0467 [0.0411, 0.0527] | 0.0468 [0.0411, 0.0525] | 0.0470 [0.0413, 0.0529] | 0.0472 [0.0415, 0.0530] | 0.0471 [0.0411, 0.0526] | 0.0468 [0.0412, 0.0526] |
| `new_dx_365d/diabetes` | 0.1554 [0.1497, 0.1616] | 0.1470 [0.1414, 0.1532] | 0.1450 [0.1395, 0.1500] | 0.1455 [0.1398, 0.1503] | 0.1459 [0.1402, 0.1514] | 0.1492 [0.1437, 0.1550] | 0.1491 [0.1434, 0.1543] | 0.1498 [0.1442, 0.1554] |
| `new_dx_365d/heart_failure` | 0.1060 [0.0998, 0.1110] | 0.0994 [0.0943, 0.1035] | 0.0979 [0.0928, 0.1026] | 0.0985 [0.0934, 0.1029] | 0.0979 [0.0929, 0.1024] | 0.1012 [0.0962, 0.1060] | 0.1018 [0.0965, 0.1062] | 0.1007 [0.0955, 0.1052] |
| `new_dx_365d/ckd` | 0.0835 [0.0779, 0.0887] | 0.0813 [0.0760, 0.0861] | 0.0797 [0.0750, 0.0843] | 0.0797 [0.0747, 0.0843] | 0.0798 [0.0750, 0.0847] | 0.0812 [0.0761, 0.0860] | 0.0821 [0.0765, 0.0869] | 0.0814 [0.0762, 0.0860] |
| `new_dx_365d/copd` | 0.1062 [0.0999, 0.1124] | 0.1011 [0.0949, 0.1067] | 0.0999 [0.0938, 0.1050] | 0.0998 [0.0937, 0.1046] | 0.1000 [0.0941, 0.1049] | 0.1024 [0.0965, 0.1079] | 0.1023 [0.0961, 0.1075] | 0.1036 [0.0974, 0.1089] |

## CALIBRATION SLOPE

| task | lr | gbm | ckpt:hybrid_final_s0 | ckpt:hybrid_final_s1 | ckpt:hybrid_final_s2 | ckpt:ar | ckpt:ar_s1 | ckpt:ar_s2 |
|---|---|---|---|---|---|---|---|---|
| `mortality_365d` | 3.109 [0.490, 5.414] | 0.413 [0.100, 0.708] | 0.646 [0.365, 0.900] | 0.626 [0.345, 0.890] | 0.632 [0.316, 0.884] | 0.649 [0.337, 0.925] | 0.675 [0.393, 0.955] | 0.589 [0.304, 0.841] |
| `inpatient_365d` | 1.169 [1.070, 1.256] | 1.064 [0.985, 1.128] | 1.001 [0.925, 1.071] | 0.997 [0.917, 1.061] | 0.971 [0.903, 1.035] | 1.016 [0.936, 1.100] | 0.963 [0.878, 1.032] | 1.024 [0.946, 1.096] |
| `readmission_30d` | 1.033 [0.764, 1.339] | 0.717 [0.541, 0.914] | 0.784 [0.588, 0.971] | 0.815 [0.636, 0.997] | 0.756 [0.582, 0.947] | 0.790 [0.524, 1.025] | 0.833 [0.601, 1.065] | 0.854 [0.612, 1.092] |
| `new_dx_365d/diabetes` | 0.990 [0.904, 1.075] | 1.018 [0.942, 1.087] | 0.951 [0.886, 1.034] | 0.945 [0.875, 1.023] | 0.893 [0.829, 0.963] | 0.904 [0.833, 0.976] | 0.947 [0.873, 1.027] | 0.964 [0.887, 1.039] |
| `new_dx_365d/heart_failure` | 1.249 [1.137, 1.364] | 1.021 [0.940, 1.111] | 1.004 [0.929, 1.079] | 0.955 [0.885, 1.042] | 0.960 [0.884, 1.035] | 0.985 [0.900, 1.064] | 1.005 [0.932, 1.087] | 1.010 [0.915, 1.083] |
| `new_dx_365d/ckd` | 1.048 [0.952, 1.155] | 0.990 [0.898, 1.072] | 1.000 [0.915, 1.104] | 1.016 [0.924, 1.110] | 1.001 [0.912, 1.105] | 1.026 [0.943, 1.118] | 1.001 [0.924, 1.102] | 1.008 [0.925, 1.110] |
| `new_dx_365d/copd` | 1.406 [1.295, 1.522] | 1.017 [0.925, 1.099] | 0.962 [0.869, 1.050] | 0.965 [0.879, 1.053] | 0.964 [0.881, 1.042] | 0.984 [0.891, 1.072] | 0.989 [0.905, 1.082] | 0.960 [0.857, 1.063] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | 0.009 | [-0.037, 0.056] | 0.710 |
| `mortality_365d` | `lr` - `ckpt:hybrid_final_s0` | -0.035 | [-0.084, 0.025] | 0.250 |
| `mortality_365d` | `lr` - `ckpt:hybrid_final_s1` | -0.037 | [-0.084, 0.027] | 0.220 |
| `mortality_365d` | `lr` - `ckpt:hybrid_final_s2` | -0.034 | [-0.082, 0.027] | 0.280 |
| `mortality_365d` | `lr` - `ckpt:ar` | -0.038 | [-0.095, 0.018] | 0.240 |
| `mortality_365d` | `lr` - `ckpt:ar_s1` | -0.046 | [-0.095, 0.004] | 0.070 |
| `mortality_365d` | `lr` - `ckpt:ar_s2` | -0.035 | [-0.092, 0.024] | 0.200 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_final_s0` | -0.045 | [-0.093, 0.003] | 0.080 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_final_s1` | -0.046 | [-0.094, 0.001] | 0.070 |
| `mortality_365d` | `gbm` - `ckpt:hybrid_final_s2` | -0.043 | [-0.090, 0.003] | 0.090 |
| `mortality_365d` | `gbm` - `ckpt:ar` | -0.047 | [-0.098, -0.000] | 0.050 |
| `mortality_365d` | `gbm` - `ckpt:ar_s1` | -0.055 | [-0.104, -0.007] | 0.000 |
| `mortality_365d` | `gbm` - `ckpt:ar_s2` | -0.044 | [-0.097, 0.011] | 0.090 |
| `mortality_365d` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s1` | -0.001 | [-0.024, 0.027] | 0.940 |
| `mortality_365d` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s2` | 0.001 | [-0.020, 0.029] | 0.740 |
| `mortality_365d` | `ckpt:hybrid_final_s0` - `ckpt:ar` | -0.003 | [-0.041, 0.035] | 0.900 |
| `mortality_365d` | `ckpt:hybrid_final_s0` - `ckpt:ar_s1` | -0.010 | [-0.045, 0.023] | 0.530 |
| `mortality_365d` | `ckpt:hybrid_final_s0` - `ckpt:ar_s2` | 0.001 | [-0.036, 0.035] | 0.970 |
| `mortality_365d` | `ckpt:hybrid_final_s1` - `ckpt:hybrid_final_s2` | 0.003 | [-0.024, 0.029] | 0.850 |
| `mortality_365d` | `ckpt:hybrid_final_s1` - `ckpt:ar` | -0.001 | [-0.046, 0.039] | 0.940 |
| `mortality_365d` | `ckpt:hybrid_final_s1` - `ckpt:ar_s1` | -0.009 | [-0.041, 0.023] | 0.580 |
| `mortality_365d` | `ckpt:hybrid_final_s1` - `ckpt:ar_s2` | 0.002 | [-0.033, 0.041] | 0.980 |
| `mortality_365d` | `ckpt:hybrid_final_s2` - `ckpt:ar` | -0.004 | [-0.042, 0.031] | 0.740 |
| `mortality_365d` | `ckpt:hybrid_final_s2` - `ckpt:ar_s1` | -0.012 | [-0.053, 0.020] | 0.450 |
| `mortality_365d` | `ckpt:hybrid_final_s2` - `ckpt:ar_s2` | -0.001 | [-0.037, 0.039] | 0.850 |
| `mortality_365d` | `ckpt:ar` - `ckpt:ar_s1` | -0.008 | [-0.042, 0.024] | 0.630 |
| `mortality_365d` | `ckpt:ar` - `ckpt:ar_s2` | 0.003 | [-0.040, 0.044] | 0.850 |
| `mortality_365d` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.011 | [-0.031, 0.053] | 0.520 |
| `inpatient_365d` | `lr` - `gbm` | -0.033 | [-0.043, -0.026] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_final_s0` | -0.047 | [-0.059, -0.038] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_final_s1` | -0.044 | [-0.055, -0.036] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:hybrid_final_s2` | -0.043 | [-0.054, -0.034] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar` | -0.030 | [-0.040, -0.021] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar_s1` | -0.028 | [-0.037, -0.019] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:ar_s2` | -0.031 | [-0.042, -0.021] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_final_s0` | -0.014 | [-0.021, -0.008] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_final_s1` | -0.011 | [-0.017, -0.006] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:hybrid_final_s2` | -0.010 | [-0.017, -0.004] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:ar` | 0.003 | [-0.005, 0.010] | 0.460 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s1` | 0.006 | [-0.003, 0.014] | 0.180 |
| `inpatient_365d` | `gbm` - `ckpt:ar_s2` | 0.002 | [-0.008, 0.008] | 0.650 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s1` | 0.003 | [-0.002, 0.007] | 0.230 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s2` | 0.004 | [-0.001, 0.008] | 0.100 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` - `ckpt:ar` | 0.017 | [0.011, 0.024] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` - `ckpt:ar_s1` | 0.020 | [0.013, 0.027] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` - `ckpt:ar_s2` | 0.016 | [0.009, 0.022] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` - `ckpt:hybrid_final_s2` | 0.001 | [-0.003, 0.005] | 0.600 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` - `ckpt:ar` | 0.014 | [0.007, 0.022] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` - `ckpt:ar_s1` | 0.017 | [0.010, 0.024] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` - `ckpt:ar_s2` | 0.013 | [0.006, 0.020] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s2` - `ckpt:ar` | 0.013 | [0.008, 0.021] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s2` - `ckpt:ar_s1` | 0.016 | [0.009, 0.023] | 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s2` - `ckpt:ar_s2` | 0.012 | [0.005, 0.019] | 0.010 |
| `inpatient_365d` | `ckpt:ar` - `ckpt:ar_s1` | 0.003 | [-0.004, 0.010] | 0.390 |
| `inpatient_365d` | `ckpt:ar` - `ckpt:ar_s2` | -0.001 | [-0.009, 0.006] | 0.710 |
| `inpatient_365d` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.004 | [-0.012, 0.004] | 0.290 |
| `readmission_30d` | `lr` - `gbm` | -0.014 | [-0.044, 0.021] | 0.390 |
| `readmission_30d` | `lr` - `ckpt:hybrid_final_s0` | -0.034 | [-0.063, -0.005] | 0.030 |
| `readmission_30d` | `lr` - `ckpt:hybrid_final_s1` | -0.042 | [-0.070, -0.011] | 0.000 |
| `readmission_30d` | `lr` - `ckpt:hybrid_final_s2` | -0.030 | [-0.061, 0.004] | 0.080 |
| `readmission_30d` | `lr` - `ckpt:ar` | 0.009 | [-0.031, 0.048] | 0.540 |
| `readmission_30d` | `lr` - `ckpt:ar_s1` | -0.013 | [-0.042, 0.025] | 0.590 |
| `readmission_30d` | `lr` - `ckpt:ar_s2` | -0.012 | [-0.045, 0.020] | 0.600 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_final_s0` | -0.019 | [-0.055, 0.010] | 0.190 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_final_s1` | -0.028 | [-0.061, 0.006] | 0.090 |
| `readmission_30d` | `gbm` - `ckpt:hybrid_final_s2` | -0.016 | [-0.048, 0.017] | 0.340 |
| `readmission_30d` | `gbm` - `ckpt:ar` | 0.023 | [-0.014, 0.066] | 0.280 |
| `readmission_30d` | `gbm` - `ckpt:ar_s1` | 0.002 | [-0.033, 0.038] | 0.860 |
| `readmission_30d` | `gbm` - `ckpt:ar_s2` | 0.002 | [-0.029, 0.039] | 0.880 |
| `readmission_30d` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s1` | -0.009 | [-0.025, 0.006] | 0.170 |
| `readmission_30d` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s2` | 0.004 | [-0.011, 0.019] | 0.610 |
| `readmission_30d` | `ckpt:hybrid_final_s0` - `ckpt:ar` | 0.043 | [0.016, 0.072] | 0.000 |
| `readmission_30d` | `ckpt:hybrid_final_s0` - `ckpt:ar_s1` | 0.021 | [-0.002, 0.046] | 0.090 |
| `readmission_30d` | `ckpt:hybrid_final_s0` - `ckpt:ar_s2` | 0.022 | [-0.002, 0.048] | 0.090 |
| `readmission_30d` | `ckpt:hybrid_final_s1` - `ckpt:hybrid_final_s2` | 0.012 | [0.000, 0.028] | 0.040 |
| `readmission_30d` | `ckpt:hybrid_final_s1` - `ckpt:ar` | 0.051 | [0.023, 0.081] | 0.000 |
| `readmission_30d` | `ckpt:hybrid_final_s1` - `ckpt:ar_s1` | 0.030 | [0.001, 0.054] | 0.050 |
| `readmission_30d` | `ckpt:hybrid_final_s1` - `ckpt:ar_s2` | 0.030 | [0.007, 0.055] | 0.020 |
| `readmission_30d` | `ckpt:hybrid_final_s2` - `ckpt:ar` | 0.039 | [0.006, 0.073] | 0.020 |
| `readmission_30d` | `ckpt:hybrid_final_s2` - `ckpt:ar_s1` | 0.017 | [-0.010, 0.048] | 0.210 |
| `readmission_30d` | `ckpt:hybrid_final_s2` - `ckpt:ar_s2` | 0.018 | [-0.008, 0.042] | 0.220 |
| `readmission_30d` | `ckpt:ar` - `ckpt:ar_s1` | -0.022 | [-0.047, 0.006] | 0.120 |
| `readmission_30d` | `ckpt:ar` - `ckpt:ar_s2` | -0.021 | [-0.045, 0.002] | 0.080 |
| `readmission_30d` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.001 | [-0.024, 0.026] | 0.980 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.034 | [-0.045, -0.024] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_final_s0` | -0.042 | [-0.054, -0.033] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_final_s1` | -0.041 | [-0.053, -0.031] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:hybrid_final_s2` | -0.039 | [-0.052, -0.029] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar` | -0.027 | [-0.039, -0.017] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s1` | -0.027 | [-0.038, -0.017] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:ar_s2` | -0.024 | [-0.036, -0.012] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_final_s0` | -0.008 | [-0.017, -0.000] | 0.040 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_final_s1` | -0.007 | [-0.014, 0.001] | 0.100 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:hybrid_final_s2` | -0.005 | [-0.012, 0.003] | 0.140 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar` | 0.008 | [-0.002, 0.017] | 0.140 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s1` | 0.007 | [-0.002, 0.017] | 0.150 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:ar_s2` | 0.010 | [0.001, 0.020] | 0.040 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s1` | 0.001 | [-0.002, 0.005] | 0.490 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s2` | 0.003 | [-0.002, 0.007] | 0.250 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` - `ckpt:ar` | 0.016 | [0.009, 0.023] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` - `ckpt:ar_s1` | 0.016 | [0.009, 0.023] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` - `ckpt:ar_s2` | 0.019 | [0.011, 0.027] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` - `ckpt:hybrid_final_s2` | 0.002 | [-0.003, 0.006] | 0.530 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` - `ckpt:ar` | 0.015 | [0.008, 0.022] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` - `ckpt:ar_s1` | 0.014 | [0.009, 0.021] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` - `ckpt:ar_s2` | 0.017 | [0.010, 0.026] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s2` - `ckpt:ar` | 0.013 | [0.005, 0.021] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s2` - `ckpt:ar_s1` | 0.013 | [0.006, 0.020] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s2` - `ckpt:ar_s2` | 0.016 | [0.007, 0.024] | 0.000 |
| `new_dx_365d/diabetes` | `ckpt:ar` - `ckpt:ar_s1` | -0.000 | [-0.007, 0.008] | 0.960 |
| `new_dx_365d/diabetes` | `ckpt:ar` - `ckpt:ar_s2` | 0.003 | [-0.005, 0.010] | 0.510 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.003 | [-0.005, 0.011] | 0.390 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | -0.043 | [-0.053, -0.031] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_final_s0` | -0.057 | [-0.070, -0.044] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_final_s1` | -0.056 | [-0.067, -0.044] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:hybrid_final_s2` | -0.059 | [-0.072, -0.045] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar` | -0.028 | [-0.041, -0.015] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s1` | -0.031 | [-0.042, -0.019] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:ar_s2` | -0.033 | [-0.045, -0.020] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_final_s0` | -0.014 | [-0.022, -0.006] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_final_s1` | -0.013 | [-0.020, -0.005] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:hybrid_final_s2` | -0.016 | [-0.024, -0.008] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar` | 0.015 | [0.006, 0.023] | 0.010 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s1` | 0.013 | [0.002, 0.024] | 0.030 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:ar_s2` | 0.010 | [-0.001, 0.019] | 0.080 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s1` | 0.001 | [-0.003, 0.005] | 0.630 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s2` | -0.002 | [-0.006, 0.003] | 0.420 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` - `ckpt:ar` | 0.029 | [0.022, 0.037] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` - `ckpt:ar_s1` | 0.027 | [0.017, 0.036] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` - `ckpt:ar_s2` | 0.024 | [0.016, 0.033] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` - `ckpt:hybrid_final_s2` | -0.003 | [-0.007, 0.002] | 0.170 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` - `ckpt:ar` | 0.028 | [0.020, 0.036] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` - `ckpt:ar_s1` | 0.026 | [0.017, 0.034] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` - `ckpt:ar_s2` | 0.023 | [0.014, 0.031] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s2` - `ckpt:ar` | 0.031 | [0.022, 0.039] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s2` - `ckpt:ar_s1` | 0.028 | [0.019, 0.038] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s2` - `ckpt:ar_s2` | 0.026 | [0.018, 0.034] | 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar` - `ckpt:ar_s1` | -0.002 | [-0.012, 0.006] | 0.650 |
| `new_dx_365d/heart_failure` | `ckpt:ar` - `ckpt:ar_s2` | -0.005 | [-0.016, 0.003] | 0.210 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.003 | [-0.013, 0.007] | 0.570 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.032 | [-0.043, -0.020] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_final_s0` | -0.045 | [-0.058, -0.030] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_final_s1` | -0.045 | [-0.060, -0.032] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:hybrid_final_s2` | -0.043 | [-0.055, -0.030] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar` | -0.031 | [-0.044, -0.016] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s1` | -0.027 | [-0.039, -0.014] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:ar_s2` | -0.027 | [-0.041, -0.013] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_final_s0` | -0.013 | [-0.023, -0.004] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_final_s1` | -0.013 | [-0.024, -0.005] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:hybrid_final_s2` | -0.011 | [-0.022, -0.001] | 0.040 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar` | 0.001 | [-0.011, 0.010] | 0.930 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s1` | 0.006 | [-0.006, 0.015] | 0.390 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:ar_s2` | 0.005 | [-0.006, 0.015] | 0.500 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s1` | -0.001 | [-0.005, 0.003] | 0.780 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s2` | 0.002 | [-0.003, 0.007] | 0.320 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` - `ckpt:ar` | 0.014 | [0.005, 0.023] | 0.010 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` - `ckpt:ar_s1` | 0.018 | [0.009, 0.027] | 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` - `ckpt:ar_s2` | 0.018 | [0.009, 0.028] | 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` - `ckpt:hybrid_final_s2` | 0.002 | [-0.002, 0.008] | 0.290 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` - `ckpt:ar` | 0.014 | [0.005, 0.025] | 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` - `ckpt:ar_s1` | 0.019 | [0.009, 0.028] | 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` - `ckpt:ar_s2` | 0.018 | [0.008, 0.029] | 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s2` - `ckpt:ar` | 0.012 | [0.002, 0.021] | 0.030 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s2` - `ckpt:ar_s1` | 0.016 | [0.006, 0.024] | 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s2` - `ckpt:ar_s2` | 0.016 | [0.007, 0.025] | 0.000 |
| `new_dx_365d/ckd` | `ckpt:ar` - `ckpt:ar_s1` | 0.004 | [-0.005, 0.013] | 0.340 |
| `new_dx_365d/ckd` | `ckpt:ar` - `ckpt:ar_s2` | 0.004 | [-0.006, 0.015] | 0.510 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` - `ckpt:ar_s2` | -0.001 | [-0.010, 0.009] | 0.820 |
| `new_dx_365d/copd` | `lr` - `gbm` | -0.035 | [-0.045, -0.023] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_final_s0` | -0.040 | [-0.051, -0.028] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_final_s1` | -0.039 | [-0.052, -0.030] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:hybrid_final_s2` | -0.039 | [-0.051, -0.028] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar` | -0.027 | [-0.040, -0.015] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s1` | -0.031 | [-0.042, -0.019] | 0.000 |
| `new_dx_365d/copd` | `lr` - `ckpt:ar_s2` | -0.026 | [-0.037, -0.014] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_final_s0` | -0.005 | [-0.015, 0.004] | 0.340 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_final_s1` | -0.004 | [-0.015, 0.006] | 0.340 |
| `new_dx_365d/copd` | `gbm` - `ckpt:hybrid_final_s2` | -0.004 | [-0.013, 0.005] | 0.330 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar` | 0.008 | [-0.002, 0.019] | 0.160 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s1` | 0.004 | [-0.006, 0.013] | 0.510 |
| `new_dx_365d/copd` | `gbm` - `ckpt:ar_s2` | 0.010 | [-0.001, 0.020] | 0.070 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s1` | 0.000 | [-0.005, 0.005] | 1.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` - `ckpt:hybrid_final_s2` | 0.000 | [-0.004, 0.006] | 0.960 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` - `ckpt:ar` | 0.013 | [0.003, 0.020] | 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` - `ckpt:ar_s1` | 0.009 | [0.001, 0.017] | 0.050 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` - `ckpt:ar_s2` | 0.014 | [0.007, 0.023] | 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` - `ckpt:hybrid_final_s2` | 0.000 | [-0.005, 0.005] | 0.870 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` - `ckpt:ar` | 0.013 | [0.004, 0.021] | 0.010 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` - `ckpt:ar_s1` | 0.009 | [0.001, 0.017] | 0.050 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` - `ckpt:ar_s2` | 0.014 | [0.006, 0.023] | 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s2` - `ckpt:ar` | 0.013 | [0.004, 0.020] | 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s2` - `ckpt:ar_s1` | 0.008 | [-0.001, 0.016] | 0.070 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s2` - `ckpt:ar_s2` | 0.014 | [0.006, 0.022] | 0.000 |
| `new_dx_365d/copd` | `ckpt:ar` - `ckpt:ar_s1` | -0.004 | [-0.012, 0.004] | 0.350 |
| `new_dx_365d/copd` | `ckpt:ar` - `ckpt:ar_s2` | 0.001 | [-0.007, 0.010] | 0.720 |
| `new_dx_365d/copd` | `ckpt:ar_s1` - `ckpt:ar_s2` | 0.005 | [-0.003, 0.015] | 0.240 |

## Few-shot (k positives + k negatives from train, 5 seeds)

| task | model | k | n train | AUROC mean ± sd | AUPRC mean ± sd |
|---|---|---|---|---|---|
| `mortality_365d` | `lr` | 32 | 64 | 0.529 ± 0.031 | 0.022 ± 0.002 |
| `mortality_365d` | `lr` | 128 | 256 | 0.508 ± 0.038 | 0.021 ± 0.003 |
| `mortality_365d` | `lr` | 512 | 1024 | 0.552 ± 0.018 | 0.023 ± 0.001 |
| `mortality_365d` | `lr` | all | 58192 | 0.566 ± 0.000 | 0.024 ± 0.000 |
| `mortality_365d` | `ckpt:hybrid_final_s0` | 32 | 64 | 0.539 ± 0.031 | 0.024 ± 0.003 |
| `mortality_365d` | `ckpt:hybrid_final_s0` | 128 | 256 | 0.569 ± 0.021 | 0.028 ± 0.002 |
| `mortality_365d` | `ckpt:hybrid_final_s0` | 512 | 1024 | 0.572 ± 0.013 | 0.026 ± 0.002 |
| `mortality_365d` | `ckpt:hybrid_final_s0` | all | 58192 | 0.601 ± 0.000 | 0.028 ± 0.000 |
| `mortality_365d` | `ckpt:hybrid_final_s1` | 32 | 64 | 0.546 ± 0.030 | 0.024 ± 0.002 |
| `mortality_365d` | `ckpt:hybrid_final_s1` | 128 | 256 | 0.561 ± 0.022 | 0.027 ± 0.002 |
| `mortality_365d` | `ckpt:hybrid_final_s1` | 512 | 1024 | 0.572 ± 0.011 | 0.025 ± 0.002 |
| `mortality_365d` | `ckpt:hybrid_final_s1` | all | 58192 | 0.603 ± 0.000 | 0.029 ± 0.000 |
| `mortality_365d` | `ckpt:hybrid_final_s2` | 32 | 64 | 0.539 ± 0.033 | 0.024 ± 0.003 |
| `mortality_365d` | `ckpt:hybrid_final_s2` | 128 | 256 | 0.565 ± 0.022 | 0.028 ± 0.001 |
| `mortality_365d` | `ckpt:hybrid_final_s2` | 512 | 1024 | 0.574 ± 0.010 | 0.026 ± 0.002 |
| `mortality_365d` | `ckpt:hybrid_final_s2` | all | 58192 | 0.600 ± 0.000 | 0.029 ± 0.000 |
| `mortality_365d` | `ckpt:ar` | 32 | 64 | 0.539 ± 0.035 | 0.025 ± 0.004 |
| `mortality_365d` | `ckpt:ar` | 128 | 256 | 0.559 ± 0.022 | 0.025 ± 0.002 |
| `mortality_365d` | `ckpt:ar` | 512 | 1024 | 0.583 ± 0.013 | 0.026 ± 0.002 |
| `mortality_365d` | `ckpt:ar` | all | 58192 | 0.604 ± 0.000 | 0.027 ± 0.000 |
| `mortality_365d` | `ckpt:ar_s1` | 32 | 64 | 0.541 ± 0.037 | 0.025 ± 0.005 |
| `mortality_365d` | `ckpt:ar_s1` | 128 | 256 | 0.551 ± 0.027 | 0.026 ± 0.003 |
| `mortality_365d` | `ckpt:ar_s1` | 512 | 1024 | 0.577 ± 0.014 | 0.029 ± 0.005 |
| `mortality_365d` | `ckpt:ar_s1` | all | 58192 | 0.612 ± 0.000 | 0.028 ± 0.000 |
| `mortality_365d` | `ckpt:ar_s2` | 32 | 64 | 0.530 ± 0.047 | 0.024 ± 0.005 |
| `mortality_365d` | `ckpt:ar_s2` | 128 | 256 | 0.552 ± 0.024 | 0.026 ± 0.003 |
| `mortality_365d` | `ckpt:ar_s2` | 512 | 1024 | 0.578 ± 0.012 | 0.026 ± 0.002 |
| `mortality_365d` | `ckpt:ar_s2` | all | 58192 | 0.601 ± 0.000 | 0.026 ± 0.000 |
| `inpatient_365d` | `lr` | 32 | 64 | 0.598 ± 0.039 | 0.276 ± 0.018 |
| `inpatient_365d` | `lr` | 128 | 256 | 0.661 ± 0.006 | 0.317 ± 0.013 |
| `inpatient_365d` | `lr` | 512 | 1024 | 0.677 ± 0.002 | 0.329 ± 0.007 |
| `inpatient_365d` | `lr` | all | 58192 | 0.712 ± 0.000 | 0.377 ± 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` | 32 | 64 | 0.634 ± 0.019 | 0.297 ± 0.016 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` | 128 | 256 | 0.694 ± 0.009 | 0.354 ± 0.015 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` | 512 | 1024 | 0.728 ± 0.003 | 0.404 ± 0.015 |
| `inpatient_365d` | `ckpt:hybrid_final_s0` | all | 58192 | 0.760 ± 0.000 | 0.454 ± 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` | 32 | 64 | 0.642 ± 0.018 | 0.305 ± 0.017 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` | 128 | 256 | 0.695 ± 0.008 | 0.361 ± 0.019 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` | 512 | 1024 | 0.730 ± 0.002 | 0.402 ± 0.012 |
| `inpatient_365d` | `ckpt:hybrid_final_s1` | all | 58192 | 0.757 ± 0.000 | 0.452 ± 0.000 |
| `inpatient_365d` | `ckpt:hybrid_final_s2` | 32 | 64 | 0.646 ± 0.016 | 0.308 ± 0.013 |
| `inpatient_365d` | `ckpt:hybrid_final_s2` | 128 | 256 | 0.698 ± 0.008 | 0.357 ± 0.013 |
| `inpatient_365d` | `ckpt:hybrid_final_s2` | 512 | 1024 | 0.725 ± 0.003 | 0.395 ± 0.010 |
| `inpatient_365d` | `ckpt:hybrid_final_s2` | all | 58192 | 0.756 ± 0.000 | 0.446 ± 0.000 |
| `inpatient_365d` | `ckpt:ar` | 32 | 64 | 0.618 ± 0.025 | 0.289 ± 0.022 |
| `inpatient_365d` | `ckpt:ar` | 128 | 256 | 0.682 ± 0.004 | 0.344 ± 0.008 |
| `inpatient_365d` | `ckpt:ar` | 512 | 1024 | 0.710 ± 0.004 | 0.375 ± 0.013 |
| `inpatient_365d` | `ckpt:ar` | all | 58192 | 0.743 ± 0.000 | 0.423 ± 0.000 |
| `inpatient_365d` | `ckpt:ar_s1` | 32 | 64 | 0.611 ± 0.031 | 0.284 ± 0.024 |
| `inpatient_365d` | `ckpt:ar_s1` | 128 | 256 | 0.672 ± 0.003 | 0.332 ± 0.007 |
| `inpatient_365d` | `ckpt:ar_s1` | 512 | 1024 | 0.704 ± 0.004 | 0.363 ± 0.011 |
| `inpatient_365d` | `ckpt:ar_s1` | all | 58192 | 0.740 ± 0.000 | 0.409 ± 0.000 |
| `inpatient_365d` | `ckpt:ar_s2` | 32 | 64 | 0.609 ± 0.021 | 0.286 ± 0.018 |
| `inpatient_365d` | `ckpt:ar_s2` | 128 | 256 | 0.678 ± 0.010 | 0.343 ± 0.011 |
| `inpatient_365d` | `ckpt:ar_s2` | 512 | 1024 | 0.712 ± 0.004 | 0.376 ± 0.014 |
| `inpatient_365d` | `ckpt:ar_s2` | all | 58192 | 0.743 ± 0.000 | 0.425 ± 0.000 |
| `readmission_30d` | `lr` | 32 | 64 | 0.568 ± 0.028 | 0.066 ± 0.008 |
| `readmission_30d` | `lr` | 128 | 256 | 0.580 ± 0.022 | 0.076 ± 0.005 |
| `readmission_30d` | `lr` | 512 | 1024 | 0.620 ± 0.005 | 0.081 ± 0.002 |
| `readmission_30d` | `lr` | all | 24417 | 0.653 ± 0.000 | 0.097 ± 0.000 |
| `readmission_30d` | `ckpt:hybrid_final_s0` | 32 | 64 | 0.618 ± 0.034 | 0.079 ± 0.010 |
| `readmission_30d` | `ckpt:hybrid_final_s0` | 128 | 256 | 0.652 ± 0.021 | 0.091 ± 0.005 |
| `readmission_30d` | `ckpt:hybrid_final_s0` | 512 | 1024 | 0.673 ± 0.007 | 0.104 ± 0.004 |
| `readmission_30d` | `ckpt:hybrid_final_s0` | all | 24417 | 0.687 ± 0.000 | 0.111 ± 0.000 |
| `readmission_30d` | `ckpt:hybrid_final_s1` | 32 | 64 | 0.623 ± 0.025 | 0.078 ± 0.008 |
| `readmission_30d` | `ckpt:hybrid_final_s1` | 128 | 256 | 0.644 ± 0.020 | 0.083 ± 0.004 |
| `readmission_30d` | `ckpt:hybrid_final_s1` | 512 | 1024 | 0.678 ± 0.005 | 0.101 ± 0.002 |
| `readmission_30d` | `ckpt:hybrid_final_s1` | all | 24417 | 0.695 ± 0.000 | 0.108 ± 0.000 |
| `readmission_30d` | `ckpt:hybrid_final_s2` | 32 | 64 | 0.607 ± 0.028 | 0.075 ± 0.008 |
| `readmission_30d` | `ckpt:hybrid_final_s2` | 128 | 256 | 0.647 ± 0.012 | 0.083 ± 0.005 |
| `readmission_30d` | `ckpt:hybrid_final_s2` | 512 | 1024 | 0.674 ± 0.005 | 0.096 ± 0.005 |
| `readmission_30d` | `ckpt:hybrid_final_s2` | all | 24417 | 0.683 ± 0.000 | 0.099 ± 0.000 |
| `readmission_30d` | `ckpt:ar` | 32 | 64 | 0.547 ± 0.028 | 0.065 ± 0.011 |
| `readmission_30d` | `ckpt:ar` | 128 | 256 | 0.570 ± 0.034 | 0.067 ± 0.008 |
| `readmission_30d` | `ckpt:ar` | 512 | 1024 | 0.612 ± 0.012 | 0.081 ± 0.002 |
| `readmission_30d` | `ckpt:ar` | all | 24417 | 0.644 ± 0.000 | 0.086 ± 0.000 |
| `readmission_30d` | `ckpt:ar_s1` | 32 | 64 | 0.574 ± 0.016 | 0.065 ± 0.004 |
| `readmission_30d` | `ckpt:ar_s1` | 128 | 256 | 0.588 ± 0.034 | 0.069 ± 0.008 |
| `readmission_30d` | `ckpt:ar_s1` | 512 | 1024 | 0.637 ± 0.005 | 0.089 ± 0.005 |
| `readmission_30d` | `ckpt:ar_s1` | all | 24417 | 0.665 ± 0.000 | 0.088 ± 0.000 |
| `readmission_30d` | `ckpt:ar_s2` | 32 | 64 | 0.581 ± 0.017 | 0.072 ± 0.007 |
| `readmission_30d` | `ckpt:ar_s2` | 128 | 256 | 0.591 ± 0.027 | 0.073 ± 0.008 |
| `readmission_30d` | `ckpt:ar_s2` | 512 | 1024 | 0.639 ± 0.011 | 0.103 ± 0.007 |
| `readmission_30d` | `ckpt:ar_s2` | all | 24417 | 0.665 ± 0.000 | 0.102 ± 0.000 |
| `new_dx_365d/diabetes` | `lr` | 32 | 64 | 0.679 ± 0.018 | 0.338 ± 0.011 |
| `new_dx_365d/diabetes` | `lr` | 128 | 256 | 0.701 ± 0.002 | 0.357 ± 0.005 |
| `new_dx_365d/diabetes` | `lr` | 512 | 1024 | 0.708 ± 0.001 | 0.367 ± 0.004 |
| `new_dx_365d/diabetes` | `lr` | all | 39458 | 0.737 ± 0.000 | 0.421 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` | 32 | 64 | 0.678 ± 0.032 | 0.344 ± 0.026 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` | 128 | 256 | 0.731 ± 0.010 | 0.400 ± 0.014 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` | 512 | 1024 | 0.758 ± 0.006 | 0.435 ± 0.006 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s0` | all | 39458 | 0.779 ± 0.000 | 0.486 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` | 32 | 64 | 0.679 ± 0.028 | 0.346 ± 0.023 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` | 128 | 256 | 0.730 ± 0.006 | 0.401 ± 0.008 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` | 512 | 1024 | 0.756 ± 0.005 | 0.433 ± 0.009 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s1` | all | 39458 | 0.778 ± 0.000 | 0.482 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s2` | 32 | 64 | 0.682 ± 0.030 | 0.341 ± 0.025 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s2` | 128 | 256 | 0.727 ± 0.012 | 0.399 ± 0.017 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s2` | 512 | 1024 | 0.754 ± 0.006 | 0.430 ± 0.010 |
| `new_dx_365d/diabetes` | `ckpt:hybrid_final_s2` | all | 39458 | 0.776 ± 0.000 | 0.481 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:ar` | 32 | 64 | 0.656 ± 0.035 | 0.332 ± 0.032 |
| `new_dx_365d/diabetes` | `ckpt:ar` | 128 | 256 | 0.707 ± 0.010 | 0.380 ± 0.012 |
| `new_dx_365d/diabetes` | `ckpt:ar` | 512 | 1024 | 0.737 ± 0.004 | 0.415 ± 0.010 |
| `new_dx_365d/diabetes` | `ckpt:ar` | all | 39458 | 0.763 ± 0.000 | 0.454 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | 32 | 64 | 0.649 ± 0.036 | 0.328 ± 0.028 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | 128 | 256 | 0.706 ± 0.012 | 0.382 ± 0.012 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | 512 | 1024 | 0.736 ± 0.006 | 0.418 ± 0.016 |
| `new_dx_365d/diabetes` | `ckpt:ar_s1` | all | 39458 | 0.764 ± 0.000 | 0.454 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | 32 | 64 | 0.645 ± 0.048 | 0.322 ± 0.031 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | 128 | 256 | 0.702 ± 0.015 | 0.371 ± 0.013 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | 512 | 1024 | 0.736 ± 0.006 | 0.407 ± 0.011 |
| `new_dx_365d/diabetes` | `ckpt:ar_s2` | all | 39458 | 0.761 ± 0.000 | 0.446 ± 0.000 |
| `new_dx_365d/heart_failure` | `lr` | 32 | 64 | 0.659 ± 0.021 | 0.201 ± 0.011 |
| `new_dx_365d/heart_failure` | `lr` | 128 | 256 | 0.672 ± 0.007 | 0.205 ± 0.009 |
| `new_dx_365d/heart_failure` | `lr` | 512 | 1024 | 0.695 ± 0.005 | 0.224 ± 0.004 |
| `new_dx_365d/heart_failure` | `lr` | all | 48770 | 0.741 ± 0.000 | 0.263 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` | 32 | 64 | 0.680 ± 0.046 | 0.235 ± 0.029 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` | 128 | 256 | 0.725 ± 0.010 | 0.260 ± 0.022 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` | 512 | 1024 | 0.770 ± 0.002 | 0.313 ± 0.009 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s0` | all | 48770 | 0.798 ± 0.000 | 0.358 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` | 32 | 64 | 0.685 ± 0.045 | 0.236 ± 0.034 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` | 128 | 256 | 0.723 ± 0.007 | 0.257 ± 0.020 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` | 512 | 1024 | 0.772 ± 0.002 | 0.314 ± 0.008 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s1` | all | 48770 | 0.797 ± 0.000 | 0.342 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s2` | 32 | 64 | 0.680 ± 0.048 | 0.230 ± 0.033 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s2` | 128 | 256 | 0.721 ± 0.004 | 0.257 ± 0.016 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s2` | 512 | 1024 | 0.772 ± 0.003 | 0.311 ± 0.014 |
| `new_dx_365d/heart_failure` | `ckpt:hybrid_final_s2` | all | 48770 | 0.800 ± 0.000 | 0.350 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | 32 | 64 | 0.648 ± 0.062 | 0.215 ± 0.028 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | 128 | 256 | 0.698 ± 0.007 | 0.240 ± 0.014 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | 512 | 1024 | 0.740 ± 0.005 | 0.273 ± 0.009 |
| `new_dx_365d/heart_failure` | `ckpt:ar` | all | 48770 | 0.769 ± 0.000 | 0.317 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | 32 | 64 | 0.654 ± 0.051 | 0.216 ± 0.031 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | 128 | 256 | 0.698 ± 0.008 | 0.236 ± 0.009 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | 512 | 1024 | 0.736 ± 0.004 | 0.271 ± 0.011 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s1` | all | 48770 | 0.771 ± 0.000 | 0.302 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | 32 | 64 | 0.644 ± 0.050 | 0.207 ± 0.028 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | 128 | 256 | 0.690 ± 0.011 | 0.240 ± 0.018 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | 512 | 1024 | 0.742 ± 0.002 | 0.280 ± 0.009 |
| `new_dx_365d/heart_failure` | `ckpt:ar_s2` | all | 48770 | 0.774 ± 0.000 | 0.319 ± 0.000 |
| `new_dx_365d/ckd` | `lr` | 32 | 64 | 0.640 ± 0.057 | 0.159 ± 0.016 |
| `new_dx_365d/ckd` | `lr` | 128 | 256 | 0.680 ± 0.002 | 0.168 ± 0.006 |
| `new_dx_365d/ckd` | `lr` | 512 | 1024 | 0.697 ± 0.004 | 0.187 ± 0.003 |
| `new_dx_365d/ckd` | `lr` | all | 51106 | 0.739 ± 0.000 | 0.246 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` | 32 | 64 | 0.657 ± 0.026 | 0.176 ± 0.029 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` | 128 | 256 | 0.728 ± 0.012 | 0.216 ± 0.019 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` | 512 | 1024 | 0.756 ± 0.006 | 0.249 ± 0.015 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s0` | all | 51106 | 0.784 ± 0.000 | 0.299 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` | 32 | 64 | 0.661 ± 0.029 | 0.180 ± 0.033 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` | 128 | 256 | 0.731 ± 0.008 | 0.218 ± 0.014 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` | 512 | 1024 | 0.759 ± 0.005 | 0.253 ± 0.013 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s1` | all | 51106 | 0.784 ± 0.000 | 0.297 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s2` | 32 | 64 | 0.664 ± 0.027 | 0.184 ± 0.026 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s2` | 128 | 256 | 0.733 ± 0.011 | 0.217 ± 0.019 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s2` | 512 | 1024 | 0.756 ± 0.005 | 0.252 ± 0.012 |
| `new_dx_365d/ckd` | `ckpt:hybrid_final_s2` | all | 51106 | 0.782 ± 0.000 | 0.297 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:ar` | 32 | 64 | 0.637 ± 0.017 | 0.166 ± 0.019 |
| `new_dx_365d/ckd` | `ckpt:ar` | 128 | 256 | 0.702 ± 0.010 | 0.202 ± 0.014 |
| `new_dx_365d/ckd` | `ckpt:ar` | 512 | 1024 | 0.730 ± 0.009 | 0.220 ± 0.011 |
| `new_dx_365d/ckd` | `ckpt:ar` | all | 51106 | 0.770 ± 0.000 | 0.275 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | 32 | 64 | 0.639 ± 0.030 | 0.166 ± 0.021 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | 128 | 256 | 0.703 ± 0.008 | 0.198 ± 0.010 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | 512 | 1024 | 0.730 ± 0.005 | 0.221 ± 0.006 |
| `new_dx_365d/ckd` | `ckpt:ar_s1` | all | 51106 | 0.766 ± 0.000 | 0.259 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | 32 | 64 | 0.633 ± 0.034 | 0.160 ± 0.020 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | 128 | 256 | 0.712 ± 0.008 | 0.210 ± 0.009 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | 512 | 1024 | 0.731 ± 0.004 | 0.231 ± 0.004 |
| `new_dx_365d/ckd` | `ckpt:ar_s2` | all | 51106 | 0.766 ± 0.000 | 0.271 ± 0.000 |
| `new_dx_365d/copd` | `lr` | 32 | 64 | 0.629 ± 0.069 | 0.195 ± 0.027 |
| `new_dx_365d/copd` | `lr` | 128 | 256 | 0.672 ± 0.005 | 0.218 ± 0.005 |
| `new_dx_365d/copd` | `lr` | 512 | 1024 | 0.687 ± 0.006 | 0.236 ± 0.006 |
| `new_dx_365d/copd` | `lr` | all | 48015 | 0.733 ± 0.000 | 0.310 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` | 32 | 64 | 0.665 ± 0.046 | 0.228 ± 0.027 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` | 128 | 256 | 0.718 ± 0.006 | 0.277 ± 0.009 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` | 512 | 1024 | 0.748 ± 0.009 | 0.321 ± 0.016 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s0` | all | 48015 | 0.772 ± 0.000 | 0.371 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` | 32 | 64 | 0.668 ± 0.036 | 0.234 ± 0.018 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` | 128 | 256 | 0.719 ± 0.007 | 0.293 ± 0.008 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` | 512 | 1024 | 0.752 ± 0.007 | 0.334 ± 0.016 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s1` | all | 48015 | 0.772 ± 0.000 | 0.370 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s2` | 32 | 64 | 0.672 ± 0.041 | 0.235 ± 0.020 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s2` | 128 | 256 | 0.721 ± 0.009 | 0.285 ± 0.009 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s2` | 512 | 1024 | 0.748 ± 0.009 | 0.324 ± 0.019 |
| `new_dx_365d/copd` | `ckpt:hybrid_final_s2` | all | 48015 | 0.772 ± 0.000 | 0.365 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:ar` | 32 | 64 | 0.642 ± 0.051 | 0.211 ± 0.031 |
| `new_dx_365d/copd` | `ckpt:ar` | 128 | 256 | 0.702 ± 0.011 | 0.266 ± 0.008 |
| `new_dx_365d/copd` | `ckpt:ar` | 512 | 1024 | 0.733 ± 0.005 | 0.302 ± 0.009 |
| `new_dx_365d/copd` | `ckpt:ar` | all | 48015 | 0.759 ± 0.000 | 0.341 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | 32 | 64 | 0.646 ± 0.040 | 0.209 ± 0.025 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | 128 | 256 | 0.700 ± 0.012 | 0.260 ± 0.006 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | 512 | 1024 | 0.735 ± 0.007 | 0.302 ± 0.015 |
| `new_dx_365d/copd` | `ckpt:ar_s1` | all | 48015 | 0.763 ± 0.000 | 0.334 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | 32 | 64 | 0.630 ± 0.058 | 0.211 ± 0.035 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | 128 | 256 | 0.691 ± 0.011 | 0.250 ± 0.012 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | 512 | 1024 | 0.728 ± 0.006 | 0.287 ± 0.009 |
| `new_dx_365d/copd` | `ckpt:ar_s2` | all | 48015 | 0.758 ± 0.000 | 0.316 ± 0.000 |
