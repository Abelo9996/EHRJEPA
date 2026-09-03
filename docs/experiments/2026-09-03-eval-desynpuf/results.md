# Downstream evaluation -- desynpuf-s1

**2026-09-03 addendum:** `lr` below is a refit, not the original run. The
original `baselines.fit_logistic` scaled sparse count features with
`StandardScaler(with_mean=False)`, which inflated rare (`min_df`-edge)
columns' scale and forced the tuned `C` to the low edge of `LOGISTIC_GRID` on
every task, capping `lr` around .55-.59 AUROC -- below even the `random_init`
control. The fix swaps in a `TfidfTransformer` for sparse inputs (dense
embedding probes are unaffected); see the README's addendum section for the
full comparison and root-cause writeup. Every other column (`gbm`,
`random_init`, both checkpoint probes) is unchanged from the original run.

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `6ea4b31` |
| created | 2026-09-03T15:14:43+00:00 |
| runtime (s) | 1615.4 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |
| `random_init` | probe | `runs/sanity-A-default/final.pt` | cls+mean |
| `ckpt:sanity-A-default` | probe | `runs/sanity-A-default/final.pt` | cls+mean |
| `ckpt:sanity-C-ema` | probe | `runs/sanity-C-ema/final.pt` | cls+mean |

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

| task | lr | gbm | random_init | ckpt:sanity-A-default | ckpt:sanity-C-ema |
|---|---|---|---|---|---|
| `mortality_365d` | 0.554 [0.483, 0.608] | 0.572 [0.481, 0.636] | 0.591 [0.516, 0.664] | 0.582 [0.517, 0.661] | 0.602 [0.534, 0.674] |
| `inpatient_365d` | 0.708 [0.683, 0.727] | 0.744 [0.719, 0.764] | 0.672 [0.646, 0.693] | 0.682 [0.659, 0.704] | 0.681 [0.656, 0.702] |
| `readmission_30d` | 0.658 [0.610, 0.708] | 0.672 [0.624, 0.717] | 0.652 [0.604, 0.686] | 0.623 [0.573, 0.670] | 0.657 [0.614, 0.707] |
| `new_dx_365d/diabetes` | 0.740 [0.721, 0.760] | 0.772 [0.755, 0.792] | 0.737 [0.716, 0.758] | 0.734 [0.713, 0.757] | 0.735 [0.716, 0.758] |
| `new_dx_365d/heart_failure` | 0.744 [0.722, 0.766] | 0.789 [0.773, 0.810] | 0.699 [0.678, 0.725] | 0.682 [0.657, 0.707] | 0.693 [0.670, 0.719] |
| `new_dx_365d/ckd` | 0.736 [0.715, 0.765] | 0.765 [0.741, 0.789] | 0.700 [0.675, 0.728] | 0.695 [0.666, 0.722] | 0.692 [0.662, 0.720] |
| `new_dx_365d/copd` | 0.726 [0.696, 0.751] | 0.767 [0.738, 0.791] | 0.700 [0.673, 0.726] | 0.703 [0.676, 0.725] | 0.700 [0.674, 0.724] |

## AUPRC

| task | lr | gbm | random_init | ckpt:sanity-A-default | ckpt:sanity-C-ema |
|---|---|---|---|---|---|
| `mortality_365d` | 0.026 [0.019, 0.043] | 0.028 [0.021, 0.042] | 0.033 [0.023, 0.057] | 0.030 [0.022, 0.046] | 0.031 [0.023, 0.047] |
| `inpatient_365d` | 0.363 [0.329, 0.405] | 0.411 [0.369, 0.453] | 0.318 [0.284, 0.350] | 0.351 [0.318, 0.387] | 0.345 [0.307, 0.378] |
| `readmission_30d` | 0.098 [0.076, 0.140] | 0.107 [0.082, 0.145] | 0.091 [0.069, 0.121] | 0.087 [0.067, 0.123] | 0.093 [0.074, 0.128] |
| `new_dx_365d/diabetes` | 0.432 [0.393, 0.471] | 0.469 [0.432, 0.510] | 0.417 [0.383, 0.459] | 0.409 [0.374, 0.447] | 0.417 [0.383, 0.460] |
| `new_dx_365d/heart_failure` | 0.276 [0.243, 0.313] | 0.344 [0.303, 0.396] | 0.259 [0.229, 0.302] | 0.237 [0.208, 0.273] | 0.252 [0.220, 0.294] |
| `new_dx_365d/ckd` | 0.236 [0.200, 0.279] | 0.259 [0.223, 0.308] | 0.174 [0.152, 0.210] | 0.178 [0.156, 0.211] | 0.175 [0.154, 0.206] |
| `new_dx_365d/copd` | 0.317 [0.270, 0.363] | 0.344 [0.296, 0.393] | 0.235 [0.210, 0.269] | 0.240 [0.209, 0.277] | 0.233 [0.206, 0.269] |

## BRIER

| task | lr | gbm | random_init | ckpt:sanity-A-default | ckpt:sanity-C-ema |
|---|---|---|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] | 0.0216 [0.0173, 0.0266] | 0.0215 [0.0173, 0.0265] | 0.0215 [0.0173, 0.0266] | 0.0216 [0.0174, 0.0266] |
| `inpatient_365d` | 0.1474 [0.1406, 0.1554] | 0.1406 [0.1342, 0.1468] | 0.1512 [0.1441, 0.1588] | 0.1497 [0.1426, 0.1567] | 0.1497 [0.1425, 0.1571] |
| `readmission_30d` | 0.0461 [0.0403, 0.0520] | 0.0461 [0.0399, 0.0522] | 0.0464 [0.0403, 0.0522] | 0.0465 [0.0407, 0.0523] | 0.0463 [0.0407, 0.0519] |
| `new_dx_365d/diabetes` | 0.1529 [0.1459, 0.1615] | 0.1458 [0.1387, 0.1530] | 0.1532 [0.1462, 0.1608] | 0.1537 [0.1471, 0.1612] | 0.1532 [0.1461, 0.1607] |
| `new_dx_365d/heart_failure` | 0.1101 [0.1016, 0.1168] | 0.1028 [0.0947, 0.1094] | 0.1112 [0.1023, 0.1185] | 0.1124 [0.1038, 0.1196] | 0.1116 [0.1026, 0.1188] |
| `new_dx_365d/ckd` | 0.0818 [0.0739, 0.0888] | 0.0799 [0.0719, 0.0867] | 0.0841 [0.0756, 0.0907] | 0.0841 [0.0758, 0.0911] | 0.0842 [0.0759, 0.0914] |
| `new_dx_365d/copd` | 0.1038 [0.0962, 0.1117] | 0.0988 [0.0910, 0.1068] | 0.1071 [0.0986, 0.1154] | 0.1069 [0.0986, 0.1154] | 0.1072 [0.0988, 0.1157] |

## CALIBRATION SLOPE

| task | lr | gbm | random_init | ckpt:sanity-A-default | ckpt:sanity-C-ema |
|---|---|---|---|---|---|
| `mortality_365d` | 2.421 [-2.027, 5.716] | 0.425 [-0.285, 0.861] | 0.654 [0.242, 1.142] | 0.677 [0.231, 1.221] | 0.463 [0.183, 0.745] |
| `inpatient_365d` | 1.154 [1.006, 1.290] | 1.063 [0.931, 1.194] | 0.948 [0.801, 1.091] | 1.048 [0.900, 1.229] | 1.033 [0.875, 1.184] |
| `readmission_30d` | 1.062 [0.756, 1.393] | 0.741 [0.523, 0.941] | 0.856 [0.581, 1.083] | 0.873 [0.528, 1.234] | 0.988 [0.716, 1.302] |
| `new_dx_365d/diabetes` | 1.032 [0.924, 1.138] | 1.026 [0.933, 1.146] | 0.924 [0.825, 1.043] | 0.975 [0.862, 1.125] | 0.983 [0.884, 1.123] |
| `new_dx_365d/heart_failure` | 1.302 [1.176, 1.474] | 1.055 [0.961, 1.188] | 0.958 [0.840, 1.139] | 0.881 [0.760, 1.034] | 0.935 [0.820, 1.086] |
| `new_dx_365d/ckd` | 1.032 [0.914, 1.186] | 0.972 [0.875, 1.099] | 0.882 [0.737, 1.041] | 0.947 [0.761, 1.163] | 0.899 [0.755, 1.077] |
| `new_dx_365d/copd` | 1.444 [1.263, 1.611] | 1.011 [0.879, 1.124] | 0.972 [0.820, 1.106] | 0.996 [0.826, 1.168] | 0.976 [0.824, 1.131] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `gbm` - `random_init` | -0.019 | [-0.110, 0.050] | 0.630 |
| `mortality_365d` | `gbm` - `ckpt:sanity-A-default` | -0.010 | [-0.112, 0.058] | 0.730 |
| `mortality_365d` | `gbm` - `ckpt:sanity-C-ema` | -0.030 | [-0.122, 0.036] | 0.380 |
| `mortality_365d` | `random_init` - `ckpt:sanity-A-default` | 0.009 | [-0.036, 0.056] | 0.810 |
| `mortality_365d` | `random_init` - `ckpt:sanity-C-ema` | -0.011 | [-0.059, 0.048] | 0.690 |
| `mortality_365d` | `ckpt:sanity-A-default` - `ckpt:sanity-C-ema` | -0.020 | [-0.066, 0.036] | 0.460 |
| `mortality_365d` | `lr` - `gbm` | -0.019 | [-0.085, 0.062] | 0.680 |
| `mortality_365d` | `lr` - `random_init` | -0.038 | [-0.149, 0.052] | 0.400 |
| `mortality_365d` | `lr` - `ckpt:sanity-A-default` | -0.028 | [-0.118, 0.058] | 0.430 |
| `mortality_365d` | `lr` - `ckpt:sanity-C-ema` | -0.048 | [-0.140, 0.036] | 0.240 |
| `inpatient_365d` | `gbm` - `random_init` | 0.072 | [0.051, 0.093] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:sanity-A-default` | 0.062 | [0.040, 0.085] | 0.000 |
| `inpatient_365d` | `gbm` - `ckpt:sanity-C-ema` | 0.063 | [0.043, 0.085] | 0.000 |
| `inpatient_365d` | `random_init` - `ckpt:sanity-A-default` | -0.010 | [-0.022, 0.004] | 0.120 |
| `inpatient_365d` | `random_init` - `ckpt:sanity-C-ema` | -0.009 | [-0.020, 0.003] | 0.190 |
| `inpatient_365d` | `ckpt:sanity-A-default` - `ckpt:sanity-C-ema` | 0.001 | [-0.009, 0.012] | 0.870 |
| `inpatient_365d` | `lr` - `gbm` | -0.036 | [-0.048, -0.023] | 0.000 |
| `inpatient_365d` | `lr` - `random_init` | 0.035 | [0.014, 0.055] | 0.000 |
| `inpatient_365d` | `lr` - `ckpt:sanity-A-default` | 0.026 | [0.004, 0.047] | 0.020 |
| `inpatient_365d` | `lr` - `ckpt:sanity-C-ema` | 0.027 | [0.006, 0.049] | 0.030 |
| `readmission_30d` | `gbm` - `random_init` | 0.020 | [-0.030, 0.069] | 0.430 |
| `readmission_30d` | `gbm` - `ckpt:sanity-A-default` | 0.049 | [-0.002, 0.094] | 0.070 |
| `readmission_30d` | `gbm` - `ckpt:sanity-C-ema` | 0.015 | [-0.030, 0.061] | 0.580 |
| `readmission_30d` | `random_init` - `ckpt:sanity-A-default` | 0.029 | [-0.005, 0.062] | 0.080 |
| `readmission_30d` | `random_init` - `ckpt:sanity-C-ema` | -0.005 | [-0.036, 0.020] | 0.520 |
| `readmission_30d` | `ckpt:sanity-A-default` - `ckpt:sanity-C-ema` | -0.034 | [-0.061, -0.009] | 0.020 |
| `readmission_30d` | `lr` - `gbm` | -0.015 | [-0.047, 0.018] | 0.460 |
| `readmission_30d` | `lr` - `random_init` | 0.006 | [-0.036, 0.049] | 0.690 |
| `readmission_30d` | `lr` - `ckpt:sanity-A-default` | 0.035 | [-0.006, 0.077] | 0.140 |
| `readmission_30d` | `lr` - `ckpt:sanity-C-ema` | 0.000 | [-0.038, 0.041] | 1.000 |
| `new_dx_365d/diabetes` | `gbm` - `random_init` | 0.035 | [0.021, 0.052] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:sanity-A-default` | 0.038 | [0.021, 0.053] | 0.000 |
| `new_dx_365d/diabetes` | `gbm` - `ckpt:sanity-C-ema` | 0.037 | [0.021, 0.053] | 0.000 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:sanity-A-default` | 0.003 | [-0.006, 0.011] | 0.560 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:sanity-C-ema` | 0.002 | [-0.007, 0.011] | 0.660 |
| `new_dx_365d/diabetes` | `ckpt:sanity-A-default` - `ckpt:sanity-C-ema` | -0.001 | [-0.009, 0.008] | 0.810 |
| `new_dx_365d/diabetes` | `lr` - `gbm` | -0.032 | [-0.046, -0.020] | 0.000 |
| `new_dx_365d/diabetes` | `lr` - `random_init` | 0.002 | [-0.013, 0.016] | 0.760 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:sanity-A-default` | 0.005 | [-0.010, 0.022] | 0.470 |
| `new_dx_365d/diabetes` | `lr` - `ckpt:sanity-C-ema` | 0.004 | [-0.013, 0.019] | 0.540 |
| `new_dx_365d/heart_failure` | `gbm` - `random_init` | 0.090 | [0.068, 0.110] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:sanity-A-default` | 0.107 | [0.086, 0.130] | 0.000 |
| `new_dx_365d/heart_failure` | `gbm` - `ckpt:sanity-C-ema` | 0.097 | [0.073, 0.117] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:sanity-A-default` | 0.017 | [0.008, 0.029] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:sanity-C-ema` | 0.007 | [-0.003, 0.016] | 0.200 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-A-default` - `ckpt:sanity-C-ema` | -0.010 | [-0.020, -0.001] | 0.040 |
| `new_dx_365d/heart_failure` | `lr` - `gbm` | -0.045 | [-0.060, -0.031] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `random_init` | 0.044 | [0.022, 0.067] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:sanity-A-default` | 0.062 | [0.039, 0.085] | 0.000 |
| `new_dx_365d/heart_failure` | `lr` - `ckpt:sanity-C-ema` | 0.051 | [0.028, 0.074] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `random_init` | 0.065 | [0.038, 0.090] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:sanity-A-default` | 0.071 | [0.044, 0.095] | 0.000 |
| `new_dx_365d/ckd` | `gbm` - `ckpt:sanity-C-ema` | 0.073 | [0.047, 0.101] | 0.000 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:sanity-A-default` | 0.006 | [-0.008, 0.020] | 0.490 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:sanity-C-ema` | 0.008 | [-0.006, 0.026] | 0.340 |
| `new_dx_365d/ckd` | `ckpt:sanity-A-default` - `ckpt:sanity-C-ema` | 0.003 | [-0.009, 0.014] | 0.620 |
| `new_dx_365d/ckd` | `lr` - `gbm` | -0.029 | [-0.049, -0.012] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `random_init` | 0.036 | [0.011, 0.062] | 0.010 |
| `new_dx_365d/ckd` | `lr` - `ckpt:sanity-A-default` | 0.041 | [0.013, 0.067] | 0.000 |
| `new_dx_365d/ckd` | `lr` - `ckpt:sanity-C-ema` | 0.044 | [0.014, 0.074] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `random_init` | 0.067 | [0.041, 0.091] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `ckpt:sanity-A-default` | 0.065 | [0.036, 0.088] | 0.000 |
| `new_dx_365d/copd` | `gbm` - `ckpt:sanity-C-ema` | 0.067 | [0.042, 0.091] | 0.000 |
| `new_dx_365d/copd` | `random_init` - `ckpt:sanity-A-default` | -0.002 | [-0.015, 0.010] | 0.800 |
| `new_dx_365d/copd` | `random_init` - `ckpt:sanity-C-ema` | 0.000 | [-0.012, 0.013] | 0.900 |
| `new_dx_365d/copd` | `ckpt:sanity-A-default` - `ckpt:sanity-C-ema` | 0.002 | [-0.008, 0.015] | 0.610 |
| `new_dx_365d/copd` | `lr` - `gbm` | -0.042 | [-0.057, -0.025] | 0.000 |
| `new_dx_365d/copd` | `lr` - `random_init` | 0.026 | [-0.002, 0.048] | 0.060 |
| `new_dx_365d/copd` | `lr` - `ckpt:sanity-A-default` | 0.023 | [0.000, 0.045] | 0.050 |
| `new_dx_365d/copd` | `lr` - `ckpt:sanity-C-ema` | 0.026 | [0.004, 0.049] | 0.030 |

## Few-shot (k positives + k negatives from train, 5 seeds)

| task | model | k | n train | AUROC mean ± sd | AUPRC mean ± sd |
|---|---|---|---|---|---|
| `mortality_365d` | `lr` | 32 | 64 | 0.527 ± 0.044 | 0.026 ± 0.004 |
| `mortality_365d` | `lr` | 128 | 256 | 0.533 ± 0.048 | 0.027 ± 0.006 |
| `mortality_365d` | `lr` | 512 | 1024 | 0.556 ± 0.026 | 0.029 ± 0.002 |
| `mortality_365d` | `lr` | all | 58192 | 0.554 ± 0.000 | 0.026 ± 0.000 |
| `mortality_365d` | `random_init` | 32 | 64 | 0.524 ± 0.040 | 0.027 ± 0.003 |
| `mortality_365d` | `random_init` | 128 | 256 | 0.524 ± 0.025 | 0.028 ± 0.007 |
| `mortality_365d` | `random_init` | 512 | 1024 | 0.570 ± 0.020 | 0.033 ± 0.004 |
| `mortality_365d` | `random_init` | all | 58192 | 0.591 ± 0.000 | 0.033 ± 0.000 |
| `mortality_365d` | `ckpt:sanity-A-default` | 32 | 64 | 0.505 ± 0.051 | 0.025 ± 0.008 |
| `mortality_365d` | `ckpt:sanity-A-default` | 128 | 256 | 0.507 ± 0.025 | 0.025 ± 0.003 |
| `mortality_365d` | `ckpt:sanity-A-default` | 512 | 1024 | 0.569 ± 0.010 | 0.028 ± 0.002 |
| `mortality_365d` | `ckpt:sanity-A-default` | all | 58192 | 0.582 ± 0.000 | 0.030 ± 0.000 |
| `mortality_365d` | `ckpt:sanity-C-ema` | 32 | 64 | 0.542 ± 0.029 | 0.027 ± 0.004 |
| `mortality_365d` | `ckpt:sanity-C-ema` | 128 | 256 | 0.524 ± 0.035 | 0.025 ± 0.003 |
| `mortality_365d` | `ckpt:sanity-C-ema` | 512 | 1024 | 0.566 ± 0.022 | 0.031 ± 0.006 |
| `mortality_365d` | `ckpt:sanity-C-ema` | all | 58192 | 0.602 ± 0.000 | 0.031 ± 0.000 |
| `inpatient_365d` | `lr` | 32 | 64 | 0.591 ± 0.036 | 0.268 ± 0.017 |
| `inpatient_365d` | `lr` | 128 | 256 | 0.657 ± 0.009 | 0.308 ± 0.016 |
| `inpatient_365d` | `lr` | 512 | 1024 | 0.674 ± 0.004 | 0.319 ± 0.009 |
| `inpatient_365d` | `lr` | all | 58192 | 0.708 ± 0.000 | 0.363 ± 0.000 |
| `inpatient_365d` | `random_init` | 32 | 64 | 0.568 ± 0.034 | 0.240 ± 0.017 |
| `inpatient_365d` | `random_init` | 128 | 256 | 0.623 ± 0.013 | 0.267 ± 0.006 |
| `inpatient_365d` | `random_init` | 512 | 1024 | 0.641 ± 0.013 | 0.281 ± 0.011 |
| `inpatient_365d` | `random_init` | all | 58192 | 0.672 ± 0.000 | 0.318 ± 0.000 |
| `inpatient_365d` | `ckpt:sanity-A-default` | 32 | 64 | 0.549 ± 0.042 | 0.229 ± 0.020 |
| `inpatient_365d` | `ckpt:sanity-A-default` | 128 | 256 | 0.612 ± 0.015 | 0.261 ± 0.005 |
| `inpatient_365d` | `ckpt:sanity-A-default` | 512 | 1024 | 0.635 ± 0.015 | 0.277 ± 0.017 |
| `inpatient_365d` | `ckpt:sanity-A-default` | all | 58192 | 0.682 ± 0.000 | 0.351 ± 0.000 |
| `inpatient_365d` | `ckpt:sanity-C-ema` | 32 | 64 | 0.575 ± 0.042 | 0.244 ± 0.022 |
| `inpatient_365d` | `ckpt:sanity-C-ema` | 128 | 256 | 0.626 ± 0.015 | 0.273 ± 0.010 |
| `inpatient_365d` | `ckpt:sanity-C-ema` | 512 | 1024 | 0.653 ± 0.004 | 0.298 ± 0.006 |
| `inpatient_365d` | `ckpt:sanity-C-ema` | all | 58192 | 0.681 ± 0.000 | 0.345 ± 0.000 |
| `readmission_30d` | `lr` | 32 | 64 | 0.568 ± 0.031 | 0.065 ± 0.008 |
| `readmission_30d` | `lr` | 128 | 256 | 0.588 ± 0.018 | 0.077 ± 0.005 |
| `readmission_30d` | `lr` | 512 | 1024 | 0.628 ± 0.003 | 0.082 ± 0.002 |
| `readmission_30d` | `lr` | all | 24417 | 0.658 ± 0.000 | 0.098 ± 0.000 |
| `readmission_30d` | `random_init` | 32 | 64 | 0.538 ± 0.016 | 0.059 ± 0.006 |
| `readmission_30d` | `random_init` | 128 | 256 | 0.549 ± 0.024 | 0.062 ± 0.004 |
| `readmission_30d` | `random_init` | 512 | 1024 | 0.603 ± 0.015 | 0.072 ± 0.007 |
| `readmission_30d` | `random_init` | all | 24417 | 0.652 ± 0.000 | 0.091 ± 0.000 |
| `readmission_30d` | `ckpt:sanity-A-default` | 32 | 64 | 0.520 ± 0.024 | 0.056 ± 0.005 |
| `readmission_30d` | `ckpt:sanity-A-default` | 128 | 256 | 0.523 ± 0.009 | 0.053 ± 0.001 |
| `readmission_30d` | `ckpt:sanity-A-default` | 512 | 1024 | 0.573 ± 0.012 | 0.072 ± 0.004 |
| `readmission_30d` | `ckpt:sanity-A-default` | all | 24417 | 0.623 ± 0.000 | 0.087 ± 0.000 |
| `readmission_30d` | `ckpt:sanity-C-ema` | 32 | 64 | 0.528 ± 0.026 | 0.060 ± 0.008 |
| `readmission_30d` | `ckpt:sanity-C-ema` | 128 | 256 | 0.573 ± 0.025 | 0.068 ± 0.005 |
| `readmission_30d` | `ckpt:sanity-C-ema` | 512 | 1024 | 0.619 ± 0.016 | 0.079 ± 0.004 |
| `readmission_30d` | `ckpt:sanity-C-ema` | all | 24417 | 0.657 ± 0.000 | 0.093 ± 0.000 |
| `new_dx_365d/diabetes` | `lr` | 32 | 64 | 0.684 ± 0.020 | 0.344 ± 0.016 |
| `new_dx_365d/diabetes` | `lr` | 128 | 256 | 0.703 ± 0.001 | 0.360 ± 0.005 |
| `new_dx_365d/diabetes` | `lr` | 512 | 1024 | 0.711 ± 0.002 | 0.373 ± 0.006 |
| `new_dx_365d/diabetes` | `lr` | all | 39458 | 0.740 ± 0.000 | 0.432 ± 0.000 |
| `new_dx_365d/diabetes` | `random_init` | 32 | 64 | 0.644 ± 0.026 | 0.321 ± 0.016 |
| `new_dx_365d/diabetes` | `random_init` | 128 | 256 | 0.678 ± 0.005 | 0.330 ± 0.007 |
| `new_dx_365d/diabetes` | `random_init` | 512 | 1024 | 0.702 ± 0.006 | 0.356 ± 0.011 |
| `new_dx_365d/diabetes` | `random_init` | all | 39458 | 0.737 ± 0.000 | 0.417 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:sanity-A-default` | 32 | 64 | 0.611 ± 0.040 | 0.295 ± 0.021 |
| `new_dx_365d/diabetes` | `ckpt:sanity-A-default` | 128 | 256 | 0.667 ± 0.013 | 0.323 ± 0.011 |
| `new_dx_365d/diabetes` | `ckpt:sanity-A-default` | 512 | 1024 | 0.694 ± 0.006 | 0.340 ± 0.009 |
| `new_dx_365d/diabetes` | `ckpt:sanity-A-default` | all | 39458 | 0.734 ± 0.000 | 0.409 ± 0.000 |
| `new_dx_365d/diabetes` | `ckpt:sanity-C-ema` | 32 | 64 | 0.643 ± 0.017 | 0.321 ± 0.015 |
| `new_dx_365d/diabetes` | `ckpt:sanity-C-ema` | 128 | 256 | 0.679 ± 0.014 | 0.347 ± 0.016 |
| `new_dx_365d/diabetes` | `ckpt:sanity-C-ema` | 512 | 1024 | 0.711 ± 0.006 | 0.376 ± 0.017 |
| `new_dx_365d/diabetes` | `ckpt:sanity-C-ema` | all | 39458 | 0.735 ± 0.000 | 0.417 ± 0.000 |
| `new_dx_365d/heart_failure` | `lr` | 32 | 64 | 0.648 ± 0.027 | 0.208 ± 0.018 |
| `new_dx_365d/heart_failure` | `lr` | 128 | 256 | 0.666 ± 0.008 | 0.215 ± 0.015 |
| `new_dx_365d/heart_failure` | `lr` | 512 | 1024 | 0.692 ± 0.004 | 0.241 ± 0.007 |
| `new_dx_365d/heart_failure` | `lr` | all | 48770 | 0.744 ± 0.000 | 0.276 ± 0.000 |
| `new_dx_365d/heart_failure` | `random_init` | 32 | 64 | 0.636 ± 0.017 | 0.196 ± 0.003 |
| `new_dx_365d/heart_failure` | `random_init` | 128 | 256 | 0.649 ± 0.013 | 0.201 ± 0.011 |
| `new_dx_365d/heart_failure` | `random_init` | 512 | 1024 | 0.673 ± 0.010 | 0.219 ± 0.007 |
| `new_dx_365d/heart_failure` | `random_init` | all | 48770 | 0.699 ± 0.000 | 0.259 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-A-default` | 32 | 64 | 0.614 ± 0.010 | 0.186 ± 0.007 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-A-default` | 128 | 256 | 0.632 ± 0.014 | 0.192 ± 0.013 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-A-default` | 512 | 1024 | 0.658 ± 0.010 | 0.211 ± 0.008 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-A-default` | all | 48770 | 0.682 ± 0.000 | 0.237 ± 0.000 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-C-ema` | 32 | 64 | 0.625 ± 0.023 | 0.202 ± 0.015 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-C-ema` | 128 | 256 | 0.640 ± 0.019 | 0.205 ± 0.021 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-C-ema` | 512 | 1024 | 0.674 ± 0.007 | 0.232 ± 0.008 |
| `new_dx_365d/heart_failure` | `ckpt:sanity-C-ema` | all | 48770 | 0.693 ± 0.000 | 0.252 ± 0.000 |
| `new_dx_365d/ckd` | `lr` | 32 | 64 | 0.632 ± 0.053 | 0.151 ± 0.013 |
| `new_dx_365d/ckd` | `lr` | 128 | 256 | 0.670 ± 0.006 | 0.161 ± 0.008 |
| `new_dx_365d/ckd` | `lr` | 512 | 1024 | 0.689 ± 0.006 | 0.177 ± 0.003 |
| `new_dx_365d/ckd` | `lr` | all | 51106 | 0.736 ± 0.000 | 0.236 ± 0.000 |
| `new_dx_365d/ckd` | `random_init` | 32 | 64 | 0.591 ± 0.039 | 0.130 ± 0.014 |
| `new_dx_365d/ckd` | `random_init` | 128 | 256 | 0.637 ± 0.015 | 0.139 ± 0.004 |
| `new_dx_365d/ckd` | `random_init` | 512 | 1024 | 0.664 ± 0.006 | 0.150 ± 0.005 |
| `new_dx_365d/ckd` | `random_init` | all | 51106 | 0.700 ± 0.000 | 0.174 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:sanity-A-default` | 32 | 64 | 0.565 ± 0.043 | 0.125 ± 0.012 |
| `new_dx_365d/ckd` | `ckpt:sanity-A-default` | 128 | 256 | 0.615 ± 0.013 | 0.130 ± 0.005 |
| `new_dx_365d/ckd` | `ckpt:sanity-A-default` | 512 | 1024 | 0.652 ± 0.014 | 0.146 ± 0.010 |
| `new_dx_365d/ckd` | `ckpt:sanity-A-default` | all | 51106 | 0.695 ± 0.000 | 0.178 ± 0.000 |
| `new_dx_365d/ckd` | `ckpt:sanity-C-ema` | 32 | 64 | 0.595 ± 0.021 | 0.133 ± 0.005 |
| `new_dx_365d/ckd` | `ckpt:sanity-C-ema` | 128 | 256 | 0.633 ± 0.012 | 0.145 ± 0.007 |
| `new_dx_365d/ckd` | `ckpt:sanity-C-ema` | 512 | 1024 | 0.669 ± 0.009 | 0.162 ± 0.009 |
| `new_dx_365d/ckd` | `ckpt:sanity-C-ema` | all | 51106 | 0.692 ± 0.000 | 0.175 ± 0.000 |
| `new_dx_365d/copd` | `lr` | 32 | 64 | 0.628 ± 0.065 | 0.193 ± 0.021 |
| `new_dx_365d/copd` | `lr` | 128 | 256 | 0.667 ± 0.003 | 0.215 ± 0.007 |
| `new_dx_365d/copd` | `lr` | 512 | 1024 | 0.682 ± 0.006 | 0.234 ± 0.009 |
| `new_dx_365d/copd` | `lr` | all | 48015 | 0.726 ± 0.000 | 0.317 ± 0.000 |
| `new_dx_365d/copd` | `random_init` | 32 | 64 | 0.600 ± 0.062 | 0.176 ± 0.025 |
| `new_dx_365d/copd` | `random_init` | 128 | 256 | 0.649 ± 0.010 | 0.201 ± 0.009 |
| `new_dx_365d/copd` | `random_init` | 512 | 1024 | 0.669 ± 0.008 | 0.204 ± 0.004 |
| `new_dx_365d/copd` | `random_init` | all | 48015 | 0.700 ± 0.000 | 0.235 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:sanity-A-default` | 32 | 64 | 0.574 ± 0.040 | 0.158 ± 0.018 |
| `new_dx_365d/copd` | `ckpt:sanity-A-default` | 128 | 256 | 0.634 ± 0.017 | 0.191 ± 0.009 |
| `new_dx_365d/copd` | `ckpt:sanity-A-default` | 512 | 1024 | 0.668 ± 0.009 | 0.205 ± 0.012 |
| `new_dx_365d/copd` | `ckpt:sanity-A-default` | all | 48015 | 0.703 ± 0.000 | 0.240 ± 0.000 |
| `new_dx_365d/copd` | `ckpt:sanity-C-ema` | 32 | 64 | 0.605 ± 0.048 | 0.175 ± 0.019 |
| `new_dx_365d/copd` | `ckpt:sanity-C-ema` | 128 | 256 | 0.652 ± 0.014 | 0.201 ± 0.010 |
| `new_dx_365d/copd` | `ckpt:sanity-C-ema` | 512 | 1024 | 0.681 ± 0.008 | 0.210 ± 0.007 |
| `new_dx_365d/copd` | `ckpt:sanity-C-ema` | all | 48015 | 0.700 ± 0.000 | 0.233 ± 0.000 |
