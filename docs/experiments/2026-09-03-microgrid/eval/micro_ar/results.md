# Downstream evaluation -- desynpuf-s1

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `desynpuf-s1` |
| MEDS | `data/meds/desynpuf-s1` |
| cache | `data/cache/desynpuf-s1` |
| tasks | `data/tasks/desynpuf-s1` |
| anchor seed | 20260903 |
| commit | `4deb9c4` |
| created | 2026-09-03T16:46:48+00:00 |
| runtime (s) | 338.1 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid/micro_ar/final.pt` | cls_mean@final |
| `ckpt:micro_ar` | probe | `/Users/abelyagubyan/Downloads/EHRJEPA/.worktrees/task/ehrjepa-47c80f/runs/2026-09-03-microgrid/micro_ar/final.pt` | cls_mean@final |

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

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.616 [0.541, 0.678] | 0.614 [0.545, 0.676] |
| `inpatient_365d` | 0.675 [0.651, 0.694] | 0.694 [0.672, 0.714] |
| `readmission_30d` | 0.646 [0.593, 0.695] | 0.638 [0.588, 0.688] |
| `new_dx_365d/diabetes` | 0.721 [0.699, 0.741] | 0.738 [0.718, 0.759] |
| `new_dx_365d/heart_failure` | 0.690 [0.667, 0.720] | 0.704 [0.683, 0.729] |
| `new_dx_365d/ckd` | 0.694 [0.665, 0.721] | 0.705 [0.679, 0.735] |
| `new_dx_365d/copd` | 0.689 [0.665, 0.713] | 0.699 [0.672, 0.721] |

## AUPRC

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.032 [0.023, 0.047] | 0.030 [0.022, 0.045] |
| `inpatient_365d` | 0.335 [0.301, 0.368] | 0.344 [0.312, 0.383] |
| `readmission_30d` | 0.093 [0.067, 0.123] | 0.099 [0.068, 0.133] |
| `new_dx_365d/diabetes` | 0.402 [0.370, 0.439] | 0.422 [0.383, 0.464] |
| `new_dx_365d/heart_failure` | 0.243 [0.215, 0.281] | 0.260 [0.225, 0.300] |
| `new_dx_365d/ckd` | 0.175 [0.151, 0.211] | 0.179 [0.152, 0.213] |
| `new_dx_365d/copd` | 0.221 [0.196, 0.256] | 0.230 [0.204, 0.266] |

## BRIER

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.0215 [0.0173, 0.0266] | 0.0215 [0.0173, 0.0265] |
| `inpatient_365d` | 0.1507 [0.1436, 0.1590] | 0.1486 [0.1418, 0.1556] |
| `readmission_30d` | 0.0463 [0.0402, 0.0521] | 0.0464 [0.0404, 0.0522] |
| `new_dx_365d/diabetes` | 0.1558 [0.1486, 0.1630] | 0.1530 [0.1459, 0.1600] |
| `new_dx_365d/heart_failure` | 0.1121 [0.1032, 0.1189] | 0.1110 [0.1018, 0.1176] |
| `new_dx_365d/ckd` | 0.0843 [0.0759, 0.0917] | 0.0838 [0.0757, 0.0906] |
| `new_dx_365d/copd` | 0.1081 [0.0999, 0.1168] | 0.1073 [0.0988, 0.1160] |

## CALIBRATION SLOPE

| task | random_init | ckpt:micro_ar |
|---|---|---|
| `mortality_365d` | 0.750 [0.309, 1.159] | 1.113 [0.419, 1.810] |
| `inpatient_365d` | 1.067 [0.924, 1.208] | 1.032 [0.884, 1.193] |
| `readmission_30d` | 1.158 [0.761, 1.558] | 1.203 [0.741, 1.692] |
| `new_dx_365d/diabetes` | 0.982 [0.864, 1.094] | 0.949 [0.832, 1.086] |
| `new_dx_365d/heart_failure` | 0.915 [0.795, 1.094] | 0.902 [0.799, 1.056] |
| `new_dx_365d/ckd` | 0.957 [0.806, 1.129] | 0.954 [0.806, 1.172] |
| `new_dx_365d/copd` | 0.951 [0.798, 1.116] | 0.922 [0.768, 1.069] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `random_init` - `ckpt:micro_ar` | 0.002 | [-0.066, 0.063] | 0.910 |
| `inpatient_365d` | `random_init` - `ckpt:micro_ar` | -0.019 | [-0.033, -0.004] | 0.010 |
| `readmission_30d` | `random_init` - `ckpt:micro_ar` | 0.008 | [-0.027, 0.037] | 0.670 |
| `new_dx_365d/diabetes` | `random_init` - `ckpt:micro_ar` | -0.016 | [-0.027, -0.005] | 0.000 |
| `new_dx_365d/heart_failure` | `random_init` - `ckpt:micro_ar` | -0.014 | [-0.025, 0.001] | 0.090 |
| `new_dx_365d/ckd` | `random_init` - `ckpt:micro_ar` | -0.011 | [-0.025, 0.002] | 0.080 |
| `new_dx_365d/copd` | `random_init` - `ckpt:micro_ar` | -0.010 | [-0.022, 0.003] | 0.140 |
