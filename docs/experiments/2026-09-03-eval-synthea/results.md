# Downstream evaluation -- synthea-coherent

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 1000 resamples, 95%.

|  |  |
|---|---|
| source | `synthea-coherent` |
| MEDS | `data/meds/synthea-coherent` |
| cache | `data/cache/synthea-coherent` |
| tasks | `data/tasks/synthea-coherent` |
| anchor seed | 20260903 |
| commit | `3c7c019` |
| created | 2026-09-03T10:37:35+00:00 |
| runtime (s) | 17.9 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `lr` | lr | -- | counts |
| `gbm` | gbm | -- | counts |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) |
|---|---|---|---|
| `mortality_365d` | 2718 (0.0743) | 326 (0.0798) | 337 (0.0801) |
| `inpatient_365d` | 2718 (0.1144) | 326 (0.1288) | 337 (0.1068) |
| `new_dx_365d/diabetes` | 2414 (0.0004) | 292 (0.0000) | 311 (0.0000) |
| `new_dx_365d/heart_failure` | 2581 (0.0058) | 311 (0.0096) | 314 (0.0000) |
| `new_dx_365d/ckd` | 2540 (0.0004) | 308 (0.0000) | 325 (0.0000) |
| `new_dx_365d/copd` | 2601 (0.0012) | 312 (0.0032) | 324 (0.0031) |

## AUROC

| task | lr | gbm |
|---|---|---|
| `mortality_365d` | 0.867 [0.779, 0.939] | 0.931 [0.870, 0.974] |
| `inpatient_365d` | 0.879 [0.787, 0.958] | 0.917 [0.843, 0.977] |
| `new_dx_365d/diabetes` | -- | -- |
| `new_dx_365d/heart_failure` | -- | -- |
| `new_dx_365d/ckd` | -- | -- |
| `new_dx_365d/copd` | 0.294 [0.245, 0.347] | 0.687 [0.638, 0.736] |

## AUPRC

| task | lr | gbm |
|---|---|---|
| `mortality_365d` | 0.525 [0.336, 0.710] | 0.675 [0.496, 0.817] |
| `inpatient_365d` | 0.772 [0.646, 0.880] | 0.828 [0.714, 0.921] |
| `new_dx_365d/diabetes` | -- | -- |
| `new_dx_365d/heart_failure` | -- | -- |
| `new_dx_365d/ckd` | -- | -- |
| `new_dx_365d/copd` | 0.004 [0.004, 0.017] | 0.010 [0.009, 0.036] |

## BRIER

| task | lr | gbm |
|---|---|---|
| `mortality_365d` | 0.0534 [0.0363, 0.0719] | 0.0461 [0.0283, 0.0657] |
| `inpatient_365d` | 0.0459 [0.0302, 0.0649] | 0.0364 [0.0199, 0.0549] |
| `new_dx_365d/diabetes` | -- | -- |
| `new_dx_365d/heart_failure` | -- | -- |
| `new_dx_365d/ckd` | -- | -- |
| `new_dx_365d/copd` | 0.0049 [0.0000, 0.0129] | 0.0031 [0.0000, 0.0093] |

## CALIBRATION SLOPE

| task | lr | gbm |
|---|---|---|
| `mortality_365d` | 0.984 [0.702, 1.382] | 0.678 [0.534, 0.945] |
| `inpatient_365d` | 0.836 [0.640, 1.129] | -- |
| `new_dx_365d/diabetes` | -- | -- |
| `new_dx_365d/heart_failure` | -- | -- |
| `new_dx_365d/ckd` | -- | -- |
| `new_dx_365d/copd` | -- | -- |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `mortality_365d` | `lr` - `gbm` | -0.064 | [-0.139, -0.012] | 0.006 |
| `inpatient_365d` | `lr` - `gbm` | -0.038 | [-0.085, -0.003] | 0.036 |
| `new_dx_365d/copd` | `lr` - `gbm` | -0.393 | [-0.464, -0.320] | 0.000 |

## Few-shot (k positives + k negatives from train, 5 seeds)

| task | model | k | n train | AUROC mean ± sd | AUPRC mean ± sd |
|---|---|---|---|---|---|
| `mortality_365d` | `lr` | 32 | 64 | 0.842 ± 0.029 | 0.502 ± 0.093 |
| `mortality_365d` | `lr` | 128 | 256 | 0.864 ± 0.021 | 0.491 ± 0.035 |
| `mortality_365d` | `lr` | 512 | 714 | 0.874 ± 0.012 | 0.520 ± 0.048 |
| `mortality_365d` | `lr` | all | 2718 | 0.867 ± 0.000 | 0.525 ± 0.000 |
| `inpatient_365d` | `lr` | 32 | 64 | 0.843 ± 0.029 | 0.530 ± 0.031 |
| `inpatient_365d` | `lr` | 128 | 256 | 0.893 ± 0.017 | 0.613 ± 0.062 |
| `inpatient_365d` | `lr` | 512 | 823 | 0.883 ± 0.005 | 0.671 ± 0.034 |
| `inpatient_365d` | `lr` | all | 2718 | 0.879 ± 0.000 | 0.772 ± 0.000 |
| `new_dx_365d/copd` | `lr` | 32 | 35 | 0.292 ± 0.181 | 0.005 ± 0.002 |
| `new_dx_365d/copd` | `lr` | 128 | 131 | 0.355 ± 0.088 | 0.005 ± 0.001 |
| `new_dx_365d/copd` | `lr` | 512 | 515 | 0.291 ± 0.095 | 0.004 ± 0.001 |
| `new_dx_365d/copd` | `lr` | all | 2601 | 0.294 ± 0.000 | 0.004 ± 0.000 |

## Skipped

| task | reason |
|---|---|
| `readmission_30d` | inpatient_discharge |
