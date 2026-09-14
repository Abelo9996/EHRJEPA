# Downstream evaluation -- physionet2019

Metrics are on the `held_out` split. Intervals are percentile bootstrap over subjects, 200 resamples, 95%.

|  |  |
|---|---|
| source | `physionet2019` |
| MEDS | `data/meds/physionet2019` |
| cache | `data/cache/physionet2019` |
| tasks | `data/tasks/physionet2019` |
| anchor seed | 20260903 |
| commit | `89bdf94` |
| created | 2026-09-11T03:02:29+00:00 |
| runtime (s) | 1085.7 |

## Models

| model | kind | checkpoint | features |
|---|---|---|---|
| `random_init` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/hybrid_bins_s1/final.pt` | last@final |
| `ckpt:hybrid_bins_s1` | probe | `/home/gaming_pc/EHRJEPA/runs/a2-physionet2019/hybrid_bins_s1/final.pt` | last@final |

## Cohorts

| task | train n (rate) | tuning n (rate) | held_out n (rate) | note |
|---|---|---|---|---|
| `sepsis_6h` | 251991 (0.0153) | 32181 (0.0138) | 31324 (0.0171) |  |
| `sepsis_stay` | 31134 (0.0558) | 3992 (0.0569) | 3869 (0.0556) |  |

## AUROC

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `sepsis_6h` | 0.686 [0.663, 0.707] | 0.822 [0.804, 0.840] |
| `sepsis_stay` | 0.627 [0.593, 0.661] | 0.766 [0.736, 0.799] |

## AUPRC

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `sepsis_6h` | 0.041 [0.035, 0.050] | 0.086 [0.072, 0.101] |
| `sepsis_stay` | 0.096 [0.078, 0.126] | 0.189 [0.159, 0.240] |

## BRIER

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `sepsis_6h` | 0.0167 [0.0153, 0.0181] | 0.0162 [0.0148, 0.0176] |
| `sepsis_stay` | 0.0519 [0.0462, 0.0573] | 0.0488 [0.0436, 0.0540] |

## CALIBRATION SLOPE

| task | random_init | ckpt:hybrid_bins_s1 |
|---|---|---|
| `sepsis_6h` | 1.270 [1.123, 1.418] | 1.021 [0.956, 1.085] |
| `sepsis_stay` | 0.744 [0.539, 0.943] | 1.076 [0.946, 1.245] |

## Paired bootstrap (AUROC difference, identical subjects)

| task | comparison | diff | 95% CI | boot p |
|---|---|---|---|---|
| `sepsis_6h` | `random_init` - `ckpt:hybrid_bins_s1` | -0.136 | [-0.154, -0.117] | 0.000 |
| `sepsis_stay` | `random_init` - `ckpt:hybrid_bins_s1` | -0.139 | [-0.177, -0.100] | 0.000 |

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
| `mortality_inhospital/24h` | not defined on source family 'physionet2019' |
| `mortality_inhospital/48h` | not defined on source family 'physionet2019' |
