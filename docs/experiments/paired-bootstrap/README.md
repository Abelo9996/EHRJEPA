# paired-bootstrap — fine-tuned encoders vs count-GBM, and hybrid vs AR

`scripts/paired_bootstrap.py` on the committed `predictions.parquet` of `mimic-ft-1b`,
`mimic-ft-1b-s12` and `mimic-ft-full` (8% source held-out split, identical anchors and labels;
all files checked to match GBM's anchors and labels row by row).

- 2000 resamples, `np.random.default_rng(0)`, subjects drawn with replacement as in
  `ehrjepa.eval.metrics`. One draw over all 1,346 held-out subjects that have an anchor
  (1,025–1,120 rows per task) is applied **jointly** to both models and all tasks.
- Diff = AUROC(a) − AUROC(b). mean-of-6 = the six non-mortality tasks; mortality_365d separate.
- 95% percentile CI; P(>0) = fraction of resamples with diff > 0.
- Pooled = mean of the 3 seeds' diffs, averaged within each resample (seed-pooling does not
  shrink the test-set sampling variance, since all seeds share the same subjects).
- Point estimates reproduce the summaries: hybrid_25pct mean-of-6 0.8183 / 0.8201 / 0.8246
  (s0/s1/s2), GBM 0.8081.

## vs GBM (mean-of-6 diff)

| cell | s0 | s1 | s2 | pooled (3 seeds) |
|---|---|---|---|---|
| hybrid, 25% pretrain | +0.010 [−0.004, +0.025] P=0.91 | +0.012 [−0.003, +0.028] P=0.95 | +0.017 [+0.002, +0.032] P=0.98 | **+0.013 [−0.001, +0.028] P=0.97** |
| AR, 25% pretrain | +0.006 [−0.009, +0.022] P=0.79 | +0.002 [−0.014, +0.018] P=0.60 | +0.007 [−0.007, +0.022] P=0.86 | +0.005 [−0.009, +0.020] P=0.76 |
| hybrid, 8% pretrain | +0.011 [−0.004, +0.027] P=0.93 | +0.000 [−0.016, +0.017] P=0.53 | +0.003 [−0.012, +0.018] P=0.64 | +0.005 [−0.010, +0.020] P=0.73 |
| AR, 8% pretrain | −0.005 [−0.021, +0.012] P=0.28 | +0.002 [−0.014, +0.018] P=0.60 | −0.002 [−0.018, +0.012] P=0.39 | −0.002 [−0.017, +0.013] P=0.41 |
| hybrid, full pretrain | +0.014 [−0.001, +0.028] P=0.97 | — | — | — |
| AR, full pretrain | +0.006 [−0.009, +0.021] P=0.79 | — | — | — |

Mortality_365d, pooled hybrid_25pct − GBM: +0.009 [−0.003, +0.021] P=0.94.

## Hybrid − AR at matched seed / pretraining data (mean-of-6 diff)

| pretrain | s0 | s1 | s2 | pooled |
|---|---|---|---|---|
| 25% | +0.004 [−0.005, +0.012] P=0.82 | +0.010 [+0.002, +0.019] P=0.995 | +0.009 [+0.001, +0.017] P=0.99 | **+0.008 [+0.002, +0.014] P=0.995** |
| 8% | +0.016 [+0.007, +0.025] P=0.999 | −0.001 [−0.012, +0.009] P=0.40 | +0.005 [−0.006, +0.016] P=0.82 | +0.007 [−0.000, +0.013] P=0.97 |
| full | +0.008 [+0.000, +0.016] P=0.98 | — | — | — |

## Reading

- Hybrid (25% pretrain) is above GBM on every seed, but the paired 95% CI includes 0 for s0 and s1,
  excludes it only for s2, and **includes 0 for the 3-seed pooled estimate** (lower bound −0.001).
- Hybrid > AR at 25% pretraining: pooled CI excludes 0.
- Per-task pooled hybrid_25pct − GBM: inpatient +0.028, readmission +0.023, ckd +0.011,
  copd +0.019, diabetes +0.002, heart failure −0.005; only inpatient's CI lower bound reaches 0.
- Full per-task results for every comparison: `results.json`.

Note: earlier READMEs describe the split as "3,038 held-out subjects"; the predictions contain
1,346 distinct subjects (those with at least one anchor), and that is the bootstrap unit here.
