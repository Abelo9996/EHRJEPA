# 2026-09-03 — downstream evaluation on Synthea (baselines only)

Count-feature baselines on the `synthea-coherent` MEDS extract. **No encoder is
evaluated here**: no checkpoint has been pretrained on Synthea, and scoring a
DE-SynPUF-pretrained encoder on a different vocabulary would be measuring the
tokenizer, not the model. The point of this directory is that the harness runs
unchanged on a second source with different codes.

## What was run

```bash
python -m ehrjepa.eval.run --source synthea-coherent --tasks all \
    --models lr,gbm --out docs/experiments/2026-09-03-eval-synthea/
```

`data/tasks/synthea-coherent/` is built first if it is missing, by
`python -m ehrjepa.eval.tasks --source synthea-coherent`.

| | |
|---|---|
| subjects | 3,539 (2,846 train / 341 tuning / 352 held_out) |
| anchored | 3,381 subjects clear the anchor rule |
| features | per-code counts over 30d / 365d / all history, last `value_z`, numeric count; 887 columns kept of 2,495 at `min_df` 10 |
| models | logistic regression (`C` over 6 values) and xgboost (3 settings), both tuned on `tuning` |
| metrics | `held_out`, 1,000 bootstrap resamples over subjects |
| runtime | 18 s |
| commit | see `results.json` |

Results: [`results.md`](results.md), [`results.json`](results.json),
per-subject held-out scores in `predictions.parquet`.

## What is and is not usable here

* `readmission_30d` is **skipped**: Synthea's ETL emits a class-less
  `END//ENCOUNTER`, so there is no inpatient discharge event to anchor on.
* The four `new_dx_365d` tasks have 0–1 positives in `held_out`
  (0.00–0.31%). Three of them report no metrics at all — the harness declines
  to fit when a split has one class — and `new_dx_365d/copd` reports numbers
  computed from a single positive subject. Nothing should be read off those
  rows.
* That leaves `mortality_365d` (8.0% positive in `held_out`, n=337) and
  `inpatient_365d` (10.7%, n=337) as the two tasks with enough events for the
  intervals to mean anything, and even there `held_out` is 337 subjects.

Synthea is generated from disease-progression models, so its labels are
predictable by construction from the codes that precede them; the AUROCs here
say that the featuriser and the fitting code work, not that these tasks are
hard.
