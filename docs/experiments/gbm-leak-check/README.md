# gbm-leak-check — does count-GBM lean on near-synonyms of a new diagnosis?

`scripts/gbm_leak_check.py` on the 8% source. The new-dx tasks already exclude the outcome's own codes
(`configs/tasks/predicates.yaml`); this checks codes that document the condition *outside* those
prefixes (e.g. I12 hypertensive CKD, E11.2x, Z99.2 for CKD; Z79.4 for diabetes; I11.0/402.x1/I42 for
HF; J47/J96.1/Z99.81 for COPD). Full numbers in `report.json`.

| task | GBM held-out AUROC | without near-synonyms | near-synonym gain share | positives / negatives with one |
|---|---|---|---|---|
| ckd | 0.939 | 0.939 | 0.000 | 1.1% / 0.3% |
| diabetes | 0.919 | 0.919 | 0.000 | 1.2% / 0.9% |
| heart_failure | 0.900 | 0.900 | 0.000 | 0.7% / 0.4% |
| copd | 0.789 | 0.789 | 0.000 | 2.0% / 0.4% |

Verdict: clean. Near-synonyms are too rare to survive the `min_df=10` column filter or to carry any
gain; GBM's top features are many weak signals (meds, ICU drips, transfers, labs; max 0.75% gain each).
GBM's 0.808 (8%) / 0.841 (25%) are legitimate targets.
