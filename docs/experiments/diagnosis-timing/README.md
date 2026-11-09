# New-diagnosis label timing audit

2026-11-06. The MIMIC-IV demo reproduces the clinical review's timing finding:
**27/33 positive task labels (81.8%) have their first matching diagnosis at the
discharge of a hospital stay containing the prediction anchor.** The four tasks
have 168 labelled rows, 157 with an anchor strictly inside a stay. These are
task-label counts, not 168 distinct patients. No label inconsistencies or
overlapping active stays were observed.

| Task | Positive labels | At current-stay discharge | Fraction |
| --- | ---: | ---: | ---: |
| Diabetes | 8 | 6 | 75.0% |
| Heart failure | 9 | 8 | 88.9% |
| CKD | 12 | 9 | 75.0% |
| COPD | 4 | 4 | 100.0% |
| Pooled task rows | 33 | 27 | 81.8% |

The machine-readable [demo report](mimic-demo.json) includes split-level counts
and hashes of the labels, task config, predicates, admissions file and audit
code. The report contains no patient IDs, dates, or individual records.

This confirms the demo result in [R2](../../reviews/R2_clinical.md). It does not
establish the fraction on full MIMIC, establish when disease began, measure
AUROC inflation, or demonstrate that post-anchor events enter model features.
The current task predicts first *recorded* diagnosis and can include discharge
coding from an ongoing admission. A claim about incident disease needs a
different validated task definition. Existing tasks and reported model results
have not been changed or re-scored.

## Reproduce locally

From the repository root, with the existing demo MEDS extract and ACES installed:

```bash
.venv/bin/python -m ehrjepa.eval.tasks \
  --source mimic-demo \
  --tasks new_dx_365d/diabetes,new_dx_365d/heart_failure,new_dx_365d/ckd,new_dx_365d/copd

.venv/bin/python -m ehrjepa.eval.diagnosis_timing \
  --meds-dir data/meds/mimic-demo \
  --tasks-dir data/tasks/mimic-demo \
  --admissions data/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/hosp/admissions.csv.gz \
  --out docs/experiments/diagnosis-timing/mimic-demo.json
```

The audit uses the existing saved labels without changing their anchors. It
scans only `subject_id`, `time`, and `code` from MEDS, filters to the four
diagnosis predicates and aggregates first times before collecting the result.
Admission membership is `[admittime, dischtime)`; the diagnosis label window is
`(anchor, anchor + 365 days]`. Prior diagnoses at or before the anchor invalidate
a labelled row, matching the existing ACES exclusion. Duplicate anchors and
invalid admission intervals are rejected. Patients without admission rows are
counted as such, not discarded. Overlapping stays are flagged and cannot double
count a positive. A first-diagnosis timestamp match is the audit criterion;
canonical MEDS does not retain diagnosis-level admission IDs.

The command returns exit status 1 if saved labels disagree with the first
diagnosis timestamps. Resolve that mismatch before interpreting the report.
An exit status of 0 means labels are internally consistent, **not** that the
task is clinically validated or free of leakage.

## Full MIMIC: unfinished

Two read-only SSH attempts to `gpu` on November 6 timed out on port 22. No remote
command ran, and no experiment was resumed. The GPU gate remains subject to the
owner's pause for the other paper and the unresolved state-objective bugs.

Next, when the PC is reachable, inspect `/home/gaming_pc/EHRJEPA/data/tasks/` and
the corresponding MEDS directories before choosing the cohort. The existing
paper evaluates labels from the **8% source `mimic-iv`**, including models
pretrained on the full source. Audit those labels first, then audit the complete
`mimic-iv-full` cohort separately if its labels have been built. Do not present
the 8% audit as a full-population result. Missing full-source labels require a
separate CPU label build; its memory use has not been verified here.

After deploying the audit module to the existing PC checkout, the command for
the current paper cohort is:

```bash
cd /home/gaming_pc/EHRJEPA
.venv/bin/python -m ehrjepa.eval.diagnosis_timing \
  --meds-dir data/meds/mimic-iv \
  --tasks-dir data/tasks/mimic-iv \
  --admissions data/mimic-iv-3.1/hosp/admissions.csv.gz \
  --out docs/experiments/diagnosis-timing/mimic-iv.json
```

For the full cohort, use `mimic-iv-full` for both the MEDS and tasks directories
and `mimic-iv-full.json` for output, after verifying those inputs exist. No GPU is
needed. Only aggregate reports should be copied back; keep source records on
the PC.
