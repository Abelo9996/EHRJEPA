# 2026-09-03 — downstream evaluation on DE-SynPUF (scope-cut run)

The first evaluation in this repository that touches a label. It compares two
of the sanity checkpoints from
[`../2026-09-03-sanity/`](../2026-09-03-sanity/README.md) — A (shared target,
SIGReg) and C (EMA target) — against count-feature baselines on seven tasks, on
identical subjects and identical anchor times. Run B (SIGReg off) is not part
of this comparison.

**The checkpoints being probed are 750–900 training steps on one demo-scale
dataset — roughly 0.4 epochs, no tuning, no convergence.** Nothing here is a
statement about what the architecture can do. What this directory establishes is
that the harness exists, that the baselines it has to beat are computed on the
same cohort, and what the numbers are today, on a fixed cut of the evaluation
budget described below.

## Scope cut

This run deliberately restricts the `held_out` split to a **seeded random
3,000-subject subset**, the same 200-resample bootstrap budget for every task,
and five models: `lr`, `gbm`, `random_init`, `ckpt:sanity-A-default`,
`ckpt:sanity-C-ema`. `train` and `tuning` are untouched — only the split being
scored is capped, so every model is still fit on the full training data.

```bash
python -m ehrjepa.eval.run --source desynpuf-s1 --tasks all \
    --models lr,gbm,random_init,\
ckpt:runs/sanity-A-default/final.pt,\
ckpt:runs/sanity-C-ema/final.pt \
    --out docs/experiments/2026-09-03-eval-desynpuf/ \
    --bootstrap 200 --eval-subject-limit 3000 --eval-subject-seed 0
```

Tasks are built first (and reused, since `data/tasks/desynpuf-s1/` already
exists from the full-cohort labelling pass):

```bash
python -m ehrjepa.eval.tasks --source desynpuf-s1
```

The 3,000-subject draw is deterministic in `(seed, subject_id)` via the same
`blake2b`-hash construction the anchor draw uses
(`ehrjepa.eval.run.restrict_eval_split`): the kept subjects do not depend on
row order or on which task's held-out set they are drawn from. Because each
task keeps a different subset of subjects (`readmission_30d` and the four
`new_dx_365d` tasks only cover the subjects eligible for that task), `held_out`
lands at exactly 3,000 for every task except where the anchored cohort itself
has fewer than 3,000 held-out subjects, which does not happen here — every
task's `held_out` before the cut is well over 3,000 (smallest is
`readmission_30d` at 3,116).

| | |
|---|---|
| MEDS | `data/meds/desynpuf-s1` — 116,352 subjects, 14.2M events |
| cache | `data/cache/desynpuf-s1` — vocabulary 30,000, `--ndc-digits 9 --max-vocab 30000` |
| anchors | one per subject, seed 20260903, ≥32 events before, 365d follow-up or death inside it, strictly before `MEDS_DEATH` |
| eval-subject cut | `held_out` restricted to 3,000 subjects, seed 0, `train`/`tuning` unrestricted |
| baselines | per-code counts over 30d / 365d / all history + last `value_z` + numeric count, ~150,000 columns pruned to those with ≥10 training rows (103k–110k kept per task) |
| probes | logistic regression on `concat(CLS, masked mean of token outputs)` = 512 dimensions, encoder frozen, `eval()`, fp32, no masking |
| control | `random_init` — run A's architecture with untrained weights and the same probe |
| metrics | `held_out` (capped at 3,000), 200 bootstrap resamples over subjects; paired bootstrap for every model pair |
| runtime | 1615.4s (~27 minutes) |
| hardware | Apple M4, 16 GB; encoder forward passes on MPS, everything else CPU |

Results: [`results.md`](results.md), [`results.json`](results.json). Held-out
scores per subject are in `predictions.parquet` (5 models × 7 tasks × 3,000
rows), so a metric can be changed without refitting. No task was skipped.

## Cohort construction, in one place

Labels come from the MEDS extract, never from the tensor cache — the cache rolls
rare codes up to an ancestor, so `ICD9CM//25001` and `ICD9CM//25002` can share
an id there and a label built on cache ids would not be the label its name
claims. The windowing is evaluated by [ACES](https://github.com/justin13601/ACES)
from the YAMLs in `configs/tasks/`, through its `direct` predicates dataframe.
The anchor rule is not expressible as an ACES trigger (its triggers are
predicate matches on the data, not seeded draws with a censoring condition), so
`ehrjepa.eval.tasks` selects the anchors and hands ACES a synthetic `anchor`
predicate set at exactly those timestamps.

Everything a model sees comes through
`EventSequenceDataset.windows_at`, which binary-searches the cache's `time_min`
with `side="left"`: an event whose timestamp equals the anchor is excluded along
with everything after it. `tests/test_eval.py` builds this cohort twice, the
second time with a `MEDS_DEATH` one day after each held-out subject's anchor —
an event that flips `mortality_365d` from 0 to 1 — and asserts the count
features and the encoder embeddings come out bit-identical.

## Cohort sizes and prevalence (before the eval-subject cut)

Anchoring keeps 72,835 of 116,352 subjects for the 365-day tasks: 88,196 have
≥32 events before some candidate time, 88,124 of those have a candidate time
strictly before death, and 72,835 of those have a full year of record after the
drawn anchor or die inside it — so the follow-up requirement is what removes
most of the cohort (15,289 subjects), not the history requirement (28,156) plus
death (72). `readmission_30d` draws from inpatient discharges only: 37,780
subjects have one, 33,302 with enough history, 30,592 with 30 days of follow-up.
The `new_dx_365d` tasks then drop subjects with a prevalent diagnosis at or
before the anchor, which is where their smaller counts come from — 23,542
prevalent diabetes cases, 11,811 heart failure, 8,836 CKD, 12,836 COPD.

| task | anchored | labelled | held_out n (full) | held_out rate (full) |
|---|---|---|---|---|
| `mortality_365d` | 72,835 | 72,835 | 7,358 | 0.0190 |
| `inpatient_365d` | 72,835 | 72,835 | 7,358 | 0.2098 |
| `readmission_30d` | 30,592 | 30,592 | 3,116 | 0.0501 |
| `new_dx_365d/diabetes` | 72,835 | 49,293 | 4,939 | 0.2264 |
| `new_dx_365d/heart_failure` | 72,835 | 61,024 | 6,164 | 0.1308 |
| `new_dx_365d/ckd` | 72,835 | 63,999 | 6,487 | 0.0999 |
| `new_dx_365d/copd` | 72,835 | 59,999 | 6,009 | 0.1340 |

Full per-split counts (before the eval-subject cut) are in
`data/tasks/desynpuf-s1/tasks.json`. After the cut, every task's `held_out` is
exactly 3,000 subjects; the per-task rate on that 3,000-subject draw is in
[`results.md`](results.md) (Cohorts table) — it moves by a few hundredths of a
point against the full-cohort rate above, as expected from subsampling.

## Reading the numbers

DE-SynPUF is a synthetic public-use file built by sampling and swapping fields
across real beneficiaries specifically to break re-identifiable associations.
Predictive structure between a patient's history and their future is therefore
weakened by construction. Low AUROCs on this source are a property of the
source; they are not comparable to published numbers on MIMIC or EHRSHOT, and
this directory is not the place to conclude anything about the architecture.
