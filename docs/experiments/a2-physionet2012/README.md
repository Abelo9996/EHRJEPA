# a2-physionet2012: the same pre-registered test, on the mortality challenge

**Not run.** This document is the protocol, written before any cell of
`configs/grids/a2_physionet2012.yaml` started. It is the second source of the
test specified in [`docs/experiments/A2_PLAN.md`](../A2_PLAN.md); the design
rationale, the decision rule and every choice worth defending are in
[`../a2-physionet2019/README.md`](../a2-physionet2019/README.md), which this
document does not repeat. Read that one first. What follows is what differs, and
one thing that does not travel between the two grids.

## The pre-registered question

Unchanged: on continuous ICU state, do the latent objectives with no code loss
(`latent_cont`, `latent_only`) beat the binned AR control (`ar_bins`) at matched
compute; does the hybrid (`hybrid_bins`) beat it; and what does a pretrained
language-model encoder (`hybrid_bins_lm`) add. Same six cells, same overrides,
same two seeds, same base config.

PhysioNet-2012 is the same regime as 2019 — 51 distinct codes, 99.3% of events
numeric — reached through a different outcome. Its two tasks are in-hospital
mortality predicted from the first 24 and the first 48 hours, at 13.9%
prevalence, against 2019's 1.7% and 5.6% sepsis onset. So the pair is not two
samples of one experiment: it is the same objectives asked about a common,
well-powered outcome and a rare, time-critical one.

## What differs from the 2019 grid

| | a2-physionet2019 | a2-physionet2012 |
|---|---|---|
| train | 32,229 stays / 9.78M events | 9,645 stays / 4.27M events |
| held-out | 4,006 stays / 1.19M events | 1,156 stays / 0.51M events |
| vocabulary | 47 (43 seen in train) | 55 |
| budget per cell | 100M nominal token slots (3,052 steps) | **40M** (1,221 steps) |
| tasks | `sepsis_6h`, `sepsis_stay` | `mortality_inhospital/24h`, `/48h` |
| prevalence | 1.7%, 5.6% | 13.9%, 13.9% |

The budget is 40M rather than 100M because the train split is 4.27M events
against 2019's 9.78M. At 100M slots a cell here would make roughly twenty passes
over the same 4.27M events, which measures how well an objective memorises a
small corpus rather than how well it learns from a stream. 40M keeps the
passes-over-data roughly comparable between the two grids, which is the quantity
that matters when the comparison is read across them.

Task selection needs no configuration: `tasks: all` with
`source: physionet2012`, and `ehrjepa.eval.tasks` declares each ICU task's
`families`, so the sepsis tasks are simply not defined here.

## Reference baselines

Count-feature `lr` and `gbm`, fit once on this cohort by whichever cell's eval
command runs first, and should reproduce
[`docs/experiments/2026-09-10-eval-physionet2012/`](../2026-09-10-eval-physionet2012/):

| task | prevalence | `lr` | `gbm` |
|---|---|---|---|
| `mortality_inhospital/24h` | 13.9% | 0.789 | 0.829 |
| `mortality_inhospital/48h` | 13.9% | 0.828 | 0.859 |

One anchor per stay in both tasks, so unlike `sepsis_6h` on the 2019 grid the
bootstrap intervals here are over independent rows.

## The one thing that does not travel between the two grids

1,156 held-out stays at 13.9% prevalence is 161 positives. A 200-resample
bootstrap on 161 positives gives an AUROC interval roughly ±0.03 wide, which is
larger than every effect this project has measured between objectives on claims
data (the hybrid's lead over AR at 1B tokens is about 0.005–0.008 mean AUROC).

So: **a null result on this grid is weak evidence.** A difference that appears on
the 2019 grid and not here is as likely to be power as regime, and will be
reported that way. The direction this grid can speak to is a difference that
appears *here* — where the interval is wide enough that clearing it means
something — or a difference that appears on both grids at both seeds, which is
four independent draws agreeing.

That is stated now, before the numbers exist, because it is exactly the kind of
thing that becomes tempting to argue afterwards in whichever direction the
numbers happen to fall.

## Reproducing

```
python -m ehrjepa.data.etl physionet2012 --source data/physionet2012_raw --out data/meds/physionet2012
python -m ehrjepa.data.tokenize build --meds data/meds/physionet2012 --out data/cache/physionet2012

pip install -e '.[lm,eval]'   # the LM cell needs the `lm` extra

python scripts/ablate.py configs/grids/a2_physionet2012.yaml --dry-run
nohup python scripts/ablate.py configs/grids/a2_physionet2012.yaml &
tail -f runs/a2-physionet2012/ablate.log
```
