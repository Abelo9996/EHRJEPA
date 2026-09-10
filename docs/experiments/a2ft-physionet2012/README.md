# a2ft-physionet2012: the same fine-tuning question, on the mortality challenge

**Not run.** This document is the protocol, written before any cell of
`configs/grids/a2ft_physionet2012.yaml` was evaluated. It is the second source
of the follow-up specified in
[`../a2ft-physionet2019/README.md`](../a2ft-physionet2019/README.md), which
carries the rationale, the fine-tuning settings and the decision rule and which
this document does not repeat. Read that one first. Nothing here is a result,
and no cell is trained: every row reuses an `a2-physionet2012` `final.pt`.

## The two pre-registered questions

Unchanged from the 2019 grid:

1. **Does fine-tuning close the gap to `gbm`?** No frozen probe in this project
   has beaten count-feature gradient boosting on any source. The reference
   numbers here, full held-out split
   ([`2026-09-10-eval-physionet2012/`](../2026-09-10-eval-physionet2012/)):

   | task | prevalence | `lr` | `gbm` |
   |---|---|---|---|
   | `mortality_inhospital/24h` | 13.9% | 0.789 | 0.829 |
   | `mortality_inhospital/48h` | 13.9% | 0.828 | 0.859 |

2. **Does the objective ordering survive fine-tuning?** The a2 decision rule
   ranks the objectives by *probed* AUROC. A pretraining objective that produces
   the more probeable representation need not produce the better initialisation,
   and if the two orderings disagree the a2 conclusion is a statement about
   probes and has to be written as one. The probed ordering for these same
   checkpoints is in [`../a2-physionet2012/`](../a2-physionet2012/)`summary.md`,
   scored on the same cohort, seed and bootstrap count.

## What differs from the 2019 fine-tuning grid

| | a2ft-physionet2019 | a2ft-physionet2012 |
|---|---|---|
| checkpoints | `runs/a2-physionet2019/<cell>/final.pt` | `runs/a2-physionet2012/<cell>/final.pt` |
| tasks | `sepsis_6h`, `sepsis_stay` | `mortality_inhospital/24h`, `/48h` |
| prevalence | 1.7%, 5.6% | 13.9%, 13.9% |
| anchors per stay | up to 8 (`sepsis_6h`) | one |
| train anchors | 251,991 (`sepsis_6h`) | 9,626 |
| gap to close | 0.836 / 0.769 | 0.829 / 0.859 |

Everything else is the same file with a different `source`: five cells
(`ar_bins_s1`, `hybrid_bins_s1`, `latent_cont_s1`, `latent_only_s1`,
`hybrid_bins_lm_s1`), seed 1, `eval_modes: [finetune]`, `ft_epochs: 5` with
early stopping at patience 2, `ft_balanced: false`, two from-scratch controls
(`ft_random@ar_bins_s1` for the four from-scratch cells, whose fine-tuned
architecture is one and the same 6L/256d causal encoder, and
`ft_random@hybrid_bins_lm_s1` for the LM cell), the full held-out split and 200
bootstrap resamples.

One anchor per stay in both tasks, so unlike `sepsis_6h` every row here is an
independent subject — in the fine-tuning loss as well as in the bootstrap.

## The one thing that does not travel between the two grids

Unchanged from [`../a2-physionet2012/README.md`](../a2-physionet2012/README.md),
and it applies to a fine-tuned row exactly as it does to a probed one: 1,156
held-out stays at 13.9% prevalence is 160 positives, and a 200-resample
bootstrap on 160 positives gives an AUROC interval roughly ±0.03 wide. That is
wider than every between-objective effect this project has measured. **A null
result on this grid is weak evidence**; a difference that appears on the 2019
grid and not here is as likely to be power as regime, and will be reported that
way.

The cheap side of that trade: 9,626 train anchors at batch 64 is 151 optimizer
steps per epoch, so five epochs of a from-scratch cell is 755 steps — minutes on
the 4060, against the 19,690 steps five epochs of the 2019 grid's multi-anchor
sepsis split costs. The LM cell remains the same order of magnitude slower per
step it is in pretraining, and is the cell to launch last and alone.

## Reproducing

```
python scripts/ablate.py configs/grids/a2ft_physionet2012.yaml --dry-run
nohup python scripts/ablate.py configs/grids/a2ft_physionet2012.yaml &
tail -f runs/a2ft-physionet2012/ablate.log
```

Rows land in `summary.md`/`summary.json` in this directory as each cell
finishes, one row per cell with `mode = finetune`; a cell already in
`summary.json` is skipped and `--only ar_bins_s1` runs a subset.
