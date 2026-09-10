# a2ft-physionet2019: does fine-tuning close the gap to GBM, and does the objective ordering survive it?

**Not run.** This document is the protocol, written before any cell of
`configs/grids/a2ft_physionet2019.yaml` was evaluated. It is a follow-up to
[`a2-physionet2019`](../a2-physionet2019/README.md), which pre-registers the
same six objectives under a frozen linear probe, and it re-scores five of those
checkpoints a second way: fine-tuned end to end on each task. Nothing here is a
result, and no cell is trained — every row reuses an `a2-physionet2019`
`final.pt`.

## The two pre-registered questions

Every encoder number this repository has published is a **frozen probe**: a
logistic regression on pooled features, tuned only in its L2 strength. That is
the right instrument for "what is already linearly readable in the
representation", and it is not the instrument the ICU literature uses. A paper
reporting a sepsis or mortality model fine-tunes the encoder on the task and
reports that number against two things: a gradient-boosting model on tabular
features, and the same architecture trained from scratch. Two questions follow,
and neither can be answered from `a2-physionet2019/summary.md`:

1. **Does fine-tuning close the gap to `gbm`?** On the frozen probe, no
   checkpoint in this project has beaten count-feature gradient boosting on any
   source. The reference numbers here (full held-out split,
   [`2026-09-10-eval-physionet2019/`](../2026-09-10-eval-physionet2019/)):

   | task | prevalence | `lr` | `gbm` |
   |---|---|---|---|
   | `sepsis_6h` | 1.7% | 0.785 | 0.836 |
   | `sepsis_stay` | 5.6% | 0.768 | 0.769 |

   A pretrained encoder that reaches or passes `gbm` after fine-tuning, while
   its probe does not, says the representation was a good *initialisation* and a
   poor *linear feature* — a different claim from either half of the a2 grid,
   and the claim ICU papers actually make.

2. **Does the objective ordering survive fine-tuning?** The a2 decision rule
   ranks the objectives by probed AUROC. There is no reason a pretraining
   objective that produces the more probeable representation also produces the
   better initialisation: fine-tuning is free to move the encoder wherever the
   task wants it, and the probe's advantage — a representation already shaped
   into linearly separable directions — is exactly the property fine-tuning can
   manufacture for itself. If `latent_cont`/`latent_only` sit below `ar_bins`
   frozen and above it fine-tuned, or the reverse, the a2 conclusion is a
   statement about probes and has to be written as one.

## Decision rule, fixed before results

Read per task on the full held-out split, seed 1, against the fine-tuned
from-scratch control (`ft_random@ar_bins_s1`) and against `gbm`:

- A cell **beats its from-scratch control** → its pretrained weights were a
  better initialisation than none, on this task. A cell that does not is
  reported as contributing nothing a fine-tune could use, exactly as a probe
  that fails to beat `random_init` is.
- A cell **at or above `gbm`** → the gap the frozen probes leave open is closed
  by fine-tuning, on this task. Below it, it is not; "closer" is not a finding
  and is reported as a number, not as progress.
- The **fine-tuned ordering of the five cells** is compared to the probed
  ordering in `a2-physionet2019/summary.md`. Agreement → the a2 ranking is a
  property of the objectives. Disagreement → the a2 ranking is a property of
  probes, and both tables have to be reported together.

One seed, so no ordering difference smaller than the seed-to-seed spread of the
a2 probes is claimed as an effect. The paired bootstrap in each cell's
`results.json` compares the arms scored in one command on identical anchors;
across cells (each its own command) only the intervals are comparable.

## Protocol

- **No training.** Five `reuse_checkpoint:` rows pointing at
  `runs/a2-physionet2019/<cell>/final.pt` on the GPU host. `--dry-run` works on
  a machine that has none of them.
- **Cells** `ar_bins_s1`, `hybrid_bins_s1`, `latent_cont_s1`, `latent_only_s1`,
  `hybrid_bins_lm_s1`. `ar_cont_s1` is left out: it is a control on the
  auxiliary term, its probe row already answers what it is for, and a
  fine-tuning run costs the same as any other cell's.
- **Mode** `eval_modes: [finetune]` — one `ft:` row per cell and no probe row.
  The probe numbers for these same checkpoints already exist in
  `a2-physionet2019/summary.md` on the same cohort, seed and bootstrap count.
- **Fine-tuning**, all of it fixed in `src/ehrjepa/eval/finetune.py` and none of
  it tuned per cell: `LayerNorm + Linear` head on the same `auto` pooling the
  probe resolves (`last@final`, every cell is causal); AdamW at 2e-5 on the
  encoder, 1e-3 on the head, 1e-4 on the LM cell's LoRA adapters; 5% linear
  warmup then cosine; gradient clip 1.0; batch 64 windows at each checkpoint's
  own `max_len` (512, and 160 for the LM cell, which fine-tunes at `ft_batch: 4`
  because 64 windows through a 0.5B trunk do not fit 8 GB — its own control
  rides in the same command and shares that batch); at most 5 epochs with early
  stopping on tuning-split AUROC at patience 2 and the best epoch's weights
  restored before the held-out pass. The tuning split chooses **when to stop**
  and nothing else.
- **What is trained** the embedding, the encoder and the head. Every pretraining
  head, the predictor and the EMA target copy are frozen: they are not on the
  path from a window to a logit. For the LM cell that means the LoRA adapters,
  the projection, its LayerNorm and the CLS row — the same 1.87M parameters
  pretraining moved, 0.38% of the model.
- **`ft_balanced: false`.** `sepsis_6h` is a 1.7%-prevalence task, but the probe
  it is read against is a logistic regression at `class_weight=None`, and
  balancing one arm would change the calibration of the scores Brier and the
  calibration slope are computed on. `--ft-balanced` exists for a follow-up that
  asks that of every arm at once.
- **Controls** two, not five. A fine-tune trains only the embedding, the encoder
  and the head, so the from-scratch twin of an `ar` cell and of a `nextlatent`
  cell is the same 6L/256d causal encoder at the same initialisation — the
  classes differ only in modules a fine-tune freezes. `ft_random@ar_bins_s1` is
  therefore the from-scratch arm for all four from-scratch cells, and
  `ft_random@hybrid_bins_lm_s1` — a pretrained Qwen trunk with *untrained*
  adapters and projection, fine-tuned on the task — is the LM cell's. Note what
  the latter is the control for: what the a2 LoRA *pretraining* added, not what
  Qwen's pretraining added.
- **Eval** the full held-out split (`eval_subject_limit: null`), 200 bootstrap
  resamples, both tasks. `sepsis_6h` draws up to eight anchors per stay: every
  train anchor is a training row and every held-out anchor a scored row, as in
  the probe, so rows within a stay are not independent in the loss either and
  the bootstrap interval is optimistic — identical across every row of the grid.
- **Baselines** `lr` and `gbm`, fit once by whichever cell's eval runs first.
  They should reproduce the table above. `reuse_predictions` is deliberately
  unset: the a2 grid's baseline scores live in one of its per-cell
  `predictions.parquet` files, which need not exist on the machine that runs
  this grid, and a silently missing baseline block is worse than one refit.

## What a fine-tuned row costs

Each cell is a second training run, not an embedding pass, and on this source
that is not a rounding error. Measured off `data/tasks/physionet2019/tasks.json`
(anchor seed 20260903):

| task | train anchors | tuning | held-out (positives) | steps/epoch at batch 64 |
|---|---|---|---|---|
| `sepsis_6h` | 251,991 | 32,181 | 31,324 (536) | 3,938 |
| `sepsis_stay` | 31,134 | 3,992 | 3,869 (215) | 487 |

Five epochs of `sepsis_6h` is 19,690 optimizer steps at 64 x 512 = **645M
nominal token slots**, six times the 100M the cell was *pretrained* on. Early
stopping (patience 2) can cut that to three epochs, and will whenever the tuning
AUROC peaks early, but the worst case is the number to plan against: per cell,
five arms plus two controls, both tasks. The LM cell is the same three orders of
magnitude slower per event slot it is in pretraining (see
[the a2 README's throughput table](../a2-physionet2019/README.md)) and is the
cell to launch last and alone.

**Verify before committing the grid**, as `configs/pretrain_scale.yaml` says to
for the from-scratch cells: run one cell at `--ft-epochs 1` on the real card,
read the per-epoch `seconds` out of `eval/<cell>/results.json`, and lower
`ft_epochs` in the grid file — the same number for every cell and both sources,
so the arms stay comparable — if five epochs of `sepsis_6h` does not fit the
window. `--only ar_bins_s1` runs one cell.

## Reproducing

```
python scripts/ablate.py configs/grids/a2ft_physionet2019.yaml --dry-run
nohup python scripts/ablate.py configs/grids/a2ft_physionet2019.yaml &
tail -f runs/a2ft-physionet2019/ablate.log
```

Rows land in `summary.md`/`summary.json` in this directory as each cell
finishes, one row per cell with `mode = finetune`; a cell already in
`summary.json` is skipped and `--only ar_bins_s1` runs a subset. A single cell
outside the grid, for a quick look:

```
python -m ehrjepa.eval.run --source physionet2019 --tasks all \
  --models ft:runs/a2-physionet2019/ar_bins_s1/final.pt,ft_random \
  --ft-epochs 5 --bootstrap 200 --no-few-shot \
  --out docs/experiments/a2ft-physionet2019/eval/ar_bins_s1/
```
