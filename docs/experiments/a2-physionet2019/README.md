# a2-physionet2019: does latent prediction earn its keep on continuous ICU state?

**Not run.** This document is the protocol, written before any cell of
`configs/grids/a2_physionet2019.yaml` started. The design and the decision rule
come from [`docs/experiments/A2_PLAN.md`](../A2_PLAN.md) and the corresponding
section of `docs/paper/main.tex`, both also written before training. When the
grid runs, `scripts/ablate.py` appends one row per cell to `summary.md` and
`summary.json` in this directory, and the findings go to
`docs/experiments/ABLATION_RESULTS.md`. Nothing here is a result.

## The pre-registered question

Every objective this repository has measured was measured on DE-SynPUF, where an
event is a discrete, low-entropy claims code. There, next-code AR wins: no pure
latent-prediction variant beats it at any budget it was run at
([`PILOT_RESULTS.md`](../PILOT_RESULTS.md), [`SCALE_RESULTS.md`](../SCALE_RESULTS.md)),
and the hybrid's ~2-point lead over AR at 1B tokens
([`ABLATION_RESULTS.md`](../ABLATION_RESULTS.md)) is measured with the tied code
loss present in every winning cell. The diagnosed mechanism is that the latent
target on claims data is largely a time-and-context prior — a ridge map from the
mask token's time features alone recovers R² = 0.58 of the predictor's output —
which a predictor can satisfy without carrying code identity.

PhysioNet-2019 is the opposite regime. 43 distinct codes, 99.1% of events
numeric, one channel (`VAR//HR`) 11% of all events: the *code* is nearly
uninformative and the *number* is the record. If predicting in representation
space is itself the useful part, this is where it should show.

Three questions, in the order the grid answers them:

1. **Do the latent objectives with no code loss — `latent_cont` (4) and
   `latent_only` (5) — beat the binned AR control `ar_bins` (1) at matched
   compute?** This is the question. If yes, latent prediction is doing work that
   token prediction is not, on this evidence. If no, the claim available from
   this project is limited to the hybrid as a regularizer riding on a
   code-prediction loss, and the JEPA framing — that predicting a representation
   is itself the useful part — is not supported by anything measured here.
2. **Does the hybrid `hybrid_bins` (3) beat `ar_bins` (1) here, as it does on
   claims data?** A yes generalises the ABLATION_RESULTS finding across regimes.
   A no localises it to claims.
3. **What does a pretrained language-model encoder add?** `hybrid_bins_lm` (6) is
   objective (3) with the from-scratch stack replaced by a frozen Qwen2.5-0.5B
   with LoRA, reading each event as `heart rate 92 (+1h)`. **It is not
   compute-matched** — see "The LM cell is not a matched-compute row" below — so
   it can answer "does a pretrained encoder help at all, at a budget that fits on
   the hardware in this project" and cannot answer "does it win at equal FLOPs".

`ar_cont` (2) is the fourth row and a control on the *auxiliary* rather than on
the objective: it is `ar_bins` with the decile head replaced by a continuous
Huber regression on the same next event's value. It separates "predicting the
number helps" from "predicting a latent helps", so a win for `latent_cont` over
`ar_bins` cannot be attributed to the value term alone.

## Decision rule, fixed before results

Read on the mean of the four held-out AUROCs, and separately per task, at both
seeds:

- (4) or (5) above (1) **at both seeds** and by more than the seed-to-seed spread
  of (1) → latent prediction does work of its own on continuous state.
- (4) and (5) at or below (1) → it does not, on this evidence, and the paper's
  claim stays limited to the hybrid.
- (3) above (1) at both seeds → the claims-data hybrid result generalises.

Two seeds distinguish "an effect" from "a draw". They do not establish a small
effect; a difference smaller than the gap between seeds 1 and 2 of the same cell
is reported as indistinguishable and nothing else.

## Protocol

- **Base config** `configs/pretrain_default.yaml`: 6-layer / 256-wide causal
  encoder, `max_len: 512`, `batch_size: 64`, `lr: 3e-4`, bf16. Every hybrid knob
  is the one chosen in [`ABLATION_RESULTS.md`](../ABLATION_RESULTS.md) (EMA
  target, horizons `[1, 4, 16]`, `lambda_recon: 0.1`, `lambda_sigreg: 0`).
- **Budget** 100,000,000 nominal token slots per cell =
  `ceil(100e6 / (64 x 512 x 1))` = 3,052 optimizer steps. Nominal, not real:
  batches are right-padded and a step consumes this many *slots* and somewhat
  fewer events. Budgeting on slots is what keeps two cells comparable when one
  changes its window length.
- **Source** `physionet2019`. Train 32,229 stays / 9.78M events; held-out 4,006
  stays / 1.19M events.
- **Eval** the full held-out split (`eval_subject_limit: null`), 200 bootstrap
  resamples, `probe_features: auto` → `last@final` (every cell is causal), all
  four tasks the source defines. Probes are logistic regression on frozen
  features; nothing is fine-tuned.
- **Seeds** 1 and 2 for every cell, every seed-1 cell listed first.
- **Controls** `random_init` once per architecture (`ar_bins_s1`,
  `hybrid_bins_s1`, `hybrid_bins_lm_s1`). A pretrained checkpoint that does not
  beat its own untrained twin has learned nothing a probe can use. Note what the
  LM control *is*: a pretrained Qwen trunk with untrained adapters and an
  untrained projection. It is the control for "what did the LoRA training add",
  not for "what did the pretraining add" — the latter has no control in this grid.
- **Baselines** count-feature `lr` and `gbm`, fit once on this cohort by
  whichever cell's eval runs first. They should reproduce
  [`docs/experiments/2026-09-10-eval-physionet2019/`](../2026-09-10-eval-physionet2019/):

  | task | prevalence | `lr` | `gbm` |
  |---|---|---|---|
  | `sepsis_6h` | 1.7% | 0.785 | 0.836 |
  | `sepsis_stay` | 5.6% | 0.768 | 0.769 |
  | `mortality_inhospital/24h` | — | not defined on this source | |
  | `mortality_inhospital/48h` | — | not defined on this source | |

  `sepsis_6h` draws up to eight anchors per stay, so its rows are not
  independent within a stay and its bootstrap interval is optimistic. That is a
  property of the task, identical across every row of the grid.

## The six cells

| cell | objective | code term | value term | latent term |
|---|---|---|---|---|
| `ar_bins` | `ar` | next-code CE (tied) | 11-way decile CE, weight 1 | — |
| `ar_cont` | `ar` | next-code CE (tied) | Huber on `value_z`, weight 1 | — |
| `hybrid_bins` | `nextlatent` | next-code CE x 0.1 | decile CE x 0.1 | horizons `[1, 4, 16]`, EMA |
| `latent_cont` | `nextlatent` | **none** | Huber on `value_z`, weight 1 | horizons `[1, 4, 16]`, EMA |
| `latent_only` | `nextlatent` | **none** | **none** | horizons `[1, 4, 16]`, EMA |
| `hybrid_bins_lm` | `nextlatent`, `encoder: lm` | next-code CE x 0.1 | decile CE x 0.1 | horizons `[1, 4, 16]`, shared |

Three notes on the arrangement, each a choice that could have gone otherwise:

- **`lambda_sigreg: 0` in every latent cell, `latent_only` included.** Pure latent
  prediction with an EMA target and no anti-collapse term is the collapse-prone
  configuration, and SIGReg is the repository's answer to it — but
  `ABLATION_RESULTS.md` found `lambda_sigreg: 0` the *best* hybrid knob on claims
  data, and turning it on for one arm only would confound "no code loss" with
  "different regularizer". So every cell runs the same setting and
  `effective_rank`, `mean_std` and `cos_gap` in each cell's `metrics.csv` are the
  collapse evidence. A `latent_only` row whose effective rank has fallen to ~1 is
  reported as collapsed, not as a fair loss.
- **The decile head weight.** For `ar_bins` the code softmax carries weight 1, so
  the decile head carries weight 1: "predict the next code and its decile" is one
  two-part task and a second weight would be a knob with nothing behind it. For
  `hybrid_bins` both sit inside the `lambda_recon: 0.1` term, which is the
  arrangement `JEPAObjective` already used for `recon_value`.
- **`value_z`, not the raw value, is the regression target.** It is what the cache
  stores, it is per-code standardised so a Huber delta of 1.0 means the same thing
  for a heart rate and a bilirubin, and it is clipped to ±5 so one mis-entered lab
  cannot own the gradient.

## The LM cell is not a matched-compute row

Measured on this repository's M4 (16 GB, MPS, bf16), Qwen2.5-0.5B + LoRA r=8 on
`q,k,v,o` with gradient checkpointing, batch 4 x 256-event windows from
`physionet2019`:

<!-- throughput table: filled from logs/throughput-lm-icu.log -->

`configs/grids/a2_physionet2019.yaml` carries the resulting budget and the
arithmetic behind it. The short version: the LM cell's budget is one to two
orders of magnitude below the 100M the from-scratch cells get, because a
0.5B-parameter encoder over ~2,000 tokens per window is that much more expensive
per event than a 13M-parameter one over 512 events. So:

- A **win** for `hybrid_bins_lm` over `hybrid_bins` would be a strong result: a
  pretrained encoder beating a from-scratch one on far less training.
- A **loss** says almost nothing. It is consistent with "pretrained text
  knowledge does not transfer to ICU waveforms" and equally consistent with "the
  budget was too small", and this grid cannot separate them.

That asymmetry is the reason the cell is in the grid at all rather than left out:
the cheap half of the answer is worth having, and the expensive half is honestly
out of reach here.

## Reproducing

```
# Build the caches (see src/ehrjepa/data/etl/physionet2019.py for the raw data).
python -m ehrjepa.data.etl physionet2019 --source data/physionet2019_raw --out data/meds/physionet2019
python -m ehrjepa.data.tokenize build --meds data/meds/physionet2019 --out data/cache/physionet2019

# The LM cell needs the optional extra.
pip install -e '.[lm,eval]'

python scripts/ablate.py configs/grids/a2_physionet2019.yaml --dry-run
nohup python scripts/ablate.py configs/grids/a2_physionet2019.yaml &
tail -f runs/a2-physionet2019/ablate.log
```

A cell interrupted mid-run resumes from its own `latest.pt`; a cell already in
`summary.json` is skipped. `--only ar_bins_s1,hybrid_bins_s1` runs a subset.
