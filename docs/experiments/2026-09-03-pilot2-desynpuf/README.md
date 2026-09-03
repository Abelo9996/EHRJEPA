# 2026-09-03 — phase-5b pilot ablation on DE-SynPUF

Seven cells: five pretraining runs at grid 1's matched token budget, and two
re-evaluations of grid 1's own checkpoints that train nothing. The question,
stated before the runs started:

**Grid 1's JEPA cells tied their own untrained control. Is that because the
prediction task can be solved without the encoder — and if the shortcuts are
closed, does the probe read anything different off the representation?**

Nothing in this README is a result. This file is the protocol; the numbers land
in [`summary.md`](summary.md) and `summary.json`, appended by `scripts/ablate.py`
as each cell finishes.

## What the diagnostics found

Four measurements on grid 1's checkpoints, which are what this grid is designed
against. Each is a route by which the objective can be satisfied without the
encoder carrying anything a probe would want:

1. **The task is mostly a time-conditional prior.** A ridge map from the
   predictor's mask-token time features *alone* reaches R² 0.58 on the
   layer-normed targets; the trained predictor, with the context as well, reaches
   0.79. Most of what the predictor produces is recoverable from "what time is
   it" without looking at the patient.
2. **The encoders discard code identity.** A linear probe recovering a token's
   own `code_id` from the frozen encoder output scores top-1 0.50 on the JEPA
   checkpoints — against **0.60 for the untrained encoder** and 0.71 for the AR
   one. Pretraining made this *worse* than initialisation.
3. **Full-sequence targets admit a context-copy shortcut.** Without SIGReg the
   target embeddings are 45% between-window variance, and a plain context-mean
   baseline reaches cosine 0.73 to the targets. The target encoder attends over
   the whole window, including the context the predictor was handed, so a target
   latent can contain what the predictor already has.
4. **Pooling was reported to be costing the causal arm 2.5–3.3 AUROC points.**
   `last@final` against `cls_mean@final` on the AR checkpoint, which is a fact
   about how the representation is read, not about how it was trained.

   **That fourth figure did not survive re-measurement.** The pipeline test for
   this grid re-scored the same `ar` checkpoint on the full seven-task,
   3,000-subject held-out cut: `last` wins four tasks of seven, ranges from
   −2.07 to +2.77 points, and averages **+0.17**. See
   [the micro-grid README](../2026-09-03-microgrid2/README.md#one-thing-the-pipeline-test-found)
   for the per-task table. The pooling default still changed, on the
   architectural argument — a causal encoder's CLS row is a constant it cannot
   reach, so half of its `cls_mean` vector carries nothing — but the *size* of
   the effect is not what the diagnostic claimed, and rows 6 and 7 below exist so
   that whatever it really is stays separable from the objective changes.

## The grid

`configs/grids/pilot2_desynpuf.yaml`, run in this order. Every trained cell is
`jepa` / `ema` / `lambda_sigreg` 0.05 — the same reference cell as grid 1's
`jepa_ema` — and differs from it only in the columns shown.

| # | run | closes | mask token time | target | `lambda_recon` | `p_future` | trains |
|---|---|---|---|---|---|---|---|
| 1 | `jepa_notime` | (1) | **off** | full window, timed | 0 | 0.6 | yes |
| 2 | `jepa_content` | (2)(3) | on | **span only, content** | 0 | 0.6 | yes |
| 3 | `jepa_content_recon` | (2)(3) | on | **span only, content** | **0.1** | 0.6 | yes |
| 4 | `jepa_recon` | (2) | on | full window, timed | **0.1** | 0.6 | yes |
| 5 | `jepa_content_future` | (2)(3) | on | **span only, content** | 0 | **1.0** | yes |
| 6 | `ar_last` | (4) | — | — | — | — | **no** — grid 1 `ar` |
| 7 | `jepa_ema_mean` | (4) | on | full window, timed | 0 | 0.6 | **no** — grid 1 `jepa_ema` |

Cells 2, 3 and 5 also set `train.time_feature_dropout: 0.3`. Content targets mean
the target encoder is shown tokens with no age and no log-gap; under EMA the
target weights are a copy of the online ones, so the online encoder is trained
with the same terms dropped 30% of the time rather than being handed an input
distribution it has never seen.

Cell 4 exists to make cell 3 readable. Without it, any movement in
`jepa_content_recon` could be the target change or the auxiliary loss and there
would be no way to say which.

### What each flag does

* `predictor.mask_token_time: false` — a mask token becomes the bare learned
  `MASK` embedding. RoPE at the target's original index is the only positional
  information left, so the predictor knows *which* target it is producing but not
  *when* it falls. The time encoders stay allocated and never run.
* `target.time_features: false` — the target encoder's `EventEmbedding` omits the
  age and log-gap terms, so a target latent is a function of content: code, value
  bin, and the gated value residual.
* `target.span_only: true` — the target encoder runs on the target positions
  compacted into their own sequence, with their own attention mask and their own
  CLS, and the rows are scattered back into window coordinates. A target latent
  therefore cannot contain the context. The cost is honest and worth naming: RoPE
  now sees within-span offsets, so contiguous targets keep their true relative
  positions and the gaps *between* multi-block targets are lost.
* `objective.lambda_recon: 0.1` — cross-entropy from the predictor's output at
  each target to that event's `code_id`, through the code embedding table (the AR
  next-code head, reused). A predicted latent that must name its event's code
  cannot be a code-free summary, and the gradient reaches the encoder through the
  context. Logged in its own `recon_loss` column.

## Pooling — read this before comparing to grid 1

This grid runs `probe_features: auto`, which resolves per checkpoint: `mean@final`
for a bidirectional encoder, `last@final` for a causal one. Grid 1 ran
`cls_mean@final` for everything.

**The trained rows here are therefore not directly comparable to grid 1's rows.**
That is what cells 6 and 7 are for: they are grid 1's `ar` and `jepa_ema`
checkpoints, unchanged, re-scored under this grid's pooling, and they are the
rows the five trained cells should be read against. Each row's `probe` column
states the pooling it actually got.

`random_init` is recomputed here rather than reused, for the same reason: grid 1's
controls were computed at `cls_mean`, and a control is only a control if it was
probed the way the thing it controls for was probed. Two are computed — one
against `jepa_notime` (bidirectional, `mean`) and one against `ar_last` (causal,
`last`).

## Config and budget

Identical to grid 1 except where the table above says otherwise.

| | |
|---|---|
| base config | `configs/pretrain_pilot.yaml` — 4×192 encoder, 2×96 predictor, SwiGLU, dropout 0.1 |
| data | `data/cache/desynpuf-s1`, `train` split, 78,997 subjects, vocabulary 30,000 |
| window | `max_len` 256, batch 64, `random_window` sampling |
| optimizer | AdamW, lr 3e-4, cosine to 1%, 100-step warmup, wd 0.05, clip 1.0 |
| **budget** | **48,005,120 nominal token slots per trained cell = 2,930 steps × 64 × 256** |
| seed | 0, every cell |
| precision | float32 (`precision: auto` on MPS) |
| hardware | Apple M4, 16 GB, MPS, torch 2.14 |
| evaluation | seeded 3,000-subject `held_out` cut (seed 0), 200 bootstrap resamples, all seven tasks, no few-shot |
| baselines | `lr`/`gbm` reused from `../2026-09-03-eval-desynpuf/predictions.parquet`; `random_init` computed once per (architecture, pooling) |

Five trained cells at ≈33–40 min idle, plus seven evaluations at ≈4–7 min each
once the count caches are warm — roughly 4 h idle, longer if the machine is busy.
Cells 6 and 7 cost evaluation only.

## Reproduce

```bash
python scripts/ablate.py configs/grids/pilot2_desynpuf.yaml --dry-run
nohup python scripts/ablate.py configs/grids/pilot2_desynpuf.yaml &
tail -f runs/2026-09-03-pilot2-desynpuf/ablate.log
```

This grid was queued behind grid 1 rather than run alongside it — one laptop, one
GPU, and two grids sharing it produce two sets of tok/s numbers that mean nothing:

```bash
nohup scripts/queue_after.sh <grid-1-pid> \
    python scripts/ablate.py configs/grids/pilot2_desynpuf.yaml \
    > runs/2026-09-03-pilot2-desynpuf/wrapper.log 2>&1 &
```

The runner is resumable at cell granularity; a cell already in `summary.json` is
skipped. Checkpoints for the five trained cells land under
`runs/2026-09-03-pilot2-desynpuf/<run>/`; cells 6 and 7 read grid 1's checkpoints
and write nothing outside this directory. Per-cell evaluation output is under
`eval/<run>/`.

The pipeline test for all of this is
[`../2026-09-03-microgrid2/`](../2026-09-03-microgrid2/README.md): two cells at
100 steps, one of them a `reuse_checkpoint` row, run before this grid was
launched.

## How to read the numbers, when they arrive

Everything in grid 1's closing section still holds. DE-SynPUF is synthetic data
built by swapping fields across beneficiaries specifically to break
re-identifiable associations, so low AUROCs are a property of the source, and 2.0
epochs of a 7.9M-parameter model is not a statement about what any objective can
do.

What this grid can support is narrower than "does closing the shortcuts help".
The diagnostics establish that the shortcuts *exist*; these cells establish
whether removing them moves the held-out probe at this budget, against a
random-init control probed the same way. A cell that closes a shortcut and does
not move the probe has said something — that the shortcut was not what was
limiting the representation — and that is a result, not a failure.
