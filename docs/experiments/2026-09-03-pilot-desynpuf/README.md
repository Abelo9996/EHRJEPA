# 2026-09-03 — phase-5a pilot ablation on DE-SynPUF

Six pretraining runs at one matched token budget, each probed on the same
held-out cohort every earlier number in this repository was scored on. The two
questions, stated before the runs started:

1. **Does the JEPA objective learn anything a next-code autoregressive objective
   does not, at matched compute?** Cell `ar` against cell `jepa_ema`. Same
   embedding, same encoder, same data order, same optimizer and schedule, same
   token budget — the encoder is causal in one and bidirectional in the other,
   and the loss is cross-entropy over `code_id[i+1]` in one and smooth-L1 against
   latent targets in the other. That is the whole difference.
2. **Does either beat a random-init probe with more training than the sanity
   runs got?** The phase-4 checkpoints (900 steps, ≈14.7M nominal tokens) tied
   with an untrained encoder at .67–.74 AUROC. This grid spends 3.3× that.

The remaining four cells vary one JEPA knob each against `jepa_ema`: the target
network (`shared` vs `ema`), the anti-collapse term (`lambda_sigreg` 0.05 vs 0),
and the masking mix (`p_future` 0.6 vs 1.0 vs 0.0).

**Results are appended automatically to [`summary.md`](summary.md) and
`summary.json` by `scripts/ablate.py` as each run finishes.** A row appears when
that run's training *and* its evaluation are both done; a partially-finished grid
has fewer rows, not wrong ones. Nothing in this README is a result and nothing
here should be read as one — this file is the protocol, written before the
numbers existed.

## The grid

`configs/grids/pilot_desynpuf.yaml`, run in this order:

| # | run | objective | target | `lambda_sigreg` | `p_future` |
|---|---|---|---|---|---|
| 1 | `ar` | next-code CE, causal | — | — | — |
| 2 | `jepa_ema` | JEPA | ema | 0.05 | 0.6 |
| 3 | `jepa_ema_nosig` | JEPA | ema | 0.0 | 0.6 |
| 4 | `jepa_shared_sig` | JEPA | shared | 0.05 | 0.6 |
| 5 | `jepa_ema_future` | JEPA | ema | 0.05 | 1.0 |
| 6 | `jepa_ema_block` | JEPA | ema | 0.05 | 0.0 |

## Config and budget

| | |
|---|---|
| base config | `configs/pretrain_pilot.yaml` |
| data | `data/cache/desynpuf-s1`, `train` split, 78,997 subjects after `min_len=16` (13,995 dropped), vocabulary 30,000 |
| encoder | 4 layers × 192 wide, 4 heads (head dim 48), SwiGLU, dropout 0.1 |
| predictor | 2 layers × 96 wide, 4 heads (JEPA cells only) |
| AR head | next-code softmax over 30,000 codes, output projection tied to the code embedding table |
| window | `max_len` 256, batch 64, `random_window` sampling |
| parameters | 7,959,680 trainable for a JEPA cell — embedding 5,892,288 (74%, of which 5.76M is the 30,000 × 192 code table), encoder 1,781,056, predictor 286,336. AR: 7,703,728 — same embedding and encoder, head 30,384 (a LayerNorm and the output bias; the projection matrix *is* the code table). An `ema` cell holds a second frozen copy of embedding+encoder, 15,633,024 parameters in total but the same 7,959,680 trainable. |
| optimizer | AdamW, lr 3e-4, cosine to 1% over the run, 100-step warmup, wd 0.05, clip 1.0, no accumulation |
| **budget** | **48,005,120 nominal token slots per run = 2,930 steps × 64 × 256, identical for every cell** |
| in real events | mean window is 121 events of 256 on this cache (47% fill, measured over 2,000 draws), so ≈22.7M real events ≈ 2.0 epochs of the 11.3M-event train split |
| seed | 0, every cell |
| precision | float32 (`precision: auto` on MPS) |
| hardware | Apple M4, 16 GB, MPS, torch 2.14 |

### Why 48M and not 12M

12M slots is 733 steps, which is *less* compute than the 900-step sanity runs
that already tied with a random-init probe — a grid at that budget could only
reproduce the tie, and half the point of this phase is what happens with more
training. 48M is the largest round budget that keeps the slowest cell inside a
45-minute wall-clock ceiling — 3.3x the sanity runs, about 2.0 epochs of real
events. Measured on this machine (`scripts/throughput.py`,
`runs/throughput/throughput.json`, 50 steps per cell, first logging window
dropped):

| config | objective | precision | tok/s | peak MB |
|---|---|---|---|---|
| 6×256, `max_len` 512, batch 32 | jepa | fp32 | 4,152 | 10,367 |
| 6×256, `max_len` 512, batch 32 | jepa | bf16 | 3,506 | 9,375 |
| 4×192, `max_len` 256, batch 64 | jepa | fp32 | 12,172 | 3,292 |
| 4×192, `max_len` 256, batch 64 | jepa | bf16 | 14,380 | 3,366 |
| 4×192, `max_len` 256, batch 64 | ar | fp32 | 10,044 | 5,548 |
| 4×192, `max_len` 256, batch 64 | ar | bf16 | 9,810 | 6,730 |

`tok/s` counts real (non-padding) events, so it is not `batch × max_len / step`.
At those rates 2,930 steps is ≈40 min for `ar` and ≈33 min for a JEPA cell.
float32 is used throughout: bf16 helps the JEPA cell and hurts the other three,
and a precision that varies by cell would be a second difference in a comparison
that only tolerates one.

## Evaluation

Identical to [`../2026-09-03-eval-desynpuf/`](../2026-09-03-eval-desynpuf/README.md),
which is the point — the rows have to land in the same frame as the phase-4
table:

* the same seeded 3,000-subject `held_out` cut (seed 0), `train` and `tuning`
  untouched;
* 200 bootstrap resamples, seed 0;
* logistic-regression probe on frozen `concat(CLS, masked mean of token outputs)`,
  encoder in `eval()`, fp32, no masking, history strictly before the anchor;
* all seven tasks; few-shot curves are off, since a per-cell few-shot sweep would
  cost more than the training it describes.

`lr` and `gbm` are count-feature models with no dependence on the encoder, so
their held-out scores are reused from the phase-4 `predictions.parquet` rather
than refit six times. `random_init` *is* architecture-dependent and is computed
twice — once against the `ar` cell and once against `jepa_ema`, appearing as
`random_init@ar` and `random_init@jepa_ema`. The control for this model has to be
an untrained copy of *this* model, not of the phase-4 6×256 one, and an untrained
causal encoder is not an untrained bidirectional one: its CLS row is a constant,
so half of its `cls_mean` vector is a constant column. One grid-wide control
would quietly compare the AR cell against a different probe than it got. All of
them land in `baselines.json` and in the "Reference models" table of
`summary.md`.

### One caveat on the AR cell's features

Under causal attention the CLS token sits at index 0 and can attend to nothing
but itself, so its output row is a constant and the `cls_mean` probe is
effectively mean-only for the `ar` cell. This is not a bug to route around: a
prefix token that read the sequence would feed the future back into every later
position at the next layer. `--probe-features last` (the final valid token)
exists for exactly this case and `--probe-layer penultimate` alongside it, but
the grid runs the default `cls_mean@final` so that every cell — and the phase-4
table — is scored on one pooling. Reading the AR row as "the mean-pooled
representation of a causal encoder" is the honest reading.

## Reproduce

```bash
python scripts/ablate.py configs/grids/pilot_desynpuf.yaml --dry-run
nohup python scripts/ablate.py configs/grids/pilot_desynpuf.yaml &
tail -f runs/2026-09-03-pilot-desynpuf/ablate.log
```

The runner is resumable at cell granularity: a cell already in `summary.json` is
skipped, so relaunching the same command after an interruption picks up where it
stopped. Per-run checkpoints are under `runs/2026-09-03-pilot-desynpuf/<run>/`
and per-run evaluation output (including `predictions.parquet`) under
`eval/<run>/`.

## How to read the numbers, when they arrive

DE-SynPUF is a synthetic public-use file built by sampling and swapping fields
across real beneficiaries specifically to break re-identifiable associations, so
predictive structure between a patient's history and their future is weakened by
construction. Low AUROCs are a property of the source. Nothing in this directory
is comparable to published numbers on MIMIC or EHRSHOT, and 2.0 epochs of a
7.9M-parameter model is not a statement about what any of these objectives can
do. What the grid can support is a *relative* claim at matched compute, with a
random-init control and a count-feature ceiling on the same rows.
