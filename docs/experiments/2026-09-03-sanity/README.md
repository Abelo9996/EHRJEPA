# 2026-09-03 — pretraining sanity runs on DE-SynPUF

Three short runs whose only purpose is to check that the loop trains, that the
collapse diagnostics move, and that removing the anti-collapse term visibly
breaks something. **These are not results.** Nothing here is evaluated on a
downstream task, no run is tuned, none reaches a converged state, and 900 steps
at batch 32 is roughly 0.4 epochs of one demo-scale dataset.

## Setup

| | |
|---|---|
| data | `data/cache/desynpuf-s1`, `train` split, 78,997 subjects after `min_len=16` (13,995 dropped), vocabulary 30,000 (`--ndc-digits 9 --max-vocab 30000`) |
| model | encoder 6 x 256, 4 heads, SwiGLU; predictor 4 x 128, 4 heads; `max_len` 512; dropout 0.1 |
| parameters | 13,540,256 trainable — embedding 7,905,536 (58%, of which 7.68M is the 30,000 x 256 code table), encoder 4,729,056, predictor 905,664 |
| objective | smooth L1 (beta 1.0) + `lambda * (sigreg_tokens + sigreg_cls)`; SIGReg with 256 directions, 8192 rows, 17-point grid on [-5, 5] |
| masking | `p_future` 0.6; future spans U(8, 64) after a cut in [0.3L, 0.9L]; multi-block 2-4 blocks of 5-15% with 0-30% context drop |
| optimizer | AdamW, lr 3e-4, cosine to 1% over 900 steps, 100-step warmup, wd 0.05, clip 1.0, batch 32, no accumulation |
| hardware | Apple M4, 16 GB, MPS, float32 (`precision: auto`), torch 2.14 |
| bf16 | `precision: bf16` does run end to end on MPS under torch 2.14 (checked, 10 steps, losses within noise of fp32); these runs are fp32 so the numbers are not mixed-precision numbers |
| budget | `run.steps=900 run.max_seconds=1150`; A finished its 900 steps, B and C hit the wall-clock cap |
| commit | `1d67896` |

Reproduce:

```bash
python -m ehrjepa.train.pretrain --config configs/pretrain_small.yaml \
  --override run.steps=900 run.max_seconds=1150 run.log_every=25 run.ckpt_every=0 \
             run.out_dir=runs/sanity-A-default
#   B adds: objective.lambda_sigreg=0.0
#   C adds: model.target_mode=ema
```

## Numbers (final logged step of each run)

| run | steps | pred_loss | sigreg_tokens | sigreg_cls | eff. rank | per-dim std | cos gap | tok/s | peak MB | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| A default (shared, lambda 0.05) | 900 | 0.0408 | 0.0591 | 0.0419 | 114.1 | 0.888 | +0.0350 | 2,542 | 10,415 | 1116 |
| B lambda_sigreg = 0 | 775 | 0.0133 | — | — | 232.2 | 0.138 | +0.0001 | 2,327 | 10,404 | 1127 |
| C target_mode = ema | 750 | 0.1205 | 0.0103 | 0.0210 | 172.9 | 0.955 | +0.2326 | 2,537 | 11,436 | 1114 |

B's SIGReg columns are zero because the term is switched off, not because the
statistic is zero; the run computes neither.

`peak MB` is `torch.mps.driver_allocated_memory()` maxed over steps, which is the
size of the MPS driver pool including cached free blocks, not the live working
set. It is the number that matters for "will this fit", not for "how much does a
step need".

`cos gap` is mean cosine(prediction, own target) minus mean cosine(prediction,
rolled target) — the part of the cosine that cannot be produced by predicting the
batch mean.

## Trajectory shape, one sentence each

**A** — `pred_loss` falls 0.39 → 0.032 over the first ~475 steps and then flattens
between 0.03 and 0.04, with per-dimension std holding near 0.89, effective rank
settling around 100-115 after an initial drop from 196, and the cosine gap rising
from +0.004 to a noisy +0.03-0.05.

**B** — `pred_loss` falls faster and further (to 0.013) while per-dimension std
decays monotonically 0.51 → 0.14 and the cosine gap returns to +0.0001, i.e. the
loss improves as the representation shrinks and the predictions stop
distinguishing their own target from a mismatched one.

**C** — `pred_loss` drops to ~0.12 by step 400 and stays there, with the largest
cosine gap of the three (+0.23), the highest per-dimension std (0.955), and
SIGReg values an order of magnitude below A's, on a target network still early in
its 0.996 → 1.0 momentum schedule.

## What these runs do and do not show

They show the loop runs, the diagnostics are computable and non-degenerate, and
that the run without SIGReg is the one whose per-dimension std collapses and
whose cosine gap goes to zero — which is the failure mode the term exists to
prevent, and the reason `pred_loss` alone is not a metric here (B has the lowest
prediction loss and the least usable representation).

They do not show that A's representation is *good*. Nothing in this directory
touches a label. The three runs also differ in steps reached (900 / 775 / 750),
so the columns are not matched-budget comparisons, and A is the only one that
completed its cosine schedule.
