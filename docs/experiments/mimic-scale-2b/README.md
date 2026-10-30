# mimic-scale-2b — AR vs hybrid at 2B tokens, seed 0

Extends the seed-0 scaling curve to a third budget (300M, 1B, 2B) on the ~30k-subject
MIMIC-IV cache. Question: does the hybrid-over-AR gap keep growing with scale? Fresh run
(LR cosine stretched over 61,036 steps), not a continuation of the 1B checkpoint. Full
held-out split, 200 bootstrap, probe `auto@final`. All numbers from `summary.md`.

## Seed-0 scaling curve (mean-of-6, non-mortality AUROC)

| budget | AR | hybrid | gap (hybrid − AR) |
|---|---|---|---|
| 300M | 0.768 | 0.770 | +0.002 |
| 1B   | 0.772 | 0.798 | +0.027 |
| 2B   | 0.764 | 0.798 | +0.034 |

- The gap grows across scale (+0.002 → +0.027 → +0.034), now ~3.8× the 300M noise floor (0.009).
- hybrid is flat 1B→2B (0.798 → 0.798); AR is flat-to-down (0.772 → 0.764, inside the noise
  floor). The widening gap at 2B is driven by AR not improving, not by hybrid climbing — both
  objectives have plateaued on this subset, hybrid higher.
- hybrid wins all 7 tasks at 2B (per-task in `summary.md`).

## Caveat
Single seed (run.seed 0). The gap is well above the AR noise floor, but hybrid's own seed
spread at 2B is unmeasured. For error bars, resume `mimic-scale-1b` seeds 1,2 (checkpointed)
or extend 2B to seeds.

## Count baselines on the same split (reference, not the decision metric)
GBM mean-of-6 = 0.808, LR = 0.795 — both above the pretrained models on this subset. Per the
re-validation plan the decision metric is hybrid-vs-AR at matched compute; beating GBM on
MIMIC counts is a separate, harder bar. random_init@ar_s0 = 0.666 (both trained models clear
the untrained control: hybrid +0.131, AR +0.097).

## Crash/resume note
The run was paused once (gaming) and crashed once (WSL segfault, user-DB I/O error — cleared
by `wsl --shutdown`). Both recovered from checkpoints (ckpt_every 2000): ar_s0 resumed from
step 50k, hybrid_s0 from step 48k. Final numbers are from the completed `final.pt` of each.
