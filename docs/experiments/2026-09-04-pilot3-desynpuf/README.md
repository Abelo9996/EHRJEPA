# 2026-09-04 — phase-5c pilot ablation on DE-SynPUF

Four cells, all trained at grid 2's matched token budget. The question, stated
before the runs started:

**With code reconstruction present, does the latent smooth-L1 prediction term
still add anything to what a probe can read off the representation -- and do
the notime and recon gains stack, or is closing one shortcut just closing the
other one twice?**

Nothing in this README is a result. This file is the protocol; the numbers land
in [`summary.md`](summary.md) and `summary.json`, appended by `scripts/ablate.py`
as each cell finishes.

## Why this grid

Grid 2 moved two flags, each on its own, off the unchanged `jepa_ema` reference:
`jepa_notime` (`predictor.mask_token_time: false`) and `jepa_recon`
(`objective.lambda_recon: 0.1`). Neither grid ran them together, and neither
grid could ask whether the latent prediction term is doing anything once the
auxiliary code-CE term exists to carry gradient into the encoder — every JEPA
cell so far has had both terms on, in a fixed 1:1 ratio.

`objective.lambda_pred` (new: default `1.0`, weights the latent smooth-L1 term;
`0` drops it from the total and skips the target-encoder forward pass that would
have fed it, so the run trains nothing it does not use) makes that separable.
Setting it to `0` alongside `lambda_recon: 0.1` gives a pure masked-code-prediction
objective, through the same predictor, with the same masking and the same
budget — the reconstruction term isolated from the latent one, rather than
inferred by subtracting two runs that also differ in what else moved.

## The grid

`configs/grids/pilot3_desynpuf.yaml`, run in this order. Every cell is `jepa`,
`lambda_recon` 0.1, and — except where the table says otherwise — the same
`ema` / `lambda_sigreg 0.05` reference as grid 2's `jepa_ema`.

| # | run | `lambda_pred` | `lambda_sigreg` | mask token time | closes | trains |
|---|---|---|---|---|---|---|
| a | `jepa_recon_notime` | 1 (default) | 0.05 | **off** | notime + recon together | yes |
| b | `recon_only` | **0** | 0.05 | on | recon alone, latent term gone | yes |
| c | `recon_only_notime` | **0** | 0.05 | **off** | recon alone, notime too | yes |
| d | `jepa_recon_nosig` | 1 (default) | **0** | off | notime + recon, SIGReg gone instead | yes |

Reading order:

* **(a) vs. grid 2's `jepa_notime` and `jepa_recon`.** If closing the notime
  shortcut and adding reconstruction move the probe independently, (a) should
  land near their combined effect; if they are the same shortcut, (a) should
  look like whichever single flag moved more.
* **(b)/(c) vs. (a).** (b) and (c) have no latent term at all — whatever they
  score, they score on code reconstruction alone. If (a) does not clearly beat
  (b)/(c), the latent smooth-L1 term is not contributing once reconstruction is
  present at this budget.
* **(c) vs. (b).** The same additive question grid 2 asked between `jepa_notime`
  and `jepa_ema`, asked again with the latent term already gone: does removing
  the mask-token clock still help when the objective is pure reconstruction, or
  was its effect in grid 2 only visible because the latent shortcut was still
  open.
* **(d) vs. (a).** (d) drops `lambda_sigreg` instead of `lambda_pred` on top of
  the same notime + recon config, so a difference between (a) and grid 2's
  unchanged reference that might look like "recon helped" can be checked against
  whether it was actually "the anti-collapse term stopped mattering."

`recon_only` and `recon_only_notime` set `objective.lambda_pred: 0` but leave
`model.target_mode` at the base config's `ema`: the EMA modules still exist (a
checkpoint from any cell in this grid deserializes the same way), they are just
never called forward or updated while `lambda_pred` is `0` — see
`EHRJEPA.forward`'s `compute_targets` and the EMA-update guard in
`ehrjepa.train.pretrain.Trainer.train`.

## Config, budget and controls

Identical to grid 2 in every respect not named above: same
`configs/pretrain_pilot.yaml` base (4x192 encoder, 2x96 predictor, SwiGLU,
dropout 0.1), same 48,005,120-token budget per cell (2,930 steps at 64x256),
same seed (0), same held-out evaluation subset (3,000 subjects, seed 0, 200
bootstrap resamples, all seven tasks), same `lr`/`gbm` baselines reused from
`../2026-09-03-eval-desynpuf/predictions.parquet`.

Every cell here builds a `recon_head` (`lambda_recon` is 0.1 throughout), so the
architecture is identical across all four rows — one `random_init` control,
computed against `jepa_recon_notime`, covers the grid; grid 3 does not repeat
grid 2's re-scored `ar_last`/`jepa_ema_mean` rows because nothing here changes
pooling or reuses another grid's checkpoint.

## Reproduce

```bash
python scripts/ablate.py configs/grids/pilot3_desynpuf.yaml --dry-run
nohup python scripts/ablate.py configs/grids/pilot3_desynpuf.yaml &
tail -f runs/2026-09-04-pilot3-desynpuf/ablate.log
```

Queued behind grid 2, same reasoning as grid 2 was queued behind grid 1 — one
GPU, one grid at a time:

```bash
nohup scripts/queue_after.sh <grid-2-pid> \
    python scripts/ablate.py configs/grids/pilot3_desynpuf.yaml \
    > runs/2026-09-04-pilot3-desynpuf/wrapper.log 2>&1 &
```

The runner is resumable at cell granularity. The pipeline test for this grid,
run before it was launched, is
[`../2026-09-04-microgrid3/`](../2026-09-04-microgrid3/README.md).

## How to read the numbers, when they arrive

Everything grid 1 and grid 2's closing sections say still holds: DE-SynPUF is
synthetic, low AUROCs are a property of the source, and 2.0 epochs of a
7.9M-parameter model is not a statement about what any objective can do at
scale. A cell that closes a shortcut and does not move the probe has said
something — that the shortcut was not what was limiting the representation —
and that is a result, not a failure.
