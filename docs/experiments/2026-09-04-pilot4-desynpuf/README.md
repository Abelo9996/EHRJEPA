# 2026-09-04 — phase-5d pilot ablation on DE-SynPUF

Five pretraining cells at grids 1–3's matched token budget, on the same cache,
seed, held-out subject subset and probe protocol. The question, stated before
the runs started:

**Does dense or window-level latent prediction transfer — alone, and with code
supervision?**

Nothing in this README is a result. This file is the protocol; the numbers land
in [`summary.md`](summary.md) and `summary.json`, appended by `scripts/ablate.py`
as each cell finishes.

## What grids 1–3 measured

Mean over the seven tasks of a cell's held-out AUROC minus its own untrained
control's, at 48M token slots on `desynpuf-s1` with 4-layer / 192-wide encoders
and linear probes on 3,000 held-out subjects:

| objective | gain over its own `random_init` |
|---|---|
| next-code AR | **+5** |
| masked-span latent JEPA (every target / masking / SIGReg variant tried) | **0** |
| JEPA + auxiliary code reconstruction on the predictor's outputs | **+2.5** |
| code reconstruction alone (`lambda_pred: 0`, no latent term at all) | **+2.5** |

The last two rows are the finding this grid exists because of: the latent
prediction term contributes nothing at this scale that the auxiliary code term
does not already contribute. Two things separate AR from masked-span JEPA — a
loss term at **every** position rather than at the 10–30% a mask covers, and a
**discrete** target. This grid keeps the latent target and takes the density.

## The two designs

Both are causal. Implementation and the exact leakage argument are in
`src/ehrjepa/models/latent.py`; the loss is in `src/ehrjepa/objectives/latent.py`.

### A. `nextlatent` — dense causal next-latent prediction

At every valid position `i`, a two-layer GELU MLP at the encoder's own width maps
`h_i` (plus the *time* of event `i + k`, and nothing else about it) to the target
encoder's latent at `i + k`, for each `k` in `objective.horizons`. One head per
horizon; the loss is the mean of the per-horizon smooth L1s against
layer-normalised targets, so adding a longer horizon does not reweight the
shorter ones. As many terms per window as the AR loss has.

`lambda_recon` here is the AR loss itself — next-code cross-entropy from the same
`h_i` through the tied code head — which makes cell (3) a literal AR+JEPA hybrid
and makes `lambda_pred: 0` reduce to next-code AR exactly (pinned by a test).

The known risk, stated in advance: the target encoder is causal too, so `z_{i+1}`
contains a prefix `h_i` already summarises and a predictor can score well by
copying. `cos_gap` — the mean cosine of predictions with their own targets minus
with shuffled ones — is the per-step diagnostic that says whether it did more
than that.

### B. `window` — future-window pooled latent

Eight anchors per window are drawn without replacement from `[0.3L, 0.9L]`. The
context summary for anchor `a` is the causal encoder's output at `a - 1`, so
"strictly before the anchor" is enforced by the attention mask rather than by a
second forward pass over a truncated window; this is a deliberate simplification
over re-encoding `events[:a]` bidirectionally, which would cost `K` times the
compute for a context differing only in direction. The target is the mean of the
target encoder's outputs over the events whose timestamp lies in
`(t_a, t_a + H]`, for each `H` in `objective.window_horizons` (days), with a
learned horizon embedding summed into the shared MLP's input.

An anchor/horizon pair is **skipped** when the horizon runs past the last event
in the window (the future is not observed, and a truncated pool would be
indistinguishable from a quiet one) or when no event falls inside it. Measured on
this cache at `max_len: 256` with 8 anchors, before the grid was queued:

| horizon | anchors kept | events per kept pool |
|---|---|---|
| 30 days | 0.71 | 6.5 |
| 365 days | 0.47 | 44 |

The realised fraction is logged every step as `skipped_frac`.

`lambda_recon` here is a multi-label BCE (TransformEHR-style) from the *predicted
pooled latent*, through the same tied code head, to the multi-hot set of distinct
codes occurring inside that anchor's horizon. `positives_per_anchor` — the mean
number of distinct codes per row, which is not the number of events — is logged
alongside it.

## The cells

| # | run | design | horizons | `lambda_recon` |
|---|---|---|---|---|
| 1 | `nextlatent_h1` | A | `[1]` | 0 |
| 2 | `nextlatent_h1416` | A | `[1, 4, 16]` | 0 |
| 3 | `nextlatent_h1416_recon` | A | `[1, 4, 16]` | 0.1 (next-code CE) |
| 4 | `window_30_365` | B | `[30, 365]` days | 0 |
| 5 | `window_30_365_recon` | B | `[30, 365]` days | 0.1 (multi-label BCE) |

All five: `model.target_mode: ema`, `objective.lambda_sigreg: 0.05`,
`model.causal: true`, 48M token slots at 64 × 256, seed 0.

## What is not rerun

`ar` and `recon_only` already have rows in grids 1–3 on the identical protocol
(`docs/experiments/2026-09-03-pilot-desynpuf/`,
`docs/experiments/2026-09-04-pilot3-desynpuf/`). Read this grid against those
rows and against `random_init@nextlatent_h1`, which is the causal 4×192 encoder
at initialisation. One control covers all five cells: a probe reads only the
embedding and the encoder, both are the same stack in every cell, and both are
constructed before any head, so an untrained `window` model and an untrained
`nextlatent` model hand a probe the same weights.

## How to read a row

The `pooling` column resolves to `last@final` for every cell here — the CLS row
of a causal encoder is a function of the CLS parameter alone, so the last valid
token is the summary a probe wants. A cell has learned something a probe can use
only if it beats `random_init@nextlatent_h1`; the count-feature `lr` and `gbm`
rows are the other reference, and neither is a target to beat at this budget.

## Provenance

- Grid file: `configs/grids/pilot4_desynpuf.yaml`
- Base config: `configs/pretrain_pilot.yaml`
- Pipeline test before the grid: `docs/experiments/2026-09-04-microgrid4/`
- Log: `runs/2026-09-04-pilot4-desynpuf/ablate.log`
