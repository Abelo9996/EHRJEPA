# 2026-09-06 -- scale-desynpuf: 200M-token grid on an RTX 4060

Four cells at 200M nominal token slots each (~4x the 48M-token pilot budget of
grids 1-4), on `configs/pretrain_scale.yaml`'s 6-layer/256-wide encoder rather
than the pilot's 4-layer/192-wide one. First grid in this repository run on
CUDA hardware (an RTX 4060, 8 GB VRAM) rather than the Apple M4/MPS the pilot
grids used. Nothing in this README is a result beyond what is stated as the
question below; the numbers land in [`summary.md`](summary.md) and
`summary.json`, appended by `scripts/ablate.py` as each cell finished.

**Question, stated before the runs started:** do the pilot grids' headline
objectives -- `ar`, the `nextlatent`+recon hybrid, `recon_only`, and the
original masked-span `jepa_ema` -- hold their ranking and their gain over
`random_init` at roughly 4x the token budget and a larger (6L/256d) encoder?

## The four cells

Carried forward from the pilot grids rather than re-derived (see
`configs/grids/scale_desynpuf.yaml` for the full rationale comment):

| run | objective | pilot equivalent | pilot mean AUROC | pilot gain |
|---|---|---|---|---|
| `ar` | next-code AR, causal | grid 1 `ar` | 0.7205 | +0.0445 |
| `hybrid` | dense causal next-latent (horizons [1,4,16]) + AR next-code loss (`lambda_recon` 0.1) | grid 4 `nextlatent_h1416_recon` | 0.7287 | +0.0923 |
| `recon_only` | masked-span, `lambda_pred: 0`, code recon only | grid 3 `recon_only` | 0.6990 | +0.0254 |
| `jepa_ema` | original masked-span latent JEPA, EMA target | grid 1 `jepa_ema` | 0.6829 | +0.0008 |

Same held-out 3,000-subject subset, anchor seed, and 200 bootstrap resamples
as the pilot grids, so a row here reads against `docs/experiments/PILOT_RESULTS.md`
directly. `lr` and `gbm` are reused from the same `predictions.parquet` as the
pilot grids (count-feature baselines do not depend on the encoder).

## Provenance

- Grid file: `configs/grids/scale_desynpuf.yaml`
- Base config: `configs/pretrain_scale.yaml`
- Hardware: RTX 4060, 8 GB VRAM (see `docs/CUDA_SETUP.md`)
- Consolidated across grids in [`docs/experiments/SCALE_RESULTS.md`](../SCALE_RESULTS.md)
