# 2026-09-06 -- scale1b-desynpuf: 1B-token grid on an RTX 4060

Two cells, `ar` and `hybrid`, at 1B nominal token slots each (~20x the
48M-token pilot budget, 5x the 200M-token `scale-desynpuf` grid), on the same
`configs/pretrain_scale.yaml` 6-layer/256-wide encoder as `scale-desynpuf`.
`recon_only` and `jepa_ema` are not part of this grid. Nothing in this README
is a result beyond what is stated as the question below; the numbers land in
[`summary.md`](summary.md) and `summary.json`, appended by `scripts/ablate.py`
as each cell finished.

**Question, stated before the runs started:** at 5x `scale-desynpuf`'s token
budget, does `ar`'s gain over `hybrid` narrow, hold, or reverse?

Same held-out 3,000-subject subset, anchor seed, and 200 bootstrap resamples
as the pilot and `scale-desynpuf` grids.

## Re-evaluation note

The `ar` row in this grid was re-evaluated after commit `8721a62` ("Fix
checkpoint eval cache collision by fingerprinting checkpoint content"). Before
that fix, `ModelSpec.cache_name` keyed a `ckpt:` model's embedding cache on its
display name alone, so a second grid whose cell was also named `ar` shared one
cache file with `scale-desynpuf`'s `ar` and would have silently reported that
grid's numbers instead of its own. `cache_name` now includes a fingerprint of
the checkpoint file itself, so this grid's `ar` row in `summary.md` is from its
own checkpoint, not a collision with `scale-desynpuf`.

## Provenance

- Grid file: `configs/grids/scale1b_desynpuf.yaml`
- Base config: `configs/pretrain_scale.yaml`
- Hardware: RTX 4060, 8 GB VRAM (see `docs/CUDA_SETUP.md`)
- Cache-collision fix: commit `8721a62`
- Seed replication at this budget: `configs/grids/scale1b_seeds_desynpuf.yaml`
  (`scale1b-seeds-desynpuf`, seeds 1 and 2 for `ar` and `hybrid`), running now
  -- not yet in `docs/experiments/SCALE_RESULTS.md`.
- Consolidated across grids in [`docs/experiments/SCALE_RESULTS.md`](../SCALE_RESULTS.md)
