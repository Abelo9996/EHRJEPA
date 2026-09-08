# 2026-09-07 -- ablate2-desynpuf: size, hybrid design, and the full held-out split

`configs/grids/ablate2_desynpuf.yaml`, complete as of 2026-09-08. Eighteen
rows: sixteen trained cells at 200M nominal token slots each (the same
per-cell budget as `scale-desynpuf`), two `reuse_checkpoint` rows that
re-score `scale-desynpuf`'s existing `ar` and `hybrid` checkpoints. Nothing
in this README is a result; the numbers are in `summary.md` and
`summary.json`, appended by `scripts/ablate.py` as each cell finished. The
analysis answering the three questions below, the figure, and the resulting
default-config decision are in
[`docs/experiments/ABLATION_RESULTS.md`](../ABLATION_RESULTS.md).

## Three questions, stated before the runs started

1. **Does the hybrid's lead over AR depend on model size?** `scale-desynpuf`
   and `scale1b-seeds-desynpuf` both trained the `nextlatent` + code-recon
   hybrid and the plain AR objective only at 6L/256d. This grid trains both at
   4L/192d/4h (`ar_small`, `hybrid_small`) and 8L/384d/6h (`ar_large`,
   `hybrid_large`) and reuses the 6L/256d checkpoints (`ar_base_s0`,
   `hybrid_base_s0`) so the three sizes read against each other under an
   identical eval protocol.
2. **Which of the hybrid's own knobs matter?** Four single-knob variants of
   the base-size hybrid: `hybrid_shared` (shared instead of EMA targets),
   `hybrid_nosig` (SIGReg off), `hybrid_h1` (a single horizon instead of
   `[1, 4, 16]`), `hybrid_recon03` (recon weight 0.1 -> 0.3).
3. **Do the 3,000-subject results hold on the full held-out set?** Every
   earlier desynpuf grid scored a seeded 3,000-subject subset of `held_out`
   (`eval_subject_limit: 3000`). This grid sets `eval_subject_limit: null`,
   which scores every held-out subject instead (`restrict_eval_split` returns
   its input unchanged when the limit is `None` -- verified in
   `tests/test_eval.py::test_restrict_eval_split_keeps_train_and_tuning_whole`
   and, at the runner level, in
   `tests/test_ablate.py::test_eval_one_omits_the_subject_limit_flag_when_none`).
   Because that cohort does not match `2026-09-03-eval-desynpuf/predictions.parquet`'s
   3,000-subject one, `reuse_predictions` is left unset here rather than
   pointed at that file: `lr` and `gbm` are fit fresh, once, on the first cell
   whose eval command runs (`scripts/ablate.py`'s `baselines_needed`), and
   every later cell finds them already cached in `baselines.json`.

## The two-seed rule

Every trained cell runs at `run.seed: 1` and `run.seed: 2` -- two independent
draws through data order, mask sampling, and initialization, so a size or
design difference has to clear "distinguishable from seed noise" before it is
read as a size or design effect. The grid file lists every seed-1 cell before
any seed-2 cell (size axis, then the hybrid design axis, at each seed) so a
`--only` restricted run or an interrupted grid trains one seed's worth of
cells before starting the second. The two `reuse_checkpoint` rows carry no
seed of their own -- they re-score whatever checkpoint `scale-desynpuf`
already produced.

## Protocol

- Base config: `configs/pretrain_scale.yaml` (6L/256d encoder, `max_len: 512`,
  `batch_size: 64`).
- Budget: 200,000,000 nominal token slots per trained cell
  (`steps = ceil(budget_tokens / (batch_size x max_len x accum_steps))` --
  see `scripts/ablate.py::steps_for`).
- Source: `desynpuf-s1`. Eval: `--eval-subject-limit` omitted (full
  `held_out` split), 200 bootstrap resamples, `probe_features: auto`
  (`last@final` for the causal encoders here), all tasks.
- Size overrides:
  - **small** -- `model.dim: 192`, `model.depth: 4`, `model.heads: 4`, plus
    the predictor shape from `configs/pretrain_pilot.yaml`
    (`pred_dim: 96`, `pred_depth: 2`, `pred_heads: 4` -- unused by `ar` and
    `nextlatent`, which build no transformer predictor, but set for the
    record).
  - **large** -- `model.dim: 384`, `model.depth: 8`, `model.heads: 6`,
    `run.batch_size: 32`, `optim.accum_steps: 2` (same effective batch as the
    unaccumulated cells, half the activation memory, to stay under the 8 GB
    VRAM budget in `docs/CUDA_SETUP.md`). `steps_for` multiplies by
    `accum_steps`, so this plans to the same 6,104 steps and the same
    200,015,872 nominal tokens as every batch-64 cell, not double.
  - **base** -- unchanged 6L/256d, re-scored from
    `runs/scale-desynpuf/{ar,hybrid}/final.pt` via `reuse_checkpoint`. Those
    files live on the GPU machine that trained them; `--dry-run` here plans a
    placeholder row for them (`scripts/ablate.py::_reuse_entry` no longer
    requires a local `config.json` to plan, only to actually evaluate).
- Objective overrides:
  - `ar_*`: `objective.kind: ar`, `model.causal: true`, `model.tie_embeddings: true`.
  - `hybrid_*` (and the base-size design axis, except where noted):
    `objective.kind: nextlatent`, `model.causal: true`, `model.target_mode: ema`,
    `objective.horizons: [1, 4, 16]`, `objective.lambda_recon: 0.1`,
    `objective.lambda_sigreg: 0.05`.
- `control_runs`: `ar_small_s1`, `hybrid_small_s1`, `ar_large_s1`,
  `hybrid_large_s1`, `ar_base_s0` -- one untrained `random_init` control per
  distinct encoder architecture (size x causal-ness). `ar` and `hybrid` share
  an architecture at a given size, so one control per size covers both; the
  base-size control also stands in for the four base-size hybrid-design-axis
  cells, which share that same 6L/256d causal shape.

## Dry-run plan

```
$ python scripts/ablate.py configs/grids/ablate2_desynpuf.yaml --dry-run
grid ablate2-desynpuf: base configs/pretrain_scale.yaml, source desynpuf-s1
summary docs/experiments/ablate2-desynpuf/summary.json  log runs/ablate2-desynpuf/ablate.log
  RUN         ar_small_s1          6104 steps x 64x512 =  200,015,872 tokens  ar/-- lambda=-- p_future=--
  RUN         hybrid_small_s1      6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         ar_large_s1          6104 steps x 32x512 =  200,015,872 tokens  ar/-- lambda=-- p_future=--
  RUN         hybrid_large_s1      6104 steps x 32x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         hybrid_shared_s1     6104 steps x 64x512 =  200,015,872 tokens  nextlatent/shared lambda=0.05 p_future=--
  RUN         hybrid_nosig_s1      6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0 p_future=--
  RUN         hybrid_h1_s1         6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         hybrid_recon03_s1    6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         ar_small_s2          6104 steps x 64x512 =  200,015,872 tokens  ar/-- lambda=-- p_future=--
  RUN         hybrid_small_s2      6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         ar_large_s2          6104 steps x 32x512 =  200,015,872 tokens  ar/-- lambda=-- p_future=--
  RUN         hybrid_large_s2      6104 steps x 32x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         hybrid_shared_s2     6104 steps x 64x512 =  200,015,872 tokens  nextlatent/shared lambda=0.05 p_future=--
  RUN         hybrid_nosig_s2      6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0 p_future=--
  RUN         hybrid_h1_s2         6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         hybrid_recon03_s2    6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  REUSE?      ar_base_s0              0 steps x 0x0 =            0 tokens  ?/-- lambda=-- p_future=--  (checkpoint not found locally -- plan only)
  REUSE?      hybrid_base_s0          0 steps x 0x0 =            0 tokens  ?/-- lambda=-- p_future=--  (checkpoint not found locally -- plan only)
  total outstanding: 3,200,253,952 tokens
```

16 trained cells (8 base-size, 4 small, 4 large), 2 reused, 18 rows total.
Every trained cell plans to the same 6,104 steps / 200,015,872 nominal tokens
regardless of size, confirming the accumulated large cells are matched to the
same budget as the unaccumulated ones. `REUSE?` (rather than `REUSE`) flags
the two rows whose checkpoint is not present in this worktree; the two `?`
placeholders in the objective/lambda columns come from the same missing-file
path.

## Wall-time estimate

Throughput assumptions (not measured on this grid's hardware): 33k tok/s at
6L/256d ("base"), ~50k tok/s at 4L/192d ("small"), ~18k tok/s at 8L/384d
("large") -- plus ~15 min eval per cell on the full held-out split (11.7k
subjects).

| size | cells | tokens/cell | tok/s | train time/cell | total train |
|---|---|---|---|---|---|
| base (design axis) | 8 | 200,015,872 | 33,000 | ~101 min | ~13.5 h |
| small | 4 | 200,015,872 | 50,000 | ~67 min | ~4.4 h |
| large | 4 | 200,015,872 | 18,000 | ~185 min | ~12.3 h |

Training total: ~30.3 h. Eval: 18 rows x ~15 min = ~4.5 h (the one-time
`lr`/`gbm` fit on the first cell is folded into that estimate rather than
added separately). **Total: ~34.8 h (~1.5 days)** of sequential wall time on
the RTX 4060 this grid was sized for.

## Provenance

- Grid file: `configs/grids/ablate2_desynpuf.yaml`
- Base config: `configs/pretrain_scale.yaml`
- Reused checkpoints: `runs/scale-desynpuf/ar/final.pt`,
  `runs/scale-desynpuf/hybrid/final.pt` (see `docs/experiments/scale-desynpuf/README.md`)
- Hardware target: RTX 4060, 8 GB VRAM (see `docs/CUDA_SETUP.md`)
