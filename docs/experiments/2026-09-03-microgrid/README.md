# 2026-09-03 — micro-grid: the pipeline test for `scripts/ablate.py`

**This is not an experiment.** It is the two-cell smoke test that had to pass
before the phase-5a pilot grid was launched, kept because "the runner worked once,
end to end, on this commit" is a claim worth being able to check — and because it
caught a bug that would have invalidated the pilot.

Two runs of 200 steps each — one per objective — through the whole path the pilot
takes: train to a token budget, evaluate the checkpoint on the same
3,000-subject `held_out` cut every other number in this repository uses, append a
row to [`summary.md`](summary.md) and `summary.json`. 200 steps is 3,276,800
nominal token slots, 7% of the pilot's budget and about 14% of one epoch of real
events. **The AUROCs are what a barely-initialised encoder scores. Reading the
two rows against each other as a comparison of objectives would be a category
error** — that comparison is what the pilot is for, at 14.6× the compute.

```bash
python scripts/ablate.py configs/grids/micro_desynpuf.yaml
```

## What it establishes

* the AR arm trains — cross-entropy falls from 10.26 to ~7.5 over 200 steps,
  top-1 rises from 0.003 to ~0.09, and the chunked softmax stays inside 5.5 GB;
* the JEPA arm's cosine gap moves off zero (+0.009 at step 50, +0.096 at step
  175), so the diagnostics are live on this config too;
* an AR checkpoint loads through `probe.load_encoder` and embeds through the
  *unchanged* `probe.embed` path — `EHRAR` and `EHRJEPA` really are
  interchangeable to the evaluation harness;
* the count-baseline reuse reproduces the phase-4 `lr`/`gbm` numbers exactly
  (`gbm` .744 / .572 / .765 / .767 / .772 / .789 / .672);
* rows are appended as each cell finishes, and re-running the grid skips cells
  already in `summary.json`.

## The bug it caught

On the first pass both cells' `random_init` controls came back byte-identical to
each other **and** to the phase-4 numbers. `probe.embedding_path` keyed the
embedding cache on a model's display name, and `random_init` is named after what
it is rather than after the checkpoint whose architecture it copies — so it read
back `emb__random_init.parquet`, left behind by the phase-4 evaluation of a
6×256 encoder. The control for the 4×192 grid was a different model.

Nothing in the output looked wrong: the numbers were plausible, the run
succeeded, and every `ckpt:` model was unaffected because those are already named
after their run directory. The pilot would have compared six checkpoints against
a control for an architecture none of them has.

Fixed by `ModelSpec.cache_name` (`random_init@<run dir>`), which changes the
cache key and leaves the display name — and therefore every results table —
alone. `test_random_init_cache_name_is_keyed_on_the_architecture_it_copies` in
`tests/test_eval.py` pins it. This directory holds the re-run, after the fix.

## Setup

| | |
|---|---|
| base config | `configs/pretrain_pilot.yaml` — 4×192 encoder, 2×96 predictor, `max_len` 256, batch 64 |
| grid | `configs/grids/micro_desynpuf.yaml` |
| budget | 3,276,800 nominal token slots per cell = 200 steps × 64 × 256 |
| data | `data/cache/desynpuf-s1`, `train` split, 78,997 subjects, vocabulary 30,000 |
| evaluation | 3,000-subject `held_out` cut (seed 0), 200 bootstrap resamples, probe `cls_mean@final`, no few-shot |
| baselines | `lr`/`gbm` reused from `../2026-09-03-eval-desynpuf/predictions.parquet`; `random_init` computed once per architecture |
| hardware | Apple M4, 16 GB, MPS, float32, torch 2.14 |

Results: [`summary.md`](summary.md), `summary.json`, `baselines.json`, and
per-cell evaluation output under `eval/<run>/`.
