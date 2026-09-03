# configs/

Plain YAML run configurations (no hydra). Every key is checked against a
dataclass field at load time, so a typo is an error rather than a silently
ignored setting — see `ehrjepa.train.config`.

| file | what it is |
|---|---|
| `pretrain_debug.yaml` | tiny CPU config for tests and CI: 50 steps, seconds not minutes |
| `pretrain_small.yaml` | 6×256 encoder, 4×128 predictor, `max_len` 512, batch 32 |
| `pretrain_pilot.yaml` | 4×192 encoder, 2×96 predictor, `max_len` 256, batch 64 — sized so a six-cell ablation grid fits an evening on one 16 GB M4 |
| `tasks/` | ACES window definitions for the downstream tasks |
| `grids/` | ablation grids for `scripts/ablate.py` |

`--override key=value` uses dotted paths (`optim.lr=1e-4`, `model.depth=8`,
`objective.kind=ar`); values parse as YAML scalars.

## Objectives

`objective.kind` selects what the encoder is trained against:

* `jepa` (default) — latent prediction against a `shared` or `ema` target
  encoder, plus SIGReg. The rest of the `objective` section and all of `masking`
  apply.
* `ar` — next-code cross-entropy. Implies `model.causal: true` (the trainer sets
  it and says so); reads `model.tie_embeddings` and `objective.ar_chunk`; ignores
  `masking` and every `sigreg_*` key.

## grids/

A grid file is a base config, a token budget, and a list of named runs, each a
set of dotted-path overrides. `scripts/ablate.py` derives steps from the budget
(`ceil(budget_tokens / (batch × max_len))`), trains, evaluates and appends one
row per run to `docs/experiments/<grid name>/summary.md`.

| file | what it is |
|---|---|
| `micro_desynpuf.yaml` | two 200-step runs — a pipeline test for the runner, not an experiment |
| `pilot_desynpuf.yaml` | the phase-5a pilot: six cells at 48M token slots each |
