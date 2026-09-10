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
* `nextlatent` / `window` — the two causal latent objectives
  (`ehrjepa.models.latent`). Both imply `model.causal: true`, read
  `objective.horizons` / `window_horizons`, and ignore `masking`.

Three auxiliary terms stack on top of whichever objective runs, all off by
default and all building no parameters at their default:

| key | term | reads |
|---|---|---|
| `objective.lambda_recon` | next-code (or masked-code) cross-entropy through the tied code table | `jepa`, `nextlatent`, `window` |
| `objective.recon_value` | 11-way cross-entropy on the target event's decile `value_bin` | `jepa`, `nextlatent`, `ar` |
| `objective.lambda_value` | Huber regression on the target event's continuous `value_z`, masked to the events that carry a number | every objective |

`recon_value` sits at the same weight as the code term it accompanies —
`lambda_recon` for the latent and masked objectives, `1.0` for `ar`, whose code
softmax is the objective itself. `lambda_value` carries its own weight.

## Encoders

`model.encoder` selects the event stack:

* `scratch` (default) — this repository's `EventEmbedding` (code + decile +
  gated value + two Fourier time channels) and RoPE transformer `Encoder`.
* `lm` — each event is serialised to a short text span (`heart rate 92 (+1h)`)
  from its `ehrjepa.data.code_text` description and read off a frozen pretrained
  causal LM with LoRA adapters; see `ehrjepa.models.lm` for the mechanism, the
  `lm_*` keys and what it refuses (`target_mode` other than `shared`,
  `objective.kind: jepa`, `target.span_only`, `share_time_encoders`,
  `code_init: text`). Needs the optional `lm` extra: `pip install -e '.[lm]'`.

## grids/

A grid file is a base config, a token budget, and a list of named runs, each a
set of dotted-path overrides. `scripts/ablate.py` derives steps from the budget
(`ceil(budget_tokens / (batch × max_len))`), trains, evaluates and appends one
row per run to `docs/experiments/<grid name>/summary.md`.

| file | what it is |
|---|---|
| `micro_desynpuf.yaml` | two 200-step runs — a pipeline test for the runner, not an experiment |
| `pilot_desynpuf.yaml` | the phase-5a pilot: six cells at 48M token slots each |
| `a2_physionet2019.yaml` | six objectives × two seeds on continuous ICU state, 100M slots each; pre-registered in `docs/experiments/a2-physionet2019/README.md` |
| `a2_physionet2012.yaml` | the same six on the mortality challenge, 40M slots each |
