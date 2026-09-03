# EHRJEPA

A joint-embedding predictive architecture for longitudinal electronic health
records. Each patient is a stream of clinical events in
[MEDS](https://github.com/Medical-Event-Data-Standard/meds) format — one token
per event, whose embedding fuses a learned code embedding with a continuous
value embedding and is positioned by continuous time rather than by ordinal
index, so irregular sampling and long gaps are represented as they actually
occurred. A context encoder reads the observed events; a latent predictor,
conditioned on the time positions of a masked future span, predicts that span's
representations; an EMA copy of the encoder produces the targets. Nothing is
reconstructed in input space — prediction happens entirely in latent space, so
the model is not forced to model unpredictable per-event detail. Collapse is
prevented by a SIGReg-style distributional regularizer following LeJEPA
([arXiv:2511.08544](https://arxiv.org/abs/2511.08544)), which pushes embeddings
toward an isotropic Gaussian via random one-dimensional projections instead of
relying on architectural asymmetry or negatives.

## Architecture

```mermaid
flowchart LR
    events["MEDS events<br/>code · value · timestamp"] --> embed["EventEmbedding<br/>code + value_bin<br/>+ value_z Fourier<br/>+ age Fourier + log_delta Fourier"]
    embed --> ctx["context Encoder<br/>(bidirectional, RoPE)"]
    ctx --> pred["Predictor<br/>(mask tokens + target time features)"]
    pred --> plat["predicted latents"]

    embed -. full window or span-only .-> tgt["target Encoder<br/>(EMA or shared, stop-grad)"]
    tgt --> tlat["layer-normed target latents"]

    plat --> loss["loss = smooth L1(pred, target)<br/>+ SIGReg(tokens) + SIGReg(CLS)<br/>+ optional code-reconstruction"]
    tlat --> loss
```

*The context encoder and predictor carry gradients; the target encoder runs
under stop-gradient (`target_mode: shared` or `ema`) and is never told what
code or value the masked span actually holds.*

## Status

**Under active rebuild. No results yet.**

There are no pretrained weights, no benchmark numbers, and no claims of
performance in this repository. The previous visit-vector prototype has been
archived under [`legacy/`](legacy/README.md); it has known label leakage and
none of its numbers are valid.

What exists now: the MEDS ETL and tensor cache (`data/`), the three networks
(`models/`), the SIGReg + latent-prediction objective and a next-code
autoregressive baseline that shares its embedding and encoder (`objectives/`), a
resumable pretraining loop (`train/`), a downstream evaluation harness (`eval/`)
with ACES-defined tasks, count-feature baselines, frozen-encoder probes and
bootstrap intervals, and an ablation runner that trains a grid of objectives to a
matched token budget and evaluates each (`scripts/ablate.py`). The runs recorded
under `docs/experiments/` are short MPS sanity runs, one evaluation of the
checkpoints they produced, and a pilot ablation grid; none of the checkpoints
involved is trained long enough for its numbers to be a statement about the
architecture.

## Data access

No data is included or committed; everything under `data/` is gitignored.

- **MIMIC-IV** — credentialed access via
  [PhysioNet](https://physionet.org/content/mimiciv/) (CITI training + signed
  DUA). Lowered to MEDS with `meds_etl`.
- **EHRSHOT** — Stanford Redivis, under its own data use agreement.
- **Development** — the MIMIC-IV *demo* subset and Synthea-generated records are
  open and are what the pipeline is developed against; they are small and
  synthetic/partial, so they are for plumbing only, never for evaluation.
- **Evaluation** — downstream tasks will be defined through
  [MEDS-DEV](https://github.com/mmcdermott/MEDS-DEV) / ACES so cohorts and label
  windows are declarative and reproducible, plus the EHRSHOT task suite.

### Data pipeline

```mermaid
flowchart LR
    src["MIMIC-IV · DE-SynPUF · Synthea"] --> etl["ehrjepa.data.etl"]
    etl --> meds["canonical MEDS layout<br/>train / tuning / held_out, codes, splits"]
    meds --> tok["ehrjepa.data.tokenize<br/>vocab (hierarchical fallback)<br/>+ per-code decile quantizer"]
    tok --> cache["memmap tensor cache"]
    cache --> ds["EventSequenceDataset / windows_at"]
    ds --> use["pretrain or eval"]
```

*Each source is lowered to the same canonical MEDS layout once; the tensor
cache and vocabulary are fit only on the `train` split and then reused for
`tuning` and `held_out`.*

## Pilot results

<p align="center">
  <img src="docs/figures/pilot_desynpuf_auroc.png" alt="Held-out AUROC by task for gbm, lr, ar, best jepa_*, and their random_init controls, DE-SynPUF sample 1 pilot" width="700">
</p>

*At this 48M-token budget, `ar` beats its untrained `random_init@ar` control on
6 of 7 tasks (mean AUROC 0.721 vs. 0.676); each `jepa_*` variant's mean AUROC
(0.681–0.690) sits within 0.01 of its `random_init@jepa_ema` control (0.682).
Grid 2 (content-only targets) is in progress.*

**Current findings** (from
[`docs/experiments/2026-09-03-pilot-desynpuf/summary.md`](docs/experiments/2026-09-03-pilot-desynpuf/summary.md),
2,930 steps / 48,005,120 tokens per run, 3,000 held-out DE-SynPUF subjects, 200
bootstrap resamples, `cls_mean@final` probe):

- `ar` scores 0.606–0.759 AUROC across the 7 tasks (mean 0.721); its untrained
  `random_init@ar` control scores 0.616–0.721 (mean 0.676).
- `jepa_ema`, `jepa_ema_nosig`, and `jepa_shared_sig` score 0.609–0.742,
  0.641–0.738, and 0.629–0.729 AUROC respectively (means 0.683, 0.690, 0.682);
  their shared `random_init@jepa_ema` control scores 0.602–0.726 (mean 0.682).
- The count-feature baselines score 0.572–0.789 (`gbm`, mean 0.726) and
  0.554–0.744 (`lr`, mean 0.695) across the same tasks.
- All four pretraining runs (`ar`, `jepa_ema`, `jepa_ema_nosig`,
  `jepa_shared_sig`) were trained to the same token budget so the comparison is
  matched.

## Quickstart

```bash
git clone https://github.com/Abelo9996/EHRJEPA.git
cd EHRJEPA

uv venv --python 3.12
uv pip install -e ".[dev]"

uv run pytest
uv run ruff check .
uv run pre-commit install   # optional
```

Verify the install:

```bash
uv run python -c "import ehrjepa; print(ehrjepa.__version__)"
```

Pretrain (needs a tensor cache built by `python -m ehrjepa.data.tokenize build`):

```bash
uv run python -m ehrjepa.train.pretrain --config configs/pretrain_small.yaml \
  --override run.steps=2000 optim.lr=3e-4
```

## Layout

```
src/ehrjepa/
  data/        MEDS ETL, vocabulary, tokenization, datasets and collators
  models/      event encoder (bidirectional or causal), latent predictor,
               EMA target encoder, next-code AR model
  objectives/  latent prediction loss, SIGReg anti-collapse regularizer,
               next-code cross-entropy
  train/       YAML config, training loop, checkpointing
  eval/        MEDS-DEV/ACES + EHRSHOT tasks, frozen-encoder probes, baselines
  utils/       seeding, device selection, logging, run directories
scripts/       ablate.py (ablation grids), throughput.py (tok/s measurement)
configs/       YAML run configs and ablation grids (configs/grids/)
docs/          experiment logs
tests/         test suite
legacy/        archived pre-rebuild visit-vector pipeline — do not build on it
```

## Evaluation protocol

```mermaid
flowchart LR
    t["anchor time t"] --> hist["history strictly before t<br/>(windows_at)"]
    hist --> feat["features / embeddings"]
    feat --> probe["LR / GBM / probe"]
    labels["labels from (t, t+365d]"] --> probe
    probe --> ci["bootstrap CIs"]
    rnd["random_init control<br/>(untrained architecture)"] -. same features/probe .-> feat
```

*Anchors are drawn strictly before any death event, so no event at or after
`MEDS_DEATH` ever enters a history window; `random_init` runs the same probe
through the same, but untrained, architecture as its control.*

## Roadmap

- Grid 2: content-only targets (`target.time_features: false`), following up
  on the JEPA models in grid 1.
- Scale up on an RTX 4070 with the full MIMIC-IV v3.1 extract.
- Evaluate against the EHRSHOT task suite and MEDS-DEV.
- Release: pretrained weights, benchmark numbers, and a citable preprint.

## Development

Ruff (line length 100) for lint and formatting, pytest for tests, pre-commit for
both on commit. CI runs ruff and pytest on Ubuntu with uv.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgments

The design draws on the JEPA line of work — I-JEPA, V-JEPA 2, and LeJEPA
(arXiv:2511.08544) — for the latent-prediction formulation and the SIGReg
anti-collapse objective, and on MEDS and MEDS-DEV/ACES for the data and task
standards. **No code from those projects is included here.** An earlier version
of this repository contained files derived from Meta's I-JEPA, which is
CC-BY-NC-licensed; all of it has been removed so this project can be released
under Apache-2.0.

## Citation

See [CITATION.cff](CITATION.cff) — a placeholder. There is no release or
preprint to cite yet.
