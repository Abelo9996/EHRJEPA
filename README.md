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

## Status

**Under active rebuild. No results yet.**

There are no pretrained weights, no benchmark numbers, and no claims of
performance in this repository. The previous visit-vector prototype has been
archived under [`legacy/`](legacy/README.md); it has known label leakage and
none of its numbers are valid.

What exists now: the MEDS ETL and tensor cache (`data/`), the three networks
(`models/`), the SIGReg + latent-prediction objective (`objectives/`), a
resumable pretraining loop (`train/`), and a downstream evaluation harness
(`eval/`) with ACES-defined tasks, count-feature baselines, frozen-encoder
probes and bootstrap intervals. The runs recorded under `docs/experiments/` are
short MPS sanity runs and one evaluation of the checkpoints they produced;
those checkpoints are ~900 steps on one demo-scale dataset, so the evaluation
measures the harness on real cohorts, not a trained model.

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
  models/      event encoder, latent predictor, EMA target encoder
  objectives/  latent prediction loss, SIGReg anti-collapse regularizer
  train/       YAML config, training loop, checkpointing
  eval/        MEDS-DEV/ACES + EHRSHOT tasks, frozen-encoder probes, baselines
  utils/       seeding, device selection, logging, run directories
scripts/       CLI entry points (placeholder)
configs/       YAML run configs: pretrain_small.yaml, pretrain_debug.yaml
docs/          experiment logs
tests/         test suite
legacy/        archived pre-rebuild visit-vector pipeline — do not build on it
```

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
