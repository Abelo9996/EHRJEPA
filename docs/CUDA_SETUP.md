# CUDA setup: RTX 4070, Linux

Everything in this repository so far — grids 1-4, the sanity runs, the
throughput table — was measured on an Apple M4 over MPS. This is the setup
procedure for a 12 GB RTX 4070 on Linux, written and checked (config parsing,
`--dry-run` grid planning) without CUDA hardware in the loop. Nothing here has
been trained.

`configs/pretrain_scale.yaml` (6-layer/256-wide encoder) and
`configs/grids/scale_desynpuf.yaml` (four cells at 500M tokens each) are the
config and grid this setup targets; see the comments in both files for the
scale-up reasoning and the expected-memory estimate. Nothing in the training
or eval code hardcodes MPS or CPU — `ehrjepa.utils.runtime.resolve_device`
picks CUDA first when `run.device: auto` (every shipped config's default), so
no config changes are needed to move a run from the M4 to this machine beyond
picking the scale-sized config.

## 1. Environment

Requires an NVIDIA driver already installed and a `nvidia-smi` that runs.

```bash
git clone https://github.com/Abelo9996/EHRJEPA.git
cd EHRJEPA
git checkout <branch>          # whatever branch has the CUDA-side work

uv venv --python 3.12
source .venv/bin/activate
```

Install the project first, then torch. PyPI's Linux `torch` wheels ship with
CUDA support built in as of recent 2.x releases, so plain `uv pip install -e
".[dev]"` (the same command the README's Quickstart uses on macOS) is usually
enough — verify with the check below before assuming otherwise. If it reports
no GPU, or the driver on this box needs a specific CUDA toolkit version,
reinstall torch from PyTorch's own index for that version instead:

```bash
uv pip install -e ".[dev,eval]"

# If the verify step below finds no GPU, pin to a specific CUDA build --
# match the version to `nvidia-smi`'s reported CUDA version, e.g. cu124:
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
uv run python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"
uv run pytest
uv run ruff check .
```

229+ tests should pass and ruff should be clean regardless of device — the
suite has no CUDA-only or MPS-only tests, it exercises CPU tensors and the
device-selection logic directly (`tests/test_eval.py::test_probe_device_prefers_cuda_over_mps`
and friends).

## 2. Build the DE-SynPUF sample-1 cache

DE-SynPUF is CMS's public synthetic Medicare claims release — no credentials
or DUA required, unlike MIMIC-IV or EHRSHOT. Download sample 1's beneficiary
summary, inpatient, outpatient and PDE CSVs from CMS's DE-SynPUF page and
place them under one raw directory, then lower to MEDS and build the tensor
cache with the same flags the existing `data/cache/desynpuf-s1` cache (vocab
30,000) used:

```bash
uv run python -m ehrjepa.data.etl desynpuf \
    --input /path/to/raw/desynpuf-sample1 \
    --output data/meds/desynpuf-s1

uv run python -m ehrjepa.data.tokenize build data/meds/desynpuf-s1 \
    --cache data/cache/desynpuf-s1 \
    --max-vocab 30000

uv run python -m ehrjepa.data.tokenize inspect data/cache/desynpuf-s1
```

`data/` is entirely gitignored (see `.gitignore`) — raw CSVs, the MEDS layout
and the tensor cache all stay local to this machine and are never committed,
the same as on the M4.

## 3. Launch a grid, detached

Dry-run first — this only parses the grid and the config, it trains nothing
and touches no `runs/` or `docs/` files:

```bash
uv run python scripts/ablate.py configs/grids/scale_desynpuf.yaml --dry-run
```

If that prints a plan for all four cells (`ar`, `hybrid`, `recon_only`,
`jepa_ema`) with no errors, launch for real, detached from the terminal so it
survives a disconnect:

```bash
nohup uv run python scripts/ablate.py configs/grids/scale_desynpuf.yaml \
    > runs/scale-desynpuf/ablate.log 2>&1 &
disown

tail -f runs/scale-desynpuf/ablate.log
```

The runner is resumable at cell granularity (same as every grid in
`docs/experiments/`): a cell already recorded in
`docs/experiments/scale-desynpuf/summary.json` is skipped, so relaunching the
same command after a disconnect or a reboot picks up where it stopped.
`--only ar,hybrid` restricts a run to named cells; `--force` reruns a cell
already summarised.

Before committing to the full 500M-token budget on new hardware, measure
actual throughput and peak memory first — the config's expected-memory note
is an estimate, not a CUDA measurement:

```bash
uv run python scripts/throughput.py --only pilot/jepa/bf16,pilot/ar/bf16
# then, once configs/pretrain_scale.yaml is confirmed to fit:
uv run python -m ehrjepa.train.pretrain --config configs/pretrain_scale.yaml \
    --override run.steps=50 run.ckpt_every=0 run.tensorboard=false \
                run.out_dir=runs/scale-throughput-check
```

If `configs/pretrain_scale.yaml` does not fit in 12 GB at `batch_size: 64`,
fall back to `optim.accum_steps: 2` with `run.batch_size: 32` (same effective
batch, half the activation memory) via `--override` on the grid's `base`
config or by editing the file directly.

## 4. Sync results back

`runs/` is gitignored everywhere (checkpoints, `metrics.csv`, TensorBoard
logs, `ablate.log`) — it never leaves this machine and is not part of what
gets synced back. What *is* tracked, via the `!docs/experiments/**` exception
to the repo's blanket `*.parquet`/`*.csv` ignore, is everything
`scripts/ablate.py` writes under `docs/experiments/<grid-name>/`:
`summary.md`, `summary.json`, `baselines.json`, and each cell's
`eval/<run>/{results.json,results.md,predictions.parquet}`.

```bash
git status docs/experiments/scale-desynpuf/
git add docs/experiments/scale-desynpuf/
git commit -m "docs: scale-desynpuf grid results"
git push
```

Write a `docs/experiments/scale-desynpuf/README.md` protocol note the same
way grids 1-4 did (state the question before reading the numbers) if this
grid's results are meant to be read by anyone other than the person who ran
it. Pull the commit back down on the machine that maintains
`docs/experiments/PILOT_RESULTS.md` to fold the new rows in — that file is
hand-written, not regenerated, so adding a fifth grid's rows to it is a
manual edit, same as this one was.
