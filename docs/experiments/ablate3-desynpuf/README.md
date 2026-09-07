# 2026-09-07 -- ablate3-desynpuf: a frozen teacher, and code embeddings that start as words

`configs/grids/ablate3_desynpuf.yaml`, not yet run. Ten trained cells at 200M
nominal token slots each -- five configurations at two seeds -- on the same base
config, budget and full-held-out eval protocol as `ablate2-desynpuf`, so a row
here reads directly against a row there. Nothing in this README is a result; the
numbers land in `summary.md` and `summary.json`, appended by `scripts/ablate.py`
as each cell finishes.

## Four questions, stated before the runs started

1. **Can a frozen pretrained teacher replace EMA?** Every hybrid this repository
   has trained used either shared weights or an EMA copy as its target encoder.
   Both are the *same network*, which is what makes collapse a live risk and a
   momentum schedule necessary. `model.target_mode: frozen` (new here) loads the
   target embedding+encoder from `runs/scale1b-desynpuf/ar/final.pt` and never
   updates it: a fixed target function from step zero, no momentum to tune, and
   a teacher that has already seen a billion tokens. `hybrid_frozen_ar` against
   `ablate2-desynpuf`'s `hybrid_*` EMA rows is the comparison.
2. **Does initializing the student from AR help, separately?** A frozen AR
   teacher confounds two things: a better *target*, and the fact that AR weights
   exist at all. `hybrid_frozen_ar_init` adds `model.init_from` pointing at the
   same checkpoint, so the student starts as the teacher. The difference between
   the two rows is the value of the pretrained *student*; the difference between
   `hybrid_frozen_ar` and the EMA baseline is the value of the pretrained
   *teacher*. Neither row is allowed to stand in for the other.
3. **Do text-initialized code embeddings help either objective?**
   `hybrid_textinit` and `ar_textinit` replace the `N(0, 0.02)` code table with
   the projected sentence embeddings of each code's description
   (`ehrjepa.data.code_text`, below). If two diabetes codes start near each
   other, the model does not have to spend gradient discovering that. Running it
   under both objectives is what keeps "text init helps" from being a claim
   about one objective's quirk.
4. **What does freezing that table cost?** `hybrid_textinit_frozen` sets
   `model.freeze_code_embeddings: true`. The code table is 7,680,000 parameters
   -- **58% of the base-size hybrid's 13,208,336 trainable ones** (61% of the AR
   model's 12,665,104; it was 74% at the 4x192 pilot size the earlier note quotes).
   Freezing it leaves 5,528,336 trainable parameters and turns the code channel
   into a fixed lookup. The question is what that buys and what it costs, not
   whether it is free.

## The two-seed rule

Every cell runs at `run.seed: 1` and `run.seed: 2` -- two independent draws
through data order, mask sampling and initialization -- so a difference has to
clear "distinguishable from seed noise" before it is read as an effect. The grid
lists every seed-1 cell before any seed-2 cell, so an interrupted or `--only`
restricted run finishes one seed's worth first.

## Protocol

- Base config: `configs/pretrain_scale.yaml` (6L/256d encoder, `max_len: 512`,
  `batch_size: 64`).
- Budget: 200,000,000 nominal token slots per cell
  (`steps = ceil(budget_tokens / (batch_size x max_len x accum_steps))`).
- Source `desynpuf-s1`. Eval: full `held_out` split (`eval_subject_limit: null`),
  200 bootstrap resamples, `probe_features: auto` (`last@final` -- every cell
  here is causal), all tasks. `reuse_predictions` is unset, so `lr`/`gbm` are fit
  fresh, once, on the first cell whose eval command runs.
- Hybrid cells: `objective.kind: nextlatent`, `model.causal: true`,
  `objective.horizons: [1, 4, 16]`, `objective.lambda_recon: 0.1`,
  `objective.lambda_sigreg: 0.05` -- identical to `ablate2-desynpuf`'s hybrid
  arm, so only the knob under test differs.
- `control_runs: [hybrid_frozen_ar_s1]`. Every cell is the same 6L/256d causal
  encoder; the two knobs change what the weights are initialized to and what the
  loss pulls on, not the architecture the probe sees, so one untrained control
  covers the grid. `random_init` rebuilds the checkpoint's config through
  `EHRJEPAConfig.for_reload`, which drops `target_init`/`init_from`/`code_init`
  -- so the control is genuinely untrained rather than quietly carrying the
  teacher's weights or the text table.

### Two artifacts this grid needs and does not carry

- **`runs/scale1b-desynpuf/ar/final.pt`**, named by `model.target_init` (and by
  `model.init_from` in one pair). It lives on the GPU machine that trained
  `scale1b-desynpuf`. Unlike a `reuse_checkpoint` row -- which `scripts/ablate.py`
  plans around a missing file -- this is a *training* override, so `--dry-run`
  does not need it (nothing constructs a model) but the run does. A missing or
  wrong-shaped file fails at construction, naming the path and the mismatched
  field (`ehrjepa.models.pretrained.check_architecture`), not a hundred lines
  into `load_state_dict`.
- **`data/cache/desynpuf-s1/code_init_text_256.npy`**, derived from a public code
  table and a public sentence model, ~30 MB, gitignored. Rebuild it on the
  machine that will train:

  ```
  python -m ehrjepa.data.code_text build --cache data/cache/desynpuf-s1 --width 256
  ```

  and check the coverage it prints against `code_init_text_256.json` in this
  directory, which was produced by the same command on an Apple M4 and *is*
  committed. No cell names the path: `PretrainConfig.model_config` derives it
  from `data.cache_dir` and `model.dim`.

## The text initialization, and what it actually covers

Each of the 30,000 vocabulary entries is mapped to a sentence, the sentences are
embedded with `all-MiniLM-L6-v2` (384-d, CPU, about 20 s for the whole
vocabulary), the result is projected to 256-d by a PCA fitted on the vocabulary
itself, and the matrix is rescaled so its standard deviation is exactly the
`0.02` the random init would have used. `[PAD]` is zeroed, as it is under random
init. No training is involved, which is why PCA rather than a learned map.

Every description carries a **tier**. `exact` means the code string itself has a
published description; `ancestor` means a genuine parent concept in the same
coding system supplied the words (an ICD-9 category for a rolled-up code, an NDC
labeler, a CPT subsection); `fallback` means nothing but the coding system's name
and the literal code -- "drug product NDC 143530". The `ancestor` tier is not a
failure mode: the vocabulary's own rollup *creates* entries like `ICD9CM//250`
and `NDC//54868`, and a parent concept is exactly as specific as the id it names.

**21,311 of 30,000 entries (71.0%) and 83.8% of the 11,322,291 train events get a
real description. 12,793 entries (42.6%) and 44.9% of events are `exact`.**

| family | entries | train events | exact | ancestor | fallback |
|---|---|---|---|---|---|
| `NDC` | 23,199 | 4,441,842 (39.2%) | 8,196 | 6,883 | 8,120 |
| `HCPCS` | 1,612 | 3,011,377 (26.6%) | 350 | 1,261 | 1 |
| `ICD9CM` | 4,305 | 2,287,106 (20.2%) | 3,957 | 302 | 46 |
| `VISIT` | 1 | 621,428 (5.5%) | 1 | 0 | 0 |
| `SP_*` | 11 | 441,621 (3.9%) | 11 | 0 | 0 |
| `MEDS_BIRTH` | 1 | 92,992 (0.8%) | 1 | 0 | 0 |
| `RACE` | 4 | 92,992 (0.8%) | 4 | 0 | 0 |
| `SEX` | 2 | 92,992 (0.8%) | 2 | 0 | 0 |
| `ICD9PROC` | 400 | 76,996 (0.7%) | 153 | 71 | 176 |
| `ADMISSION` | 1 | 53,353 (0.5%) | 1 | 0 | 0 |
| `DISCHARGE` | 1 | 53,353 (0.5%) | 1 | 0 | 0 |
| `DRG` | 458 | 53,353 (0.5%) | 111 | 1 | 346 |
| `MEDS_DEATH` | 1 | 2,886 (0.0%) | 1 | 0 | 0 |
| `[SPECIAL]` | 4 | 0 | 4 | 0 | 0 |

Where the words come from, by event mass:

| source | entries | train events |
|---|---|---|
| `mimic-cpt-section` | 1,129 | 2,360,838 (20.9%) |
| `cms-icd9-dx` | 3,957 | 2,136,339 (18.9%) |
| `fallback` | 8,689 | 1,834,560 (16.2%) |
| `fda-ndc-labeler` | 6,882 | 1,598,637 (14.1%) |
| `handwritten` | 20 | 1,350,164 (11.9%) |
| `fda-ndc` | 8,196 | 849,566 (7.5%) |
| `mimic-d-hcpcs` | 350 | 601,653 (5.3%) |
| `desynpuf-codebook` | 11 | 441,621 (3.9%) |
| `icd9cms` | 292 | 64,126 (0.6%) |
| `cms-icd9-sg` | 153 | 31,010 (0.3%) |
| `hcpcs-group` | 111 | 24,546 (0.2%) |
| `mimic-drgcodes` | 111 | 13,021 (0.1%) |
| `cms-icd9-sg-prefix` | 70 | 11,186 (0.1%) |
| `mimic-d-hcpcs-prefix` | 20 | 4,513 (0.0%) |
| `cms-icd9-dx-prefix` | 9 | 511 (0.0%) |

Three things in that table are worth knowing before reading a result off this
grid:

- **NDC is the weak spot.** 8,120 entries and most of the 16.2% fallback mass are
  drug codes. DE-SynPUF's Part D events are from 2008-2010, and the FDA NDC
  directory lists currently-marketed products; adding the *excluded*
  (finished-marketing) file lifted exact coverage from 1,449 to 8,196 entries,
  and the remainder are simply not in either file. Those get their 5-digit
  labeler's name where the labeler is known (6,882 entries, 14.1% of events --
  "medication manufactured by Nelco Laboratories, Inc.", which says something
  real but nothing clinical) and `drug product NDC <code>` otherwise. An RxNorm
  or UMLS mapping would close most of this and needs a license and per-code API
  calls; it was not done.
- **The numeric HCPCS range is CPT, and its long titles are AMA-licensed.**
  MIMIC's `d_hcpcs` leaves `long_description` null for all 80,487 numeric codes
  and fills `short_description` with the CPT *subsection* ("cardiovascular
  system", "diagnostic imaging"). Those are read as `ancestor`, which is why
  `HCPCS` shows 350 exact against 1,261 ancestor while having almost no fallback:
  a common lab code like `HCPCS//80053` gets "pathology or laboratory test", not
  "comprehensive metabolic panel".
- **`ICD9PROC` is 44% fallback** because DE-SynPUF's synthetic generation puts
  some diagnosis-shaped codes in the procedure columns (`ICD9PROC//25000`); the
  procedure description file correctly does not carry them, and they are not
  cross-looked-up into the diagnosis table. It is 0.7% of events.

Sources are downloaded into `data/code_text` on first build: the CMS ICD-9-CM v32
master description zip, the FDA NDC directory (`ndctext.zip` and
`ndc_excluded.zip`), plus `d_hcpcs.csv.gz` and `drgcodes.csv.gz` discovered in a
local MIMIC-IV **demo** extract (public, uncredentialed) or dropped into
`data/code_text` by hand. The optional `icd9cms` package supplies the 3-digit
ICD-9 category titles the billable-leaf CMS list omits. Every source is optional:
a missing one costs coverage, which the JSON reports, not the build.
`python -m ehrjepa.data.code_text describe --cache <dir>` prints the descriptions
and the coverage report without embedding anything.

## Dry-run plan

```
$ python scripts/ablate.py configs/grids/ablate3_desynpuf.yaml --dry-run
grid ablate3-desynpuf: base configs/pretrain_scale.yaml, source desynpuf-s1
summary docs/experiments/ablate3-desynpuf/summary.json  log runs/ablate3-desynpuf/ablate.log
  RUN         hybrid_frozen_ar_s1   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/frozen lambda=0.05 p_future=--
  RUN         hybrid_frozen_ar_init_s1   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/frozen lambda=0.05 p_future=--
  RUN         hybrid_textinit_s1   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         hybrid_textinit_frozen_s1   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         ar_textinit_s1       6104 steps x 64x512 =  200,015,872 tokens  ar/-- lambda=-- p_future=--
  RUN         hybrid_frozen_ar_s2   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/frozen lambda=0.05 p_future=--
  RUN         hybrid_frozen_ar_init_s2   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/frozen lambda=0.05 p_future=--
  RUN         hybrid_textinit_s2   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         hybrid_textinit_frozen_s2   6104 steps x 64x512 =  200,015,872 tokens  nextlatent/ema lambda=0.05 p_future=--
  RUN         ar_textinit_s2       6104 steps x 64x512 =  200,015,872 tokens  ar/-- lambda=-- p_future=--
  total outstanding: 2,000,158,720 tokens
```

Ten cells, every one at the same 6,104 steps and 200,015,872 nominal tokens. The
`target` column reads `frozen` for the four frozen-teacher cells, which is
`scripts/ablate.py` reading `model.target_mode` straight out of the resolved
config -- the new mode needed no change there.

## Wall-time estimate

Assumptions carried over from `ablate2-desynpuf` (not measured on this grid's
hardware): 33k tok/s at 6L/256d, plus ~15 min eval per cell on the full held-out
split (11.7k subjects). The frozen-target cells run one extra encoder forward per
step exactly as the EMA cells do -- a frozen teacher costs the same forward and
skips the EMA update -- so the throughput assumption carries. The
`hybrid_textinit_frozen` cells have 58% fewer trainable parameters, which shrinks
optimizer state, not activations; at this shape memory is activation-dominated,
so no throughput change is assumed.

| cells | tokens/cell | tok/s | train time/cell | total train |
|---|---|---|---|---|
| 10 | 200,015,872 | 33,000 | ~101 min | ~16.8 h |

Training ~16.8 h, eval 10 x ~15 min = ~2.5 h. **Total ~19.3 h** of sequential
wall time on the RTX 4060 this grid was sized for, plus a couple of minutes to
rebuild the text-init table.

## Provenance

- Grid file: `configs/grids/ablate3_desynpuf.yaml`
- Base config: `configs/pretrain_scale.yaml`
- Frozen teacher / student init: `runs/scale1b-desynpuf/ar/final.pt`
  (see `docs/experiments/scale1b-desynpuf/README.md`)
- Text init: `ehrjepa.data.code_text`, coverage in `code_init_text_256.json`
- Hardware target: RTX 4060, 8 GB VRAM (see `docs/CUDA_SETUP.md`)
