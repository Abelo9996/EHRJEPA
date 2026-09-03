# The canonical MEDS layout

Every source we ingest -- MIMIC-IV, CMS DE-SynPUF, Synthea -- is lowered into one
identical on-disk shape, so the tokenizer and model in phase 3 never learn that
more than one source exists. This document is the contract.

```
<out>/
  data/
    train/0.parquet, 1.parquet, ...
    tuning/0.parquet, ...
    held_out/0.parquet, ...
  metadata/
    codes.parquet
    subject_splits.parquet
    subject_id_map.parquet      # only when the source uses opaque ids
    dataset.json
```

## Data shards

Each shard holds the MEDS 0.4 core columns, in this order:

| column | dtype | notes |
|---|---|---|
| `subject_id` | `Int64` | never null |
| `time` | `Datetime("us")` | null for static measurements |
| `code` | `String` | never null, never empty |
| `numeric_value` | `Float32` | null when the event carries no number |
| `text_value` | `String` | null when the event carries no text |

Rows are sorted by `(subject_id, time, code)` with **null times first**, so a
subject's static facts precede their timeline. Shards are *subject-disjoint*: all
of a subject's events live in exactly one file, in exactly one split. Every shard
is passed through `meds.DataSchema.align()` and `meds.DataSchema.validate()`
before it is written, as are `codes.parquet` (`meds.CodeMetadataSchema`) and
`subject_splits.parquet` (`meds.SubjectSplitSchema`).

Source-specific extension columns are **dropped**, not carried. `meds_etl`'s MIMIC
output has ~20 of them (`hadm_id`, `caregiver_id`, `unit`, ...); keeping them would
mean the model sees a different column set per source, which defeats the point.
They remain in `meds_etl`'s own output if a later phase wants them.

## Metadata

* `codes.parquet` -- `code`, `description`, `parent_codes`. Contains **exactly** the
  vocabulary observed in the data; descriptions and ontology parents are filled in
  from the source's own code metadata where it has any (MIMIC-IV, via `meds_etl`'s
  OMOP concept maps) and are null otherwise.
* `subject_splits.parquet` -- `subject_id`, `split`. One row per subject, including
  subjects that ended up with zero events.
* `subject_id_map.parquet` -- `source_id`, `subject_id`. Written for DE-SynPUF and
  Synthea, whose native ids are strings.
* `dataset.json` -- `dataset_name`, `etl_name`, `etl_version`, `meds_version`,
  `source`, `created`.

## Splits and shards are deterministic

`split` is a pure function of `subject_id`:
`blake2b("split:<seed>:<subject_id>")` truncated to 64 bits, taken `% 10000`, and
cut at 8000 / 9000 -- an 80/10/10 train/tuning/held_out draw. The seed is fixed at
`SPLIT_SEED = 20240917`; changing it reshuffles every dataset ever produced, so it
is pinned by a test. `hashlib` is used rather than polars' `Expr.hash` because
polars makes no cross-version stability guarantee for its hash.

Shard index uses the same construction with a different prefix, modulo the number
of shards in the split. `--shard-size N` sets the number of **subjects** per shard
(default 10,000), so shard count is `ceil(subjects_in_split / N)`.

Subjects whose native id is a string get an `Int64` via
`blake2b("subject:<id>")` truncated to 63 bits (always non-negative); each ETL
asserts the mapping is collision-free.

## How a shard gets written

`write_canonical` is two-phase, so an extract never has to fit in memory:

1. Every source table -- a `polars.LazyFrame` -- is joined to the split/shard
   assignment and streamed through `sink_parquet(pl.PartitionBy(...))` into a
   staging tree keyed by `(split, shard)`.
2. Each staging partition is read back (one shard of subjects, bounded by
   `--shard-size`), sorted, validated and written as the final file.

## Per-source code conventions

Codes are **not** harmonised across sources -- there is no shared ontology mapping
step yet, and inventing one would be guesswork. What is shared is the *shape*:
`NAMESPACE//value`, with further `//`-separated qualifiers.

| source | conventions |
|---|---|
| MIMIC-IV | whatever `meds_etl.mimic` emits: `MIMIC_IV_ITEM/<itemid>`, `MIMIC_IV_LAB/<itemid>`, `ICD9CM/<code>`, `ICD10CM/<code>`, `NDC/<ndc>`, `MIMIC_IV_Admission/<type>`, ... (single slash, `MIMIC_IV_` prefixes) |
| DE-SynPUF | `MEDS_BIRTH`, `MEDS_DEATH`, `SEX//{M,F}`, `RACE//{WHITE,BLACK,OTHER,HISPANIC}`, `SP_<FLAG>//1`, `ADMISSION//INPATIENT`, `DISCHARGE//INPATIENT` (numeric = length of stay in days), `VISIT//OUTPATIENT`, `DRG//<code>`, `ICD9CM//<code>`, `ICD9PROC//<code>`, `HCPCS//<code>`, `NDC//<code>` (numeric = days supplied) |
| Synthea | `MEDS_BIRTH`, `MEDS_DEATH`, `SEX//<gender>`, `RACE//<race>`, `ETHNICITY//<ethnicity>`, `ENCOUNTER//<class>`, `END//ENCOUNTER`, `SNOMED//<code>`, `CONDITION_STOP//<code>`, `SNOMED_PROC//<code>`, `RXNORM//<code>` (numeric = dispenses), `LOINC//<code>//<unit>` |

Observation units are folded into the LOINC code the way MEDS_transforms does it,
so a code's numeric values are always on one scale; unit-less observations get
`//UNK`.

# The tensor cache

MEDS is the interchange format; it is not what the model reads. `tokenize.py`
fits a vocabulary and a value quantizer on the **train split only**, and
`cache.py` uses them to lower every split into flat arrays. One MEDS event
becomes one token.

```
<cache>/
  vocab.parquet        code_id, code, train_count, direct_count, is_ancestor, is_special
  vocab.json           the fit summary (sizes, UNK/ancestor rates, top codes)
  quantizer.parquet    one row per vocabulary id: scope, source_key, use_log1p,
                       mean, std, edge_0..edge_8
  meta.json            vocab size, feature dtypes, source dataset.json, tokenizer
                       version, per-split fit statistics, cache size, build time
  train/               code_id.npy value_bin.npy value_z.npy age.npy
                       log_delta.npy time_min.npy subjects.parquet
  tuning/ held_out/    same
```

The feature arrays are one flat vector per split, concatenated across subjects in
shard order; `subjects.parquet` (`subject_id`, `offset`, `length`, `split`,
`has_birth`) says where each subject's slice lives. They are written through
`numpy.lib.format.open_memmap` one input shard at a time, so the writer's memory
is bounded by the largest shard rather than by the split, and read back with
`mmap_mode="r"`, so a training run pages in only the windows it samples. One file
per subject would be 116k files for DE-SynPUF; a ragged parquet could not be
sliced without decoding.

## Features, per event

| feature | dtype | meaning |
|---|---|---|
| `code_id` | int32 | vocabulary id, after hierarchical fallback |
| `value_bin` | int8 | 0 = no numeric value, 1..10 = decile of the code's train values |
| `value_z` | float32 | z-score of the (optionally `log1p`-ed) value, clipped to ±5, 0 when absent |
| `age` | float32 | years since `MEDS_BIRTH`, clipped to [0, 120] |
| `log_delta` | float32 | `log1p` of hours since the subject's previous event; 0 for the first event and for ties |
| `time_min` | int64 | raw event time, minutes since epoch, for downstream label alignment |

Subjects with no `MEDS_BIRTH` are anchored at their first event (`age` 0 there)
and flagged `has_birth = false`, so a model can learn not to trust their ages.
Events with identical timestamps get `log_delta = 0` and keep their parquet
order.

## Vocabulary: rollup before UNK

Ids 0-3 are `[PAD]`, `[UNK]`, `[CLS]`, `[MASK]`. A code with at least
`--min-count` (default 5) train occurrences gets its own id. A rarer code is not
sent straight to `[UNK]`: codes are hierarchical strings, so it is first
truncated towards an ancestor that is frequent enough to be worth an embedding.

| prefix | chain |
|---|---|
| `ICD9*`, `ICD10*` | strip one character at a time: `ICD9CM/250.01` → `250.0` → `250` → `25` → `2` (a trailing `.` goes with the character that exposed it) |
| `NDC//<digits>` | 11-digit code → first 9 (product) → first 5 (labeler) |
| `HCPCS//<code>` | first 3 characters |
| anything else | the bare `PREFIX` (e.g. `LOINC`) |

then `[UNK]`. Admission is one bottom-up sweep over the forest those chains
imply: a node holds its own count plus the mass of its *rejected* children, and
keeps that mass if it clears `min_count` or hands it to its parent if it does
not. Mass is therefore never counted at two levels at once, and an ancestor is
only admitted if the codes that actually need it are frequent enough together.
Entries admitted this way carry `is_ancestor = true`.

## Value quantizer

A raw float is meaningless without its code, so quantization is per code. Every
vocabulary id with at least `--min-value-obs` (default 20) train observations
gets its own nine interior decile edges plus a mean and standard deviation; ids
below that share their `PREFIX`'s fit, and failing that a global one. `scope` and
`source_key` in `quantizer.parquet` record which. Non-negative codes with
`|skew| > 2` are `log1p`-ed before the moments are taken. A value sitting exactly
on an edge falls in the *lower* bin, which keeps a code whose values are mostly
one constant (dose counts, "days supplied") from smearing across bins.

## Dataset

`EventSequenceDataset(cache_dir, split, max_len=512, min_len=16, sampling=...)`
is one item per **subject**. `sampling="random_window"` draws a fresh contiguous
window each time an item is requested, so an epoch sees a different crop of every
patient; `sampling="full"` deterministically returns the most recent `max_len`
events. Subjects with fewer than `min_len` events are dropped and counted in
`n_dropped`. `collate_events` right-pads to the batch maximum (`code_id` with
`PAD`, everything else with 0) and returns `attention_mask`. `windows_at(subject,
end_time)` returns the last `max_len` events *strictly before* `end_time`, which
is how phase-5 labels will get their context without leaking the labelled event.
No device logic lives in the dataset; tensors come out on CPU.

## CLI

```
python -m ehrjepa.data.etl {mimic,desynpuf,synthea} --input <dir> --output <dir> [--shard-size N] [--work-dir D]
python -m ehrjepa.data.stats <meds_dir> [--top-k K] [--json]
python -m ehrjepa.data.tokenize fit <meds_dir> --out <cache_dir> [--min-count 5] [--min-value-obs 20]
python -m ehrjepa.data.tokenize build <meds_dir> --cache <cache_dir> [--min-count 5]
python -m ehrjepa.data.tokenize inspect <cache_dir> [--events 12]
```

`--work-dir` only applies to `mimic`; it is where `meds_etl` writes its
intermediate output, and a finished run there is reused instead of recomputed.
`build` is `fit` plus tensorization of every split. `inspect` prints `meta.json`
and one decoded window per split.
