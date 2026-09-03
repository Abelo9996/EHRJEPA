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

## CLI

```
python -m ehrjepa.data.etl {mimic,desynpuf,synthea} --input <dir> --output <dir> [--shard-size N] [--work-dir D]
python -m ehrjepa.data.stats <meds_dir> [--top-k K] [--json]
```

`--work-dir` only applies to `mimic`; it is where `meds_etl` writes its
intermediate output, and a finished run there is reused instead of recomputed.
