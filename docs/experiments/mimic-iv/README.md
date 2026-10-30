# MIMIC-IV v3.1 — acquisition & characterization

MIMIC-IV v3.1 (`hosp` + `icu`) was pulled from PhysioNet's BigQuery mirror, sorted (clustered by
`subject_id`) to native compression, verified **byte-identical (SHA-256)** on transfer to the GPU box,
and characterized directly from the v3.1 `.csv.gz` files. Numbers below are exact row counts
(`gzip -dc | wc -l`, minus header); the full machine-readable record is [`stats.json`](stats.json).

This is **stage B**: the first run on *real* hospital data with labs and vitals — the modality
DE-SynPUF (all stages so far) entirely lacks. See [`docs/figures/mimic_iv_scale.png`](../../figures/mimic_iv_scale.png).

## Cohort

| quantity | count |
|---|---|
| patients | 364,627 |
| hospital admissions | 546,028 |
| ICU stays | 94,458 |
| **total events** (event tables) | **875,390,184** |

## Largest tables (rows)

| table | rows |
|---|---|
| `icu/chartevents` | 432,997,491 |
| `hosp/labevents` | 158,374,764 |
| `hosp/emar_detail` | 87,371,064 |
| `hosp/poe` | 52,212,109 |
| `hosp/emar` | 42,808,593 |
| `hosp/prescriptions` | 20,292,611 |
| `hosp/pharmacy` | 17,847,567 |
| `icu/ingredientevents` | 14,253,480 |
| `icu/inputevents` | 10,953,713 |
| `icu/datetimeevents` | 9,979,761 |

## Vocabulary (dimension tables)

| dictionary | size |
|---|---|
| ICD diagnosis codes (`d_icd_diagnoses`) | 112,107 |
| HCPCS codes (`d_hcpcs`) | 89,208 |
| ICD procedure codes (`d_icd_procedures`) | 86,423 |
| chart items (`d_items`) | 4,095 |
| lab items (`d_labitems`) | 1,650 |

## Contrast with DE-SynPUF

- DE-SynPUF train split ≈ **11.3M events, 0 labs/vitals**.
- MIMIC-IV v3.1 ≈ **875M events**, of which **158M lab events + 433M chart events** exercise the
  value-quantizer / value-embedding path that DE-SynPUF barely touched.

## Status & next step

- **Done:** acquired, verified, characterized. Data at `~/EHRJEPA/data/mimic-iv-3.1/{hosp,icu}/` on the GPU box.
- **Next (stage B):** `python -m ehrjepa.data.etl mimic --input <hosp/icu dir> --output <MEDS dir>`
  (wraps `meds_etl.mimic`; note `meds_etl` must be installed in the env first), then build the memmap
  cache and pretrain the hybrid default with the ≥2-seed rule.

*Method note on data equivalence:* the BigQuery pull is record-identical to PhysioNet (row counts match
source; `admissions` was byte-identical uncompressed) but rows are ordered by `subject_id`, not
PhysioNet's exact order — so file checksums differ from PhysioNet's, with no effect on the ETL.
