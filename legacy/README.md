# legacy/ — pre-rebuild visit-vector pipeline

Everything in this directory belongs to the **first-generation EHRJEPA prototype**,
which represented each patient as a sequence of fixed-width *visit vectors*
(one dense feature row per hospital admission) rather than as a stream of
MEDS clinical-event tokens. It is kept only for reference while the new
MEDS-native pipeline is being written, and will be deleted once that pipeline
replaces it.

**Do not build on this code, and do not treat any number it produces as a result.**

Known problems, none of which will be fixed here:

- **Label leakage.** The visit-level feature builders aggregate over the whole
  admission, including events recorded at or after the point the outcome is
  determined, so mortality / length-of-stay / readmission labels are partly
  visible in the features. Every metric this pipeline ever printed is invalid.
- Cohort construction, splitting, and evaluation were not audited against
  MEDS-DEV / ACES task definitions, so the tasks are not comparable to
  published numbers.
- The benchmark suite contains "expected results" scaffolding and SOTA
  comparison tables that were never actually produced by a validated run.

## Contents

| Path | What it was |
| --- | --- |
| `preprocess_mimic_visits.py` | MIMIC-IV → visit-vector matrices |
| `split_mimic_data.py` | Patient-level train/val/test splitting |
| `run_visit_preprocessing.sh` | Driver for the two scripts above |
| `evaluate_visit_model.py` | Linear-probe / random-forest downstream eval |
| `run_benchmark.sh` | Driver for the benchmark suite |
| `benchmark/` | Feature extraction, baselines, ablations, artifact generation |
| `configs/` | YAML configs for the old training entry points |
| `requirements_ehr.txt` | Dependency pins for the old pipeline |

The training entry points these scripts were written against
(`train_ehr.py`, `main_ehr.py`, `src/`) were derived from Meta's I-JEPA
(CC-BY-NC 4.0) and have been removed from the repository, so nothing here
runs end to end any more.
