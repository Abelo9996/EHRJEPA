# Few-shot evaluation -- DE-SynPUF sample 1

Held-out AUROC at few-shot training sizes k=32, k=128, and the full training split ("all"), on the same 3,000-subject held-out subset (`eval_subject_seed: 0`), same seeded anchors, same 200 bootstrap resamples used by grids 1-5 (`docs/experiments/PILOT_RESULTS.md`). Full protocol and what did and did not run: [`README.md`](README.md). Raw per-model output: [`results.json`](results.json), [`eval_report.md`](eval_report.md).

Each `k` draws `k` positives and `k` negatives from `train` (or all of `train` when a class has fewer than `k`), fits a probe, and scores it on the shared `held_out` split -- 5 draws per (task, k, checkpoint) (`ehrjepa.eval.probe.FEW_SHOT_SEEDS`). `ar` and `hybrid` are further averaged over their 3 training seeds (grid 1 `ar` + grid 5 `ar_s1`/`ar_s2`; grid 4 `nextlatent_h1416_recon` + grid 5 `hybrid_s1`/`hybrid_s2`), so each `ar`/`hybrid` cell below is a mean of 3 numbers, each itself a mean of 5. `lr` has one model, so its `±` is the std over the 5 few-shot seeds directly. `gbm` has no few-shot fits: `ehrjepa.eval.run.evaluate_task` only fits few-shot subsamples for `spec.kind in ("lr", "probe")`, so `gbm`'s k=32/k=128 cells are `--` and its k=all cell is the standard full-data fit (single point, no seed variation to report).

## `inpatient_365d`

| family | k=32 | k=128 | k=all |
|---|---|---|---|
| `lr` | 0.5912 ± 0.0000 | 0.6575 ± 0.0000 | 0.7077 ± 0.0000 |
| `gbm` | -- | -- | 0.7440 |
| `ar` | 0.6707 ± 0.0015 | 0.7058 ± 0.0045 | 0.7417 ± 0.0027 |
| `hybrid` | 0.6591 ± 0.0012 | 0.6976 ± 0.0023 | 0.7445 ± 0.0025 |

## `mortality_365d`

| family | k=32 | k=128 | k=all |
|---|---|---|---|
| `lr` | 0.5265 ± 0.0000 | 0.5332 ± 0.0000 | 0.5537 ± 0.0000 |
| `gbm` | -- | -- | 0.5723 |
| `ar` | 0.5510 ± 0.0019 | 0.5668 ± 0.0071 | 0.6034 ± 0.0061 |
| `hybrid` | 0.5636 ± 0.0028 | 0.5942 ± 0.0037 | 0.6219 ± 0.0109 |

## `new_dx_365d/ckd`

| family | k=32 | k=128 | k=all |
|---|---|---|---|
| `lr` | 0.6320 ± 0.0000 | 0.6701 ± 0.0000 | 0.7361 ± 0.0000 |
| `gbm` | -- | -- | 0.7654 |
| `ar` | 0.6739 ± 0.0017 | 0.7134 ± 0.0048 | 0.7480 ± 0.0044 |
| `hybrid` | 0.6758 ± 0.0020 | 0.7035 ± 0.0036 | 0.7525 ± 0.0018 |

## `new_dx_365d/copd`

| family | k=32 | k=128 | k=all |
|---|---|---|---|
| `lr` | 0.6281 ± 0.0000 | 0.6666 ± 0.0000 | 0.7258 ± 0.0000 |
| `gbm` | -- | -- | 0.7674 |
| `ar` | 0.6840 ± 0.0042 | 0.7176 ± 0.0011 | 0.7604 ± 0.0009 |
| `hybrid` | 0.6696 ± 0.0031 | 0.7036 ± 0.0025 | 0.7515 ± 0.0023 |

## `new_dx_365d/diabetes`

| family | k=32 | k=128 | k=all |
|---|---|---|---|
| `lr` | 0.6838 ± 0.0000 | 0.7031 ± 0.0000 | 0.7397 ± 0.0000 |
| `gbm` | -- | -- | 0.7721 |
| `ar` | 0.6940 ± 0.0051 | 0.7356 ± 0.0018 | 0.7618 ± 0.0032 |
| `hybrid` | 0.6967 ± 0.0021 | 0.7324 ± 0.0040 | 0.7643 ± 0.0033 |

## `new_dx_365d/heart_failure`

| family | k=32 | k=128 | k=all |
|---|---|---|---|
| `lr` | 0.6483 ± 0.0000 | 0.6663 ± 0.0000 | 0.7438 ± 0.0000 |
| `gbm` | -- | -- | 0.7893 |
| `ar` | 0.6934 ± 0.0056 | 0.7227 ± 0.0050 | 0.7660 ± 0.0069 |
| `hybrid` | 0.6876 ± 0.0017 | 0.7042 ± 0.0026 | 0.7636 ± 0.0041 |

## `readmission_30d`

| family | k=32 | k=128 | k=all |
|---|---|---|---|
| `lr` | 0.5680 ± 0.0000 | 0.5880 ± 0.0000 | 0.6578 ± 0.0000 |
| `gbm` | -- | -- | 0.6724 |
| `ar` | 0.5766 ± 0.0078 | 0.5994 ± 0.0062 | 0.6824 ± 0.0064 |
| `hybrid` | 0.6100 ± 0.0037 | 0.6487 ± 0.0009 | 0.6969 ± 0.0031 |

## Figure

![Held-out AUROC vs. few-shot training size, one panel per task, one line per model family](../figures/fewshot_desynpuf.png)

Produced by [`scripts/plot_fewshot.py`](../../scripts/plot_fewshot.py):

```bash
python scripts/plot_fewshot.py
```

## Runtime

`422.6` s end to end (all 8 models x 7 tasks, including few-shot fits), commit `d64d7ab`, on MPS. Count matrices and every checkpoint's embeddings were already cached under `data/eval_cache/desynpuf-s1/` from grids 1-5, so this run re-embedded nothing -- the time is fitting `lr`/`gbm`/probes and their few-shot subsamples.
