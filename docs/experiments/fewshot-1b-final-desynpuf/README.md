# Few-shot at 1B, full held-out split — DE-SynPUF sample 1

Few-shot AUROC (`k=32`, `k=128`, `k=512`, and the full training split, "all")
for the six 1B-token checkpoints of the paper's "final default at 1B tokens"
section (`docs/paper/main.tex`, `\label{sec:final1b}`) — `ckpt:hybrid_final_s0/s1/s2` (the repository
default) and `ckpt:ar`, `ckpt:ar_s1`, `ckpt:ar_s2` — plus `lr` and `gbm`, on
the **full held-out split** (11,708 subjects; no `--eval-subject-limit`),
not the 3,000-subject subset used by the pilot-era few-shot run in
[`docs/experiments/2026-09-04-fewshot-desynpuf/`](../2026-09-04-fewshot-desynpuf/).
Raw per-model output: [`results.json`](results.json), full metric tables:
[`results.md`](results.md).

## What ran

```bash
python -m ehrjepa.eval.run \
  --source desynpuf-s1 \
  --tasks all \
  --models lr,gbm,\
ckpt:runs/scale1b-final-desynpuf/hybrid_final_s0/final.pt,\
ckpt:runs/scale1b-final-desynpuf/hybrid_final_s1/final.pt,\
ckpt:runs/scale1b-final-desynpuf/hybrid_final_s2/final.pt,\
ckpt:runs/scale1b-desynpuf/ar/final.pt,\
ckpt:runs/scale1b-seeds-desynpuf/ar_s1/final.pt,\
ckpt:runs/scale1b-seeds-desynpuf/ar_s2/final.pt \
  --out docs/experiments/fewshot-1b-final-desynpuf/ \
  --bootstrap 200 --seed 0 --probe-features auto --device mps
```

`--eval-split` defaults to `held_out` and no `--eval-subject-limit` is passed,
so scoring runs against all 11,708 held-out subjects rather than a seeded
subset. `--probe-features auto` resolves to `last@final` for every checkpoint
(all six are causal). Anchor seed `20260903`, `n_boot=200`, `seed=0`, commit
`44fcab4`, `created` 2026-09-10T15:04:30Z, 1440.9s end to end.

Few-shot mechanics are unchanged from the pilot-era run
(`docs/experiments/2026-09-04-fewshot-desynpuf/README.md`): `ehrjepa.eval
.probe.FEW_SHOT_K = (32, 128, 512, None)`, `FEW_SHOT_SEEDS = (0, 1, 2, 3, 4)`
(5 draws per `(task, k, checkpoint)`), and only `spec.kind in ("lr", "probe")`
gets few-shot fits — `gbm` has no `k=32/128/512` cells, only its ordinary
full-data fit at `k=all`.

## How family numbers are built

`ar` and `hybrid_final` are each three independently trained 1B checkpoints
(seeds 0, 1, 2). For each `(task, k)`: read each checkpoint's own few-shot
row (already a mean over 5 few-shot-sample seeds, `auroc_mean` in
`results.json`), then take the mean and population std of those 3 numbers.
The reported `±` for `ar`/`hybrid_final` below is therefore training-seed
spread, not few-shot-sampling spread. `lr` has one model, so its `±` is the
std over the 5 few-shot seeds directly. `gbm` is a single deterministic fit
at `k=all` — no `±` to report.

## AUROC by task

| task | family | k=32 | k=128 | k=512 | k=all |
|---|---|---|---|---|---|
| `inpatient_365d` | `lr` | 0.5979 ± 0.0390 | 0.6613 ± 0.0061 | 0.6767 ± 0.0020 | 0.7124 |
| `inpatient_365d` | `ar` | 0.6128 ± 0.0040 | 0.6772 ± 0.0039 | 0.7086 ± 0.0034 | 0.7420 ± 0.0015 |
| `inpatient_365d` | `hybrid_final` | 0.6404 ± 0.0048 | 0.6959 ± 0.0018 | 0.7276 ± 0.0020 | 0.7573 ± 0.0016 |
| `inpatient_365d` | `gbm` | -- | -- | -- | 0.7455 |
| `mortality_365d` | `lr` | 0.5295 ± 0.0313 | 0.5078 ± 0.0381 | 0.5520 ± 0.0181 | 0.5659 |
| `mortality_365d` | `ar` | 0.5369 ± 0.0046 | 0.5541 ± 0.0036 | 0.5793 ± 0.0026 | 0.6053 ± 0.0046 |
| `mortality_365d` | `hybrid_final` | 0.5414 ± 0.0031 | 0.5650 ± 0.0032 | 0.5728 ± 0.0007 | 0.6011 ± 0.0012 |
| `mortality_365d` | `gbm` | -- | -- | -- | 0.5564 |
| `new_dx_365d/ckd` | `lr` | 0.6402 ± 0.0569 | 0.6795 ± 0.0024 | 0.6975 ± 0.0042 | 0.7390 |
| `new_dx_365d/ckd` | `ar` | 0.6364 ± 0.0027 | 0.7056 ± 0.0045 | 0.7305 ± 0.0005 | 0.7674 ± 0.0019 |
| `new_dx_365d/ckd` | `hybrid_final` | 0.6607 ± 0.0026 | 0.7307 ± 0.0021 | 0.7569 ± 0.0012 | 0.7834 ± 0.0010 |
| `new_dx_365d/ckd` | `gbm` | -- | -- | -- | 0.7713 |
| `new_dx_365d/copd` | `lr` | 0.6294 ± 0.0691 | 0.6722 ± 0.0051 | 0.6870 ± 0.0062 | 0.7326 |
| `new_dx_365d/copd` | `ar` | 0.6391 ± 0.0068 | 0.6976 ± 0.0047 | 0.7319 ± 0.0029 | 0.7603 ± 0.0023 |
| `new_dx_365d/copd` | `hybrid_final` | 0.6685 ± 0.0030 | 0.7192 ± 0.0010 | 0.7496 ± 0.0019 | 0.7721 ± 0.0002 |
| `new_dx_365d/copd` | `gbm` | -- | -- | -- | 0.7677 |
| `new_dx_365d/diabetes` | `lr` | 0.6794 ± 0.0183 | 0.7007 ± 0.0018 | 0.7076 ± 0.0011 | 0.7368 |
| `new_dx_365d/diabetes` | `ar` | 0.6503 ± 0.0046 | 0.7049 ± 0.0022 | 0.7364 ± 0.0008 | 0.7625 ± 0.0014 |
| `new_dx_365d/diabetes` | `hybrid_final` | 0.6796 ± 0.0018 | 0.7289 ± 0.0017 | 0.7557 ± 0.0019 | 0.7778 ± 0.0012 |
| `new_dx_365d/diabetes` | `gbm` | -- | -- | -- | 0.7709 |
| `new_dx_365d/heart_failure` | `lr` | 0.6588 ± 0.0213 | 0.6721 ± 0.0066 | 0.6949 ± 0.0046 | 0.7407 |
| `new_dx_365d/heart_failure` | `ar` | 0.6488 ± 0.0043 | 0.6957 ± 0.0039 | 0.7395 ± 0.0022 | 0.7715 ± 0.0021 |
| `new_dx_365d/heart_failure` | `hybrid_final` | 0.6816 ± 0.0021 | 0.7232 ± 0.0017 | 0.7713 ± 0.0010 | 0.7982 ± 0.0011 |
| `new_dx_365d/heart_failure` | `gbm` | -- | -- | -- | 0.7840 |
| `readmission_30d` | `lr` | 0.5680 ± 0.0284 | 0.5799 ± 0.0222 | 0.6204 ± 0.0047 | 0.6529 |
| `readmission_30d` | `ar` | 0.5675 ± 0.0147 | 0.5827 ± 0.0093 | 0.6293 ± 0.0122 | 0.6580 ± 0.0101 |
| `readmission_30d` | `hybrid_final` | 0.6160 ± 0.0064 | 0.6477 ± 0.0035 | 0.6748 ± 0.0022 | 0.6881 ± 0.0052 |
| `readmission_30d` | `gbm` | -- | -- | -- | 0.6670 |

`lr`'s `±` at `k=all` is 0 by construction (a single deterministic fit on
the full split — no few-shot-seed axis).

## Family mean over the six non-mortality tasks

| family | k=32 | k=128 | k=512 | k=all |
|---|---|---|---|---|
| `lr` | 0.6289 | 0.6610 | 0.6807 | 0.7191 |
| `ar` | 0.6258 | 0.6773 | 0.7127 | 0.7436 |
| `hybrid_final` | 0.6578 | 0.7076 | 0.7393 | 0.7628 |
| `gbm` | -- | -- | -- | 0.7511 |

`k=all` matches the paper's final-1B baselines table mean-of-6 exactly (`lr`
0.7191, `gbm` 0.7511, `ar` 0.7436, `hybrid_final` 0.7628) — this run's
`k=all` cell is that section's evaluation.

## Figure

`scripts/plot_fewshot.py --results docs/experiments/fewshot-1b-final-desynpuf/results.json`
(the default as of this run) renders one panel per task, x = training size
(log scale: `k=32`, `k=128`, `k=512`, the task's full-train-split size), y =
held-out AUROC, one line per family (`lr`, `ar`, `hybrid_final`) with a ±1
training-seed-std band, `gbm` as an unconnected marker at its single
full-data point. The pilot-era 3,000-subject run is still selectable with
`--results docs/experiments/2026-09-04-fewshot-desynpuf/results.json` (its
model names differ: `ckpt:nextlatent_h1416_recon`/`hybrid_s1`/`hybrid_s2`
rather than `hybrid_final_s0/s1/s2`).

```bash
python scripts/plot_fewshot.py
```

Output: [`../figures/fewshot_desynpuf.png`](../figures/fewshot_desynpuf.png).
