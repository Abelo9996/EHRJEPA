# Scale results — DE-SynPUF sample 1, RTX 4060

Consolidates the two token-scaling grids run on CUDA hardware (an RTX 4060,
8 GB VRAM) against the 48M-token pilot grids (`docs/experiments/PILOT_RESULTS.md`,
run on Apple M4/MPS). Source directories:
[`scale-desynpuf/`](scale-desynpuf/) (200M tokens, 6L/256d encoder),
[`scale1b-desynpuf/`](scale1b-desynpuf/) (1B tokens, same encoder), and
[`scale1b-final-desynpuf/`](scale1b-final-desynpuf/) (the repository default
at 1B tokens, plus every earlier 1B checkpoint re-scored on the full held-out
split). Each directory's `README.md` is the protocol written before its
numbers existed; each `summary.md` is the row source the tables below are
built from.

**The headline numbers in this document are now the "Final 1B comparison on
the full held-out split" section directly below** — the repository default
(`hybrid_final`) against `ar`, on the full 11,708-subject held-out split.
Every other table in this document, unless it says otherwise, is scored on
the earlier seeded 3,000-subject held-out subset and is kept below as
history; those numbers are superseded by the final section for any cell that
appears in both.

Common to every 200M/1B row below `## Final 1B comparison`:
`configs/pretrain_scale.yaml` base (6-layer, 256-wide encoder, 4-layer,
128-wide predictor, SwiGLU, dropout 0.1), `data/cache/desynpuf-s1` `train`
split, held-out evaluation on the same seeded 3,000-subject `held_out` cut
with 200 bootstrap resamples across all seven tasks (`inpatient_365d`,
`mortality_365d`, `new_dx_365d/{ckd,copd,diabetes,heart_failure}`,
`readmission_30d`) as the pilot grids. The pilot rows below (48M tokens,
4-layer/192-wide encoder, seed 0) are `ar` and `hybrid`
(`nextlatent_h1416_recon`) from `docs/experiments/PILOT_RESULTS.md`'s master
table.

The 1B `ar` row was re-evaluated after commit `8721a62` fixed an
embedding-cache key collision in the eval harness (a second grid's cell
sharing the display name `ar` had been silently reusing another grid's
cached embeddings); see [`scale1b-desynpuf/README.md`](scale1b-desynpuf/README.md)
for the mechanism.

## Final 1B comparison on the full held-out split

Source: [`scale1b-final-desynpuf/summary.md`](scale1b-final-desynpuf/summary.md);
protocol in [`scale1b-final-desynpuf/README.md`](scale1b-final-desynpuf/README.md),
complete. `configs/pretrain_scale.yaml` base (6L/256d encoder), 1,000,013,824
nominal token slots per cell, seeds 0/1/2, evaluated on the FULL held-out
split (11,708 subjects, `eval_subject_limit: null`), 200 bootstrap resamples,
`lr`/`gbm` refit on the same split. Three cells: `ar` (next-code AR),
`hybrid` (the earlier `nextlatent_h1416_recon` config, `lambda_sigreg: 0.05`
— this is the same trained checkpoint as the `hybrid` row in every table
below this one, re-scored on the full split rather than the 3,000-subject
subset), and `hybrid_final` (the repository default,
`configs/pretrain_default.yaml`: causal next-latent, horizons `[1, 4, 16]`,
`lambda_recon: 0.1`, `lambda_sigreg: 0`, EMA target — see
[`ABLATION_RESULTS.md`](ABLATION_RESULTS.md)). `std` is sample standard
deviation ($n=3$, ddof=1). `overlap` is whether `ar`'s and `hybrid_final`'s
min-max ranges across the three seeds intersect for that task.

| task | ar mean | ar std | ar range | hybrid_final mean | hybrid_final std | hybrid_final range | overlap |
|---|---|---|---|---|---|---|---|
| inpatient_365d | 0.7420 | 0.0019 | [0.7399, 0.7435] | 0.7573 | 0.0019 | [0.7558, 0.7595] | no |
| mortality_365d | 0.6053 | 0.0057 | [0.6006, 0.6116] | 0.6011 | 0.0014 | [0.5997, 0.6025] | yes |
| new_dx_365d/ckd | 0.7674 | 0.0024 | [0.7657, 0.7701] | 0.7834 | 0.0012 | [0.7821, 0.7844] | no |
| new_dx_365d/copd | 0.7603 | 0.0028 | [0.7581, 0.7634] | 0.7721 | 0.0003 | [0.7719, 0.7724] | no |
| new_dx_365d/diabetes | 0.7625 | 0.0017 | [0.7606, 0.7637] | 0.7777 | 0.0015 | [0.7762, 0.7792] | no |
| new_dx_365d/heart_failure | 0.7715 | 0.0025 | [0.7691, 0.7741] | 0.7982 | 0.0014 | [0.7970, 0.7997] | no |
| readmission_30d | 0.6580 | 0.0123 | [0.6438, 0.6655] | 0.6881 | 0.0064 | [0.6827, 0.6951] | no |
| **mean-of-6** (excl. mortality) | **0.7436** | — | — | **0.7628** | — | — | non-overlapping on all 6 |

Non-mortality seed ranges do not overlap between `ar` and `hybrid_final` on
all six tasks; `mortality_365d` overlaps.

`hybrid` (SIGReg on, `lambda_sigreg: 0.05`, the earlier configuration, same
checkpoints as the `hybrid` rows below) scores mean-of-6 0.7615 on this same
full split, 3-seed mean — between `ar` (0.7436) and `hybrid_final` (0.7628),
consistent with `ABLATION_RESULTS.md`'s Grid 2/4 finding that dropping
SIGReg is worth about +0.6 on its own.

| model | inpatient_365d | mortality_365d | ckd | copd | diabetes | heart_failure | readmission_30d | mean-of-6 |
|---|---|---|---|---|---|---|---|---|
| `gbm` | 0.7455 | 0.5564 | 0.7713 | 0.7677 | 0.7709 | 0.7840 | 0.6670 | 0.7511 |
| `lr` | 0.7124 | 0.5659 | 0.7390 | 0.7326 | 0.7368 | 0.7407 | 0.6529 | 0.7191 |
| `ar` (3-seed mean) | 0.7420 | 0.6053 | 0.7674 | 0.7603 | 0.7625 | 0.7715 | 0.6580 | 0.7436 |
| `hybrid`, SIGReg on (3-seed mean) | 0.7559 | 0.5991 | 0.7845 | 0.7725 | 0.7780 | 0.7944 | 0.6838 | 0.7615 |
| `hybrid_final`, default (3-seed mean) | 0.7573 | 0.6011 | 0.7834 | 0.7721 | 0.7777 | 0.7982 | 0.6881 | **0.7628** |

`hybrid_final` is above `gbm` (0.7511) and `lr` (0.7191) on the mean-of-6, and
above `ar` on every non-mortality task with non-overlapping seed ranges.

## Historical: 3,000-subject-subset results

Every table from here down is on the earlier seeded 3,000-subject held-out
subset, not the full split above; kept for the record of what was measured
at each stage. Where a cell also appears in the full-split table above, that
table supersedes the numbers here.

## All trained cells

`control` is the row's own `random_init` (untrained, architecture- and
pooling-matched) reference. `gain` is mean AUROC across the seven tasks minus
the same mean for `control`. `mean` is the row's own mean AUROC across the
seven tasks. `gbm` and `lr` are count-feature baselines, reused from the same
`predictions.parquet` at every token budget (they do not depend on the
encoder). Sorted by `scale`, then `mean` descending within each scale.

| cell | scale | family | control | inpatient_365d | mortality_365d | ckd | copd | diabetes | heart_failure | readmission_30d | mean | gain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid` | 48M | nextlatent | `random_init@nextlatent_h1` | 0.7417 | 0.6332 | 0.7531 | 0.7487 | 0.7618 | 0.7642 | 0.6985 | 0.7287 | +0.0923 |
| `ar` | 48M | ar | `random_init@ar` | 0.7265 | 0.6060 | 0.7586 | 0.7552 | 0.7590 | 0.7437 | 0.6946 | 0.7205 | +0.0445 |
| `gbm` | — | count-feature | — | 0.7440 | 0.5723 | 0.7654 | 0.7674 | 0.7721 | 0.7893 | 0.6724 | 0.7261 | — |
| `lr` | — | count-feature | — | 0.7077 | 0.5537 | 0.7361 | 0.7258 | 0.7397 | 0.7438 | 0.6578 | 0.6949 | — |
| `hybrid` | 200M | nextlatent | `random_init@ar` | 0.7511 | 0.5919 | 0.7761 | 0.7676 | 0.7729 | 0.7767 | 0.6930 | 0.7328 | +0.0781 |
| `ar` | 200M | ar | `random_init@ar` | 0.7515 | 0.5946 | 0.7624 | 0.7747 | 0.7714 | 0.7908 | 0.6674 | 0.7304 | +0.0757 |
| `recon_only` | 200M | recon-only | `random_init@jepa_ema` | 0.7326 | 0.6508 | 0.7336 | 0.7242 | 0.7568 | 0.7453 | 0.6894 | 0.7190 | +0.0448 |
| `jepa_ema` | 200M | masked-span jepa | `random_init@jepa_ema` | 0.6948 | 0.6413 | 0.7108 | 0.7189 | 0.7499 | 0.7159 | 0.6580 | 0.6985 | +0.0243 |
| `hybrid` | 1B | nextlatent | `random_init@hybrid` | 0.7553 | 0.5767 | 0.7741 | 0.7781 | 0.7779 | 0.7997 | 0.6874 | **0.7356** | +0.0809 |
| `ar` | 1B | ar | `random_init@ar` | 0.7506 | 0.6028 | 0.7586 | 0.7629 | 0.7618 | 0.7804 | 0.6490 | 0.7237 | +0.0691 |

`random_init@ar` is 0.6547 mean AUROC at both 200M and 1B (same untrained 6L/256d
causal encoder, same seed); `random_init@hybrid` at 1B is numerically identical
to `random_init@ar` for the same reason grids 1-4 saw this (a probe reads only
the untrained embedding and encoder, and neither the AR head nor the
`nextlatent` heads exist at initialisation). `random_init@jepa_ema` at 200M is
0.6742.

The `hybrid`/`ar` 1B rows above (mean 0.7356 / 0.7237) are seed 0 only, as
their `scale` column implies for every row in this table; the 3-seed means at
1B (0.7363 / 0.7251) are in the "Scaling" and "1B seed replication" sections
below.

## Scaling: ar vs. hybrid, 48M → 200M → 1B

The 48M and 200M rows are seed 0 on the 3,000-subject subset, unchanged from
before. **The 1B row is now the full-held-out-split, 3-seed mean from the
"Final 1B comparison" section above** (`ar` and `hybrid_final`, not the
3,000-subject-subset numbers reported for 1B previously); see that section
for per-seed values, per-task std, and overlap, and the seed-replication
table further down for the superseded 3,000-subject-subset 1B numbers.

| tokens | split | ar mean AUROC | ar readmission_30d | hybrid mean AUROC | hybrid readmission_30d |
|---|---|---|---|---|---|
| 48M | 3,000-subject subset | 0.7205 | 0.6946 | 0.7287 | 0.6985 |
| 200M | 3,000-subject subset | 0.7304 | 0.6674 | 0.7328 | 0.6930 |
| 1B (3-seed mean) | full split (11,708 subjects) | 0.7436 | 0.6580 | **0.7628** | **0.6881** |

(`hybrid` in the 1B row is `hybrid_final`, the repository default; mean AUROC
here is mean-of-6, excluding `mortality_365d`, to match the full-split table
above — the 48M/200M rows above are mean-of-7, so the two are not directly
comparable across rows; see the "Final 1B comparison" section for the
full-split mean-of-6 vs. mean-of-7 distinction.)

## Figure

![Left: mean AUROC vs. token budget (log x) for ar, hybrid, recon_only, jepa_ema on the 3,000-subject subset (circles), with gbm as a horizontal reference and full-split 1B points for ar_full/hybrid_final overlaid as diamonds. Right: per-task AUROC at 1B tokens on the full held-out split, ar vs. hybrid_final, paired bars.](../figures/scale_desynpuf.png)

Produced by [`scripts/plot_scale.py`](../../scripts/plot_scale.py), which
parses the seven committed `summary.md` files directly (the three pilot grids
that contributed a 48M-token point, both scale grids, and
`scale1b-final-desynpuf`) — nothing plotted is a number not already committed
in one of those tables. The left panel's circles (48M, 200M, and the earlier
1B point) are the 3,000-subject subset; its diamonds, and the entire right
panel, are the full held-out split (11,708 subjects) — the two splits are
never drawn as the same marker shape so they cannot be misread as
comparable. Regenerate with:

```bash
python scripts/plot_scale.py
```

## 1B seed replication (3,000-subject subset, historical)

Superseded by the full-held-out-split "Final 1B comparison" section above
for `ar` and `hybrid_final`; kept here for the record of the 3,000-subject-subset
1B numbers this document reported before the full-split re-score.

`scale1b-seeds-desynpuf` (seeds 1 and 2, `ar` and `hybrid`) plus the seed-0
rows from `scale1b-desynpuf` give three seeds per cell at 1B tokens. Std is
sample std (n=3, ddof=1). "Overlap" is whether the ar and hybrid min-max
ranges across the three seeds intersect for that task.

| task | ar s0 | ar s1 | ar s2 | ar mean | ar std | hybrid s0 | hybrid s1 | hybrid s2 | hybrid mean | hybrid std | seed ranges overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inpatient_365d | 0.7506 | 0.7410 | 0.7441 | 0.7452 | 0.0049 | 0.7553 | 0.7491 | 0.7556 | 0.7533 | 0.0037 | yes |
| mortality_365d | 0.6028 | 0.6031 | 0.5933 | 0.5997 | 0.0056 | 0.5767 | 0.5862 | 0.5969 | 0.5866 | 0.0101 | yes |
| new_dx_365d/ckd | 0.7586 | 0.7584 | 0.7581 | 0.7584 | 0.0003 | 0.7741 | 0.7717 | 0.7752 | 0.7737 | 0.0018 | no |
| new_dx_365d/copd | 0.7629 | 0.7738 | 0.7656 | 0.7674 | 0.0057 | 0.7781 | 0.7798 | 0.7760 | 0.7780 | 0.0019 | no (gap 0.0022) |
| new_dx_365d/diabetes | 0.7618 | 0.7639 | 0.7589 | 0.7615 | 0.0025 | 0.7779 | 0.7787 | 0.7784 | 0.7783 | 0.0004 | no |
| new_dx_365d/heart_failure | 0.7804 | 0.7796 | 0.7787 | 0.7796 | 0.0009 | 0.7997 | 0.8015 | 0.7950 | 0.7987 | 0.0034 | no |
| readmission_30d | 0.6490 | 0.6742 | 0.6679 | 0.6637 | 0.0131 | 0.6874 | 0.6846 | 0.6840 | 0.6853 | 0.0018 | no |

Ranges do not overlap between `ar` and `hybrid` on four of seven tasks (ckd,
diabetes, heart_failure, readmission_30d); `mortality_365d` and
`inpatient_365d` overlap; `new_dx_365d/copd` does not overlap by the min-max
definition above, with the smallest gap of any non-overlapping task (ar max
0.7738 vs. hybrid min 0.7760, a gap of 0.0022).

## Findings (3,000-subject subset, historical)

The token-budget trend below was measured on the 3,000-subject subset at
every budget, including the 1B point; it is retained for the record of what
this document reported before the full-split 1B re-score. The "Final 1B
comparison" section above is the current full-split 1B result and is not a
drop-in replacement for every number below, since 48M and 200M have no
full-split counterpart.

- `ar` mean AUROC (3-seed mean at 1B): 0.7205 (48M) → 0.7304 (200M) → 0.7251
  (1B). It does not improve from 200M to 1B (-0.0053) and drops on six of
  seven tasks over that step — every task except `mortality_365d`
  (0.5946 → 0.5997, +0.0051): `inpatient_365d` (0.7515 → 0.7452, -0.0063),
  `new_dx_365d/ckd` (0.7624 → 0.7584, -0.0040), `new_dx_365d/copd`
  (0.7747 → 0.7674, -0.0073), `new_dx_365d/diabetes` (0.7714 → 0.7615,
  -0.0099), `new_dx_365d/heart_failure` (0.7908 → 0.7796, -0.0112),
  `readmission_30d` (0.6674 → 0.6637, -0.0037).
- `hybrid` mean AUROC (3-seed mean at 1B) improves at every step: 0.7287
  (48M) → 0.7328 (200M) → 0.7363 (1B), +0.0041 then +0.0035.
- At 1B (3-seed means), `hybrid` leads `ar` on six of the seven tasks — every
  task except `mortality_365d` — by 0.81 to 2.16 AUROC points:
  `inpatient_365d` +0.81, `new_dx_365d/ckd` +1.53, `new_dx_365d/copd` +1.05,
  `new_dx_365d/diabetes` +1.68, `new_dx_365d/heart_failure` +1.92,
  `readmission_30d` +2.16. `ar` leads on `mortality_365d` (0.5997 vs. 0.5866,
  +1.31 points).
- `hybrid` at 1B (0.7363 mean AUROC, 3-seed mean) is the highest mean AUROC
  of any cell in this table, above `gbm` (0.7261) and `lr` (0.6949).
- `recon_only` and `jepa_ema` were run through 200M only (0.7190 and 0.6985
  mean AUROC, gains +0.0448 and +0.0243) and have no 1B row.
- 1B is now three seeds per cell (`ar`, `hybrid`); see the seed-replication
  table above. 200M and 48M rows above remain single-seed (seed 0), except
  that the pilot grid separately reseeded `ar` and `hybrid` at 48M (see
  `PILOT_RESULTS.md`, grid 5) — `recon_only` and `jepa_ema` are single-seed
  at every budget they were run at.

## Caveats

- **The "Final 1B comparison" section's `ar`/`hybrid_final`/`hybrid` numbers
  are on the full held-out split (11,708 subjects); every other table in
  this document, including "All trained cells", "Scaling"'s 48M/200M rows,
  and "1B seed replication", is on the 3,000-subject subset.** They are not
  interchangeable, though for the one cell scored on both (`hybrid`, SIGReg
  on) the mean-of-7 difference between splits is small: 0.7356 (3,000-subject
  subset, 3-seed mean) vs. 0.7383 (full split, 3-seed mean, computed the same
  way including `mortality_365d`).
- **DE-SynPUF has no labs.** It is a CMS claims-derived public-use file with
  no lab results, vitals, or notes — the same caveat as the pilot grids.
- **3,000-subject held-out subset**, not the full `held_out` split — same
  subset, anchors, and seed as the pilot grids.
- **The embedding table is a smaller majority of trainable parameters at this
  scale than at the pilot's.** For the 6L/256d `hybrid` cell: 7,905,536 of
  13,208,336 trainable parameters (60%, a 30,000 × 256 code table plus
  value/age/delta encoders, EMA target copy excluded). For `ar`: 7,905,536 of
  12,665,104 (62%). Both are down from 74% at the pilot's 4L/192d scale —
  encoder and predictor capacity grows faster than the embedding table as
  depth and width scale, at a fixed 30,000-code vocabulary.
- **Single seed at 200M.** Every 200M row (`ar`, `hybrid`, `recon_only`,
  `jepa_ema`) is seed 0 only; the 48M-to-200M step rests on one training run
  per cell. `recon_only` and `jepa_ema` are also single-seed at 48M — neither
  has a seed-replication grid, and neither has been run at 1B.
- **1B is three seeds per cell for `ar` and `hybrid`,** but the seed-0-only
  200M comparison point they are scaled from is not. A "200M → 1B" step is
  therefore comparing a single run at 200M to a three-seed mean at 1B, not a
  matched-seed-count comparison at both ends.
