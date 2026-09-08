# Ablation results -- model size and hybrid-knob effects (`ablate2-desynpuf`)

Source: [`docs/experiments/ablate2-desynpuf/summary.md`](ablate2-desynpuf/summary.md)
and [`summary.json`](ablate2-desynpuf/summary.json); protocol in
[`ablate2-desynpuf/README.md`](ablate2-desynpuf/README.md). DE-SynPUF sample 1,
full held-out split (11,708 subjects, not the earlier 3,000-subject subset),
200 bootstrap resamples, `probe_features: auto`, 200,000,000 nominal token
slots per trained cell. `small`/`large` cells are `run.seed` 1 and 2 (2-seed
mean below); `base` is `ar_base_s0`/`hybrid_base_s0`, seed 0 only, re-scored
from `runs/scale-desynpuf/{ar,hybrid}/final.pt`. Trainable parameter counts
are computed by instantiating each model from its recorded config overrides
(`scripts/plot_ablate2.py::n_params`), not estimated.

## (a) Size: ar vs. hybrid at small / base / large

| size | params (ar / hybrid) | inpatient | mortality | ckd | copd | diabetes | heart_failure | readmission | mean-of-6 | mean-of-7 |
|---|---|---|---|---|---|---|---|---|---|---|
| small, ar (2-seed) | 7.70M | 0.7464 | 0.6055 | 0.7681 | 0.7572 | 0.7695 | 0.7616 | 0.6731 | 0.7460 | 0.7259 |
| small, hybrid (2-seed) | 8.01M | 0.7538 | 0.6313 | 0.7717 | 0.7584 | 0.7706 | 0.7720 | 0.6968 | 0.7539 | 0.7364 |
| base, ar (seed 0 only) | 12.7M | 0.7500 | 0.6158 | 0.7725 | 0.7666 | 0.7725 | 0.7813 | 0.6614 | 0.7507 | 0.7315 |
| base, hybrid (seed 0 only) | 13.2M | 0.7523 | 0.6088 | 0.7798 | 0.7643 | 0.7720 | 0.7745 | 0.6896 | 0.7554 | 0.7345 |
| large, ar (2-seed) | 26.2M | 0.7526 | 0.6111 | 0.7740 | 0.7682 | 0.7756 | 0.7741 | 0.6793 | 0.7540 | 0.7336 |
| large, hybrid (2-seed) | 27.4M | 0.7585 | 0.6103 | 0.7841 | 0.7665 | 0.7780 | 0.7851 | 0.6847 | 0.7595 | 0.7382 |

`base` rows are a single seed (`run.seed: 0`); every other row in this table
is a 2-seed mean.

## (b) Hybrid knobs at base size (6L/256d), 2-seed

`default` is `hybrid_base_s0` (seed 0 only, no seed-spread column). The other
four rows are 2-seed means (`run.seed` 1, 2); `seed spread` is the per-task
`|s1 - s2|`, averaged over the six non-mortality tasks.

| knob | inpatient | mortality | ckd | copd | diabetes | heart_failure | readmission | mean-of-6 | seed spread (mean-of-6) |
|---|---|---|---|---|---|---|---|---|---|
| default (base_s0) | 0.7523 | 0.6088 | 0.7798 | 0.7643 | 0.7720 | 0.7745 | 0.6896 | 0.7554 | -- |
| shared target | 0.7512 | 0.6039 | 0.7730 | 0.7558 | 0.7699 | 0.7724 | 0.6838 | 0.7510 | 0.0029 |
| no SIGReg | 0.7572 | 0.5984 | 0.7819 | 0.7700 | 0.7785 | 0.7838 | 0.6977 | 0.7615 | 0.0039 |
| horizon [1] only | 0.7471 | 0.6140 | 0.7673 | 0.7550 | 0.7697 | 0.7686 | 0.6995 | 0.7512 | 0.0020 |
| recon 0.3 | 0.7558 | 0.6162 | 0.7762 | 0.7702 | 0.7754 | 0.7847 | 0.6936 | 0.7593 | 0.0046 |

Per-seed mean-of-6 (not shown per task above): shared target 0.7520 / 0.7500;
no SIGReg 0.7620 / 0.7611; horizon [1] only 0.7511 / 0.7513; recon 0.3
0.7579 / 0.7607.

## (c) Controls and baselines

| model | inpatient | mortality | ckd | copd | diabetes | heart_failure | readmission | mean-of-6 | mean-of-7 |
|---|---|---|---|---|---|---|---|---|---|
| `lr` | 0.7124 | 0.5659 | 0.7390 | 0.7326 | 0.7368 | 0.7407 | 0.6529 | 0.7191 | 0.6972 |
| `gbm` | 0.7455 | 0.5564 | 0.7713 | 0.7677 | 0.7709 | 0.7840 | 0.6670 | 0.7511 | 0.7233 |
| `random_init@ar_small_s1` (= `@hybrid_small_s1`) | 0.6570 | 0.5338 | 0.6731 | 0.6771 | 0.7062 | 0.6731 | 0.5842 | 0.6618 | 0.6435 |
| `random_init@ar_base_s0` | 0.6742 | 0.5406 | 0.6857 | 0.6906 | 0.7185 | 0.6897 | 0.6166 | 0.6792 | 0.6594 |
| `random_init@ar_large_s1` (= `@hybrid_large_s1`) | 0.6760 | 0.5637 | 0.6915 | 0.6960 | 0.7296 | 0.7020 | 0.6264 | 0.6869 | 0.6693 |

`lr` and `gbm` are count-feature baselines, shared across every cell in this
grid. `random_init@<run>` probes that run's own untrained architecture at the
same size; `ar` and `hybrid` share an encoder shape at a given size, so one
control covers both.

## (d) Findings

- Hybrid leads ar at every size on the mean-of-6: small +0.0079, base
  +0.0047, large +0.0055. The gap does not close as size increases (base's
  gap is the smallest of the three, large's is larger than base's).
- No-SIGReg is the best knob at both seeds (mean-of-6 0.7620 at seed 1,
  0.7611 at seed 2), ahead of the other three knobs at both seeds
  individually.
- Horizon-[1]-only (mean-of-6 0.7512) and shared target (0.7510) are the
  worst knobs, both below the default (0.7554).
- Recon 0.3 (mean-of-6 0.7593) is within 0.0039 of the default's 0.1
  (0.7554), against a seed spread of 0.0020-0.0046 across these knobs --
  within noise.

## (e) Decision: new default

New default (`configs/pretrain_default.yaml`): hybrid objective
(`objective.kind: nextlatent`), causal encoder (`model.causal: true`), EMA
target (`model.target_mode: ema`), horizons `[1, 4, 16]`
(`objective.horizons`), `objective.lambda_recon: 0.1`,
`objective.lambda_sigreg: 0.0` (SIGReg dropped, per the no-SIGReg result
above).

## (f) Caveats

- The base-size default row (`hybrid_base_s0`, `ar_base_s0`) is one seed
  (seed 0); every size/knob comparison against it is a 2-seed-mean-vs-1-seed
  comparison, not 2-seed-vs-2-seed.
- 200,000,000 nominal token slots per cell -- not a claim about behavior at
  other budgets.
- DE-SynPUF: a CMS claims-derived public-use file with no lab results,
  vitals, or notes.
- Differences among the base-size hybrid knobs are 0.3-0.7 points (mean-of-6,
  vs. the default) against a seed spread of about 0.3-0.5 points among those
  same knobs -- the no-SIGReg and shared-target/horizon-[1] results are
  outside that spread; recon 0.3 vs. 0.1 is inside it.
