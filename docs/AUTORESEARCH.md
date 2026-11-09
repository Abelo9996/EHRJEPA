# EHRJEPA research loop

Owner direction, 2026-11-06: use iterative autoresearch to maximize citability,
impact and novelty. Operationally, prioritize useful reusable artifacts and
testable findings. Citation counts and acceptance are outcomes, not a local
reward function. No forecast of either is used to select or report results.

## One cycle

1. State one hypothesis, closest prior work, competing explanation, primary
   measurement, falsifier, data split and compute cap **before running it**.
2. Run the cheapest control that could invalidate it. Prefer synthetic controls
   and training crops before tuning data; reserve held-out data for a frozen
   confirmatory protocol. A change in target definition creates a new experiment.
3. Make the smallest controlled change. Preserve legacy behavior and checkpoint
   semantics. Compare identical weights for structural tests; use matched seeds,
   architecture, label budget and compute for learned-performance claims.
4. Record commands, input/config/code hashes, aggregates, failures and exclusions
   in an append-only [ledger](experiments/autoresearch-ledger.jsonl). Label
   retrospective entries honestly; a git file is not external preregistration.
5. Decide **keep, revise, reject, or blocked** against the declared criterion.
   Report null/negative results. Choose the next experiment by the unresolved
   explanation it can distinguish, not by the best score observed so far.

## Current candidate contribution

**An executable validity contract for latent patient-state models:** distinguish
changes in physiological observations from direct action-token echo, changing
measurement schedules, compressed time horizons, and label timing. Pair the
diagnostics with corrected objectives, explicit controls and a reproducible
public-data evaluation recipe.

This is a candidate, **not an established novelty claim**. A bug in this repo
does not establish a flaw in another method. The diagnostics become a paper
contribution only if they expose a consequential, generalizable distinction on
trained models and survive matched comparisons. A tiny random-weight target
change alone is not sufficient.

Primary sources checked 2026-11-06:

| Prior work | What it already covers | Consequence for this project |
| --- | --- | --- |
| [SMB-Structure](https://arxiv.org/abs/2601.22128) | Hybrid token/latent learning for structured EHR | Do not claim first hybrid EHR JEPA. |
| [Clin-JEPA v5](https://arxiv.org/abs/2605.10840v5) | Co-trained EHR encoder/predictor and action-conditioned rollouts | Do not claim first action-conditioned EHR latent model; evaluate actual targets and code before attributing any shortcut to it. |
| [Gupta, Delaney, Russo](https://arxiv.org/abs/2610.29528) | Stable JEPA underperforms MLM; clinically meaningful targets remain difficult | A new target must beat controlled alternatives, not merely train stably. |
| [EHR-WM](https://arxiv.org/abs/2610.23862) | State/action separation, rollout and observed-feature forecasting | A small state model or action-sensitivity demonstration alone is insufficient differentiation. |
| [LeakBench](https://arxiv.org/abs/2610.26088) | Executable prediction-time leakage audits for clinical modeling agents | Generic leakage auditing is already occupied; distinguish latent-target and time-axis mechanisms. |

This was a focused primary-source check, not an exhaustive novelty search. Keep
the counterfactual/RCT benchmark a separate hypothesis; no treatment-effect
claim follows from predictive action sensitivity.

## Evidence and decisions

| Cycle | Hypothesis/control | Result | Decision |
| --- | --- | --- | --- |
| D01 | Do new-dx positives coincide with discharge from the anchor's current stay? | Demo: 27/33 positive task labels; full MIMIC unreachable. [Report](experiments/diagnosis-timing/README.md) | Keep timing audit; block incident-disease claims pending cohort redesign/validation. |
| S01 | Holding observations fixed, does action content change target states? Can an observation-only target remove this direct dependence without becoming constant? | Three random initializations: legacy changes nonzero; observation-only action change exactly zero, observation positive control nonzero. [Report](experiments/state-target-integrity/README.md) | Keep explicit target-view ablation; trained effect size unresolved. |
| S02 | Does k occupied windows equal k calendar bins? | Demo k=1: 205/523 mismatches. Exact-bin lookup retains 318 pairs with zero mismatches; also passes k=2/4. | Keep sparse calendar mode; preserve coverage counts, do not interpret fewer pairs as equal compute/equal sample size. |

S01 and S02 are structural controls on training crops, not trained efficacy
experiments or evidence of clinical benefit. They were run during development;
this record must not be described as prospective preregistration.

## Next hypotheses and stopping conditions

1. **D02 — generalize timing result.** On PC, audit the paper's 8% label cohort
   first and the full source separately. Stop interpretation on any label/input
   mismatch. Record all split counts. CPU only; blocked on SSH access.
2. **S03 — materiality on trained checkpoints.** Run the same target audit on the
   saved state checkpoint, then compare controlled retraining. Falsifier for a
   strong claim: action echo is negligible or removing it does not change the
   scientific conclusions. Do not compare raw latent losses across independently
   learned encoders as if they shared a calibrated metric.
3. **S04 — fixed prediction-time inputs.** Separate pre-action state from action
   plan, remove post-baseline actions/count shortcuts, audit source timestamps,
   and test noninterference before clinical rollout claims. Observation targets
   and exact bins alone do not fix these issues.
4. **E01 — controlled learned comparison.** Freeze corrected task definitions,
   then run target-view × predictor-action ablations with the same valid calendar
   pairs, seeds, regularization and token/compute accounting. Compare downstream
   results and observation forecasts against persistence, AR, MLM and a tuned
   tabular model. Use tuning for selection; report all arms in final evaluation.
5. **External applicability.** Only after a consequential result, adapt the
   diagnostic contract to one released prior method and a second site. If it is
   only our implementation bug, release a regression tool and drop the broad
   methodological claim. No outreach or public release without its own mandate.

Local structural cycles may use the existing CPU demo and tests. Do not allocate
new paid compute, alter remote infrastructure, or restart the old GPU gate.
GPU ownership remains with the other paper until explicitly released. Each
training cycle needs an explicit config, run identity, comparison and measured
resource cap; no unattended unbounded hyperparameter search is configured.
