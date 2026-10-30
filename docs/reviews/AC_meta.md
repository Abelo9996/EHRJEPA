# Area Chair meta-review

Paper: "Latent Prediction Helps EHR Foundation Models Only Next to a Token Loss, and Only Fine-Tuning Shows It: A Careful Public-Data Study and an Action-Conditioned Patient-State Model" (docs/paper/v2/main.tex, Oct 2026 draft).
Inputs: R1 (ML methods / SSL), R2 (clinical), R3 (causal inference / world models). The plan documents TRIAL_PILOT_SPEC.md and NEXT_STEPS.md were read for feasibility.

All three reviews were read in full. R2 arrived at 02:46.

## 1. Consensus decision for the NeurIPS main track, as of today

**Reject.** Scores: R1 3 (conf 4), R2 3 (conf 4), R3 3 (conf 4). The reviewers agree independently on these points.

1. **Half the paper has no results.** The action-conditioned state model is in the title, abstract, contribution 4, Table 5, the discussion and the conclusion. Every number in it is a placeholder. Under the venue norm, a placeholder counts as a missing result.
2. **The planned control for the state model is confounded by construction.** R1 (W6) and R3 (W2a, C1) found this independently. The target state is the EMA causal encoder's output over a stream that contains the very action tokens the predictor is given. A positive `state_ac − state_blind` gap or action sensitivity is therefore expected even with zero physiological effect. R3 adds further problems:
   - the action input includes post-baseline, state-responsive treatments, and the action count and Δt-to-next-event are severity and measurement-process proxies;
   - the "k = 4h/8h/16h" horizon is false, because windows are numbered over non-empty windows only (`actions.py:156–163`);
   - the MEDS timestamps (`chartdate` midnight procedures, `storetime` ICU actions, `dod` dates) cannot support 4h cause→effect order.
3. **The surviving empirical claim is small and not attributed.** Hybrid > AR (+0.008 AUROC) is significant at 1 of 3 pools. Seed variance is excluded from the interval. The 1,346-subject test set is used although ~36k held-out subjects exist.
   - R1 shows the hybrid–AR contrast is not controlled. The hybrid adds SIGReg on tokens and CLS, down-weights the code loss by 10x, and uses ~33% more compute. The "controlled addition / recovers AR exactly" sentence (L267–269) is inaccurate.
4. **Task validity is broken (R2 W1, critical).** `diagnoses_icd` is stamped at `dischtime`, and anchors fall at random in-stay event times. On the demo, ~82% of the "new diagnosis within 365 d" positives are the *current* stay's discharge coding. These four tasks make up two-thirds of the headline mean-of-6 and measure in-progress workup, not incident disease. This alone invalidates every table until the evals are rerun.
   - The GBM leakage audit is partly vacuous: CKD has 0 near-synonym columns after pruning, and the audit ran on a different GBM than the one reported.
   - The negative-label rule (a record must exist ≥ horizon after the anchor) selects on the future.
5. **The paper's own numbers contradict its title.** On the claimed effect (hybrid − AR), the probe gap exceeds the fine-tune gap at every pool (R1 W3/M7). The only probe/FT "flip" is hybrid vs GBM, which is not significant in either regime.
6. **Scope is below the main-track bar.** One site, one 10M model size, six binary tasks, one label regime, and no released-model baseline. Template and anonymity are not compliant (desk-reject risk).

The paper's strengths are real and the reviewers agree on them: unusually careful evaluation hygiene for this subfield, numbers that reproduce from committed artefacts (R1 verified), restrained claims against GBM, and explicit causal disclaimers. These make it a solid workshop or methodology note. They do not offset points 1–5 at a main track.

Reviewer disagreement:
- R2 lists "λ=0 recovers AR exactly" as a strength. R1 checked the code and configs and shows the trained hybrid differs from AR by SIGReg, loss weight and compute. The AC sides with R1.
- R2 reads the probe-vs-FT point as "well supported in direction". R1 shows the probe overstates hybrid − AR and that the only flip is non-significant. The AC sides with R1. The probe-vs-FT observation survives only as a ranking matrix, not as "the probe hides the gain".

## 2. Strongest paper achievable by late January 2027 on one RTX 4060

**Budget reality.**
- Thirteen weeks remain. The GPU is currently on another project.
- Measured rates: AR 2.7 h, hybrid 3.6 h, state cell ~7.5 h per 1B tokens, and ~17 min per fine-tune eval cell.
- Allowing for PC job deaths and the other project, a realistic usable budget is **~250–400 GPU-h**.
- The binding constraint is one person's engineering and writing time, not GPU.
- Three RCT emulations at RCT-DUPLICATE quality, plus a benchmark, plus a second site, plus 35M scaling do not fit.

**Recommended paper: "What Actually Helps EHR Pretraining? A Seeded, Compute-Controlled, Fine-Tuned Study on Public MIMIC-IV, with an Audit of Action-Conditioned State Objectives."**
- **Foundation (non-negotiable): valid tasks.**
  - Anchors move to clinical decision points: discharge for readmission and new-dx, as in EHRSHOT; admission + 24h for the 365-day tasks.
  - The current stay's codes are excluded from new-dx labels, and the tasks are renamed accordingly.
  - Censoring for negatives uses MIMIC's 1-year post-discharge death linkage rather than requiring a return visit.
  - Every table is re-scored on the ~36k full-pool held-out split.
  - A cohort table, AUPRC, calibration and basic subgroups are added; the code already computes most of these.
  - The headline numbers will drop and may reorder. That has to be known before anything else is built on them.
- **Core (must be solid).** De-confounded hybrid vs AR, with SIGReg, loss-weight and FLOP-matched ablations at 3 seeds, using a hierarchical seed × subject bootstrap. Add a properly tuned GBM (MEDS-Tab or Optuna) and one released model on identical anchors (ETHOS weights on MIMIC are the cheapest). Report a probe family (last-token, mean, multi-layer, MLP) against fine-tuning as a ranking matrix, with a label-efficiency curve.
- **State-model section, reframed as an audit, not a world model.** Report three things:
  - (a) the *echo* result: the action-sensitivity and state_ac–state_blind gaps with action-containing targets versus observation-only targets. If the gap collapses, that is a useful, citable cautionary finding for Clin-JEPA-style action-conditioned EHR models.
  - (b) the fixed objective (observation-only targets, window-t actions only, real-hour horizons, corrected timestamps) under the existing gate rule.
  - (c) a small mechanistic positive-control panel with AIPW references, positivity and negative-control outcomes: saline→chloride, PRBC→Hb, insulin→glucose, via within-patient action swaps on held-out patients.
- **Trial reproduction.** Cut from this paper. At most, keep one OSF-pre-registered albumin feasibility case if it is done before the first outcome look. Do not call anything a "benchmark".
- **EHRSHOT.** Add it only if the DUA lands by mid-December.

**Probable outcome.**
- **NeurIPS main track: 4 (borderline reject), range 3–5.** After the task fix, the core effect is still roughly +0.01 AUROC at one site with a 10M model. The echo audit is valuable but niche.
- **ML4H or CHIL full paper, or a NeurIPS workshop: a plausible 6 (accept).**
- **NeurIPS Datasets & Benchmarks:** about 5, and only if corrected tasks and checkpoints are released as a MEDS/ACES suite.

The decisive unknowns are:
- whether hybrid > AR survives the task fix and the de-confounding ablations;
- whether the echo audit gives a clean collapse.

A main-track 6+ would require a second site plus released-model baselines at a larger scale, which is unrealistic on this timeline and hardware.

## 3. Ranked must-do list

| # | Item | GPU-h | Human cost | Source |
|---|---|---|---|---|
| 1 | **Fix task validity, then re-score everything on the ~36k held-out split.** Discharge or admission+24h anchors; current-stay codes excluded from new-dx labels; death-linkage censoring; renamed tasks; redo the leak audit on the *reported* GBM and the encoders, including medication proxies and with no `min_df` pruning; cohort table, AUPRC, calibration and subgroups. | ~25 (re-fine-tune ~20 existing checkpoints) | 3–4 days | R2 W1/W2/W5/W7, R1 W5 |
| 2 | **Stop and fix the state objective before spending the remaining ~31 GPU-h of the current gate.** Observation-only target stack; window-t actions only; drop the count and Δt-to-future-event inputs; real-hour horizons with empty windows kept; timestamp fixes (`chartdate` codes moved or excluded, ICU actions at `starttime`, `dod` at end of day); invariance unit tests. Finish only `state_blind_s0` (~2.6 h) as the "echo" arm. | ~35 (2.6 + 4 × 7.5) + cache rebuild on CPU | 3–4 days | R1 W6, R3 W2–W4, C1–C9 |
| 3 | **De-confound hybrid vs AR.** On the 25% pool, 3 seeds each: AR + SIGReg 0.05; hybrid with SIGReg 0; λ_lat = 0 with λ_code = 0.1 + SIGReg; AR at 1.33B tokens (FLOP-matched). | ~40 | 1 day | R1 W2/M1/M8 |
| 4 | **Full-pool seeds 1–2 for hybrid and AR**, with a hierarchical seed × subject bootstrap, a pre-declared primary comparison, and Holm correction across pools. | ~15 | 0.5 day | R1 W5, R2 W3 |
| 5 | **Baselines.** Optuna-tuned XGBoost or MEDS-Tab, plus one released FM on identical anchors (ETHOS MIMIC weights; otherwise CEHR-GPT trained on the same pool). | ~10 (ETHOS) to ~60 (CEHR-GPT) | 2–3 days + licence (owner) | R1 W4, R2 W4 |
| 6 | **Probe family plus a label-efficiency curve** (1/8/25/100%, GBM and LR refit). Rewrite contribution 3 as a ranking matrix. | ~30 | 1–2 days | R1 W3/M4/M7 |
| 7 | **Mechanistic positive controls for the fixed state model.** Within-patient action swaps; cross-fit AIPW reference; propensity overlap with ESS; negative-control outcomes; itemids verified. This is the minimum for any "world model" wording. | ~5–10 | 1–1.5 weeks | R3 §7 |
| 8 | **Compliance and claim hygiene.** NeurIPS template, anonymisation, ≤9 pages; fix R1 M1–M9 and R2 §4 mismatches; repository URL and commit hashes. | 0 | 2–3 days | all |

**Totals.** About 160–220 GPU-h, inside the ~250–400 realistic budget. About 5–6 person-weeks of engineering, leaving January for writing.

**Cut order if time runs short:** the second half of 5 (CEHR-GPT), then 7 (the state section becomes the echo audit only), then the 1% and 100% points of 6. Items 1–4 and 8 are the floor.

## 4. Claims to cut, or rewrite until evidence exists

1. "**…and Only Fine-Tuning Shows It**" (title) and "the probe can hide the effect" (L56–57, L112–115). The probe *overstates* hybrid − AR at every pool. The only flip is non-significant.
2. "**Setting the latent weight to zero recovers AR exactly … a controlled addition to AR**" (L267–269). False for the trained configurations.
3. "**First-ever diagnosis … within 365 days**" (L331–333) as a construct. As implemented, it mostly predicts the current stay's discharge coding.
4. "**[Near-synonyms] carry no measurable share of GBM's gain**" (L347–351). This is vacuous for CKD and was run on a different GBM. Other overstatements to fix: "**tuned**" GBM for a 3-point grid, and "**we use all three**" (MEDS-Tab is not used).
5. "**Action-Conditioned Patient-State Model**" in the title, and every abstract or conclusion sentence about it, until results exist from the fixed objective. Also "**k ∈ {1,2,4} windows (4, 8 and 16 hours ahead)**" (L309), because windows are non-empty-only.
6. "**Conditioning on treatments separates [patient and clinician]**" (L572–573), and "**the treatment-blind control … isolates the use of the action input**" (L318–322). Neither holds while targets contain action tokens and the action input includes post-baseline treatments.
7. The **trial-anchored benchmark** and "to our knowledge no public benchmark yet combines…" (L126–128, L579–588).
8. Related-work claims contradicted by the cited papers ("none reports seeds, paired intervals, or a tuned tabular baseline"; "all … are probe-only": Clin-JEPA has an MLP probe, CIs and LightGBM). Also "hybrid > AR holds at every pool size" without the per-seed exceptions.
