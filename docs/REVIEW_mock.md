# Mock review: EHRJEPA (planned NeurIPS/ICLR submission)

Written 2026-10-30 against the planned paper (HANDOFF LATEST + Results), the old draft
`docs/paper/main.tex`, and three competitor papers: Imperial 2610.29528, SMB-Structure
2601.22128, Clin-JEPA 2605.10840. No `docs/paper/v2/` existed while this was written.

Paper as planned: a code-token EHR model pretrained from scratch on public MIMIC-IV
(~10M params, 1B tokens); a hybrid objective (next-code + multi-horizon next-latent JEPA); the
claim that frozen probes hide gains which fine-tuning reveals, with fine-tuned hybrid > AR
(paired CI excludes 0) and hybrid ≈ GBM (+0.013 [−0.001, +0.028]); a new action-conditioned
4h patient-state world model with a treatment-blind control (**results pending**); and a
planned public benchmark with RCT-anchored counterfactual checks (**not started**).

---

## Reviewer 1: ML methods

**Summary.** The paper compares AR, MLM, pure-JEPA and a hybrid AR+next-latent objective on
MIMIC-IV code streams. It argues that linear probes mis-rank objectives and that, under
fine-tuning, the hybrid beats AR. It adds an action-conditioned latent state predictor.

**Strengths**
1. Careful evaluation plumbing: ACES-declared tasks, leakage tests, random-init controls,
   seeds, paired subject-level bootstrap. Few EHR-SSL papers report this.
2. Honest negative results. Pure latent prediction loses to next-code prediction, which agrees
   with Imperial and the paper's own pre-registered A2 test.
3. Public data, Apache-2.0 code, consumer-GPU budget. Reproducible.

**Weaknesses (ranked)**
1. **The effect is tiny and fragile.** Hybrid − AR = +0.008 mean AUROC, significant only at
   the 25%-pretraining cell pooled over 3 seeds. At 8% the CI touches 0; at full MIMIC there is
   1 seed and the lower bound is +0.000. Many cells (8/25/full × probe/FT × seeds) were run and
   the one that clears gets reported. No multiplicity correction and no pre-registered primary
   comparison. This reads as a forking-paths result.
2. **The novelty of "probe mis-ranks, fine-tune reveals" is low.** MAE (He et al. 2022) made
   this argument for vision; SMB and Imperial are probe-only, so pointing out their protocol
   limit is a critique, not a contribution. To count, it needs a systematic ranking-flip
   analysis (Kendall τ across ≥4 objectives, readouts and label budgets), not one AR/hybrid pair.
3. **The hybrid is not novel.** SMB-Structure already trains next-token + EMA-JEPA jointly and
   calls it a world model. Clin-JEPA already has an action-conditioned latent predictor on
   MIMIC-IV with rollout evaluation. What is left is "a small from-scratch model, joint training
   works without a curriculum". That is an ablation-level delta.
4. **Fairness of the comparison.** Objectives are matched on tokens, not compute. The hybrid
   adds an EMA target forward pass and a predictor, so it gets roughly 1.3–1.5× the FLOPs (state
   it). Hyperparameters were tuned around the hybrid (SIGReg, horizons, target mode) over ~20
   grids, while AR and MLM appear to use defaults. MLM is fixed at a 15% mask with no
   mask-rate, pooling or LR sweep. A reviewer will assume the baselines are under-tuned.
5. **Scale.** About 10M params and 1B tokens. The 35M run stopped at step 8k, and there is no
   scaling curve. Gains of +0.008 at 10M say nothing about whether the auxiliary loss matters at
   the scale people deploy (100M–8B).
6. **The world-model section has no results.** A NeurIPS paper cannot rest on "results
   pending". Even when they arrive, the gate (beat hybrid/AR/MLM/GBM on 30–365-day tasks) is
   mismatched to a 4h-window dynamics model (see R2/R3).
7. Mean-pooled action embedding + log count discards order, dose and timing. With horizons
   k ∈ {1,2,4} the model predicts at most 16h ahead from 4h windows, so it is not a rollout
   model in the Clin-JEPA sense (48h autoregressive).

**Questions**
- Is the hybrid > AR gap stable under equal-FLOP training and an equal tuning budget per
  objective?
- Which split chose the hybrid's design (horizons, SIGReg, EMA)? If MIMIC held-out was used at
  any point, report a fresh split.
- Does the gap survive partial fine-tuning, MLP probes and k-shot? Kendall τ of the rankings?
- Does state_ac beat state_blind on anything other than the latent loss?

**Score: 3 (reject).** Confidence 4.

---

## Reviewer 2: Clinical informatics

**Summary.** A MIMIC-IV-only EHR foundation model, evaluated on 7 binary tasks (mortality,
inpatient admission, 30d readmission, 4 new-diagnosis tasks), with an action-conditioned ICU
state model and a proposed RCT-anchored benchmark.

**Strengths**
1. MEDS/ACES pipeline, open code. A tuned GBM on counts is used as a real bar, and a
   GBM-leakage audit is included (clean). That is good practice.
2. Fine-tuned hybrid is at par with or slightly above GBM on 8% labels.

**Weaknesses (ranked)**
1. **The test set is too small for the claims.** It has 1,346 labeled held-out subjects
   (1,025–1,120 anchors per task) out of ~36k available held-out subjects. Per-task CIs are
   ±0.015–0.03, so "≈GBM" and "> AR" are both within what this test set can resolve. Using the
   ~36k full held-out split is cheap and was not done. This is the single most damaging fact.
2. **No comparison to published EHR FMs.** CLMBR-T / MOTOR (EHRSHOT), CEHR-BERT / CEHR-GPT,
   ETHOS (trained on MIMIC-IV), Context-Clues-style long-context models, the MEDS-DEV task suite.
   "Beats our own GBM by 0.01" is not a position a 2026 FM paper can take. Clin-JEPA reports
   34 tasks on 3 benchmarks against a pretrained FM.
3. **Single site, single system.** BIDMC only. The value of an FM is transfer, and no external
   validation is shown (EHRSHOT and INSPECT are pending DUAs). The SMB head-to-head on INSPECT is
   promised but absent.
4. **Label validity on MIMIC.** MIMIC-IV has little outpatient coverage. `new_dx_365d/*` and
   `inpatient_365d` positives require a *return visit to BIDMC*, so the labels mix disease with
   utilization and loss to follow-up (informative censoring). Out-of-hospital death is captured
   only ~1 year after discharge. Restrict to subjects with observed follow-up or report censoring.
5. **Clinical relevance of +0.008–0.013 AUROC.** No AUPRC, calibration, decision curves or
   subgroup analysis (age, sex, race, insurance). Heart failure and diabetes show no gain over
   GBM.
6. **"Trial reproduction" on MIMIC is not credible as framed.** MIMIC is ICU/ED-centric with
   short stays. Most landmark ICU RCTs have 28/90-day mortality or ventilator-free-day
   endpoints, many effects are null or small, and those endpoints sit far beyond a 16h
   predictor's reach. Ventilator settings, doses and timing are partly in chartevents, not in
   the action vocabulary (7.5% of events). Notes, which carry the reasons for treatment, are
   excluded.
7. Releasing weights trained on MIMIC requires PhysioNet credentialed hosting, so "public"
   needs that caveat. Membership-inference and memorization checks are absent.

**Questions**
- Per-task results on the full held-out split with all 3 seeds?
- How does the model do with 100% labels? GBM gained +0.033 with 3× labels. Does the FM
  advantage vanish in the full-label regime?
- What does CLMBR-T or ETHOS score on the same ACES tasks?
- Which RCTs, with what eligibility, time zero, strategies and outcome windows?

**Score: 4 (borderline reject).** Confidence 4.

---

## Reviewer 3: Causal inference

**Summary.** A latent predictor f(s_t, a_{t:t+k}) is trained on observational ICU data, with a
treatment-blind control. The authors propose counterfactual rollouts checked against RCT
effects through target-trial emulation.

**Strengths**
1. Including an action-blind control and a permuted-action sensitivity test
   (`state_sensitivity.py`) is the right instinct.
2. Stating RCT anchoring as the eventual falsifier is better than unvalidated "what-if"
   demos.

**Weaknesses (ranked)**
1. **Confounding by indication makes the conditional model non-causal.** f learns
   E[s_{t+k} | s_t, a]. That is not E[s_{t+k} | s_t, do(a)] unless s_t blocks every back-door
   path (sequential exchangeability). Clinicians act on bedside findings, notes and trajectory
   cues that are not in the coded stream. Nothing in a JEPA loss makes s_t a sufficient
   adjustment set, and with EMA targets it is not even identified. Vasopressors will "predict"
   worse states because sicker patients get them. The paper must not use the word
   counterfactual for this object.
2. **Post-treatment and future-information leakage.** Actions in windows t..t+k−1 are partly
   *responses* to the intermediate states the model is asked to predict (vasopressor started at
   t+1 because MAP fell at t+1). So state_ac > state_blind and a positive permuted-action
   win-rate show that actions are **informative proxies of the future**, not that the model
   learned treatment effects. The treatment-blind control cannot separate these. Even
   within-window ordering is lost by 4h pooling: an order and the lab that prompted it fall in
   the same window.
3. **Time-varying confounding over k > 1.** Treatment–confounder feedback across windows needs
   g-methods (g-computation over the intermediate states, MSMs/IPW, or LTMLE). A one-shot
   predictor on mean-pooled action sequences is neither. Rollouts would need a validated
   transition model at every step plus an outcome model, and neither exists.
4. **Positivity.** The counterfactual action sequences in the planned queries will often be
   off-support (e.g. withholding standard care in shock). No overlap or propensity diagnostics
   are planned.
5. **RCT anchoring has low power to falsify.** Agreement with null RCTs is uninformative, and
   RCT CIs are wide. Without a pre-registered trial set, an agreement metric (sign / CI overlap /
   bias against the RCT), a comparison to standard TTE estimators (IPW/AIPW on the same
   emulation) and negative controls, any match is anecdotal. RCT-DUPLICATE-style benchmarks
   show that even well-designed TTEs disagree with RCTs in a sizeable fraction of cases. The
   question is whether *the model* adds anything over a classical emulation.
6. Selection: ICU admission itself is conditioned on, and is a collider for, the illness
   severity and treatment pathways.

**Questions**
- Does the AC advantage survive when actions are restricted to those time-stamped strictly
  before every observation in the target path (k=1, lagged actions)?
- Do pre-specified negative-control exposures (e.g. stress-ulcer prophylaxis → unrelated lab)
  or negative-control outcomes show an "effect"?
- How does model-based g-computation compare with AIPW on the same emulated trial?

**Score: 2 (strong reject) if counterfactual or RCT claims are made; 4 if the world model is
framed purely as predictive.** Confidence 5.

---

## Area chair meta-review

**Consensus.** The engineering and evaluation hygiene are above average, and the negative
results are honest. But the paper's three pillars are each under-supported.

1. Hybrid > AR: +0.008, one cell significant, no multiplicity control, 1,346-subject test set.
2. Probe vs fine-tune: known in vision and not systematically studied here.
3. Action-conditioned world model / RCT benchmark: no results yet. Already occupied by
   Clin-JEPA (action-conditioned on MIMIC-IV), SMB (hybrid world model) and medDreamer / latent
   SDE work. Causal claims are unidentified under confounding by indication.

There are also no published-FM baselines and no second site. Scores are 3 / 4 / 2(→4), so the
average is ~3.

**Likely decision today: Reject** (NeurIPS/ICLR main track). It is a credible accept at an ML4H
/ TS4H / NeurIPS workshop as a rigorous "probe vs fine-tune audit of EHR-SSL objectives on
public data".

**Ranked fixes (largest score gain per cost first)**

1. **Full held-out evaluation with a pre-registered primary endpoint.** Score all cells on the
   ~36k full-MIMIC held-out split: hybrid/AR/MLM × 3 seeds at full pretraining, plus GBM/LR.
   Use the paired subject bootstrap with Holm correction over the declared comparisons, and
   report AUPRC and calibration. Add label budgets of 1% / 8% / 100% to show where the FM gain
   lives. *Cost:* ~2 extra seeds × (3.6h hybrid + 2.7h AR + ~2.5h MLM) ≈ 18h GPU, plus ~1 day
   of eval and CPU bootstrap. Highest return: it settles "≈GBM" vs "> GBM" either way.
2. **Published-FM and stronger-baseline comparison on identical ACES tasks.** CEHR-GPT or
   CEHR-BERT and ETHOS trained in their own codebases on the same MEDS split; CLMBR-T-base /
   MOTOR applied via MIMIC-OMOP; a GBM with time-binned count features and 50-trial Optuna
   instead of the 3-config grid. *Cost:* ~1 week engineering, ~2–3 days GPU.
3. **Strip or rebuild the causal claims.** Cheap path (recommended for this submission): call
   it an "action-conditioned predictive state model", drop "counterfactual" and "trial", and
   add the leakage controls below. Full path (a separate paper): 6–10 pre-registered MIMIC TTEs
   with in-hospital or 28-day outcomes (include nulls); per trial, model g-computation vs
   AIPW/IPW vs the RCT, with negative-control exposures and outcomes, positivity diagnostics
   and E-values. *Cost:* cheap path ~2 days; full path ~2–3 months plus clinical co-author.
4. **Action-leakage controls for state_ac vs state_blind.** (a) Lagged actions only (actions
   at t, predict t+1, with all actions time-stamped before the target window's first
   observation). (b) Scheduled vs reactive action strata, e.g. prophylactic heparin vs
   vasopressor start. (c) An action-only predictor with no s_t. (d) Within-patient shuffled
   actions vs cross-patient permutation. If the AC gain disappears under (a), say so. *Cost:*
   ~1 day, mostly eval on existing checkpoints.
5. **Match the evaluation to the model's time scale.** Add short-horizon ICU tasks
   (decompensation / ICU mortality 24–48h, vasopressor or ventilation onset 24h, AKI 48h;
   `icu_tasks.py` already has sepsis-6h and mortality) plus rollout error at 4–48h against
   Clin-JEPA's degradation metric. A 4h state model judged only on 365-day outcomes will look
   weak for the wrong reason. *Cost:* ~2–3 days.
6. **Fair-objective protocol.** Compute-matched training (report FLOPs incl. EMA/predictor),
   an equal tuning budget per objective (3-point LR sweep each), and an MLM mask-rate sweep
   (15/30/40%) with mean and last pooling. *Cost:* ~3 days GPU.
7. **Turn probe-vs-FT into a measured result.** For ≥5 objectives (AR, MLM, hybrid, pure JEPA,
   state_ac) × 3 seeds × {linear probe, MLP probe, k-shot, full FT}, report Kendall τ between
   readouts. Cite MAE as precedent and claim only the EHR-specific magnitude. *Cost:* ~1–2 days
   eval on existing checkpoints.
8. **Second site plus scale.** Transfer the MIMIC-pretrained model to EHRSHOT or INSPECT vs
   CLMBR-T / SMB weights, and finish the 35M run plus one ~100M point (rented A100, ~$150–300)
   to show whether the gap holds with scale. *Cost:* DUA lead time plus ~1 week.

Fixes 1, 2, 4 and 6 plus the cheap path of 3 move the paper to roughly 5 (borderline). Adding
5 and 8 with a clear AC > blind result under lagged actions is the route to 6+. The full
RCT-anchored benchmark is its own paper and should not be promised in this one.
