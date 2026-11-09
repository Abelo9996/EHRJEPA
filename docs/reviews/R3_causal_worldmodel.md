# R3 — Causal inference / world-model review

Paper: "Latent Prediction Helps EHR Foundation Models Only Next to a Token Loss, and Only Fine-Tuning Shows It: A Careful Public-Data Study and an Action-Conditioned Patient-State Model" (docs/paper/v2/main.tex, Oct 2026 draft).
Also reviewed: docs/TRIAL_PILOT_SPEC.md, docs/NEXT_STEPS.md, and the code they rely on (`src/ehrjepa/models/state.py`, `src/ehrjepa/data/actions.py`, `src/ehrjepa/eval/state_sensitivity.py`, `configs/grids/mimic_state_gate.yaml`, the vendored `meds_etl/mimic/__init__.py` timestamp map).
Reviewer stance: world models and causal inference from observational health data. Placeholders count as missing.

Calibration set:
- **Causal Transformer** (Melnychuk et al., ICML 2022, 2204.07258). Counterfactual outcomes over time. Validated on a synthetic tumour-growth simulator and semi-synthetic MIMIC-III, where ground-truth counterfactuals exist. Uses explicit balancing (counterfactual domain confusion) against time-varying confounding. Accepted because the causal claim is checked against known counterfactuals.
- **V-JEPA 2-AC** (2506.09985). Action-conditioned latent predictor. Actions are exogenous robot commands, and the evaluation is interventional (closed-loop planning on real robots). Confounding by indication does not arise.
- **RCT-DUPLICATE** (Wang et al., JAMA 2023; Franklin et al., Circulation 2021). 32 trials emulated in claims data. Protocols were pre-registered on ClinicalTrials.gov before any outcome was looked at. Agreement metrics were fixed in advance (regulatory, estimate, standardized difference). The paper shows agreement depends on how closely the design emulates the trial.

This submission currently meets none of these bars for its world-model component. Its non-causal half (probe vs fine-tune, hybrid vs AR) is a reasonable workshop-level empirical note.

---

## 1. Summary

The paper trains 10M-parameter causal transformers from scratch on coded MIMIC-IV MEDS events.
- Latent-only objectives trail next-code AR by 0.06–0.09 mean AUROC under a frozen probe.
- A next-latent auxiliary term on top of AR (the "hybrid") beats AR under fine-tuning by +0.008 [0.002, 0.014], with 25% pool and 3 seeds.
- The frozen probe ranks the hybrid below XGBoost on counts, while fine-tuning ranks it above. The margin is +0.013 [−0.001, 0.028], not significant.
- The paper then proposes an action-conditioned patient-state objective, `state_ac`: 4h windows, a direct k-step MLP predictor from the window state plus a mean embedding of the intervening treatment codes, and a treatment-blind twin as the control.
- **Every result for this objective is a placeholder** (abstract L63–64, Table 5, L539–541, L577, L625).
- A trial-anchored benchmark is described as future work (L579–588). A detailed pilot spec (TRIAL_PILOT_SPEC.md) plans to use the state model as a g-computation outcome model and to score it against ALBIOS/SAFE, SMART/PLUS/BaSICS and TRICC/TRISS.

## 2. Strengths

1. **Honest scoping in the text.** The paper repeatedly says the state model is "predictive, not causal" (L124, L573–576, L608–611). It does not claim the hybrid beats GBM. The pass rule and the fallback are fixed before any numbers are read (NEXT_STEPS §Step 2). This is unusual and creditable.
2. **Evaluation hygiene beats the EHR-JEPA prior work.** The paper uses 3 seeds, a paired subject-level bootstrap that is honest about the shared test set (L366–370), a GBM audited for leakage, and a single choke point for the anchor cut-off with a death-insertion test.
3. **The probe-vs-fine-tune re-ranking is a useful, falsifiable methodological point** (Table 2). All three cited EHR-JEPA papers are probe-only.
4. **The treatment-blind twin is the right minimal ablation in principle.** The pilot spec is unusually self-aware:
   - it gates the readout at R² ≥ 0.3;
   - it uses negative-control outcomes;
   - it mandates an AIPW reference;
   - it calls masking the window-t0 actions "required";
   - it freezes the success criteria in git.

## 3. Weaknesses (ranked)

**W1 (fatal for the world-model contribution): there are no results.** The contribution titled "Action-Conditioned Patient-State Model" has zero numbers. Table 5 is red placeholders, and the abstract's two key sentences are TODOs.
- Even once filled in, the planned gate measures **365-day downstream AUROC** (mean-of-6). That tests the state loss as an auxiliary pretraining signal. It does not test the model as a world model: there is no rollout error, no calibration of predicted state, and no interventional or counterfactual check.
- V-JEPA 2-AC is accepted because of planning success. Causal Transformer is accepted because of counterfactual RMSE on simulators. This paper offers neither.

**W2 (fatal for any causal reading, including the planned pilot): the action-sensitivity diagnostic and the `state_ac` vs `state_blind` contrast are confounded by construction.** There are two separate mechanisms, either of which is enough.
- *(a) Representational echo.* The target `s̃_{t+k}` is the EMA causal encoder's output at the last observation in window t+k. That encoder attends over the whole stream, including the action tokens in windows t..t+k−1 (`state.py:214`, `:225`; `jepa.py:600–602` runs the target stack on all tokens). So the target literally contains the action codes that the predictor is given. A predictor that learns "the target remembers that albumin was given" lowers its loss without learning any physiology.
  - The blind twin cannot see those codes, so it loses for a trivial reason.
  - `loss_perm > loss_true` (`state_sensitivity.py`) is therefore expected even when the treatment has zero effect on the patient.
  - Gate criterion 2 and pilot criterion S5 cannot separate dynamics from echo.
- *(b) Actions are proxies for the future state (confounding by indication inside the predictor input).* For k ≥ 2, `a_{t..t+k−1}` includes treatments given in windows t+1..t+k−1. Clinicians chose those in response to observations in those windows, which lie on the path to the target.
  - Examples: a transfusion in window t+1 signals that Hb dropped in t+1; insulin signals hyperglycaemia; vasopressor escalation signals shock.
  - The count term `log1p|A|` (`state.py:285`, `:309`) is a direct severity and charting-intensity proxy.
  - Δt (`state.py:290`), on by default (`train/config.py:86`), is the time to the *first event of the future window*. That leaks the future measurement process, which in the ICU signals acuity.
  - The paper's sentence "conditioning on treatments separates the two" (L572–573) is wrong. Conditioning on post-baseline, state-responsive treatments mixes the two.
  - This is time-varying confounding plus mediator conditioning in its textbook form, the exact problem that CRN and Causal Transformer exist to address. The paper cites both (L174–175) but neither balances nor weights.

**W3 (major, code–paper mismatch): "k ∈ {1,2,4} windows (4, 8 and 16 hours ahead)" is false.** `assign_windows` (`actions.py:156–163`) numbers only **non-empty** windows: a boundary appears only where the calendar key changes between consecutive *events*. So k=1 means "the next 4h block that has any event", and that can be days or months later in outpatient or between-admission stretches.
- The same false claim appears in the paper (L309), the grid header (`mimic_state_gate.yaml`), and the pilot spec (§0, which builds its whole design on "16h"). The spec's rollouts then force Δt = 4k h. Those values are in-support but are not what the model's k slots were trained to mean.
- The "94% of crops have a k=1 pair" statistic (L289) is inflated by this compression.

**W4 (major): timestamps in the MEDS ETL make "pre-treatment state" and action order unreliable at 4h resolution.** These come from the vendored `meds_etl/mimic/__init__.py`. None are discussed in the paper.
- `hosp/procedures_icd` (L304–306) and `hosp/hcpcsevents` (L227–229) are stamped at **chartdate**, which is midnight. A procedure done at 15:00 lands in the 00:00–04:00 window. These are billing codes, assigned after discharge. They are a large part of the action set (`actions.py:58–64`: `ICD9Proc/`, `ICD10PCS/`, `HCPCS/`). The result is a *future action placed before the observations it responded to*, inside the causal encoder's view of s_t.
- `icu/inputevents`, `procedureevents` and `ingredientevents` are stamped at **storetime** (L88–90, L111–112, L125–126), not starttime. Charting lag puts an infusion after the labs it already changed. The `start`/`end` metadata are dropped by the canonicaliser, so dose duration is lost. The value is `rate`, not the amount given.
- Labs are stamped at storetime, i.e. result time (L237). A specimen drawn at 03:00 and resulted at 05:00 falls in the next window.
- `MEDS_DEATH` is stamped at `dod`, a date (midnight; L281). On the death day the death token precedes every action and observation. Any window-level readout trained on target-stack states (pilot §2.2 step 4, 28-day death) is trivially informed for those windows.
- Net effect: at a 4h grain, which of cause and effect comes first in the token stream is partly arbitrary. That is fatal for a treatment → next-state model and for the pilot's "treatment = arm codes in window t0".

**W5 (major): the trial pilot's world-model estimator is not identified, as specified.**
- `s_t0` is computed with window-t0 actions masked (spec §2.2.1). The model never saw that input distribution in training, because actions were always in the stream (`state.py:212`).
- The forced action is a single code averaged into a mean embedding with the patient's other actions (`state.py:288`). The "effect" is then a function of how many concurrent actions the patient has, i.e. of severity.
- The readout heads `g_y` are trained on observational target states. Those states echo past treatments (W2a), so `g_y` learns associations such as "albumin token in history ↔ low measured albumin" (indication). The world-model ATE can then have the *wrong sign* for reasons unrelated to dynamics.
- Exchangeability given `s_t0` is asserted to be "tested empirically". But the only test proposed is agreement with AIPW on the same cohort. Two estimators sharing the same unmeasured confounders agree for the wrong reasons, so agreement is not validation.
- Doutreligne et al. show that MIMIC albumin emulations are sensitive to these exact choices.

**W6 (major): the trial-anchored benchmark is future work, yet it is the stated novelty hook** (L126–128, L579–588; NEXT_STEPS step 3–4). Nothing is pre-registered externally (no OSF or ClinicalTrials.gov-style registration; spec "frozen in git" is self-hosted and amendable).
- Every RCT number is tagged [M] (memory) and every itemid [U] (unverified).
- RCT-DUPLICATE spent years on 32 emulations. Three emulations done in ~4 days of engineering would not be credible at NeurIPS without far more design rigour.

**W7 (moderate): the headline non-causal effect is small and rests on a narrow base.**
- +0.008 AUROC on 1,346 subjects from one site, with 10M models.
- Full-pool numbers are single-seed (Tables 2–4).
- The comparison against "frozen probe" mixes encoder quality with readout capacity. The paper half-admits this ("the 256-d readout was the bottleneck", L457) but does not control it, e.g. with an MLP probe or a probe on concatenated layers.
- There is no released-model baseline (CLMBR-T, MOTOR).
- The "probe hides the gain" claim is a well-known phenomenon in vision SSL (the linear-probe vs fine-tune discrepancy in MAE, He et al. 2022). It is new only to EHR-JEPA.

**W8 (moderate): the novelty position is thin.**
- Clin-JEPA already trains an action-conditioned latent world model on MIMIC-IV ICU state–action pairs.
- SMB-Structure already proposed the hybrid.
- Gupta et al. proposed the patient-state target.
- The paper's distinctives (code tokens, small model, treatment-blind twin) are engineering variants. The trial benchmark, which could be distinctive, is absent.

**W9 (minor):**
- The paper states MIMIC-IV v3.1 (L185), but the ETL hard-codes the `2.2` layout (`etl/mimic.py:31`). The reader needs confirmation that v3.1 tables were really used and the symlink only renamed the directory.
- The tie order within a timestamp is alphabetical by code string (`canonical.py:266`). This decides which token is "last observation" and whether an action precedes it.
- The 3,038 vs 1,346 held-out subject counts are inconsistent across documents (the authors already flag this).

## 4. Code-level issues (file:line)

| # | Location | Issue | Consequence |
|---|---|---|---|
| C1 | `src/ehrjepa/models/state.py:214,225` + `models/jepa.py:600–602` | The target state is pooled from the EMA causal encoder over **all** tokens, actions included | Target echoes the given actions. The action gain and the blind-twin gap are trivially positive (W2a). Fix: run the target stack on observation tokens only (mask actions in `window_targets`), or use observation-only targets such as next-window lab values or z-scores |
| C2 | `src/ehrjepa/models/state.py:282–288` | `a_{t..t+k−1}` includes post-baseline actions in windows t+1..t+k−1 | Actions respond to intermediate observations, so they leak the future state (W2b). For a causal reading use k=1 only, or condition only on the action plan fixed at t |
| C3 | `src/ehrjepa/models/state.py:285,309` | `log1p(count)` of actions is a predictor input | Severity and charting-intensity proxy. Ablate it, or replace it with the plan-level indicator |
| C4 | `src/ehrjepa/models/state.py:290`; `train/config.py:86` (`mask_token_time: True`) | Δt = hours to the **first event of window t+k** | Leaks the future measurement process. In counterfactual use it is itself post-treatment |
| C5 | `src/ehrjepa/data/actions.py:156–163` | Windows are numbered over non-empty windows only | k ≠ 4k hours. The paper L309, the grid header and the pilot §0 are wrong (W3) |
| C6 | `src/ehrjepa/models/state.py:212,220` | The online encoder sees window-t actions before the last observation of window t, so s_t already encodes part of `a_t` | During training the predictor gets the action twice. At pilot time (masked t0 actions) the input is out of distribution |
| C7 | vendored `meds_etl/mimic/__init__.py:304–306, 227–229` | `procedures_icd` and `hcpcsevents` are stamped at `chartdate` (midnight) | Future billed procedures sit in the causal past of same-day windows (W4) |
| C8 | vendored `meds_etl/mimic/__init__.py:88–90, 111–126` | ICU actions are stamped at `storetime`. `start`/`end` metadata are not carried into the canonical cache | Treatment timing can be off by hours, and the order of cause and effect can invert at 4h |
| C9 | vendored `meds_etl/mimic/__init__.py:281` | `MEDS_DEATH` is stamped at `dod` (a date) | On the death day, death precedes that day's events. Any window-level 28d-death readout (pilot §2.2) is contaminated |
| C10 | `src/ehrjepa/models/state.py:345–355`, `eval/state_sensitivity.py` | The permuted-action contrast swaps in another patient's actions | Tests "actions are informative about the target", which C1–C3 guarantee. It does not test "actions act on physiology". The needed test is within-patient: swap two plausible actions for the same s_t and compare against known pharmacology (e.g. saline → chloride) |
| C11 | `src/ehrjepa/models/state.py:288` | Mean-pooled action embedding | One forced arm code is diluted by co-administered actions. The ATE then depends on co-treatment count, which tracks severity |
| C12 | `src/ehrjepa/data/canonical.py:266` | Ties are sorted by code string | Arbitrary action/observation order at identical timestamps. It affects `last`-obs pooling and leakage of t0 actions |

Note: the anchor choke point and the death-insertion test for the *downstream tasks* look sound. I found no label leakage in the 365-day tasks beyond sub-day `chartdate` effects, which are negligible at that horizon. The leakage problems above affect the state objective's causal interpretation and the pilot, not Tables 1–4.

## 5. Questions for the authors

1. With the target stack run on observation tokens only (C1), does `state_ac` still beat `state_blind` in held-out state loss? This is the single most informative experiment, and it costs about 2 × 7.5 GPU-h.
2. What is the distribution of real elapsed hours for k=1,2,4 under `assign_windows` on full MIMIC (C5)? What fraction of k=1 pairs span more than 24h?
3. Restricted to k=1 and with only window-t actions (no future windows), does action sensitivity survive?
4. Does the world model reproduce **mechanistic** effects on a held-out cohort *before* any trial is attempted? Examples: saline → chloride ↑ within 4–8h; PRBC → Hb ↑ about 1 g/dL/unit; insulin → glucose ↓; furosemide → urine output ↑. Each needs a within-patient action swap and a comparison with AIPW on the observed outcome.
5. Why is the gate defined by 365-day downstream AUROC rather than by any world-model property (rollout error, counterfactual agreement on a simulator)?
6. Was the v3.1 release really processed through a 2.2-named layout? Were `chartdate`-stamped codes shifted to end of day, or left at midnight?
7. For the pilot: the positivity diagnostics (propensity overlap, effective sample size after trimming) and the E-value or sensitivity analysis to unmeasured confounding are not specified. Will they be?

## 6. Score

**3 / 10 (reject), confidence 4/5.**
The non-causal half (hybrid > AR under fine-tuning, probe vs fine-tune) is honest but incremental and single-site, which would be about 4 on its own.
- The world-model half contributes no results.
- Its planned diagnostics are confounded by construction (C1–C3).
- Its stated horizon is wrong (C5).
- Its timestamping cannot support 4h causal ordering (C7–C9).

Relative to the calibration set, the paper has no simulator ground truth (Causal Transformer), no interventional evaluation (V-JEPA 2-AC), and no pre-registered emulation protocol (RCT-DUPLICATE).

## 7. What a credible world-model + trial-reproduction contribution must include (with costs on one RTX 4060 + M4)

| # | Item | Why | GPU-h | Human / eng |
|---|---|---|---|---|
| 1 | **Fix the objective**: observation-only target stack (C1), k=1 plus window-t actions only (or a plan fixed at t), drop the count and Δt-to-future-event inputs (C3–C4), real-hour horizons (fixed-calendar k, empty windows kept) (C5) | Without this, nothing about "actions" is interpretable | ~15 (2 cells × 7.5) + ~15 for seed 2 | 1–2 days code + tests (target invariant to action tokens; horizon in hours) |
| 2 | **Fix timestamps**: chartdate codes moved to end of day (or dropped from the action set); ICU actions at `starttime` from metadata; labs at `charttime`, with a result-availability lag kept as a feature; `dod` → end of day or `deathtime` | 4h causal order requires it | 0 GPU; cache rebuild ~3–6 CPU-h; retrains fold into item 1 | 1–2 days |
| 3 | **Semi-synthetic ground truth before real trials**: inject a known treatment → lab effect with confounded assignment into real MIMIC trajectories (Causal Transformer / CRN style) and show the world model recovers it while naive does not | The only place a world model's causal claim is checkable | ~10 | 3–4 days |
| 4 | **Mechanistic positive controls** (saline → Cl⁻, PRBC → Hb, insulin → glucose, furosemide → UO) via within-patient action swaps, with AIPW on observed outcomes as reference | Cheap known-direction checks; replaces the cross-patient permutation diagnostic | <2 | 3 days (cohorts, itemids verified) |
| 5 | **Externally pre-registered emulation protocols** (OSF, timestamped, before any outcome look): eligibility, t0, strategies, estimand, analysis, agreement metrics in RCT-DUPLICATE style (regulatory, estimate, standardized difference) | Matches the RCT-DUPLICATE bar; stops forking paths | 0 | 3–5 days for 3 trials, ideally with a clinician co-author |
| 6 | **DR baselines on identical cohorts**: IPW, cross-fit AIPW/TMLE with GBM nuisances, plus DR-with-latent-features; Doutreligne's forest-AIPW re-run for albumin | The world model must be shown to add something over standard TTE | 0 (CPU, a few hours per arm) | 2–3 days |
| 7 | **Positivity / overlap**: propensity histograms by arm, ESS after trimming, overlap-restricted re-estimates, and the AR-head propensity AUROC | Without it, AIPW agreement means nothing | 0 | 0.5 day |
| 8 | **Negative controls**: negative-control outcomes (as specified in the pilot) **and** negative-control exposures (e.g. an unrelated, similarly timed drug class), plus E-values | Detects residual confounding by indication | 0 | 1 day |
| 9 | **Uncertainty**: subject-level bootstrap including nuisance refits, plus ≥3 pretraining seeds | Seed variance of the world model is part of the estimator's error | (rollouts are minutes) | — |

Total for a credible minimal version: about 45–55 GPU-h and 3–4 weeks of one person's engineering. That is feasible by late January only if the trial component is cut to items 1–4 plus one pre-registered emulation (albumin, because a reference implementation exists), with SMART/TRICC deferred. A three-trial "benchmark" is not achievable credibly on this timeline.
