# Trial-reproduction pilot: spec

Status: design only, nothing implemented. Written 2026-10-30.
Model under test: the `objective.kind: state` world model (`src/ehrjepa/models/state.py`). It uses 4h calendar
windows, `state_horizons: [1, 2, 4]` (4h/8h/16h ahead), last-observation pooling, and an EMA target stack.
Actions are defined by `src/ehrjepa/data/actions.py` and `configs/actions/mimic_icu_action_items.txt`, and
the AR next-code head is kept with `lambda_recon: 0.1`. Checkpoints come from the `mimic-state-gate` grid
(`configs/grids/mimic_state_gate.yaml`, cells `state_ac_s{0,1}` and `state_blind_s{0,1}`).

Question this pilot answers: when actions are forced in the world model, do the resulting effect estimates
(a) recover effects that RCTs established, (b) recover RCT nulls, and (c) agree with a standard
doubly-robust (DR) estimator on the same emulated cohort? Every success criterion is fixed below,
before any number is computed.

Tags used below: **[V]** means checked against the cited page this session. **[M]** means taken from
memory of the primary paper and must be checked against the paper before it goes into the paper.
**[U]** means the item is unverified and must be checked against `d_items` / `d_labitems`.

---

## 0. A constraint that shapes everything

The predictor is a direct k-step MLP, `ŝ_{t+k} = f(s_t, mean-emb(a_{t..t+k-1}), log1p|a|, k, Δt)`.
Its longest horizon is k=4, which is **16h**. Two consequences follow:

1. **Primary pilot estimands are short-horizon physiological effects** that RCTs documented
   (chloride/bicarbonate, haemoglobin, serum albumin, MAP), read out 4–16h after time zero.
   These are the "non-null" tests, and the RCT tells us the direction for certain.
2. **Mortality cannot be rolled out.** Feeding ŝ back into f recursively takes the model out of
   distribution, because f was never trained on its own outputs. Mortality enters only as a
   *secondary* estimand: a frozen outcome readout (28-day death) applied to `ŝ_{t+4}`. That is a
   surrogate-at-16h estimand, **not** the RCT estimand. The trials' mortality nulls serve as the
   "null" tests: in each chosen trial, the classic observational failure is a spurious harm signal
   from confounding by indication.

The AR next-code head is **not** used for effect readout. It reads token-position hidden states, not
window states, so it cannot be applied to ŝ. Its only role is as a sanity check: next-token
likelihood of the arm codes at time zero acts as a propensity-like signal (§2.4).

---

## 1. Candidate RCTs

Code families follow `meds_etl.mimic` as used in this repo. Infusions are
`MIMIC_IV_ITEM/<itemid>/<ordercategorydescription>` (icu/inputevents, always an action). Labs are
`MIMIC_IV_LABITEM/<itemid>`. Charted vitals are bare `MIMIC_IV_ITEM/<itemid>` (observation).
Every itemid below is **[U]**.

| # | Trial(s) | Intervention vs comparator (MEDS codes) | Population / eligibility | Outcome, horizon | RCT effect (95% CI) | MIMIC emulation | Feasibility |
|---|---|---|---|---|---|---|---|
| 1 | **ALBIOS** (Caironi, NEJM 2014), **SAFE** (Finfer, NEJM 2004) | albumin 5%/25% (`MIMIC_IV_ITEM/220864/*`, `/220862/*` [U]) + crystalloid vs crystalloid only | Sepsis-3, adult, first ICU stay, ≥24h follow-up; treatment in first 24h | 28d / 90d mortality. Short-horizon: serum albumin ↑, MAP ↑, net fluid balance ↓ (ALBIOS days 1–7 [M]) | ALBIOS 90d RR 0.94 (0.85–1.05) [M]. SAFE 28d RR 0.99 (0.91–1.09) [M]. RCT meta-analysis cited by the emulation: 28d OR 0.93 (0.80–1.08) [V] | **Yes.** Doutreligne et al., PLOS Digit Health 2025, e0000721 [V], code at github.com/soda-inria/causal_ehr_mimic. IPW, AIPW, DML and T-learner were all run. Forest-AIPW came closest to the RCT null [V] | **3,559 albumin vs 14,862 crystalloid-only** (Table 2, n=18,421; the text says 18,121, an inconsistency in the paper) [V] |
| 2 | **SMART** (Semler, NEJM 2018); **BaSICS** 2021, **PLUS** 2022 | lactated Ringer's / Plasma-Lyte (`MIMIC_IV_ITEM/225828/*`, Plasma-Lyte itemid [U]) vs NaCl 0.9% (`MIMIC_IV_ITEM/225158/*` [U]) | Adult ICU patients receiving ≥ X mL crystalloid in a window; sepsis subgroup secondary | MAKE30; 30d mortality. Short-horizon: chloride ↑ and bicarbonate ↓ under saline | MAKE30 OR 0.91 (0.84–0.99); 30d mortality 10.3 vs 11.1%, OR 0.90 (0.80–1.01); hyperchloraemia 24.5 vs 35.6% [V, secondary sources]. PLUS 90d RR 0.99 (0.90–1.10) [M]. BaSICS 90d HR 0.97 (0.90–1.05) [M] | None found by search (one query, so low recall) | Both fluids are near-ubiquitous in MIMIC ICU, so a large cohort is expected; no published count |
| 3 | **TRICC** (Hébert, NEJM 1999), **TRISS** (Holst, NEJM 2014) | PRBC transfusion (`MIMIC_IV_ITEM/225168/*` [U]) vs no transfusion, in a window with Hb 7–9 g/dL | Adult ICU, Hb in [7,9] g/dL (`MIMIC_IV_LABITEM/51222` [U]), no active haemorrhage / cardiac surgery exclusions | 30d / 90d mortality. Short-horizon: Hb ↑ ≈ 1 g/dL per unit [M] | TRICC 30d 18.7 vs 23.3%, p=0.11 [M]. TRISS 90d RR 0.94 (0.78–1.09) [M]. Sepsis RCT meta-analysis 28–30d OR 0.99 (0.67–1.46) [V] | **Yes (preprint).** MIMIC-IV TTE of restrictive vs liberal strategy in sepsis: 4,013 stays, ICU-acquired infection cOR 0.954 (0.797–1.142) [V, abstract only], doi 10.64898/2026.08.14.26360449 | 4,013 stays (sepsis, strategy-level) [V] |
| 4 | **VASST** (Russell, NEJM 2008), **VANISH** (Gordon, JAMA 2016) | vasopressin (`MIMIC_IV_ITEM/222315/*` [U]) added early vs norepinephrine alone (`/221906/*` [U]) | Septic shock on NE; time zero = shock onset | 28d mortality; NE dose. Short-horizon: NE requirement ↓ | VASST 35.4 vs 39.3%, p=0.26 [M]. VANISH: kidney-failure-free days null [M] | Partial. MIMIC-IV cohort with early AVP (n=531) found no mortality difference [V]. Eastwood et al. (Crit Care 2025) g-formula TTE gave RR 0.95 (0.93–0.98) [V]; whether it used MIMIC-IV is unconfirmed | 531 (one MIMIC-IV study) [V] |
| 5 | **ADRENAL** (Venkatesh, NEJM 2018), **APROCCHSS** (Annane, NEJM 2018) | hydrocortisone (`NDC/*` / `MIMIC_IV_Drug/*` hydrocortisone family [U]) vs none | Septic shock on vasopressors | 90d mortality. Short-horizon: glucose ↑, faster shock reversal | ADRENAL 90d OR 0.95 (0.82–1.10) [M]. APROCCHSS 90d RR 0.88 (0.78–0.99) [M] | Yes: MIMIC-IV v3.1 TTE in refractory shock reports benefit [V, effect size not retrieved]. A broader MIMIC-IV sepsis cohort reports HR 1.35 (1.24–1.47) harm [V], a clear confounding-by-indication example | 1,802 vs 6,348 (broad sepsis cohort) [V] |
| 6 | **STARRT-AKI** (NEJM 2020) | RRT start (CRRT itemids in procedureevents [U]) accelerated vs standard | KDIGO stage 2–3 AKI, no urgent indication | 90d mortality | 43.9 vs 43.7%, RR 1.00 (0.93–1.09) [M]. Accelerated median 6.1h vs standard 31.1h [V] | Yes: She et al., Front Med 2025 (10.3389/fmed.2025.1645046) [V] | 7,607 (stage ≥1), 943 (stage 3) [V] |
| 7 | **ICU-ROX** 2020, **OXYGEN-ICU** (Girardis, JAMA 2016) | conservative vs liberal SpO2/FiO2 target | Mechanically ventilated adults | Ventilator-free days; ICU mortality | ICU-ROX null [M]. OXYGEN-ICU 11.6 vs 20.2% [M] | No TTE found. Observational SpO2 94–98% optimum (Chest 2020) [V]. Yarnell et al. TTE is about ventilation *thresholds*, not O2 targets [V] | **Infeasible as specified:** FiO2/O2 flow are charted (observation) codes in this repo, so they are not predictor inputs |

**Rejected for the pilot:**
- #4 (vasopressin) and #5 (hydrocortisone): the RCT ground truth is itself discordant (VASST/VANISH vs the
  g-formula TTE; ADRENAL vs APROCCHSS), so it is unclear which answer the pilot should hit.
- #6 (RRT): it is a *timing strategy* with a 31h standard-arm delay, longer than the 16h horizon.
- #7 (oxygen): the intervention is not an action token.

All four stay as stage-2 candidates.

### Chosen 3

| Pilot arm | Non-null test (short horizon, RCT-documented direction) | Null test (mortality readout) | Why |
|---|---|---|---|
| **P1 Albumin vs crystalloid (ALBIOS/SAFE)** | serum albumin ↑ and MAP ↑ at 4–16h | 28d mortality ≈ null | Only candidate with a published MIMIC-IV emulation **plus open code**, so the cohort and the IPW/AIPW reference come for free and the DR comparison is like-for-like |
| **P2 Balanced vs 0.9% saline (SMART/PLUS/BaSICS)** | chloride ↑ and bicarbonate ↓ under saline at 4–16h (mechanistic, large) | mortality ≈ null (PLUS/BaSICS). SMART's MAKE30 OR 0.91 is too small to resolve, so it is reported, not scored | Cleanest action codes, both arms common so positivity is good, and treatment is a near-routine choice (weaker confounding by indication than P1 and P3) |
| **P3 Transfusion at Hb 7–9 (TRICC/TRISS)** | Hb ↑ ≈ 1 g/dL per unit at 4–16h | 30/90d mortality ≈ null | Strongest, best-quantified positive control. Strong confounding by indication (sicker patients get transfused), so it is the hardest null to hit. A MIMIC-IV TTE exists |

Together the three give three large, known-direction short-horizon effects and three mortality nulls.
Confounding strength rises from P2 to P1 to P3. Negative-control outcomes are in §2.5.

---

## 2. Estimation protocol

### 2.1 Target-trial emulation (shared by all estimators)

For each pilot arm, write the protocol table *before* touching outcomes: eligibility, treatment
strategies, assignment, time zero, follow-up, outcome, and causal contrast (per-protocol "start
within window"). This follows Hernán & Robins target-trial emulation.

- **Time zero = the first 4h calendar window** in which the subject is eligible:
  - P1: first crystalloid in the ICU, sepsis-3 at or before this window (Doutreligne definition; reuse
    their SQL from `causal_ehr_mimic`).
  - P2: first window with ≥500 mL of either crystalloid (threshold [U]; fix it after a cohort-count
    dry run, before any outcome look).
  - P3: first window whose last Hb is in [7,9] g/dL.
- **Treatment = arm codes present in window t0** (the window whose actions enter `a_{t0..}`).
  Comparator = absent in t0. Anything in window t0+1 or later is follow-up, not assignment. This removes
  the immortal-time issue that Doutreligne et al. acknowledge for their 24h window [V].
- **Exclusions:**
  - Subjects with fewer than 2 observation windows before t0, because s_t0 needs history.
  - Subjects with neither arm code in t0 (P2).
  - P3 only: haemorrhage / cardiac-surgery ICD codes.
- **Split:** cohort subjects come from the **held-out test split only** (the same split as the
  probe/fine-tune eval). The world model never trained on them.

### 2.2 World-model estimator (g-computation through the learned dynamics)

For each subject i in the cohort and each arm a ∈ {1,0}:

1. `s_i = s_{t0}`, the encoder state from the observed history up to and including window t0's
   observations. The encoder is causal, so s_{t0} never sees actions or observations after window t0.
   One caveat: window-t0 *actions* earlier in the token stream than window-t0 observations are visible
   to s_{t0}. **Mask window-t0 action tokens out of the encoder input** when computing s_{t0} for
   effect estimation. This is required; without it, the observed treatment leaks into the "pre-treatment"
   state.
2. Forced action set for windows t0..t0+k-1:
   - **A (primary):** the arm code (with the arm's median `value_bin` in the train cohort), plus the
     subject's *observed* non-trial actions in window t0 only. For windows t0+1..t0+k-1 there are no
     extra actions beyond what the count encoding implies. Report the count fixed at the arm's median.
   - **B (sensitivity):** the observed non-trial actions across the full horizon. This is a
     controlled-direct-effect flavour, biased if those actions are mediators.
3. `ŝ_i^a(k) = f(s_i, act(A^a), k, Δt = 4k h)` for k ∈ {1,2,4}.
4. **Readout heads** `g_y` are trained once, frozen, on the train split. They take
   `LN(target-stack pooled state of window w)` as input, which is exactly the space ŝ is regressed into,
   and predict:
   - `y_w` = the last `value_z` of the outcome lab/vital in window w (chloride, bicarbonate, Hb,
     albumin, MAP), with a "measured" mask;
   - for the null test, 28d death from the window-w end time.
   Use ridge or a 1-hidden-layer MLP and report readout R²/AUROC on test windows. **The readout
   must reach R² ≥ 0.3 on observed states**, otherwise that outcome is dropped from the pilot.
5. `ATE_wm(k) = mean_i [ g_y(ŝ_i^1(k)) − g_y(ŝ_i^0(k)) ]`, converted back to native units via the
   tokenizer's per-code mean/std (and log1p if skewed). Get CIs from a subject-level bootstrap
   (2000 resamples, as in the repo's paired bootstrap), and repeat across seeds 0 and 1.

This is plug-in g-computation with the world model as the outcome model. It is unbiased **only** under
exchangeability given s_t0 (no unmeasured confounding), positivity, and correct dynamics. The pilot
tests these assumptions empirically, so it does not assume them.

### 2.3 Reference estimators on the identical cohort (mandatory)

All estimators use the same cohort, the same t0, and the same outcome `y` (the *observed* lab at t0+k,
for subjects where it is measured; IPCW for missingness, with weights from the same GBM features).

| Estimator | Features | Purpose |
|---|---|---|
| Naive difference in means | – | Shows the size of confounding |
| IPW (stabilised, trimmed at [0.01,0.99]) | GBM propensity (the repo's `eval/baselines.py` GBM on count/last-value features up to t0) | Standard |
| **AIPW / DR (primary reference)** | GBM propensity + GBM outcome, 5-fold cross-fitting | Agreement target |
| DR with **world-model features** | propensity + outcome models on s_t0 | Tests whether the latent state is a sufficient adjustment set |
| P1 only: Doutreligne forest-AIPW | Their pipeline, re-run as-is | External replication anchor |

### 2.4 Controls

- **Treatment-blind world model** (`state_blind_s*`): its ATE_wm is 0 by construction, which is the
  floor. Use the blind model's *s_t0 + readout* as the outcome model in a T-learner (fit g on
  treated and untreated separately). Any effect it shows comes from confounding through the state,
  not from the predictor.
- **Permuted-action control:** ATE under arm codes swapped with a random non-trial action of matched
  frequency. The effect must be ≈ 0 relative to the true contrast. This reuses the
  `state_sensitivity.py` permutation logic.
- **AR-head propensity check:** the next-code head's predicted P(arm code | history to t0) vs the
  GBM propensity. A large AUROC means strong indication signal, which is reported, not scored.
- **Positivity check:** a histogram of GBM propensity by arm. Report the fraction trimmed and re-run
  every estimator on the overlap-restricted cohort.

### 2.5 Negative-control outcomes (true effect = 0 by mechanism)

| Arm | Negative-control outcomes at 4–16h |
|---|---|
| P1 albumin | platelet count, INR, glucose |
| P2 saline vs balanced | haemoglobin, platelet count, ALT |
| P3 transfusion | chloride, glucose, ALT |

A non-zero estimate on any of these flags residual confounding or a readout artefact for that estimator.

### 2.6 Success criteria (fixed now)

Per arm, per estimator, at k=4 (16h; k=1,2 are reported only):

- **S1 direction:** the 95% CI of the short-horizon effect excludes 0 *in the RCT/mechanistic
  direction* (albumin ↑ / MAP ↑; chloride ↑ / bicarbonate ↓ under saline; Hb ↑).
  Magnitude check, P3 only: CI overlaps +0.5 to +1.5 g/dL per unit [M].
- **S2 null:** for the mortality readout, the 95% CI includes the RCT point estimate **and** overlaps
  the RCT 95% CI.
- **S3 DR agreement:** |ATE_wm − ATE_AIPW| is smaller than the half-width of the AIPW 95% CI, i.e. the
  bootstrap CI of the difference covers 0.
- **S4 negative controls:** all negative-control CIs include 0 (Bonferroni over outcomes in that arm).
- **S5 blind control:** the `state_ac` contrast is outside the `state_blind` T-learner's CI for at
  least the S1 outcome. This shows the action input, not the state, carries the effect.
- **S6 seeds:** S1–S4 hold for both seeds.

**Pilot pass** means S1–S6 hold in ≥2 of the 3 arms, with P3-S2 failing reported as the expected
hardest case. Any outcome is written up as-is. These thresholds are not tuned after the fact.

---

## 3. Engineering

### 3.1 Files to add

| File | Content |
|---|---|
| `configs/trials/{albumin,saline,transfusion}.yaml` | Arm code lists, value_bin per arm, eligibility rules, t0 rule, outcome / negative-control codes, horizons |
| `src/ehrjepa/eval/trials/cohort.py` | Builds the TTE cohort from the MEDS cache (polars): eligibility, t0 window index, arm labels, observed outcomes at t0+k, censoring. Output is a parquet per arm. P1 cross-checks its counts against Doutreligne Table 2 |
| `src/ehrjepa/eval/trials/readout.py` | Trains the frozen readout heads `g_y` on LN(target-stack pooled states), plus R²/AUROC report |
| `src/ehrjepa/eval/trials/rollout.py` | Computes s_t0 with **window-t0 actions masked**, builds the forced action vectors (A/B), runs `EHRStateModel.predict` per k and applies `g_y`. Needs a small `EHRStateModel.state_at(batch, window)` helper and a public way to build `act_mean/act_count` from code ids (currently internal to `build_pairs`) |
| `src/ehrjepa/eval/trials/reference.py` | Naive / IPW / AIPW (cross-fit GBM; reuse `_gbm_worker.py` features), DR-on-s_t0, IPCW for unmeasured outcomes. Optional `econml` dependency only if needed. Plain sklearn GBM is the default |
| `src/ehrjepa/eval/trials/report.py` | Bootstrap CIs, the S1–S6 table, forest plot per arm. Writes `docs/experiments/trial-pilot/<arm>.json` |
| `tests/test_trials_*.py` | Synthetic-cohort test with a planted effect and a planted confounder (DR must recover it, naive must not). Leakage test: s_t0 must be invariant to window-t0 action tokens |

### 3.2 Cost

- **GPU:** no training. The state-gate checkpoints are reused (4 cells). Inference is one encoder pass
  per subject per arm over ≤512 events. Cohorts total ~25k–60k subjects across P1–P3, so this is
  minutes on one GPU; CPU-feasible on the M4 in roughly hours.
- **CPU:**
  - cohort building takes polars over the full MIMIC MEDS cache, ~tens of minutes per arm;
  - cross-fit GBM AIPW with 2000 bootstraps is the dominant cost. Use a reduced 500-resample DR
    bootstrap with nuisance models refit per resample, or a cross-fit influence-function CI
    (cheaper, the default);
  - estimate a few CPU-hours per arm.
- **Engineering:** ~3–4 days. Cohort definitions are the long pole (P1 is ported from the
  Doutreligne SQL).

### 3.3 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Confounding by indication: sicker patients get albumin or transfusion, giving a spurious mortality harm | Mortality nulls are scored (S2) exactly to detect this. DR reference; negative-control outcomes; DR-on-s_t0 tests whether the latent state adjusts adequately |
| Treatment leakage into s_t0 (window-t0 actions precede observations in the token stream) | Mask window-t0 actions when encoding s_t0, plus a unit test. Without this the pilot is invalid |
| Positivity: near-deterministic albumin in shock, transfusion at Hb <7 | Hb band 7–9 for P3; propensity trimming and overlap-restricted re-run; report the trimmed fraction |
| Short horizon (16h): mortality is only a surrogate readout on ŝ_{t+4} | Mortality is labelled secondary; non-null tests are short-horizon by design; no recursive rollout in the pilot |
| The predictor sees a *mean* action embedding, so one albumin code among 30 actions is diluted | Report effect vs count of concurrent actions; sensitivity with action set A restricted to arm code only |
| Readout heads may be poor (labs not measured every window; sparse albumin) | R² gate (§2.2 step 4); measured-mask; drop the outcome if it fails rather than tune |
| Dose / value: forced code needs a value_bin | Arm-median bin; sensitivity at the 25th and 75th percentile bins |
| Itemid errors (all [U]) | First task is to resolve every itemid against `d_items`/`d_labitems` and count occurrences in the cache before any analysis |
| Outcome-dependent tuning | §2.6 is frozen in git before the first effect estimate; any deviation is logged as a deviation |

---

## Sources

- Doutreligne M, Struja T, Abecassis J, Morgand C, Celi LA, Varoquaux G. Step-by-step causal analysis of EHRs to ground decision-making. *PLOS Digit Health* 2025;4(2):e0000721. https://pmc.ncbi.nlm.nih.gov/articles/PMC11790099/ (preprint https://arxiv.org/abs/2308.01605). Code: https://github.com/soda-inria/causal_ehr_mimic/ [V]
- Early albumin and SA-AKI, MIMIC-IV TTE (2025). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12369241/ [V, title only]
- MIMIC-IV transfusion-strategy TTE preprint (2026). https://sciety.org/articles/activity/10.64898/2026.08.14.26360449 [V, abstract]
- Liberal vs restrictive transfusion in sepsis, RCT meta-analysis (Crit Care 2019). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6659290/ [V]
- SMART summaries: https://criticalcarereviews.com/foundational-trials/smart , https://www.mdedge.com/content/balanced-crystalloids-critically-ill ; sepsis secondary analysis https://pubmed.ncbi.nlm.nih.gov/31454263/ [V, secondary]
- Early vasopressin, MIMIC-IV (n=531) and MIMIC-III/IV timing study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10071631/ ; Eastwood et al. TTE: https://researchonline.jcu.edu.au/87757/ [V, abstract]
- Hydrocortisone MIMIC-IV: TTE https://wjarr.com/node/11625 ; broad-cohort harm HR 1.35 https://doaj.org/article/589089edab534dfcb18b07789c92a36b [V, abstract]
- RRT timing MIMIC-IV TTE (She et al., Front Med 2025): https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1645046/pdf ; STARRT-AKI timing: https://rebelem.com/starrt-aki-trial-timing-of-renal-replacement-therapy-initiation-in-acute-kidney-injury/ [V]
- Oxygen: Yarnell et al. TTE https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9944781/ ; SpO2 94–98% eICU/MIMIC https://journalfeed.org/article-a-day/2019/whats-the-ideal-spo2-range-in-critically-ill-patients/ [V]
- RCT primaries [M, verify before citing numbers]: SAFE NEJM 2004;350:2247; ALBIOS NEJM 2014;370:1412; SMART NEJM 2018;378:829; BaSICS JAMA 2021;326:818; PLUS NEJM 2022;386:815; TRICC NEJM 1999;340:409; TRISS NEJM 2014;371:1381; VASST NEJM 2008;358:877; VANISH JAMA 2016;316:509; ADRENAL NEJM 2018;378:797; APROCCHSS NEJM 2018;378:809; STARRT-AKI NEJM 2020;383:240; ICU-ROX NEJM 2020;382:989; OXYGEN-ICU JAMA 2016;316:1583.
