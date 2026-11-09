# R2: clinical ML review of `docs/paper/v2/main.tex` (draft, Oct 2026)

Reviewer profile: clinical ML (EHR foundation models and benchmarks). Venues considered: NeurIPS main track, NeurIPS Datasets & Benchmarks (D&B), CHIL, npj Digital Medicine.
TODO placeholders are treated as missing. `docs/REVIEW_mock.md` was not read.

## 1. Summary

The paper trains 10M-parameter transformers from scratch on coded MIMIC-IV events (MEDS) with several objectives: next-code AR, four latent-only JEPA variants, an AR + multi-horizon next-latent "hybrid", and MLM. It scores them on 7 ACES-defined binary tasks with a frozen linear probe and with fine-tuning. It makes three claims:
1. Latent-only objectives trail AR by 0.06–0.09.
2. The hybrid beats AR under fine-tuning by +0.008 [+0.002, +0.014], but only in the 25% pool.
3. The frozen probe ranks the hybrid below count-feature XGBoost and fine-tuning reverses that order. The hybrid's margin over GBM, +0.013 [−0.001, +0.028], is not significant.

A fourth contribution, an action-conditioned 4-hour patient-state objective with a treatment-blind control, has no results yet: Table 5, the abstract, §5.5, §6 and the conclusion are all TODO. Every downstream number comes from 1,346 held-out patients at one site.

## 2. Strengths

- **Honest statistics for this subfield.** It uses paired subject-level bootstrap over identical anchors, 3 seeds for most cells, a stated noise floor (0.009) and an explicit "we do not claim hybrid > GBM". This is better than SMB-Structure, Clin-JEPA and Gupta et al., which are all probe-only with no CIs.
- **The probe-vs-fine-tune point is useful and well supported in direction.** The probe ranks hybrid < GBM and fine-tuning ranks hybrid > GBM (Table 2). This is a protocol warning other EHR-JEPA authors should heed.
- **Clean experimental control.** Pools are nested by subject and share one tokenizer, so the effect of pretraining data is separated from the effect of label count (§3). Setting λ=0 recovers AR exactly. An untrained-encoder control is included.
- **Reproducible infrastructure.** Public data, MEDS, declarative ACES configs, a leakage unit test at a single cut point, and committed `predictions.parquet` files. I could recompute the metrics from these (§5).
- **Appropriately non-causal framing** of the state model (§4.1, §6, §7).

## 3. Weaknesses (ranked)

### W1 (critical). The "new diagnosis within 365 days" tasks mostly measure how the *current* admission will be billed, not incident disease.

In `meds_etl` 0.3.2, `hosp/diagnoses_icd` is timestamped at `dischtime` (`meds_etl/mimic/__init__.py` L206–208). Anchors are drawn uniformly over a patient's event times (`eval/tasks.py` `select_anchors`), and in MIMIC most events are in-stay labs, meds and chart events. The ongoing stay's discharge ICD codes therefore fall inside the label window.

I ran the repo's own `build_task` on the MIMIC-IV demo (`data/meds/mimic-demo`, 100 patients):
- 92–93% of anchors fall strictly inside a hospital stay.
- **27/33 new-dx positives (82%) are labelled by the discharge coding of the stay that contains the anchor.** By task: CKD 9/12, HF 8/9, diabetes 6/8, COPD 4/4.

The demo is ICU-enriched and the sample is tiny, but the mechanism is structural and will hold on full MIMIC. The GBM feature importances agree:
- CKD: the top feature is last creatinine z-score (`LABITEM/50912`).
- HF: the top feature is the 30-day NT-proBNP count (`LABITEM/50963`, 4.3% of gain, about 3× the next).
- Diabetes: the top features are 30-day insulin NDCs and `MIMIC_IV_Drug/Insulin`.

In other words, the models detect the in-progress workup of a diagnosis that will be coded at discharge a few days later. That explains the implausibly high AUROCs (CKD 0.94, diabetes 0.92, HF 0.90) for "new diagnosis in a year".

EHRSHOT's "Assignment of New Diagnoses" tasks predict at discharge, which avoids this. Four of the six tasks in the headline mean-of-6 have this flaw (§4 "Tasks"). The task names and the clinical framing must change, or anchors must move to admission/discharge with the current stay's codes excluded from labels.

### W2 (critical). The leakage audit is narrower than the paper states and partly vacuous (§4 "Baselines").

- **It only audits ICD near-synonyms in GBM *features*.** It does not audit medication proxies (insulin for diabetes, diuretics for HF, dialysis or EPO for CKD), and it does not address W1, which is label-side.
- **The CKD test is vacuous.** For CKD, `near_columns = 0`: every near-synonym column was removed by the `min_df = 10` pruning before fitting. "Removing them leaves AUROC unchanged" is trivially true. Diabetes, HF and COPD have only 4, 2 and 2 such columns.
- **It checks a different GBM.** `gbm_leak_check.py` uses fixed parameters (the grid's middle setting), and its held-out AUROCs (CKD 0.939, diabetes 0.919) differ from the reported tuned GBM (0.930, 0.913; Table 3). The audited model is not the reported baseline.
- **The encoders are never audited.** The hybrid and AR read every code, including rare near-synonyms that GBM pruned.

The sentence "carry no measurable share of GBM's gain" should be rewritten to say what was actually tested.

### W3 (major). The effect size is below clinical relevance and below the evidence the design can give.

- **The size of the effect.** Hybrid − AR is +0.008 mean AUROC and is significant only for the 25% pool. At 8% the CI is [−0.000, +0.013] and for the full pool it is [+0.000, +0.016]. The abstract headlines the one significant configuration out of three pool sizes, with no multiplicity correction and no pre-registration. The defensible summary is "hybrid ≥ AR at every pool size, with a significant pool-pooled effect"; the paper should show that analysis.
- **Clinically, an AUROC change of 0.008–0.014 averaged over six heterogeneous tasks has no interpretation.** No paper in this space reports NRI, decision curves, or sensitivity at a fixed alert rate.
- **Per-task, the gain is not uniform.** Pooled hybrid−GBM per task: diabetes +0.002, HF −0.005, inpatient +0.028. Only inpatient's CI excludes 0 (`docs/experiments/paired-bootstrap/README.md`).
- **Test-set sizes are very small.** The held-out test set has COPD **45 positives**, CKD 78, HF 84, diabetes 112, readmission 125, mortality 210 and inpatient 430 (computed from `mimic-ft-full/eval/hybrid_full_s0/predictions.parquet`). With 45 positives, a per-task AUROC SE is about 0.02–0.03.
- **The fix is already available.** Scoring on the ~36k full-pool held-out patients is already identified (§6 Limitations) and costs only inference.

### W4 (major). No published foundation-model baselines, and a weak tabular baseline.

- **Published FMs.** The paper compares against its own AR and MLM reimplementations at 10M parameters (§7). ETHOS has released MIMIC-IV weights and code. CEHR-GPT, MOTOR and CLMBR-T have public code, and CEHR-GPT and ETHOS run on MIMIC-IV MEDS. An npj DM or D&B reviewer will expect at least one of them on identical anchors.
- **GBM tuning.** The "tuned" GBM is a 3-point grid (`baselines.py` `GBM_GRID`) with no early stopping, subsampling or column sampling. The fixed middle setting beats the tuned choice on held-out CKD by 0.009 and diabetes by 0.006, so tuning-split selection is noisy. A Bayesian or Optuna search, or MEDS-Tab itself, is the standard.
- **MEDS-Tab is cited but not used.** The paper says "we use all three" (MEDS, ACES, MEDS-Tab; §2), but the repo only cites MEDS-Tab as inspiration in `baselines.py`. That is a factual mismatch.
- **The supervised-from-scratch baseline is weak.** At 0.689 it is a strawman, with no tuning reported.

### W5 (major). Single site and an uncharacterized cohort. There is no Table 1.

- **No external validation.** All data is BIDMC 2008–2019. EHRSHOT and INSPECT are "planned".
- **No cohort description.** There is no table of age, sex, race, insurance, ICU fraction, admissions per patient, history length at anchor, or label prevalence.
- **The cohort is hospital-wide but the anchors are a mix.** The cohort covers all MIMIC-IV hosp patients, yet the anchor procedure mixes in-ICU, in-ward and post-discharge anchors with no breakdown.
- **Prediction time is not deployable.** Anchors at random event times do not match any clinical decision point; standard practice is admission, 24h, or discharge.
- **Censoring selects on the future.** A negative must have a recorded event ≥365 days after the anchor (`last_time − anchor ≥ horizon`), so patients who never return to BIDMC are excluded. Meanwhile MIMIC captures out-of-hospital deaths up to 1 year after discharge for everyone. This explains the 18.8% one-year mortality prevalence, and it biases mortality and inpatient-admission labels.
- **ICU vs hospital framing is unclear.** The state model uses ICU `inputevents` as actions in a mostly non-ICU cohort, and the reported action share (7.5%) comes from the ICU-only demo, not full MIMIC (§3 "Actions").

### W6 (major). The headline novel contribution has no results.

The action-conditioned state model is the fourth contribution and is the subject of the title's subtitle. Table 5, the gate pass/fail, the action-sensitivity test and the discussion are all TODO. The novelty is also explicitly weak: Clin-JEPA already has an action-conditioned latent world model on MIMIC-IV, and the paper says so. As written, the paper is an evaluation-protocol note plus a pending experiment.

### W7 (moderate). No calibration, AUPRC, subgroup or fairness analysis, despite the code computing most of it.

`eval/metrics.py` and `report.py` already compute AUPRC, Brier and calibration slope, but none is reported. My recomputation from committed predictions (full pool, seed 0) shows hybrid − AR AUPRC gains are small and uneven:

| Task | Hybrid AUPRC | AR AUPRC |
|---|---|---|
| COPD | 0.189 | 0.190 |
| Diabetes | 0.743 | 0.687 |
| Readmission | 0.218 | 0.213 |

COPD AUPRC is 0.19 at 4.2% prevalence. There is no stratification by sex, age, race or insurance; Delphi-2M and npj DM papers routinely report these.

### W8 (minor)

- **Most headline numbers are single-seed.** The full-pool numbers behind the abstract's (iii) (0.803 probe, 0.822 fine-tuned) are seed 0. Table 2's 25% probe values are also seed 0, even though the 3-seed Eval-A values exist (HANDOFF).
- **The "1B tokens" budget is mostly padding.** It is ~550M real events (§4 "Compute"), and the 8% vs full comparisons sit at very different numbers of passes.
- **The seed-spread noise floor comes from a different setting.** The 0.009 figure is from 300M-token probes and is reused as a threshold for fine-tuned 1B results.
- **Readmission excludes the current stay's discharge codes from inputs.** Discharge ICD codes share the anchor timestamp and inputs are cut strictly before it. This is realistic (coding lags discharge) but should be stated.
- **The reproducibility statement is TODO.** There is no URL and no commit hashes (§8).

## 4. Factual mismatches (paper vs repo / HANDOFF)

| Paper claim | Repo evidence | Status |
|---|---|---|
| "MEDS, ACES and MEDS-Tab … we use all three" (§2) | MEDS-Tab only cited in a `baselines.py` docstring; own count features used | **Mismatch** |
| GBM leak audit: near-synonyms "carry no measurable share of GBM's gain, and removing them leaves held-out AUROC unchanged" (§4) | CKD `near_columns = 0` (pruned before fit); audit GBM is fixed-param, AUROC 0.939/0.919 vs reported 0.930/0.913 | **Misleading**: true, but vacuous for CKD and run on a different model |
| Near-synonyms in 0.7–2.0% of positives | `report.json`: 1.1 / 1.2 / 0.7 / 2.0% | Matches |
| "first-ever diagnosis … within 365 days" | Diagnoses timestamped at `dischtime`; ~82% of demo positives come from the anchor's own stay | **Mislabelled construct** (W1) |
| "tuned" GBM | 3-config grid | Overstated |
| Test set: 1,346 subjects, 1,025–1,120 rows/task | `predictions.parquet`: 1,346 subjects; rows 1,025–1,120 | Matches |
| Table 3 per-task values | `mimic-ft-full/summary.md` | Matches (rounded) |
| Table 4 bootstrap values | `paired-bootstrap/README.md` | Matches |
| Table 1, 1B 8% probe (0.769 / 0.794), GBM 0.808, LR 0.795 | HANDOFF "Results so far" | Matches |
| Action share "about 7.5%" | Demo cache (ICU-only, 100 patients) | Matches the source, but wrong population for a full-MIMIC claim |
| Gate uses 4h windows | HANDOFF: 4h (`b35b67c`); gate paused, state_blind_s0 at 20k/30.5k | Consistent; results do not exist |
| `mimic-ft-full/summary.md` header says "full held-out split" | Paper says 8% held-out; predictions have 1,346 subjects (8% split) | Repo doc header is wrong, paper is right; fix the README template |

## 5. Questions to the authors

1. What fraction of anchors on full MIMIC fall inside a stay, and what fraction of new-dx positives come from the ongoing stay's discharge codes? What happens to mean-of-6, and to hybrid − GBM, when anchors are at discharge (EHRSHOT-style) or when the current stay's codes are excluded from labels?
2. How many anchors are excluded by the "record extends ≥ horizon" rule, and how do labels change if negatives are defined by MIMIC's 1-year post-discharge death linkage instead of by return visits?
3. Does the hybrid's advantage hold with a properly tuned GBM (Optuna or MEDS-Tab), and against ETHOS or CEHR-GPT on the same anchors?
4. Why is the 25% pool the headline? What is the pre-specified primary comparison?
5. Do the conclusions survive per-task reporting with AUPRC and calibration? Is the hybrid gain concentrated in inpatient-365d?
6. Are any action codes, such as insulin, also outcome proxies for the diabetes task? For the state model this conflates "action" with "label signal".

## 6. Score

**NeurIPS main track: 3/10 (reject). Confidence: 4/5.**

- **Main track.** The probe-vs-fine-tune observation is real but incremental. The core effect (+0.008) is small and significant only in a selected configuration, there is one site and one small model, and the novel component has no results.
- **D&B track: not a fit as written.** There is no new dataset or benchmark; the "trial-anchored benchmark" is future work. A D&B paper built on corrected task definitions, a public MEDS/ACES task suite, released checkpoints and a probe-vs-fine-tune protocol could be a fit (EHRSHOT's bar). Without the W1 fix it would be rejected for task validity.
- **Clinical journal (npj DM / JAMIA / CHIL): reject.** It lacks a cohort table, external validation, calibration, subgroup analysis and clinically defined prediction times, and its tasks do not measure what they are named for.
- **Best realistic venue after fixes:** a CHIL or ML4H full paper framed as "evaluation protocol matters for EHR self-supervised objectives", with corrected tasks and the 36k test set.

## 7. Minimum changes to reach acceptance (with costs)

| # | Change | Fixes | Cost |
|---|---|---|---|
| 1 | Redefine anchors at discharge (or admission + 24h) and exclude the current stay's codes from new-dx labels; rename tasks. Rerun every eval. | W1 | ~1 day code + ~1 GPU-day re-eval (fine-tune 4 encoders × seeds) |
| 2 | Score on the full-pool held-out split (~36k patients) with 3 seeds; pre-specify one primary comparison (hybrid − AR, pooled over pools or full pool only) | W3 | ~1–2 GPU-days (full pool needs seeds 1–2 pretrain: ~7 h each) + eval |
| 3 | Redo the leak audit on the reported GBM *and* the encoders, including medication proxies and without `min_df` pruning; add a "drop all near-synonym and proxy codes from inputs" ablation | W2 | ~0.5 day CPU + 1 fine-tune run |
| 4 | Add MEDS-Tab (or Optuna-tuned XGBoost/LightGBM) and one published FM (ETHOS weights on MIMIC, or CEHR-GPT trained on the same pool) | W4 | MEDS-Tab ~1 day; ETHOS inference ~1–2 days; CEHR-GPT ~2–3 GPU-days |
| 5 | Table 1 cohort characteristics; per-task AUROC + AUPRC + Brier/calibration slope; subgroups (sex, age band, race, insurance, ICU vs non-ICU anchor) | W5, W7 | ~1 day (metrics already in the code) |
| 6 | Either finish the gate (Table 5, action sensitivity, pass/fail) or cut §4.1/§5.5 and retitle | W6 | ~1.5 days GPU (paused grid), or 1 hour of editing |
| 7 | External validation on EHRSHOT or INSPECT (at least the probe-vs-fine-tune flip and hybrid − AR) | W5 | Weeks (DUA) + ~2–3 GPU-days; needed for a journal, optional for CHIL |
| 8 | Fix the MEDS-Tab claim, the leak-audit wording, single-seed labelling in the abstract, and the repo URL/hashes | W8, §4 | 1 hour |

Items 1–3, 5 and 8 are the floor for any venue. Items 4 and 6 are needed for NeurIPS main or D&B. Item 7 is needed for a clinical journal.

## 8. Calibration against accepted comparable papers

Sources: EHRSHOT (arXiv 2307.02028, full HTML read) and ETHOS (arXiv 2407.21124 / npj DM 2024, HTML read). Delphi-2M (Nature 2025) is summarized from my prior reading of the paper, because nature.com and medRxiv blocked fetching (303/403).

| Criterion | EHRSHOT (NeurIPS D&B 2023) | ETHOS (npj DM 2024) | Delphi-2M (Nature 2025) | This paper | Gap |
|---|---|---|---|---|---|
| Pretraining cohort | 2.57M patients (Stanford) | ~241k MIMIC-IV patients | ~400k UK Biobank | 249k MIMIC-IV patients | OK |
| Sites / external validation | 1 site; contribution = released dataset + weights | 1 site (MIMIC); none, acknowledged | External: ~1.9M Danish registry, no retraining | 1 site; none | **Large** for a journal; moderate for ML |
| Test-set size | ~1,200 patients/task (e.g. readmission 260 pos; celiac 21 pos) | 26,758 test patients; per-task up to 43k admissions | Hundreds of thousands | 1,346 patients; 45–430 pos/task | Comparable to EHRSHOT, but EHRSHOT made no ±0.01 ranking claim; too small for this paper's claims |
| Published-FM baselines | CLMBR-T vs tuned LightGBM, LR, RF | Literature models per task (XGBoost, LSTM, GPT-4o) | Epidemiological risk scores and biomarker models | Own AR/MLM only | **Large** |
| Tabular baseline | Count LightGBM tuned on validation | XGBoost / LR on literature features | Risk scores | Count XGBoost, 3-config grid | Moderate |
| Replicates / CIs | 5 replicates per few-shot k | Bootstrap 95% CIs | CIs throughout | 3 seeds + paired bootstrap | **Better than the comparators** |
| Prediction time / task validity | Defined clinical prediction times (admission, discharge); new-dx at discharge | Clinically defined (admission, ICU, discharge) | Age-based incidence, yearly | Random event time; new-dx contaminated by the ongoing stay's coding | **Critical** |
| Calibration | None | Limited | Calibration curves reported | None (code exists) | Moderate |
| Subgroup / fairness | None (qualitative) | Clinical subgroups (sepsis, hemorrhage); demographics only descriptive | Sex, age and other subgroups; dataset-bias analysis | None | Moderate–large for a journal |
| Leakage controls | Date jitter, prediction-time cuts | Timeline-based | Explicit bias/missingness analysis | Input-cut unit test; partial GBM synonym audit | Label-side leakage unaddressed |
| Reproducibility | Code + dataset + weights (gated) | Code + weights + scripts released | Code released | Code public; URL/hashes TODO; no weights | Small (release weights) |
| Clinical relevance of tasks | Operational, labs, new dx, imaging | Mortality, readmission, LOS, SOFA, DRG | Incidence of >1,000 diseases | 1-year admission/mortality, 30-day readmission, 4 "new dx" | Tasks are standard, but the definitions undermine them |

**Bottom line of calibration:** the statistics are above the bar of all three comparators. The evidence base falls below it: one site, no published-FM baseline, no cohort table, calibration or subgroups. Task validity falls below even EHRSHOT's minimum, because new-dx labels leak from the ongoing stay.
