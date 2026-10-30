# Review R1 (ML methods / SSL, JEPA family): "Latent Prediction Helps EHR Foundation Models Only Next to a Token Loss, and Only Fine-Tuning Shows It"

Reviewed: `docs/paper/v2/main.tex` (Oct 2026 draft). Numbers were checked against `docs/experiments/*/summary.md`, `docs/experiments/paired-bootstrap/README.md`, `docs/experiments/mimic-data25-1b/README.md`, `docs/experiments/gbm-leak-check/README.md`, the HANDOFF, and the code in `src/ehrjepa/` and `configs/grids/`. Red TODO placeholders count as missing results.

## Summary

The paper trains 10M-parameter causal and bidirectional transformers from scratch on coded MIMIC-IV events (MEDS, one token per event). It compares six pretraining objectives at 300M tokens and AR against an AR + multi-horizon next-latent "hybrid" at 1B tokens over three nested patient pools (8%, 25%, full). Evaluation covers six binary tasks (mean-of-6 AUROC) on 1,346 labelled held-out subjects, with a frozen linear probe and with end-to-end fine-tuning. Baselines are count-feature LR and XGBoost, and differences are tested with a paired subject bootstrap.

The paper makes three claims:
1. Latent-only objectives trail AR by 0.06 to 0.09.
2. The hybrid beats AR under fine-tuning by +0.008 [+0.002, +0.014], significant only at the 25% pool.
3. The probe ranks the hybrid below GBM and fine-tuning ranks it above, though hybrid vs GBM is not significant in either regime.

A fourth contribution is announced: a code-token, action-conditioned patient-state objective with a treatment-blind control. Its results are entirely placeholders (abstract L63-64, L124, Table 5, L539-541, L577, L625).

## Strengths

1. **Careful evaluation hygiene for this subfield.** It has a single leakage choke point with a bit-identity test (L339-341), identical anchors across models, paired subject bootstrap applied jointly across tasks (L362-370), random-init and from-scratch controls, and a GBM near-synonym leakage audit (L347-351, `gbm-leak-check/README.md`). None of the three closest EHR-JEPA papers has all of these.
2. **Numbers are reproducible from the committed artefacts.** Every Table 1-4 cell I recomputed matches the per-run summaries to the third decimal. That covers all 18 Table 1 seeds, the fine-tuned 3-seed means (0.806/0.813/0.813/0.821), the per-task Table 3, and the bootstrap Table 4.
3. **The negative result is useful and honestly stated.** Latent-only objectives (masked-span JEPA, next-latent, window-pooled) lose to a token loss by margins far outside the seed spread (Table 1). This agrees with Gupta et al. The pure next-latent collapse diagnostic (cos_gap 0.002) is a useful mechanistic hint.
4. **Fine-tuned evaluation fills a real gap.** SMB-Structure, Clin-JEPA and Gupta et al. are all probe-only (linear, or a shallow MLP for Clin-JEPA). Fine-tuned evaluation with seeds is a gap none of them fills.
5. **Claims are restrained where the data are weak.** Hybrid vs GBM is explicitly not claimed (L113-115), and causal readings of the state model are disclaimed (L573-576, L608-611).

## Weaknesses (ranked by how much they drive the score)

**W1. Contribution 4 is absent, and the title and abstract are built around a gate with no result.**
- Half the abstract (L57-64), contribution 4 (L116-124), §3.4, Table 5 (L515-534) and the discussion and conclusion paragraphs (L570-577, L622-625) describe an experiment whose every cell is TODO.
- Per HANDOFF, `state_ac_s0` is done and `state_blind_s0` is paused at 20k of 30.5k steps.
- Even when complete, the novelty is limited. Clin-JEPA already trains an action-conditioned latent world model on MIMIC-IV, as the paper concedes (L117-118). What remains is a small code-token variant plus a treatment-blind control.
- As submitted, this is a paper about one auxiliary loss on one dataset with a promised second half.

**W2. The headline "hybrid > AR" effect is not attributed to latent prediction. The comparison is confounded, and the paper's "controlled addition" claim is inaccurate.**
- L267-269 says "Setting the latent weight to zero recovers AR exactly (a unit test pins this), so the hybrid is a controlled addition to AR." The test (`tests/test_latent.py:180`) sets `lambda_pred=0, lambda_sigreg=0, lambda_recon=1`.
- The trained hybrid differs from the trained AR in at least four ways (`configs/grids/mimic_scale_1b.yaml`, `src/ehrjepa/objectives/latent.py:193-202`):
  - SIGReg at 0.05, applied to both token outputs and CLS. AR has no SIGReg.
  - The code loss is down-weighted 10x relative to the other terms, which changes the effective optimisation of the shared encoder under AdamW with weight decay.
  - An EMA target and predictor heads.
  - About 33% more compute per token (3.6 h vs 2.7 h, L375). This is token-matched, not FLOP-matched.
- `pretrain_default.yaml` comments that no-SIGReg was the best knob on DE-SynPUF, and the HANDOFF lists "SIGReg 0 on MIMIC" as untested. The hybrid's own latent term shows a small matched-vs-shuffled gap (cos_gap 0.016-0.023, about 9x below masked-span JEPA's 0.18). By the paper's own copying diagnostic (L561-564), that is weak evidence that the latent term is what helps.
- Required cells: AR + SIGReg 0.05; hybrid without SIGReg; AR at 0.1x code weight plus SIGReg (latent off); AR at 1.33B tokens. Without them, contribution 2's title claim ("latent prediction helps") is unsupported. The data support only "this composite recipe helps."

**W3. The data contradict the title's second clause ("Only Fine-Tuning Shows It") and the framing of contribution 3.**
- On the effect the paper actually claims (hybrid > AR), the probe shows a *larger* gap than fine-tuning at every pool size:

  | pool | probe gap (hybrid − AR) | fine-tuned gap |
  |---|---|---|
  | 8% | +0.025 | +0.007 |
  | 25% | +0.012 | +0.008 |
  | full | +0.015 | +0.008 |

  The probe overstates, not hides, the hybrid's advantage.
- The only "ranking flip" is hybrid vs GBM. That comparison is non-significant under fine-tuning (+0.013 [−0.001, +0.028]) and untested under the probe (0.803 vs 0.808, no interval). A reversal between two differences, neither shown to be non-zero, is not evidence that "the protocol ... can hide the effect" (L56-57).
- The second probe/FT disagreement (pool growth helps AR under the probe and the hybrid under FT, L455-457) has no paired test either.
- That linear probing and fine-tuning rank SSL objectives differently is well established. MAE (He et al., CVPR 2022) built its argument on it. In EHRs, MOTOR (ICLR 2024) already reports linear-probe, fine-tune and from-scratch arms. Neither is cited.
- The probe is also a weak strawman: a final-token, 256-d, L2-LR readout. No mean-pool, multi-layer or attentive probe (standard since V-JEPA) is tried. The claim that "the 256-dimensional frozen readout ... was the bottleneck" (L457-458) is asserted, not tested.

**W4. Scale and external validity are far below the venue bar.**
- One dataset and one site.
- One model size (10M parameters; the 35M run was abandoned, L601-602).
- Six tasks, all binary at one anchor per subject.
- No released foundation-model baseline. CLMBR-T-base and MOTOR are public on EHRSHOT, and Clin-JEPA itself compares against frozen CLMBR-T-base.
- Calibration papers use 120-143M models, 2+ datasets, 14-19+ tasks and released-model baselines (table below).
- Every fine-tuned conclusion uses 8% train labels. Under 25% labels GBM rises to 0.841 (Eval-B, `mimic-data25-1b/README.md`), which the paper omits (see M4). The hybrid's advantage over GBM may be a low-label artefact. No label-efficiency curve is reported, although MOTOR reports one at 5/10/25%.

**W5. Statistical evidence is thin, and the bootstrap ignores training variance.**
- Hybrid > AR is significant in one of three pool configurations (25% pooled). It is borderline at 8% (lower bound −0.000) and full (+0.000, one seed).
- Per seed at 25%, s0 is not significant (+0.004 [−0.005, +0.012]). At 8%, s1 is negative (−0.001).
- The "pooled" interval averages seeds within a subject resample (L366-368), so it treats the three seeds as fixed. Seed-to-seed variance, which is the same order as the effect (0.009 spread), is excluded from the interval.
- No correction for multiple configurations and comparisons is applied, and there is no pre-registration of which pool is primary.
- The test set is 1,346 subjects although about 36k held-out subjects exist in the full pool (L509-510). This is the paper's own stated remedy, and it is cheap.
- The full-pool row, the paper's best number (0.822), is one seed.

**W6. The state-model design has a target-leakage problem that undermines the planned control.**
- The target state is the EMA causal encoder's output at the last observation token of window w+k (L284-288; `src/ehrjepa/models/state.py:220-224`). That encoder has read every action token in windows w..w+k.
- The predictor's action input is a mean embedding of exactly those actions (L295-298).
- The action-conditioned model can therefore lower its loss by re-encoding which treatments appear in the target's context, without modelling patient response. The "true vs permuted actions" sensitivity test (L319-322) will register this as action sensitivity.
- The state_ac vs state_blind gap is likewise not interpretable as "use of the action input" in any physiological sense (L319: "isolates the use of the action input").
- A fix: compute targets from an observation-only stream (action tokens masked in the target encoder), or report sensitivity against a lagged or shuffled-within-patient action control.

**W7. Presentation and compliance issues.**
- The paper is not in the NeurIPS template, not anonymised (author and e-mail on L30-32), and about 11 pages. That would be a desk reject.
- Repository URL and commit hashes are TODO (L630).
- The headline metric excludes mortality without stated justification. The hybrid is worse than AR on mortality at the full pool (0.944 vs 0.948, Table 3) and at 25% s0. The direction of the conclusions survives mean-of-7, but the exclusion should be pre-specified and justified.
- Fine-tuning hyperparameters (lr 2e-5, ≤5 epochs, patience 2) are fixed across objectives, while GBM and LR are tuned on the tuning split. This asymmetry should be addressed with a small per-objective sweep.

## Factual mismatches found

| # | Location | Paper says | Evidence says |
|---|---|---|---|
| M1 | L267-269 | Latent weight 0 "recovers AR exactly (a unit test pins this)"; the hybrid is a "controlled addition to AR" | The test requires λ_sigreg=0 and λ_code=1 (`tests/test_latent.py:180-203`). The trained hybrid uses SIGReg 0.05 on tokens + CLS and λ_code=0.1; AR has neither (`configs/grids/mimic_scale_1b.yaml`). |
| M2 | L265 (Eq.) | One SIGReg term | SIGReg is applied twice, on tokens and CLS (`latent.py:196-200`) |
| M3 | L92-97 | Prior work uses "a linear probe"; "none reports seeds, paired intervals, or a tuned tabular baseline" | Clin-JEPA uses a shallow **MLP** probe, reports 95% bootstrap CIs plus Wilcoxon/Mann-Whitney tests, and compares with LightGBM and frozen CLMBR-T-base. Gupta et al. report 95% bootstrap CIs and are a **workshop** paper (WMHS), which the paper does not say. |
| M4 | L458-459 | LR refit on 25% labels (0.830) beats the frozen hybrid probe | The same experiment has GBM = **0.841** at 25% labels (`mimic-data25-1b/README.md`); omitted. It is the most relevant baseline for the label-regime question. |
| M5 | L436-437, L455-456 | 25% probe is seed 0 only (†); "+0.016 AR, +0.003 hybrid" | 3-seed Eval-A exists: AR +0.015 (non-overlapping), hybrid +0.001 (`mimic-data25-1b/README.md`). Use the 3-seed numbers. |
| M6 | L452-453 | "the ordering hybrid > AR holds at every pool size" (FT) | True for 3-seed means only. 8% s1: AR 0.8098 > hybrid 0.8085. 25% s0 paired CI includes 0. |
| M7 | Title, L56-57, L112-115 | The probe "hides the gain" | The probe gap hybrid−AR (+0.025/+0.012/+0.015) exceeds the FT gap (+0.007/+0.008/+0.008) at every pool. Only the non-significant GBM comparison flips. |
| M8 | L101-103, L373-375 | "matched token budgets" presented as the control | Token-matched, not compute-matched: hybrid 3.6 h vs AR 2.7 h per 1B tokens |
| M9 | L218-219 | ~7.5% of events are actions (used to motivate the design) | Measured on the MIMIC-IV **demo** cache, not the full pool used for the gate |

Verified correct: Table 1 (all seeds; untrained control 0.666); 1B probe seeds (L414-416); 2B numbers; Table 2 FT means and the 25% range [0.818, 0.825]; from-scratch 0.689; Table 3 per-task; Table 4 intervals; GBM leak audit numbers; 1,346 subjects with 1,025-1,120 rows per task.

## Questions for the authors

1. What is hybrid − AR when AR also gets SIGReg 0.05, and when the hybrid drops SIGReg? Is the gain from latent prediction or from the distributional regulariser plus loss reweighting?
2. Under a seed-aware test (hierarchical bootstrap over seeds and subjects, or a mixed model with seed as a random effect), is hybrid > AR at 25% still significant? Which pool was pre-specified as primary?
3. Does the probe/FT disagreement persist with mean-pooled, multi-layer or attentive probes, and with an MLP probe as in Clin-JEPA?
4. How do fine-tuned hybrid, AR and GBM compare at 25% and 100% train labels (where GBM reaches 0.841 at 25%)? Is there a crossover?
5. For the state model: are target states computed from a stream that contains the action tokens given to the predictor? If so, how will you distinguish learned treatment response from the target re-encoding the action tokens?
6. Why is mortality excluded from the headline metric, and was this fixed before the hybrid/AR comparison was run?
7. Were fine-tuning learning rate and epochs tuned per objective? The current fixed protocol could favour one encoder's representation scale.
8. With ~36k held-out subjects available in the full pool, why is every number scored on 1,346?

## Score

**3 (Reject).** Confidence: **4/5** (familiar with JEPA/SSL evaluation methodology and EHR foundation-model benchmarks; checked numbers and code directly).

Rationale:
- The one significant positive finding (hybrid > AR, +0.008 at one of three pools) is confounded by unablated SIGReg, loss reweighting and compute differences (W2).
- The paper's own data contradict its central methodological claim (W3).
- Scale and external validity are well below accepted EHR-FM papers (W4).
- The novel-method half of the paper has no results, and even when complete it is incremental over Clin-JEPA, with a target-leakage issue in its key control (W1, W6).
- The evaluation hygiene is genuinely good and would make a solid workshop paper or Datasets & Benchmarks-style methodology note as is.

## Minimum changes to reach a 6 (weak accept)

Costs are for one RTX 4060 at the measured rates: AR 2.7 h, hybrid 3.6 h, state about 7.5 h per 1B tokens; fine-tune eval about 17 min per cell.

| # | Change | Concrete experiment | Cost |
|---|---|---|---|
| 1 | Finish or cut the gate | Complete state_blind s0, state_ac/state_blind/MLM s1, and hybrid/AR s1 on full. Fix the target leak (observation-only target stream, or mask action tokens in the EMA encoder), which needs a state_ac/state_blind re-run. Otherwise drop contribution 4 from title and abstract and keep it as an appendix pilot. | Finish: ~1.5-2 GPU-days. Re-run with fixed targets: +2 days. Cut: 0. |
| 2 | De-confound hybrid vs AR | On the 25% pool, 3 seeds each: (a) AR + SIGReg 0.05; (b) hybrid with SIGReg 0; (c) λ_lat=0, λ_code=0.1, SIGReg 0.05; (d) AR at 1.33B tokens (FLOP-matched). Fine-tune all. | 12 cells × ~3 h + FT ≈ 40 GPU-h |
| 3 | Fix the statistics and test set | Score every fine-tuned model on the full-pool held-out split (~36k subjects; labels from full train or at least the 8% train). Add full-pool seeds 1-2 for hybrid and AR. Report a hierarchical (seed × subject) bootstrap. Pre-declare the 25% or full pool as primary. Holm-correct across pools. | 4 pretrain cells ≈ 13 GPU-h + eval on 27x more subjects (~4-8 h) |
| 4 | Make the probe claim honest | Add mean-pool, concat-last-4-layer and attentive probes. Reframe contribution 3 as "probe and FT rank hybrid vs AR consistently but differ in magnitude; probe vs FT reversal vs GBM is not significant", or show a significant reversal. Cite MAE and MOTOR for LP-vs-FT. Retitle (e.g. "...Helps Only Next to a Token Loss: A Seeded, Fine-Tuned Study on Public EHRs"). | ~0.5 GPU-day + writing |
| 5 | Label-efficiency curve | Fine-tune hybrid and AR and fit GBM and LR at 1%, 8%, 25% and 100% of full-pool train labels, scored on the full held-out split | ~1 GPU-day |
| 6 | One external anchor | Either (a) EHRSHOT with released CLMBR-T-base / MOTOR baselines (DUA lead time), or (b) a public released model run on the same MIMIC tasks (e.g. CEHR-GPT, or frozen CLMBR-T-base via MEDS) | (a) weeks for access + ~2 GPU-days; (b) ~1-2 days |
| 7 | Compliance | NeurIPS template, anonymised, ≤9 pages; correct M1-M9; release URL and hashes | 1 day |

Items 2, 3, 4 and 7 are necessary. Item 1 is necessary if the state model stays in the title. Items 5 and 6 are what would move this from a borderline to a defensible 6, because the main finding is otherwise a single-site, 10M-parameter, 8%-label result.

## Gap to calibration papers

| Dimension | MOTOR (ICLR 2024) | I-JEPA (CVPR 2023) | ORA, 2602.00541 (NeurIPS 2026 poster; track unverified) | This paper |
|---|---|---|---|---|
| Model scale | 143M (released) | ViT-B → ViT-G (~1B) | ~120M Transformer and Mamba | 10M only (35M abandoned) |
| Pretraining data | 2.7M patients / 2.4B events (EHR) + 55M (claims) | IN-1K, IN-22K | MIMIC-IV ~364k + CUMC 6.7M | MIMIC-IV, 249k patients, ~1.8 passes at 1B nominal tokens |
| Datasets/sites | 2 pretraining sources + MIMIC-IV transfer | 6+ downstream benchmarks | 2 (public + private) | 1 |
| Tasks | 19 TTE + 6 transfer | linear probe, low-shot, 4 transfer, Clevr | 14 (classification, TTE, regression) | 6 binary + mortality |
| Released-model / SOTA baselines | Cox, DeepSurv, DeepHit, DSM, RSF; next-code ablation | MAE, data2vec, CAE, DINO, iBOT, MSN, BYOL | NTP, Context Clues, TPP, MOTOR, XGBoost, DeepHit | Own AR/MLM reimplementations, LR, GBM; none released |
| Protocol | linear probe + fine-tune + from scratch + label efficiency (5/10/25%) | linear probe + low-shot + FT (appendix) | linear probe | linear probe + FT (fixed hyperparameters); single label regime |
| Ablations | objective, #tasks, width, bins, data fraction, temporal shift, calibration, subgroups | 9 tables (masking, targets, predictor, etc.) | loss × architecture | objective family (6 objectives at 300M), token/pool scaling. **Missing: SIGReg, loss weight, compute match, horizons** |
| Statistics | paired bootstrap, 95% CI, significance | none | mean ± SE | 3 seeds (1 at full), paired bootstrap; seed variance excluded; n=1,346 |
| Headline effect size | consistent gains over baselines across 19 tasks | large linear-probe gains over pixel-reconstruction baselines | consistent gains across 14 tasks | +0.008 AUROC vs own AR, significant at 1 of 3 pools; n.s. vs GBM |

In short, this paper's statistics are stronger than ORA's and I-JEPA's. It falls short of all three calibration papers on scale, breadth of tasks and datasets, external baselines, and component ablations.
