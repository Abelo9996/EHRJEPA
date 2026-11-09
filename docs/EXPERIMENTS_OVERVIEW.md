# Experiments overview (as of 2026-10-30)

One row per experiment run to date. Numbers are copied from `docs/experiments/*/summary.md`,
`*_RESULTS.md`, `paired-bootstrap/README.md`, `gbm-leak-check/README.md` and the shared
HANDOFF; anything not found there is `—`. Nothing here is newly computed except mean-of-6
averages of committed per-task rows.

Conventions. **mean-of-6** = mean held-out AUROC over the 6 non-mortality tasks (inpatient_365d,
readmission_30d, new_dx_365d/{ckd, copd, diabetes, heart_failure}). **probe** = frozen encoder +
linear probe. **FT** = end-to-end fine-tune (encoder + LN/Linear head per task, BCE, AdamW,
<=5 epochs, early stop on tuning AUROC). **Tokens** = nominal token slots (about 45% padding,
so 1B nominal is about 550M real events). Model is 6L x 256 (about 10M params) unless noted.
**hybrid** = causal next-latent (EMA target, horizons [1,4,16]) + 0.1 x AR code loss.
MIMIC 8% split: GBM 0.808, LR 0.795, random-init 0.666 (same split for every 8%-scored row).

## A. DE-SynPUF, stage A (claims data, synthetic; Sept 2026)

| experiment | objectives | tokens | seeds | eval | key metric | commit | verdict |
|---|---|---|---|---|---|---|---|
| Pilot grids 1-4 (`2026-09-0{3,4}-pilot*`) | 20 cells: AR, masked-span JEPA, JEPA+recon, recon-only, next-latent variants | 48M | 1 (s0) | probe, 3,000-subj subset, mean-of-7 | best `nextlatent_h1416_recon` 0.7287; GBM 0.7261; AR 0.7205; masked-span JEPA 0.6829 | `daece7c` | hybrid config selected; masked-span / recon-only / window dropped |
| Grid 5 seeds (`pilot5-seeds`) | AR vs hybrid | 48M | 3 | probe, mean-of-7 | AR 0.7234 vs hybrid 0.7279; gap inside seed range on 4/7 tasks | `6fd64e1` | no decision at 48M |
| Microgrids 1-4, sanity, Synthea eval | various | — | — | — | — | — | exploratory; not used downstream |
| Scale 200M / 1B (`scale-desynpuf`, `scale1b-*`) | AR, hybrid (SIGReg on), hybrid_final (SIGReg 0) | 200M, 1B | 3 at 1B | probe, full held-out 11,708 subj | 1B mean-of-6: hybrid_final 0.7628, hybrid 0.7615, GBM 0.7511, AR 0.7436, LR 0.7191; ranges non-overlapping vs AR on 6/6 | `4eff6a5`, `1f98e8f` | hybrid > AR and > GBM on DE-SynPUF |
| Few-shot at 1B (`fewshot-1b-final`) | AR, hybrid_final, LR | 1B | 3 x 5 sample seeds | probe at k=32/128/512/all | k=32: hybrid 0.6578 vs AR 0.6258 vs LR 0.6289 | `bf1af4f` | hybrid ahead at every k |
| Ablate2: size + knobs (`ablate2-desynpuf`) | AR/hybrid small/base/large; shared target, no-SIGReg, horizon [1], recon 0.3 | 200M | 2 (s1,s2); base s0 | probe, full held-out | hybrid - AR: +0.0079 / +0.0047 / +0.0055 (small/base/large); no-SIGReg 0.7615 vs default 0.7554 | `2161c45` | SIGReg dropped from DE-SynPUF default |
| Ablate3: teachers + code-emb init | EMA / frozen-AR teacher, init variants | 200M | 2 | probe, full held-out | — (see `ABLATION_RESULTS.md` Grid 3) | `97e4efd` | — |
| Ablate4: no-SIGReg + frozen teacher | hybrid_nosig_frozen | 200M | 2 | probe, full held-out | 0.7625 (2-seed) | `1f98e8f` | gains do not stack; EMA kept |
| A2 continuous state (PhysioNet 2019 / 2012) | ar_bins, ar_cont, hybrid_bins, latent_cont, latent_only | 100M (2019), 40M (2012) | 2 probe; s1 FT | probe + FT, full held-out | sepsis_6h FT: hybrid 0.8403, GBM 0.8363, ar_bins 0.8264; latent_only below ft_random on 4/4 tasks | `1491da1`, `e68e4b5` | pre-registered rule: pure latent objectives NOT supported; GBM ahead on both 2012 tasks |

## B. MIMIC-IV 8% subset re-validation (30,386 subjects, scored on 8% held-out)

| experiment | objectives | tokens | seeds | eval | key metric (mean-of-6) | commit | verdict |
|---|---|---|---|---|---|---|---|
| Preliminary 20M (`mimic-hybrid-ar`) | AR, hybrid (SIGReg 0) | 20M | 2 | probe | AR 0.6996/0.6960, hybrid 0.7054/0.7016 | `a9c006e` | preliminary |
| Preliminary 100M (`mimic-hybrid-ar-100m`) | AR, hybrid (SIGReg 0) | 100M | 2 | probe | AR 0.7424/0.7425, hybrid 0.7431/0.7531 | `cf621ac` | within noise |
| Phase 0 noise floor (`mimic-ar-noise-300m`) | AR | 300M | 3 | probe | 0.7663 / 0.7607 / 0.7699; spread 0.0091 | `b31ae85` | noise floor fixed at about 0.009 |
| Phase 1 objectives (`mimic-objectives-300m`) | AR, hybrid, jepa_ema, recon_only, nextlatent, window | 300M | 3 | probe | hybrid 0.7700, AR 0.7680; latent-only families 0.675-0.713; jepa_ema - random-init +0.039 | `edefd53` | hybrid = AR (tie); code loss required |
| 1B scale (`mimic-scale-1b`) | AR, hybrid | 1B | 3 | probe | hybrid 0.7943 (0.7911-0.7983) vs AR 0.7694 (0.7668-0.7716); gap +0.0249, non-overlapping | `040e8e7` | hybrid > AR (probe) |
| 2B scale (`mimic-scale-2b`) | AR, hybrid | 2B | 1 | probe | hybrid 0.7976, AR 0.7634 | `9ac0f6a` | flat 1B -> 2B (subset exhausted, about 44 passes) |

## C. 25% data (91,157 subjects, 1B tokens, 8% tokenizer reused)

| experiment | objectives | tokens | seeds | eval | key metric (mean-of-6) | commit | verdict |
|---|---|---|---|---|---|---|---|
| Eval-A (25%-pretrained, scored on 8% test with 8% labels) | AR, hybrid | 1B | 3 | probe | s0: AR 0.785 (vs 0.769 at 8%), hybrid 0.797 (vs 0.794); 3-seed: AR +0.015 non-overlapping, hybrid +0.001 | `6bfde8f` | more patients helped AR, not the hybrid probe |
| Eval-B (`mimic-data25-1b`, all at 25%) | AR, hybrid | 1B | 3 | probe, 25% test, refit baselines | AR 0.8065/0.8059/0.8025; hybrid 0.8176/0.8149/0.8138 (mean 0.815); GBM 0.841, LR 0.830, random 0.706 | `6bfde8f` | GBM ahead by about 0.026; frozen probe likely the bottleneck |

## D. Fine-tune grids (8% labels and test; vs GBM 0.808)

| experiment | objectives | tokens | seeds | eval | key metric (mean-of-6) | commit | verdict |
|---|---|---|---|---|---|---|---|
| `mimic-ft-1b` | AR, hybrid x {8%, 25%} pretrain; ft_random | 1B | 1 (s0) | FT | hybrid 0.819 / 0.818; AR 0.803 / 0.814; from-scratch 0.689 | `6bfde8f` | FT adds +0.02-0.03 over probe |
| `mimic-ft-1b-s12` | same, seeds 1-2 | 1B | 3 (with s0) | FT | hybrid 25% 0.821 (0.818-0.825); AR 25% 0.813; hybrid 8% 0.813; AR 8% 0.806 | `105fc12` | hybrid 25% above GBM on every seed; 25% pretraining helps hybrid under FT |

## E. Full MIMIC-IV (all subjects; 248,743 usable train subjects; scored on 8% source)

| experiment | objectives | tokens | seeds | eval | key metric (mean-of-6) | commit | verdict |
|---|---|---|---|---|---|---|---|
| `mimic-full-1b` | AR, hybrid | 1B (about 1.8 passes) | 1 (s0) | probe | hybrid 0.803, AR 0.788 | `4e3f58a` | — |
| `mimic-ft-full` | AR, hybrid | 1B | 1 (s0) | FT | hybrid 0.822, AR 0.814, GBM 0.808 | `4e3f58a` | hybrid FT 8% 0.813 -> 25% 0.821 -> full 0.822 (flat 25% -> full at 1B) |
| `mimic-full-35m` (8L x 512, 37.9M params) | hybrid | 1B planned | 1 | probe + FT planned | — (stopped at step 8,000 of 30,518; ckpts kept) | `c9ad178` | incomplete |

## F. Paired subject bootstrap (`paired-bootstrap`, 2,000 resamples, 1,346 held-out subjects with anchors)

| comparison | seeds | eval | diff mean-of-6 [95% CI], P(>0) | commit | verdict |
|---|---|---|---|---|---|
| hybrid 25% - GBM | 3 pooled | FT | +0.013 [-0.001, +0.028], P=0.97 | `aaffe12` | CI includes 0; s2 alone excludes 0 |
| AR 25% - GBM | 3 pooled | FT | +0.005 [-0.009, +0.020], P=0.76 | `aaffe12` | n.s. |
| hybrid 8% - GBM | 3 pooled | FT | +0.005 [-0.010, +0.020], P=0.73 | `aaffe12` | n.s. |
| hybrid full - GBM | s0 | FT | +0.014 [-0.001, +0.028], P=0.97 | `aaffe12` | CI includes 0 |
| hybrid - AR, 25% | 3 pooled | FT | +0.008 [+0.002, +0.014], P=0.995 | `aaffe12` | CI excludes 0 |
| hybrid - AR, 8% | 3 pooled | FT | +0.007 [-0.000, +0.013], P=0.97 | `aaffe12` | borderline |
| hybrid - AR, full | s0 | FT | +0.008 [+0.000, +0.016], P=0.98 | `aaffe12` | borderline |

## G. GBM leakage check (`gbm-leak-check`)

| check | result | commit | verdict |
|---|---|---|---|
| near-synonym codes of new-dx outcomes removed from GBM features | AUROC unchanged: ckd 0.939, diabetes 0.919, HF 0.900, copd 0.789; gain share 0.000 | `3012abe`, `6bfde8f` | clean; GBM 0.808 / 0.841 are legitimate targets |

## H. Gate: action-conditioned patient-state objective (`mimic-state-gate`, full MIMIC, 1B, 4h windows)

Grid `configs/grids/mimic_state_gate.yaml`, code `7529cd7`, grid `8eaa54b`, 4h windows `b35b67c`.
Probe + FT on the 8% source. Results are on the PC only; none are committed.

| cell | status (2026-10-30) | probe | FT | sensitivity diag. |
|---|---|---|---|---|
| hybrid_s0 (reuse full-1b) | eval done | — | — | n/a |
| ar_s0 (reuse full-1b) | eval done | — | — | n/a |
| state_ac_s0 | done | — | — | — |
| state_blind_s0 | paused at step 20,000 of about 30.5k (2026-10-28, GPU given to another project); ckpt kept | — | — | — |
| mlm_s0 | not started | — | — | n/a |
| state_ac_s1, state_blind_s1, mlm_s1, hybrid_s1, ar_s1 | not started | — | — | — |

## What the evidence says

- A latent-prediction loss on its own does not beat a token loss: on MIMIC (300M, 3 seeds) every
  latent-only family is 0.06-0.09 below AR/hybrid, and on PhysioNet the pre-registered A2 rule
  rejected pure latent objectives on 4/4 tasks.
- As an auxiliary to the AR code loss, the latent term gives a small, repeatable gain over AR at
  1B tokens: +0.025 under the probe (8%, 3 seeds, non-overlapping) and +0.008 [+0.002, +0.014]
  under fine-tuning (25%, paired bootstrap). At 300M the two tie.
- The frozen linear probe understates both objectives by 0.02-0.03, and it changed the 25%-data
  conclusion: under the probe more patients did not help the hybrid; under fine-tuning they did
  (0.813 -> 0.821).
- No encoder beats count-GBM with significance yet: the best pooled difference is +0.013 with a
  CI lower bound of -0.001 on 1,346 test subjects. Under matched 25% labels with a probe, GBM
  leads (0.841 vs 0.815).
- Full MIMIC at a fixed 1B tokens added nothing over 25% under fine-tuning (0.821 -> 0.822, seed 0
  only). The 35M model and the action-conditioned state objective have no results yet.
