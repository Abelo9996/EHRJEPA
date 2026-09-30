# EHR-JEPA — Design Decisions & the Experiments That Justify Them (reference)

Companion reference to `DESIGN_DECISIONS_BRIEF.md` and the main `SLIDE_DECK_BRIEF.md`.
Every choice in the final default config (`configs/pretrain_default.yaml`) is tied to the
experiment that decided it. All numbers are copied from committed tables:
`docs/experiments/ABLATION_RESULTS.md` (ablate2/3/4, 200M tokens, full 11,708-subject held-out,
200 bootstrap), `PILOT_RESULTS.md` (48M pilot), `A2_RESULTS.md` (ICU). "mean-of-6" = mean AUROC
over the six non-mortality tasks.

---

## 1. Objective family: hybrid = causal next-latent + tied next-code
- **Question:** does predicting future events *in representation space* help, and does it need a code loss?
- **Experiment:** 20 objective cells across pilot grids 1–4 (48M tokens); Stage A2 on continuous ICU state.
- **Finding:** masked-span latent JEPA **alone** scores 0.678–0.690 vs its own untrained control 0.682 (gains −0.005 to +0.008 → does not clear control). AR transfers (+0.045). The **hybrid `nextlatent_h1416_recon` is the single best cell of all 20** (mean 0.7287, +0.092 over control, above GBM's 0.7261). On ICU, the two code-loss-free latent objectives never beat binned AR on any task/seed → "the JEPA framing … is **not supported**; the claim … is limited to the hybrid as a regularizer riding on a code-prediction loss."
- **Decision:** hybrid — the latent term is kept only *with* the code loss.

## 2. Latent scheme: causal next-latent, not masked-span
- **Experiment:** pilot masked-span (`jepa_ema*`) vs causal next-latent; grid-2 diagnostics.
- **Finding (why masked-span fails):** a ridge map from the mask token's **time features alone** recovers **R²=0.58** of the predictor output (0.79 with context); a code-identity probe reads **worse** off the trained JEPA encoder (top-1 **0.50**) than off an untrained one (**0.60**) or an AR encoder (**0.71**); context-mean→target cosine 0.73 without SIGReg. The latent target is dominated by time/context priors over low-entropy discrete events. *(Caveat: these diagnostics are one-off, not reproducible from committed code.)*
- **Decision:** causal next-latent, dense (one prediction per position per horizon).

## 3. Horizons [1, 4, 16], not a single horizon
- **Experiment:** ablate2 `horizon [1] only`.
- **Finding:** mean-of-6 **0.7512** (both seeds 0.7511/0.7513) — one of the two worst knobs, below the default 0.7554.
- **Decision:** multi-horizon [1,4,16].

## 4. Drop SIGReg (λ_sigreg = 0), against the LeJEPA default 0.05
- **Experiment:** ablate2 `no SIGReg`.
- **Finding:** mean-of-6 **0.7615 vs default 0.7554 (+0.6)** — the **single best knob**, and best at *both* seeds individually (0.7620 / 0.7611).
- **Decision:** drop SIGReg. **Interpretation:** the tied code loss already prevents collapse, so the anti-collapse regularizer only distorts the representation. *(This overturns the canonical JEPA recipe.)*

## 5. Target encoder: EMA, not shared or frozen-AR-teacher
- **Experiment:** ablate2 (`shared target`), ablate3 (`hybrid_frozen_ar`), ablate4 (stacking).
- **Finding:** shared-target 0.7510 (worst); frozen-AR-teacher **0.7615** (matches no-SIGReg's +0.6); ablate4 — no-SIGReg + frozen-teacher **do not stack** (both = 0.7625 vs 0.7615 for either alone).
- **Decision:** EMA — equal accuracy to the frozen teacher, but **self-contained**: no separately trained 1B-token checkpoint has to be built and shipped to the training machine first.

## 6. λ_recon = 0.1 (light code loss), not 0.3
- **Experiment:** ablate2 `recon 0.3`.
- **Finding:** 0.7593 vs default's 0.1 at 0.7554 — within **0.004**, inside the 0.002–0.005 seed spread across these knobs (i.e. within noise).
- **Decision:** 0.1 (a light code loss is enough).

## 7. Code embeddings: trainable random-init, not text-init, not frozen
- **Experiment:** ablate3 `hybrid_textinit`, `hybrid_textinit_frozen`, `ar_textinit`.
- **Finding:** text-init **neutral** — `hybrid_textinit` 0.7561 (+0.001 vs 0.7554), `ar_textinit` 0.7490 (−0.0017), both inside seed spread. **Freezing** the text table costs 0.1 (0.7546) while removing **58%** of trainable params (the 30k-code table is 7.68M of 13.21M).
- **Decision:** trainable random-init. Text-init doesn't earn its complexity; freezing is a documented accuracy-for-size trade, not the default. *(This overturns the intuition that text-init embeddings should help.)*

## 8. Value representation: per-code decile bins + gate; no continuous-latent objective
- **Experiment:** Stage A2 `ar_cont` (Huber on z-scored values) vs `ar_bins`; `latent_cont` (next-latent + Huber, no code loss).
- **Finding:** continuous-value AR ≈ binned AR (mixed; e.g. sepsis_6h 0.8206 vs 0.8175); `latent_cont` is below every `ar_bins` number on all four ICU tasks. 
- **Decision:** keep per-code decile binning with a "has-value" gate; no continuous latent objective.

## 9. Encoder size: 6-layer / 256-wide
- **Experiment:** ablate2 size sweep — small (4L/192d), base (6L/256d), large (8L/384d).
- **Finding:** hybrid > AR at **every** size (mean-of-6 +0.0079 / +0.0047 / +0.0055) and **the gap does not close** as size grows.
- **Decision:** 6L/256d as the scale default — the hybrid advantage is size-robust, and this size fits the RTX 4060 at bf16.

---

## The two ablation tables behind Slides 4a/4b

**Hybrid knobs at base size (6L/256d), 200M tokens, mean-of-6 (from `ABLATION_RESULTS.md` (b)):**

| knob | mean-of-6 | vs default | note |
|---|---|---|---|
| default (EMA, SIGReg 0.05, h[1,4,16], recon 0.1) | 0.7554 | — | seed 0 |
| **no SIGReg** | **0.7615** | **+0.0061** | best knob (both seeds) |
| recon 0.3 | 0.7593 | +0.0039 | within seed noise |
| horizon [1] only | 0.7512 | −0.0042 | worst tier |
| shared target | 0.7510 | −0.0044 | worst tier |

**Teacher / init (200M tokens, mean-of-6, from `ABLATION_RESULTS.md` grid 3):**

| config | mean-of-6 | vs EMA default | takeaway |
|---|---|---|---|
| `hybrid_frozen_ar` | 0.7615 | +0.0061 | = no-SIGReg gain; doesn't stack (ablate4) |
| `hybrid_textinit` | 0.7561 | +0.0007 | text-init neutral |
| `hybrid_textinit_frozen` | 0.7546 | −0.0008 | freezing table costs ~0.1, −58% params |
| `ar_textinit` | 0.7490 | −0.0064 | text-init neutral for AR too |

Baselines on the same split: `gbm` 0.7511, `lr` 0.7191 (mean-of-6).

## Talk framing
- **Headline:** the design is *earned*, not assumed — 20 objective variants + a size sweep + knob/teacher/init grids.
- **The two overturned assumptions** (SIGReg hurt; text-init neutral) are the credibility hook — they show the choices came from measurement, not convention.
- **Honest caveats:** the base-size default row is a single seed (some comparisons are 2-seed-vs-1-seed); all at 200M tokens on DE-SynPUF (no labs/vitals); knob differences are 0.3–0.7 pts against a 0.3–0.5 pt seed spread, so no-SIGReg / shared-target / horizon-[1] are outside noise while recon-0.3 is inside it.
