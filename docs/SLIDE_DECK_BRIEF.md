# EHR-JEPA — Slide-Deck Brief (for a deck-generating agent)

**How to use this file.** Each `## Slide N` block is one slide. It gives the slide
**title**, the **one-line message** (the takeaway), **bullets** (put on the slide,
tighten as needed), the **figure** to place (filename in `figures/`), the **exact
numbers** to use (do not alter — they are copied from committed result tables), and
**speaker notes**. Every AUROC is real and traceable; the negative result (Slide 11)
is intentional and should be kept. Do **not** invent numbers or inflate claims.

Project in one sentence: *we test whether a JEPA-style objective — predicting future
clinical events in representation space instead of predicting the next code — helps
for EHR foundation models, and report honestly what does and doesn't work.*

Suggested length: 12–14 slides. Tone: measured, evidence-first, no hype.

---

## Slide 1 — Title
- **Title:** EHR-JEPA: Does predicting in *representation space* help EHR foundation models?
- **Subtitle:** A joint-embedding predictive architecture for longitudinal electronic health records — what worked, what didn't.
- **Message:** A rigorous, controlled study of the JEPA idea on EHR, not a leaderboard claim.
- Speaker notes: Frame as a measurement paper. The interesting result is *where the idea holds and where it breaks*.

## Slide 2 — The problem
- **Message:** The dominant recipe spends all its capacity predicting the *identity* of the next code — much of which is arbitrary.
- Bullets:
  - EHR = irregular stream of timestamped clinical events over years.
  - Standard self-supervision = next-token autoregression over serialized codes (CLMBR, ETHOS, CEHR-GPT, EHRMamba).
  - Which billable code a clinician picks / which interchangeable drug NDC is dispensed is administrative noise — "not facts a representation of the patient needs to carry."
- Speaker notes: Motivates predicting in latent space, which can discard unpredictable detail.

## Slide 3 — The idea: JEPA for EHR
- **Message:** Predict the *representation* of future events, not the events themselves.
- Bullets:
  - Context encoder reads observed events; a predictor (told *where/when* future events fall) predicts the target encoder's *latent* for them.
  - Nothing reconstructed in input space → free to drop unpredictable detail.
  - Collapse prevented by SIGReg (LeJEPA).
- Figure: (optional) a simple schematic if the agent can draw one; else none.

## Slide 4 — Design
- **Message:** A compact, controllable architecture where "add the JEPA term to AR" is a one-line change.
- Bullets:
  - **Event embedding** = sum of code + value-bin + value + time features (age & log-gap via Fourier features); time enters twice (content + RoPE position).
  - **Encoder**: pre-norm transformer, RoPE, SwiGLU, runs causal or bidirectional; **predictor** carries only the target's *time*; **EMA target encoder**.
  - **5 objective families**: masked-span JEPA · causal next-latent · window-pooled · next-code AR · **hybrid = next-latent + tied next-code**.
  - Setting the latent weight to 0 reduces the hybrid to plain AR *exactly* (unit-tested) → every comparison is controlled.
- **Final default:** causal next-latent, horizons [1,4,16], EMA target, λ_recon 0.1, **SIGReg dropped**; 6-layer/256-wide encoder.

## Slide 5 — How we measure (protocol)
- **Message:** Relative comparisons at matched compute, on identical rows, with real controls — not absolute leaderboard numbers.
- Bullets:
  - Datasets: **DE-SynPUF** (synthetic Medicare claims; no labs/vitals), **PhysioNet 2019/2012** (ICU), and now **MIMIC-IV v3.1**.
  - **≥2-seed rule** for any decision; strict leakage discipline; **LR & GBM count-feature baselines**; **random-init controls** (same untrained architecture); 200× bootstrap CIs; frozen linear probe (+ fine-tune for ICU).
  - 7 tasks: inpatient/mortality/readmission + first-ever CKD/COPD/diabetes/HF within 365 d.
- Speaker notes: DE-SynPUF absolute AUROCs are low by construction — the story is *relative* hybrid-vs-AR at matched compute.

## Slide 6 — Result 1: which objectives transfer?
- **Message:** Masked-span latent JEPA **alone fails**; a next-latent + code hybrid is the strongest objective.
- Figure: `figures/pilot_grids_gain.png` (and optionally `figures/pilot_desynpuf_auroc.png`)
- Numbers (48M tokens, 20 cells, mean AUROC over 7 tasks):
  - Masked-span `jepa_ema*` variants: **0.678–0.690** vs their own untrained control **0.682** (gains −0.005 to +0.008 → does not clear its control).
  - AR: +0.045 over control. **Hybrid (`nextlatent_h1416_recon`) = top cell of all 20**, mean **0.7287**, +0.092 over control (above GBM's 0.7261).
- Speaker notes (the "why"): the masked-span latent target is dominated by time/context priors over low-entropy codes; a code-identity probe reads *worse* off the trained JEPA encoder (top-1 0.50) than an untrained one (0.60) or an AR encoder (0.71). *(Caveat: these diagnostics are one-off, not reproducible from committed code.)*

## Slide 7 — Result 2: scaling
- **Message:** **AR plateaus; the hybrid keeps improving.**
- Figure: `figures/scale_desynpuf.png`
- Numbers (mean AUROC):
  - 48M → 200M → 1B: AR **0.7205 → 0.7304 → 0.7436**; hybrid **0.7287 → 0.7328 → 0.7628** (1B = 3-seed, full held-out, mean-of-6 non-mortality).
  - AR does **not** improve 200M→1B on the fixed subset (drops on 6/7 tasks); hybrid improves at every step.

## Slide 8 — Result 3: the headline (money slide)
- **Message:** At 1B tokens, the hybrid beats AR on **all 6 non-mortality tasks with non-overlapping 3-seed ranges**, and beats the count baselines.
- Figure: `figures/headline_desynpuf.png`  ← **make this the centerpiece**
- Numbers (1B, 3 seeds, full 11,708-subject held-out split, mean-of-6 non-mortality): **hybrid 0.7628 vs AR 0.7436**; GBM 0.7511, LR 0.7191.
- Per task (AR → hybrid): inpatient .742→.757 · CKD .767→.783 · COPD .760→.772 · T2DM .763→.778 · HF .772→.798 · readmit .658→.688. Mortality is the one tie (.605 vs .601).
- Speaker notes: Paired bootstrap on identical subjects confirms significance on all six (p≈0.00–0.05), not mortality.

## Slide 9 — Result 4: what actually helps (ablations)
- **Message:** Two knobs each help by +0.6, and they **don't stack**.
- Figure: `figures/ablate2_desynpuf.png`
- Numbers (200M, full split, mean-of-6): default 0.7554 → **drop SIGReg 0.7615** → **frozen AR teacher 0.7615** → both together 0.7625 (only +0.001). Text-init embeddings neutral; freezing the code table costs 0.1 but removes 58% of trainable params. Default = EMA + no SIGReg (self-contained).

## Slide 10 — Result 5: few-shot
- **Message:** The hybrid leads at **every** label budget — biggest edge in the low-data regime.
- Figure: `figures/fewshot_desynpuf.png`
- Numbers (mean-of-6 non-mortality): k=32/128/512/all — LR .629/.661/.681/.719 · AR .626/.677/.713/.744 · **hybrid .658/.708/.739/.763**.

## Slide 11 — Result 6: the honest negative (keep this slide)
- **Message:** On continuous ICU state, latent objectives **without** a code loss never beat binned AR → the strong JEPA claim is **not supported**.
- Figure: `figures/a2_physionet.png`
- Numbers: on PhysioNet 2019/2012, the two code-loss-free objectives are below *every* binned-AR number on all four tasks (probe and fine-tune); `latent_only` even falls below its random-init control on the sepsis tasks. The one win: fine-tuned hybrid = **0.8403** on `sepsis_6h`, the only trained cell above GBM (0.8363) on any ICU task.
- Speaker notes (the honest boundary, quote it): *"the JEPA framing — that predicting in representation space is itself the useful part — is not supported; the claim this evidence carries is limited to the hybrid as a regularizer riding on a code-prediction loss."*

## Slide 12 — Scaling to real hospital data: MIMIC-IV v3.1
- **Message:** We've moved from synthetic claims to **real hospital data with labs & vitals** — the modality DE-SynPUF lacks.
- Figure: `figures/mimic_iv_scale.png`
- Numbers: **364,627 patients · 546,028 admissions · 94,458 ICU stays · 875 M events** (chartevents 433 M, labevents 158 M). Vocab: 112 K ICD-dx / 4,095 chart / 1,650 lab items. ~77× more events than DE-SynPUF **plus** a whole lab/vital modality.
- Status: acquired via BigQuery, verified byte-identical, characterized, ETL + tokenizer pipeline validated end-to-end on MIMIC.

## Slide 13 — Preliminary MIMIC-IV result
- **Message:** On real hospital data, the same shape holds — the hybrid edges the AR baseline on the chronic-diagnosis tasks — but at a short budget both trail strong count baselines, and the margin is within seed noise.
- Figure: `figures/mimic_hybrid_vs_ar.png`
- Numbers (~30k-subject subset, **20M tokens, 2 seeds**, full held-out; end-to-end pipeline validated on MIMIC-IV v3.1):
  - **hybrid ahead of AR on 5 of 6 non-mortality tasks** — mean-of-6 **0.7035 vs 0.6978 (Δ +0.006)**; biggest gains **HF +0.017, CKD +0.014** (same shape as the DE-SynPUF finding). Per task (AR→hybrid): inpatient .602→.596 · CKD .766→.780 · COPD .712→.713 · T2DM .736→.741 · HF .792→.810 · readmit .579→.582; mortality .861→.872.
  - Both trail the count baselines (GBM/LR reach **.88–.94** on the diagnosis tasks) — expected on MIMIC at this short budget (the diagnosis code sits directly in the count features; MEDS-Tab baselines are a famously high bar).
- **Say this honestly:** *preliminary* — a subject subset + a short in-session budget, not the full stage-B run; the +0.006 margin is within seed noise. A larger-budget run is in progress; if it finishes before the talk, swap in `figures/mimic_hybrid_vs_ar.png` (this file is regenerated) and its numbers.
- Speaker notes: the headline here is that **the whole pipeline now runs on real MIMIC-IV** (ETL→cache→train→eval) and the hybrid's direction of advantage carries over; the clean magnitude needs the full run.

## Slide 14 — Takeaways, limitations, next steps
- **Takeaways:** (1) masked-span latent JEPA alone doesn't clear its control; (2) a next-latent + code hybrid is the strongest objective at every budget; (3) AR plateaus, the hybrid scales; (4) it wins few-shot at every k; (5) on continuous ICU state the strong JEPA claim is not supported — the honest boundary.
- **Limitations (state them):** most DE-SynPUF numbers on a 3,000-subject subset; uneven seeds; DE-SynPUF has no labs/vitals; embedding table dominates params; diagnostics not reproducible from the repo; MIMIC pretraining is preliminary/partial.
- **Next:** full stage-B pretraining on MIMIC-IV; reseed single-seed cells; (pending DUA) EHRSHOT.

---

### Asset index (files in this package)
| figure | slide | shows |
|---|---|---|
| `headline_desynpuf.png` | 8 | 1B/3-seed per-task hybrid vs AR vs GBM/LR (money slide) |
| `scale_desynpuf.png` | 7 | scaling 48M→1B; AR plateau |
| `pilot_grids_gain.png` | 6 | pilot gains by objective family |
| `pilot_desynpuf_auroc.png` | 6 | pilot per-task AUROC |
| `ablate2_desynpuf.png` | 9 | ablations (SIGReg drop, teacher freeze) |
| `fewshot_desynpuf.png` | 10 | few-shot at k=32→all |
| `a2_physionet.png` | 11 | Stage A2 ICU negative result |
| `mimic_iv_scale.png` | 12 | MIMIC-IV scale + labs/vitals contrast |
| `mimic_hybrid_vs_ar.png` | 13 | preliminary MIMIC hybrid-vs-AR *(added when run finishes)* |

`PRESENTATION.md` (also in this package) is the fuller prose write-up if the agent needs more context than the per-slide notes.
