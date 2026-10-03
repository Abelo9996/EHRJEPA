# EHR-JEPA — Design-Decisions Slide Brief (companion to the main deck)

**How this fits with the main package.** This is a companion to `SLIDE_DECK_BRIEF.md`
(in `ehrjepa-slidedeck-package.zip`). It adds a short **"why these choices"** block to
the same talk. Slot these slides **right after Slide 4 (Design)** and before Slide 5
(Protocol) — they justify the design just introduced. Same format, same tone (evidence-first,
no hype). Figure `figures/ablate2_desynpuf.png` is included here and is the same file as in
the main package, so nothing conflicts. Every number is copied from committed ablation tables
(`docs/experiments/ABLATION_RESULTS.md`, `PILOT_RESULTS.md`, `A2_RESULTS.md`); the fuller
per-decision detail is in `DESIGN_DECISIONS_REFERENCE.md`.

The one-line meta-message: **the architecture is ablation-driven, and two "obvious" choices
were overturned by evidence** — SIGReg (the canonical JEPA anti-collapse term) *hurt*, and
text-initialized code embeddings were *neutral*.

---

## Slide 4a — Every choice is earned (the objective)
- **Message:** The objective isn't asserted — 20 objective variants were run, and the data picked the hybrid.
- Bullets (choice → what the experiment showed):
  - **Latent-only fails, code loss is required.** Masked-span JEPA alone (0.678–0.690) does not clear its untrained control (0.682); AR transfers (+0.045); the **hybrid is the single best of 20 cells** (0.7287, +0.092). On ICU, latent objectives *without* a code loss never beat binned AR.
  - **Causal next-latent, not masked-span** — masked-span fails *because* the target is dominated by time/context priors: a map from the mask token's **time alone** recovers R²≈0.58 of the predictor output; a code-identity probe reads worse off the trained JEPA encoder (top-1 0.50) than untrained (0.60) or AR (0.71).
  - **Multiple horizons [1,4,16]** — horizon-[1]-only (0.7512) is one of the two worst variants (vs default 0.7554).
  - **Value binning, not a continuous-latent objective** — on ICU, continuous-value AR ≈ binned AR, and the code-loss-free continuous latent objective fails ("not supported").
- Speaker note: this is the slide that says "we didn't guess the objective — we measured it."

## Slide 4b — Every choice is earned (the knobs) + two overturned assumptions
- **Message:** The training knobs were ablated at matched compute; two textbook choices lost.
- Figure: `figures/ablate2_desynpuf.png`
- Bullets (all mean-of-6 AUROC, 200M tokens, full held-out):
  - **Drop SIGReg (λ=0)** — **0.7615 vs 0.7554 default (+0.6, the single best knob)**, both seeds. *Overturned assumption #1:* the canonical anti-collapse regularizer hurts once a tied code loss already anchors the representation.
  - **EMA target encoder** — a frozen-AR teacher matches it (0.7615) but the gains **don't stack** (both = 0.7625); shared-target is worst (0.7510). EMA wins on being **self-contained** (no staged checkpoint to ship).
  - **Text-init code embeddings: neutral** (0.7561, +0.001). *Overturned assumption #2* — initializing the 30k-code table from text embeddings doesn't help. Freezing it costs only 0.1 but removes **58% of trainable params** (a documented size trade).
  - **λ_recon 0.1 ≈ 0.3** (0.7554 vs 0.7593, within seed noise) → light code loss suffices.
  - **Size-robust:** hybrid > AR at small/base/large (+0.008 / +0.005 / +0.006); the gap **doesn't close** with size.
- Speaker note: land the two overturned assumptions — they make the design story credible rather than lucky.

---

### If you want a single slide instead of two
Title: **"The design is ablation-driven (and evidence overturned two obvious choices)."**
Keep: the hybrid-is-best-of-20 line, the drop-SIGReg (+0.6) result, text-init-neutral, and the
`ablate2_desynpuf.png` figure. Put the rest in speaker notes / the reference MD.

### Asset
| figure | shows |
|---|---|
| `figures/ablate2_desynpuf.png` | AR vs hybrid at small/base/large + the base-size hybrid-knob variants (SIGReg drop, shared target, horizon-[1], recon-0.3) — the evidence behind Slides 4a/4b |
