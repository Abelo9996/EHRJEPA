# MIMIC-IV Re-Validation Protocol

Goal: re-decide **every** design choice on MIMIC-IV, assuming **no** DE-SynPUF conclusion
transfers. Nothing is "already decided." Each decision = one experiment + one pass/fail gate.
Run in dependency order (later phases depend on earlier ones). Only scale on big GPU after Phase 4.

Hardware: RTX 4060 (8 GB), 15 GB RAM. Budgets in nominal token slots. "3 seeds" = run.seed 0,1,2.
"mean-of-6" = mean AUROC over the 6 non-mortality tasks. "Gate" = the rule that must pass to accept.

---

## Phase 0 — Calibrate (do first; everything trusts this)

Purpose: know the noise floor before reading any ablation. Without this, every later Δ is unreadable.

| # | Decision / question | Experiment | Gate (pass if) | ~GPU-h |
|---|---|---|---|---|
| 0.1 | Subject-set size | ETL 30k vs ~80k subset; compare AR mean-of-6 + its seed spread | pick smallest set whose 3-seed spread < 0.01 | 4 |
| 0.2 | Noise floor vs budget | AR, 3 seeds, at 100M / 300M / 600M; plot seed spread vs budget | fixes the **budget + seed count** every later cell must use | 12 |

Output of Phase 0: the standard `(subject set, budget, n_seeds)` used for all later phases.

---

## Phase 1 — Objective (most fundamental; do not assume hybrid)

Purpose: re-run the full objective comparison. DE-SynPUF eliminated masked-span/recon-only/window;
that elimination was claims-specific and must be re-tested where labs exist.

| # | Decision | Experiment | Gate | ~GPU-h |
|---|---|---|---|---|
| 1.1 | Which objective family wins | masked-span JEPA, recon-only, window-pooled, next-latent, AR, hybrid — Phase-0 budget, 3 seeds | winner's mean-of-6 beats 2nd by > seed spread; all beat random-init | 20 |
| 1.2 | Causal next-latent vs masked-span | the two best latent variants head-to-head | winner > other by > spread | 8 |
| 1.3 | (if masked-span competitive) re-diagnose | time-only R² + code-identity probe on MIMIC | — (explanatory) | 2 |

Gate for the phase: **a single objective is chosen on MIMIC evidence.** Everything downstream uses it.

---

## Phase 2 — Input representation (feeds the objective)

Run with the Phase-1 winning objective. These were never tested with codes + labs together.

| # | Decision | Experiment | Gate | ~GPU-h |
|---|---|---|---|---|
| 2.1 | Value handling | none vs binned vs continuous (Huber) vs gate on/off | best beats others by > spread | 10 |
| 2.2 | Time features | full vs no-age vs no-gap vs no-time; RoPE on/off | each kept feature must earn > spread | 10 |
| 2.3 | Vocab / tokenizer | 30k cap vs larger; hierarchical fallback on/off (MIMIC has 112k ICD codes) | larger/fallback must earn its cost | 8 |
| 2.4 | Window length / sampling | max_len 512 vs 1024/2048; random vs last-window | best beats 512 by > spread | 10 |

Gate: the input config is re-confirmed on MIMIC. If 2.x changed materially, **re-run Phase 1.1** once.

---

## Phase 3 — Training knobs (tune the chosen objective + input)

Re-run the DE-SynPUF ablations on MIMIC. Prioritize the two choices most likely to be DE-SynPUF artifacts.

| # | Decision | Experiment | Gate | ~GPU-h |
|---|---|---|---|---|
| 3.1 | SIGReg drop vs keep (priority) | λ_sigreg 0 vs 0.05 (claims-code anchoring may not hold with labs) | keep the better by > spread | 6 |
| 3.2 | Code-embedding init (priority) | random vs text-init; trainable vs frozen (MIMIC real ICD codes) | keep the better by > spread | 8 |
| 3.3 | Target encoder | EMA vs shared vs frozen-AR-teacher | keep the better by > spread | 8 |
| 3.4 | Horizons + λ_recon | [1] vs [1,4,16]; recon 0.1 vs 0.3 | keep the better by > spread | 8 |
| 3.5 | Encoder size | 4L/192 vs 6L/256 vs 8L/384 | pick best cost/accuracy on 4060 | 10 |

(Lower-tier, optional sweep: lr / weight-decay / warmup / batch — only if Phase 3 leaves accuracy on the table.)

---

## Phase 4 — Lock & scale

| # | Step | Gate |
|---|---|---|
| 4.1 | Full-config confirmation run (winning objective + input + knobs), 3 seeds, clean | every chosen option reproduces its advantage |
| 4.2 | **Decision: commit large GPU** for full-MIMIC 1B+ scale-up | only if 4.1 passes; else revise the failing decision first |

---

## Rules that apply to every cell
1. Always 3 seeds (the 2-seed spread was too large at 20-100M).
2. Report **effect size vs seed spread**, not just the mean — a Δ inside the spread is "no decision."
3. Metric is **relative** (chosen-vs-alternative at matched compute), not "beat GBM" (MIMIC count baselines are a separate, harder bar).
4. Pre-register each gate before running; a result inside the spread means "keep the simpler/cheaper option."
5. One `ablate.py` grid per phase; everything resumable via systemd on the 4060.

## Rough total
~170 GPU-h on the one 4060 (~1-2 weeks wall-clock, unattended). That buys a fully MIMIC-justified
config before any large-GPU spend.
