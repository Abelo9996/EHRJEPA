# Next steps to an arXiv preprint (target: late January 2027)

Written 2026-10-30, about 13 weeks before the target. Results to date are in
`docs/EXPERIMENTS_OVERVIEW.md`. The GPU budget assumes the single RTX 4060 (8 GB) at the
measured rates: hybrid 1B about 3.6 h, AR about 2.7 h, a 4h-window state cell about 7.5 h,
eval about 17 min per cell. GPU-h figures for steps never run before are planning estimates.
Owner = Abel (human actions: approvals, DUAs, email, GPU hand-back). Me = Claude sessions.

Ground rules for every step: a claimed difference must exceed the seed spread, or the
paired-bootstrap CI must exclude 0. 3 seeds for any headline number. Don't claim a first
hybrid, a first EHR world model, that JEPA alone helps, or significance over GBM until step 6
shows it (positioning decided 2026-10-27 vs SMB-Structure 2601.22128 and Imperial 2610.29528).

## Ordered plan

| # | step | owner | GPU-h | exit criterion | depends on |
|---|---|---|---|---|---|
| 0 | Hand the GPU back to EHRJEPA | owner | 0 | owner says the other paper is done | — |
| 1 | **Finish the gate grid** (`mimic-state-gate`) | me | about 31 (see below) | all 10 cells probe + FT scored; sensitivity JSON for each state cell; results committed | 0 |
| 2 | **Gate decision** | me, owner approves | 0 | pass/fail written into HANDOFF using the rule below | 1 |
| 3 | Trial-reproduction pilot (only if 2 passes) | me | about 2-4 (rollouts) + CPU | one target-trial emulation on MIMIC (candidate: albumin, which already has a MIMIC-IV vs RCT emulation to compare against): model counterfactual rollouts give the RCT effect's sign, with a CI | 2 |
| 4 | Benchmark build (RCT-anchored, public MIMIC) | me | about 5-10 | trial list + emulation protocols + scoring script committed; each trial's observational effect reproduces the published emulation before any model is scored; baselines (GBM, AR, hybrid) scored | 3 |
| 5 | EHRSHOT DUA + INSPECT access | **owner** | 0 | both approved, data on the PC | — (submit now; long lead) |
| 6 | **Full held-out eval (about 36k subjects)** for significance vs GBM | me | about 6-12 | hybrid, AR, GBM, LR scored on the full-cache held-out split (36,516 subj) for 3 seeds; paired bootstrap hybrid - GBM and hybrid - AR | 7 (seeds) |
| 7 | Seeds on full MIMIC: hybrid/AR s1 (from the gate grid) + s2 | me | about 6.3 for s2 | 3 full-MIMIC seeds each for hybrid and AR, FT | 1 |
| 8 | Comparisons vs published pretrained models: CLMBR-T, MOTOR (EHRSHOT), frozen SMB-v1 Qwen3-1.7B (MIMIC, local only per its DUA) | me (owner signs licences) | about 5-10 | each baseline scored on our tasks or on the EHRSHOT tasks with the same protocol; licence terms recorded | 5 for EHRSHOT models; SMB is HF-gated |
| 9 | 35M scale | me | about 7.5 resume + 1 FT (seed 0); about 21 per extra seed | 35M hybrid FT mean-of-6 vs 10M 0.822, with 3 seeds if it is to be claimed | 0; lower priority than 6-7 |
| 10 | Second dataset: pretrain/eval on EHRSHOT (and INSPECT if granted) | me | about 10-20 | hybrid vs AR vs GBM (and the gate winner) under FT, 3 seeds, on a second site | 5 |
| 11 | Paper writing | me drafts, owner edits | 0 | `docs/paper/main.tex` updated: SMB cited as prior hybrid (about L202-207, L270-275), probe-vs-FT ranking matrix, every number traced to a committed summary | 6, 7, 2 (+3/4 if gate passed) |
| 12 | Outreach follow-ups | **owner** | 0 | replies logged in HANDOFF; collaboration scope agreed or declined | emails sent 2026-10-27 |
| 13 | arXiv submission | owner | 0 | submitted late Jan 2027 | 11 |

Suggested calendar: Nov wk 1-2 = steps 1-2 (and 5 submitted on day 1); Nov wk 3 - Dec wk 2 =
6, 7, 3 (or the fallback); Dec = 4 or 8, 9 if GPU allows; Jan = 10 if data arrived, then 11.
Follow-ups in step 12: nudge after about 2 weeks of silence (about 2026-11-10).

### Step 1 detail: resume the gate

Command (from HANDOFF: relaunch the same unit/grid; `ablate.py` skips done cells and resumes
state_blind_s0 from step 20,000). Run on the PC only after step 0, after checking the
keep-alive clients (`ps -ef | grep -c '[s]leep infinity'` >= 1):

```
sudo -n systemd-run --unit=gate-ac -p User=gaming_pc -p WorkingDirectory=/home/gaming_pc/EHRJEPA \
  -p MemoryHigh=16G \
  -p StandardOutput=append:/home/gaming_pc/EHRJEPA/runs/mimic-state-gate/service.log \
  .venv/bin/python scripts/ablate.py configs/grids/mimic_state_gate.yaml
```

The PC checkout must be on `b35b67c` (4h windows). If systemd still holds the old unit name,
`sudo -n systemctl reset-failed gate-ac` first. After each state cell, run
`python -m ehrjepa.eval.state_sensitivity` (true vs permuted actions) as the grid header says.

Remaining GPU time (estimate): state_blind_s0 about 2.6 h; mlm_s0 about 2.5 h (old estimate,
unverified); state_ac_s1 and state_blind_s1 about 7.5 h each; mlm_s1 about 2.5 h; hybrid_s1
3.6 h; ar_s1 2.7 h; evals about 2-3 h. Total about 31 GPU-h. The seed-0 read (state_ac,
state_blind, hybrid, ar, mlm) is available after about 6 h.

### Step 2: gate pass criteria (fixed before reading the numbers)

Pass only if all of these hold:
1. **state_ac > hybrid under fine-tuning**: mean-of-6 higher on both seeds, and the pooled
   paired-bootstrap CI for state_ac - hybrid excludes 0.
2. **state_blind < state_ac** under fine-tuning by more than the seed spread (the action input
   must matter), and the sensitivity diagnostic shows held-out true actions predict better than
   permuted actions.
3. state_ac >= AR and >= MLM under fine-tuning (no regression vs the simpler baselines).

The seed-0 read alone can only fail the gate early (state_ac below hybrid by more than the
1B seed spread of about 0.007). It cannot pass it. The grid has 2 seeds; if the result is
borderline, add s2 for state_ac, state_blind and hybrid (about 19 GPU-h) before deciding.

**Fallback if the gate fails:** drop the world-model framing and steps 3-4. The paper becomes
the evaluation-protocol paper: (a) probe vs FT re-ranks objectives (AR / hybrid / pure JEPA /
MLM / state, 2+ seeds, from the gate grid, which is useful either way); (b) hybrid > AR under FT
with paired CIs; (c) the hybrid vs GBM comparison on the 36k held-out, reported whatever the
outcome; (d) a second site (step 10) if access arrives. The negative state result is reported
as a pre-registered negative.

## Risks and mitigations

| risk | mitigation |
|---|---|
| GPU stays with the other project into late Nov | Steps 5, 8 (licences), 11 (related work, methods) and the step 4 trial list need no GPU; start them now. Cut 9 first, then 10. |
| PC job deaths (Windows commit limit, keep-alive tasks disabled, Tailscale drops) | `-p MemoryHigh=16G` on every unit; check keep-alive before launch; the bigger fixed pagefile applies at the next reboot; ablate resumes from checkpoints. |
| Gate fails or is borderline at 2 seeds | Pre-registered fallback above; pass rule fixed before reading; add s2 rather than reinterpret. |
| Hybrid vs GBM stays non-significant even on 36k | Report it as found; the thesis rests on hybrid vs AR and probe vs FT, not on beating GBM. |
| EHRSHOT / INSPECT DUA not approved before January | Submit now; paper is MIMIC-only with a stated limitation; CLMBR-T/MOTOR comparison moves to v2. |
| Licence terms block a baseline (SMB CC-BY-NC gated, CLMBR-T/MOTOR terms) | Local-only use per DUA; report only aggregate metrics; record terms per model before running. |
| Scooping (SMB, Imperial, NeurIPS WMHS workshop, the world-model papers in HANDOFF) | Cite them as prior work; scope claims to the protocol and the public, RCT-anchored benchmark; check arXiv weekly. |
| Eval mismatch: 1,346 vs 3,038 "held-out subjects" in different docs | Always state the bootstrap unit (subjects with at least one anchor); full held-out eval in step 6 replaces it. |
| Full MIMIC at 1B is under one pass of real events (about 550M) | Any scale claim needs a matched budget; treat 35M/2B+ as optional, not a headline. |
| Collaborators change the scope (Imperial / SMB replies) | Owner decides; record in HANDOFF Decisions; do not re-plan without it. |
