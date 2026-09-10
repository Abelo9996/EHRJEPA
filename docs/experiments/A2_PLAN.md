# A2: a pre-registered test of latent prediction on continuous ICU state

Written before training. Same design and decision rule as
`docs/paper/main.tex`, Section "Where latent prediction stands, and a
pre-registered test on continuous state".

## Why

Every pure latent-prediction variant measured on DE-SynPUF loses to next-code
AR at matched compute (`docs/experiments/PILOT_RESULTS.md`,
`SCALE_RESULTS.md`). The hybrid's latent term is worth about two points of
mean AUROC over AR at 1B tokens on the full held-out split, three points on
`readmission_30d`, and about three points at `k=32` few-shot
(`SCALE_RESULTS.md`), but every one of those numbers is measured with the
tied code loss present; the dense and window-pooled latent cells with no code
loss do not beat AR at any budget they were run at
(`docs/experiments/PILOT_RESULTS.md`). The diagnosed mechanism is that a
ridge map from mask-token time features alone recovers most of what the
predictor outputs ($R^2 = 0.58$ vs. $0.79$ with context), and a linear probe
for code identity reads *worse* off the trained JEPA encoder (0.50) than off
an untrained one (0.60) — consistent with a latent target dominated by
time-and-context priors over discrete, low-entropy events, which a predictor
can satisfy without carrying code identity.

Clin-JEPA (`clinjepa` in `refs.bib`) reports a margin over LightGBM (.827 vs.
.851) and an LSTM (.865 vs. .883) of the same size as this paper's hybrid-vs-AR
margin, on continuous hourly ICU state rather than discrete low-entropy
events, with a pretrained Qwen3-8B/LoRA encoder, 430 GPU-hours on eight H200s,
and no matched-compute next-token control. Whether a latent objective does
work of its own on continuous state — rather than riding on a code-prediction
loss the way it appears to on DE-SynPUF — is untested here. This plan
specifies that test.

## Datasets

Converted to the same MEDS layout used throughout this project, one event per
measurement (`VAR//<name>` with a numeric value), via
`src/ehrjepa/data/etl/physionet2019.py` and
`src/ehrjepa/data/etl/physionet2012.py`.

| source | stays | events | % numeric |
|---|---|---|---|
| PhysioNet/CinC 2019 sepsis challenge (Reyna et al. 2019) | 40,336 | 12.2M | 99.1% |
| PhysioNet/CinC 2012 mortality challenge (Silva et al. 2012) | 12,000 | 5.3M | 99.3% |

Measured directly on `data/meds/physionet2019` and `data/meds/physionet2012`.

## Tasks

Built in `src/ehrjepa/eval/icu_tasks.py`, registered in
`src/ehrjepa/eval/tasks.py`. History strictly before the anchor, label event
strictly after — the same `windows_at` discipline as every other task in this
project.

| task | anchor | prevalence (held-out) | note |
|---|---|---|---|
| `sepsis_6h` | up to 8 hash-drawn anchors/stay, hour >= 6 | 1.7% | onset within 6h; anchors at/after onset dropped; multi-anchor, rows not independent within a stay |
| `sepsis_stay` | hour 12 | 5.6% | onset anywhere in the stay |
| `mortality_inhospital/24h` | hour 24 | 13.9% | outcome parked at hour 49 by the ETL, past both anchors |
| `mortality_inhospital/48h` | hour 48 | 13.9% | same |

## Baselines

Count-feature `lr`/`gbm` (Section "Baselines, probes and controls" of the
paper, unchanged construction), full held-out split, 200 bootstrap resamples.
Source: `docs/experiments/2026-09-10-eval-physionet2019/results.md`,
`docs/experiments/2026-09-10-eval-physionet2012/results.md`.

| task | lr | gbm |
|---|---|---|
| `sepsis_6h` | 0.785 | 0.836 |
| `sepsis_stay` | 0.768 | 0.769 |
| `mortality_inhospital/24h` | 0.789 | 0.829 |
| `mortality_inhospital/48h` | 0.828 | 0.859 |

## Objectives (matched compute, two seeds each)

1. **AR** over code + decile value bins (this project's `ar`, applied to the
   ICU vocabulary).
2. **AR + continuous value regression**: the code term unchanged, plus a
   Huber loss on each event's per-code z-scored numeric value.
3. **Hybrid**: next-latent + code loss + bin loss
   (`nextlatent_h1416_recon`, unchanged).
4. **Latent + continuous value regression, no code loss**: the next-latent
   term of (3) plus the Huber value term of (2), tied code-identity term
   removed.
5. **Pure latent**: the next-latent term alone, no code loss, no value
   regression.
6. **Hybrid with a pretrained encoder**: Qwen2.5-0.5B with LoRA, over events
   serialized from this project's code *descriptions* (the same descriptions
   `hybrid_textinit` uses, `docs/experiments/ABLATION_RESULTS.md` Grid 3),
   not free text.

## Decision rule (fixed before results)

If the latent objectives without a code loss — (4) and (5) — beat the binned
AR baseline (1) on these continuous-state tasks, latent prediction is doing
work that token prediction is not, on this evidence.

If they do not, the paper's claim is limited to the hybrid as a regularizer
riding on a code-prediction loss, and the JEPA framing — that predicting in
representation space is itself the useful part — is not supported by
anything measured in this project.

## Status

Not run. This document and Section "Where latent prediction stands, and a
pre-registered test on continuous state" of `docs/paper/main.tex` are written
before training starts.
