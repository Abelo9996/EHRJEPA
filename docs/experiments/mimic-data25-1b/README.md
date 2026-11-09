# mimic-data25-1b — 25% vs 8% of MIMIC-IV patients at a fixed 1B-token budget

25% subset = every 4th patient (91,157 subjects; train 72,900 / 138.0M events), a strict superset of
the 8% subset, tokenized with the 8% fit (`tokenize build --reuse-fit-from`). Same model, objectives,
budget and seeds as `mimic-scale-1b`. Frozen linear probe throughout; mean-of-6 AUROC.

**Eval-A** (`eval_on8/`): 25%-pretrained encoders scored on the 8% source — same 3,038 test subjects,
same probe labels, same LR/GBM (0.795/0.808). Isolates the pretraining-data effect.

| encoder | pretrained on 8% (3 seeds) | pretrained on 25% (3 seeds) | change |
|---|---|---|---|
| AR | 0.769 (0.767–0.772) | 0.784 (0.781–0.786) | +0.015, non-overlapping |
| hybrid | 0.794 (0.791–0.798) | 0.796 (0.793–0.797) | +0.001, no gain |

**Eval-B** (`summary.md`): everything at 25% (probe and LR/GBM refit on 25% labels, 25% test set):
AR 0.8065/0.8058/0.8027, hybrid 0.8177/0.8145/(s2 in summary); GBM 0.841, LR 0.830, random_init 0.706.
GBM gains +0.033 from 3x labels — more than the hybrid probe does.

Reading: more pretraining patients improve AR's frozen representation but not hybrid's; hybrid's
Eval-B gain comes from more probe labels. Fine-tuning (`mimic-ft-1b`) is the follow-up.
Run note: the grid was stopped on 2026-10-09 and finished by a relaunch outside the driving session
(08:28–09:02); hybrid_s2 resumed from its step-26000 checkpoint and completed normally.
