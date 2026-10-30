# mimic-ft-1b — fine-tuning vs frozen probe vs count-GBM (seed 0)

Fine-tunes existing seed-0 checkpoints end to end (`reuse_checkpoint`, no pretraining) on the 8%
source: the same 3,038 held-out subjects, labels and LR/GBM as `mimic-scale-1b` and the Eval-A runs
in `mimic-data25-1b`. Settings from the a2ft grids (5 epochs, early stop on tuning AUROC). Numbers
from `summary.md`; mean-of-6 = non-mortality AUROC.

| model | frozen probe | fine-tuned | mortality (ft) |
|---|---|---|---|
| hybrid, pretrained on 8% | 0.798 | **0.819** | 0.944 |
| hybrid, pretrained on 25% | 0.797 | **0.818** | 0.943 |
| AR, pretrained on 25% | 0.785 | 0.814 | 0.949 |
| AR, pretrained on 8% | 0.772 | 0.803 | 0.941 |
| same encoder from random init (`ft_random`) | — | 0.689 | 0.874 |
| GBM on counts | — | 0.808 | 0.935 |
| LR on counts | — | 0.795 | 0.939 |

- Fine-tuning adds +0.02–0.03 to every encoder; fine-tuned hybrid is +0.011 above GBM on mean-of-6.
- Pretraining is worth ~+0.13 over the identical encoder fine-tuned from scratch.
- Hybrid > AR after fine-tuning (+0.016 at 8% pretraining, +0.004 at 25%).
- 25% pretraining does not move fine-tuned hybrid (0.819 -> 0.818); it lifts AR by +0.011, matching Eval-A.

**Caveat:** single seed. The +0.011 over GBM is inside the per-task bootstrap intervals; seeds 1–2
(fine-tuning the s1/s2 checkpoints) are needed before it can be read as beating GBM.

## Seeds 1–2 (`docs/experiments/mimic-ft-1b-s12`, grid `configs/grids/mimic_ft_1b_s12.yaml`)

Same cohort and settings; fine-tunes the seed-1/2 pretraining checkpoints; lr/gbm read from the
seed-0 grid's predictions. Combined 3-seed mean-of-6 (fine-tuned, 8% labels, 3,038 test subjects):

| fine-tuned encoder | s0 | s1 | s2 | mean | vs GBM 0.808 |
|---|---|---|---|---|---|
| hybrid, pretrained on 25% | 0.818 | 0.820 | 0.825 | **0.821** | +0.013 (every seed >= +0.010) |
| AR, pretrained on 25% | 0.814 | 0.810 | 0.816 | 0.813 | +0.005 |
| hybrid, pretrained on 8% | 0.819 | 0.809 | 0.811 | 0.813 | +0.005 (s1 only +0.0004) |
| AR, pretrained on 8% | 0.803 | 0.810 | 0.806 | 0.806 | −0.002 |

- Hybrid pretrained on 25% of patients (unlabeled) and fine-tuned on the same labels GBM uses is above
  GBM on all three seeds; the gap (+0.013) is ~2x its seed range (0.006).
- At 25% pretraining hybrid > AR with non-overlapping seed ranges (min 0.818 > max 0.816); at 8% the
  ranges overlap.
- Under fine-tuning, 25% pretraining does help hybrid (0.813 -> 0.821) — the frozen probe hid this
  (probe: 0.794 -> 0.796). The seed-0-only reading above ("does not move") is superseded.
- Not yet done: a paired bootstrap of hybrid − GBM on the shared anchors.
