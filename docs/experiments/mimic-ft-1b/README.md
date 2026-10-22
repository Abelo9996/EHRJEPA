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
