# mimic-full-1b / mimic-ft-full — full MIMIC-IV at 1B tokens (seed 0)

Pretraining on all of MIMIC-IV v3.1 (train 291,495 subjects / 546.4M events; 248,743 usable after
min_len), tokenized with the 8% fit (`--reuse-fit-from`, 99.91% of events map directly). 0 of the 8%
held-out subjects are in the full train split. Each cell overrides `data.cache_dir` to the full cache
and is scored on the 8% source: same 3,038 test subjects, labels and LR/GBM (0.795/0.808) as
mimic-scale-1b, Eval-A and mimic-ft-1b. Fine-tuning in `docs/experiments/mimic-ft-full`. Mean-of-6 AUROC.

| encoder | probe | fine-tuned | fine-tuned mortality |
|---|---|---|---|
| hybrid, full | 0.803 | **0.822** | 0.944 |
| AR, full | 0.788 | 0.814 | 0.948 |
| GBM / LR (counts) | 0.808 / 0.795 | — | 0.935 / 0.939 |

Data-scaling series for hybrid (fine-tuned): 8% 0.813 (3 seeds) -> 25% 0.821 (3 seeds) -> full 0.822
(seed 0). Probe: 0.794 -> 0.796 -> 0.803. At a fixed 1B-token budget, 1B tokens is ~22 passes over the
8% train split but only ~1.8 over the full one; the flat 25%->full fine-tuned step suggests the token
budget, not the data, now limits. Single seed for the full run.
