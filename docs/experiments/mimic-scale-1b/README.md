# mimic-scale-1b — AR vs hybrid at 1B tokens, 3-seed error bars

Seed replication of the 1B point on the ~30k-subject MIMIC-IV cache, to put a spread on the
hybrid-over-AR gap. Full held-out split, 200 bootstrap, probe `auto@final`. Numbers from
`summary.md`.

## Status: COMPLETE — 3 of 3 seeds, both objectives.

## 1B mean-of-6 (non-mortality AUROC), 3 seeds

| objective | seed 0 | seed 1 | seed 2 | mean | range |
|---|---|---|---|---|---|
| AR     | 0.7716 | 0.7699 | 0.7668 | **0.7694** | 0.7668–0.7716 |
| hybrid | 0.7983 | 0.7911 | 0.7935 | **0.7943** | 0.7911–0.7983 |

- Gap (hybrid − AR, means) = **+0.0249**.
- **Seed ranges do not overlap:** worst hybrid (0.7911) > best AR (0.7716) by +0.0195 — above the
  0.009 noise floor on every pairwise comparison. The +0.027 single-seed gap holds under full
  replication.
- AR spread 0.0048, hybrid spread 0.0072 — both tight; the +0.025 gap is ~3–5× either spread.

## Reference (same split, not the decision metric)
GBM mean-of-6 = 0.808, LR = 0.795 — above the pretrained models on this subset (beating GBM on
MIMIC counts is a separate, harder bar). random_init@ar_s0 = 0.666 (trained models clear the
untrained control by +0.10–0.13).

## Combined scaling picture
300M ar 0.768 / hybrid 0.770 (+0.002) → 1B ar 0.769±0.002 / hybrid 0.794±0.003 (+0.025,
non-overlapping, 3 seeds) → 2B ar 0.764 / hybrid 0.798 (+0.034, seed 0 only). Verdict: hybrid > AR
is real and replicated at 1B; the gap is stable-to-growing with scale; hybrid plateaus 1B→2B.

## Run note (infrastructure)
Seeds 0,1 and most of seed 2 ran across repeated interruptions (gaming pauses, a transient WSL
ext4 read-only remount, and WSL VM teardown killing detached jobs). Root cause of the detached-job
deaths: WSL tears down its VM when no session is held open, and WSL ignores systemd linger; fixed
by holding a keep-alive WSL session from the driving machine while a supervisor relaunched the
systemd unit on any death. Physical SSD healthy throughout. All final numbers are from each cell's
completed `final.pt`.
