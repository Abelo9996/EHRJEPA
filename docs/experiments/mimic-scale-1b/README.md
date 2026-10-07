# mimic-scale-1b — AR vs hybrid at 1B tokens, error bars

Seed replication of the 1B point on the ~30k-subject MIMIC-IV cache, to put a spread on the
hybrid-over-AR gap. Full held-out split, 200 bootstrap, probe `auto@final`. Numbers from
`summary.md`.

## Status: 2 of 3 seeds complete
Seeds 0 and 1 finished for both objectives. Seed 2 (ar_s2, hybrid_s2) did NOT complete — the
PC's WSL ext4 virtual disk began remounting read-only from I/O errors every few minutes
(physical SSD is Healthy per SMART; the vhdx is corrupted, likely from repeated crash/shutdown
cycles). ar_s2 reached step ~22k/30.5k before the environment became unable to finish a cell.
Blocked pending a WSL vhdx repair/rebuild.

## 1B mean-of-6 (non-mortality AUROC), 2 seeds

| objective | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| AR     | 0.7716 | 0.7699 | 0.7708 | 0.0017 |
| hybrid | 0.7983 | 0.7911 | 0.7947 | 0.0072 |

- Gap (hybrid − AR, means) = **+0.0239**.
- **Seed ranges do not overlap:** worst hybrid (0.7911) > best AR (0.7716) by +0.0195. Well above
  the 300M noise floor (0.009). The +0.027 single-seed gap holds up under replication.
- AR is very tight across seeds (spread 0.0017); hybrid a bit wider (0.0072) but still separated.

## Reference (same split, not the decision metric)
GBM mean-of-6 = 0.808, LR = 0.795 — above the pretrained models on this subset (beating GBM on
MIMIC counts is a separate, harder bar). random_init@ar_s0 = 0.666.

## Combined scaling picture (seed 0, with 1B now 2-seed)
300M ar 0.768 / hybrid 0.770 (+0.002) → 1B ar 0.771±0.002 / hybrid 0.795±0.007 (+0.024,
non-overlapping) → 2B ar 0.764 / hybrid 0.798 (+0.034, seed 0 only). hybrid ≥ AR is stable;
hybrid plateaus 1B→2B.
