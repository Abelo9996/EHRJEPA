# ablate4-desynpuf

**Complete.** Two cells at seeds 1 and 2, base 6L/256d, 200M nominal token
slots, full held-out split, same protocol as `ablate2-desynpuf` /
`ablate3-desynpuf`. Results, the per-task table and findings are in
[`docs/experiments/ABLATION_RESULTS.md`](../ABLATION_RESULTS.md#grid-4-do-the-nosig-and-frozen-teacher-gains-stack-ablate4-desynpuf);
the raw numbers are in `summary.md`, appended by `scripts/ablate.py` as each
cell finished.

Question: do the two +0.6 effects from grids 2 and 3 (dropping SIGReg; a frozen 1B AR teacher instead of EMA) stack? Two cells, seeds 1 and 2, base 6L/256d, 200M token slots, full held-out split, same protocol as ablate2/ablate3. Reference rows (mean of six non-mortality AUROCs): hybrid_nosig (ablate2) .762, hybrid_frozen_ar (ablate3) .762, EMA default .755.
