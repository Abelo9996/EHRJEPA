# ablate4-desynpuf

Question: do the two +0.6 effects from grids 2 and 3 (dropping SIGReg; a frozen 1B AR teacher instead of EMA) stack? Two cells, seeds 1 and 2, base 6L/256d, 200M token slots, full held-out split, same protocol as ablate2/ablate3. Reference rows (mean of six non-mortality AUROCs): hybrid_nosig (ablate2) .762, hybrid_frozen_ar (ablate3) .762, EMA default .755.
