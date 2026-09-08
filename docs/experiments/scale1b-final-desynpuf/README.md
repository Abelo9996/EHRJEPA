# scale1b-final-desynpuf

The final DE-SynPUF default (hybrid: causal next-latent, horizons [1,4,16], lambda_recon 0.1, lambda_sigreg 0, EMA target; see ABLATION_RESULTS.md) trained to 1B token slots at seeds 0, 1, 2 on the base 6L/256d config, evaluated on the FULL held-out split with baselines refit. The six existing 1B checkpoints (ar and the SIGReg-on hybrid at seeds 0-2, from scale1b-desynpuf and scale1b-seeds-desynpuf) are re-scored here on the same full held-out split so all 1B numbers share one evaluation set.
