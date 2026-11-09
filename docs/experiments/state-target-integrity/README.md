# State target and horizon integrity

2026-11-06. Two structural audits on **MIMIC demo training crops**, with no model
training or held-out evaluation. Three seeds independently initialize a tiny
2-layer, 32-dimensional EMA model and sample 8 batches of 4 crops, up to 512
events each. Within each seed the comparison uses identical weights and inputs.

## Target content

Perturb action code/value content while preserving observations, token roles,
positions and times. Repeat with observation perturbations as a positive control.
The metric is mean `||changed_target - target|| / ||target||` over windows with
observations. These are sampled windows, not independent patients.

| Seed | Windows | Legacy action change | Observation-only action change | Observation-only observation change |
| --- | ---: | ---: | ---: | ---: |
| 0 | 264 | 0.000174 | 0 | 0.499634 |
| 1 | 194 | 0.000379 | 0 | 0.605061 |
| 2 | 257 | 0.000574 | 0 | 0.693804 |

Legacy targets contain a direct action-content path even when only observation
outputs are pooled. The effect is **small at random initialization**; this says
nothing yet about the size of the effect after training or on downstream scores.
Observation-only targets eliminate this direct dependence without being constant.
Additional regression tests remove action tokens entirely and alter cached time
gaps, verifying that token positions and action-derived gaps do not reintroduce it.

## Time horizon

The old model indexes nonempty windows. Its `k` therefore need not be a calendar
offset. On the same crops, aggregated across seeds:

| k | Legacy pairs | Wrong calendar offset | Exact-calendar pairs | Wrong offset after fix |
| --- | ---: | ---: | ---: | ---: |
| 1 | 523 | 205 (39.2%) | 318 | 0 |
| 2 | 444 | 282 (63.5%) | 240 | 0 |
| 4 | 372 | 329 (88.4%) | 179 | 0 |

New mode matches an observation-bearing bin exactly `k × 4h` later **by bin
start**, skips absent target bins and preserves crop-completeness checks. It does
not claim that every source-to-target event gap is exactly `k × 4h`. It uses
sparse lookup, so a multi-year record gap cannot allocate years of empty states.
The time input is the nominal bin offset rather than the time of a future event.
Different pair counts are part of the result and must be accounted for in training
comparisons. At k>1, a source's exact target may differ from its legacy target;
the exact-calendar arm is not just a subset of the old pairs.

## Reproduce

```bash
.venv/bin/python -m ehrjepa.eval.state_integrity \
  --cache data/cache/mimic-demo \
  --codes-file configs/actions/mimic_icu_action_items.txt \
  --out docs/experiments/state-target-integrity/mimic-demo.json
```

The [JSON](mimic-demo.json) contains per-seed counters and input/code hashes.
`--checkpoint /path/to/final.pt` evaluates fixed trained weights instead; seeds
then resample crops only. That experiment has **not** run: the trained checkpoints
are on the unreachable PC. The audit supports calendar-anchored models and returns
nonzero on failed or inconclusive corrected-arm controls.

The independent config switches are:

```yaml
objective:
  kind: state
  state_target_view: observations  # legacy default: all
  state_horizon_mode: calendar    # legacy default: occupied
```

Changing either switch requires a separate training run; resuming a checkpoint
under changed semantics is rejected. Old checkpoints still load with their old
defaults. Observation-only targets compact the retained tokens before encoding
and recompute time gaps between observations; simply masking attention keys would
leave position/count and cached-gap paths.

These changes do **not** establish causal dynamics. Online states still contain
actions; the predictor still receives intervening post-baseline actions and an
action count; action/observation definitions and source timestamps need further
validation. This “observation” set is the complement of the configured action
table, not an adjudicated set of physiological measurements. Prior methods have
not been audited. The historical GPU grid remains paused.
