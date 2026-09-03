# scripts/

Command-line entry points that orchestrate `ehrjepa.*` rather than add to it. No
modelling logic lives here; anything a test would want to assert about belongs in
`src/ehrjepa/`.

## `ablate.py`

Runs one ablation grid end to end: for each named cell, train to a token budget,
evaluate the resulting checkpoint on the shared held-out cohort, append a row to
`docs/experiments/<grid>/summary.md` and `summary.json`.

```bash
python scripts/ablate.py configs/grids/pilot_desynpuf.yaml --dry-run
nohup python scripts/ablate.py configs/grids/pilot_desynpuf.yaml &
tail -f runs/<grid>/ablate.log
```

* **Budget, not steps.** `steps = ceil(budget_tokens / (batch x max_len))`, so a
  cell that changes its window or batch keeps the same compute rather than the
  same number of updates.
* **Resumable.** A cell whose row is already in `summary.json` is skipped;
  relaunching the same command after an interruption picks up where it stopped.
  `--force` reruns anyway, `--only a,b` restricts the plan.
* **Baselines once.** `lr`/`gbm` are count-feature models and are read out of an
  earlier run's `predictions.parquet`; `random_init` is architecture-dependent
  and is computed once against the grid's own first checkpoint, then cached in
  `baselines.json`.

A grid file is a base config, a budget, and a list of runs with dotted-path
overrides -- see `configs/grids/` and the header of `ablate.py` for the full key
list.

## `throughput.py`

Steady-state tok/s for a handful of (config, precision, objective) cells, each
measured by a real 50-step training run rather than a synthetic forward pass. The
first logging window is dropped from every cell: it holds allocator growth and
the first Metal kernel compilations, and counting it understates a long run.

```bash
python scripts/throughput.py --steps 50            # writes runs/throughput/throughput.json
python scripts/throughput.py --only pilot/ar/fp32
```

This is what sizes a grid: pick the config, read the number, divide the budget.
