# Preprint source

Work-in-progress arXiv-style preprint for EHRJEPA. Everything here is generated
from numbers already committed under `docs/experiments/`; nothing in the paper
is a number that does not appear in one of those tables, except where the text
says so explicitly (the partial 1B seed replicate in Section 6.6).

## Build

```bash
cd docs/paper
tectonic -X compile main.tex
```

Produces `main.pdf` (24 pages, A4). Tectonic downloads what it needs on first
run and resolves the bibliography itself, so this is the whole build.

With a TeX Live installation instead:

```bash
cd docs/paper
latexmk -pdf main.tex        # runs pdflatex + bibtex + reruns
latexmk -c                   # clean intermediates, keep the PDF
```

The build must finish with no undefined references or citations. It currently
does, and with no overfull boxes.

## Layout

```
main.tex                     the paper
refs.bib                     bibliography (natbib / plainnat)
main.bbl                     resolved bibliography, kept for arXiv submission
figures/
  fig_architecture.tex       TikZ: embedding -> encoder -> predictor, target branch, losses
  fig_objectives.tex         TikZ: the five objective families on a common token strip
  fig_protocol.tex           TikZ: anchor, history window, label window, baselines, controls
  pilot_desynpuf_auroc.png   copied from docs/figures/
  pilot_grids_gain.png       copied from docs/figures/
  fewshot_desynpuf.png       copied from docs/figures/
  scale_desynpuf.png         copied from docs/figures/
```

The three `fig_*.tex` files are `\input` into `main.tex` and are wrapped in
`\resizebox{\textwidth}{!}{...}` there, so they scale to the text width rather
than carrying fixed dimensions. They compile standalone against a preamble with
`tikz` plus the `positioning`, `arrows.meta`, `fit`, `backgrounds`,
`decorations.pathreplacing` and `calc` libraries, and `amsmath`.

## Regenerating the PNG figures

The four PNGs are copies of `docs/figures/*.png`. To refresh them after new
results land:

```bash
python scripts/plot_grids.py      # docs/figures/pilot_grids_gain.png
python scripts/plot_pilot.py      # docs/figures/pilot_desynpuf_auroc.png
python scripts/plot_fewshot.py    # docs/figures/fewshot_desynpuf.png
python scripts/plot_scale.py      # docs/figures/scale_desynpuf.png
cp docs/figures/*.png docs/paper/figures/
```

Each plotting script parses the committed `summary.md` tables directly, so a
regenerated figure cannot contain a number that is not already recorded.

## Contents

7 figures, 10 tables.

| | |
|---|---|
| Fig. 1 | architecture (TikZ) |
| Fig. 2 | the five objective families (TikZ) |
| Fig. 3 | evaluation protocol (TikZ) |
| Fig. 4 | pilot held-out AUROC by task |
| Fig. 5 | pilot gain over control, by cell and family |
| Fig. 6 | few-shot AUROC against training size |
| Fig. 7 | scaling: mean AUROC against token budget, and per-task at 1B |
| Tab. 1 | per-event features in the tensor cache |
| Tab. 2 | cohort sizes and held-out prevalence |
| Tab. 3 | pilot grids 1–4, all 20 trained cells and 4 controls |
| Tab. 4 | grid 5, three-seed spread at 48M tokens |
| Tab. 5 | few-shot AUROC |
| Tab. 6 | all trained cells across 48M / 200M / 1B |
| Tab. 7 | `ar` against the hybrid across budgets |
| Tab. 8 | 1B seed replication (partial — one of four cells) |
| Tab. 9 | the two base configurations |
| Tab. 10 | commit hashes for the key result commits |

## Keeping it honest

The paper reports point estimates and, where they were computed, seed spreads.
It does not claim significance, importance or comparability to published
benchmark numbers. When new results land — in particular the three outstanding
cells of `scale1b-seeds-desynpuf` — Section 6.6 and Table 8 are what need
updating, along with the corresponding caveats in Sections 7 and 8.
