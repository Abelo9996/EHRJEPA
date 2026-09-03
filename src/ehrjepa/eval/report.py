"""Turn an evaluation result dict into markdown.

The JSON written next to the markdown is the record; this module only formats
it. Numbers are reported with their bootstrap interval and nothing else -- no
ranking, no highlighting, no adjective.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = ["render", "write"]


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def _ci(entry: Mapping | None, digits: int = 3) -> str:
    if not entry:
        return "--"
    point = _fmt(entry.get("point"), digits)
    if point == "--":
        return "--"
    return f"{point} [{_fmt(entry.get('lo'), digits)}, {_fmt(entry.get('hi'), digits)}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def render(results: Mapping) -> str:
    """The markdown report for one :mod:`ehrjepa.eval.run` invocation."""
    lines: list[str] = []
    source = results.get("source", "?")
    split = results.get("eval_split", "held_out")
    lines.append(f"# Downstream evaluation -- {source}")
    lines.append("")
    lines.append(
        f"Metrics are on the `{split}` split. Intervals are percentile bootstrap over "
        f"subjects, {results.get('n_boot', 1000)} resamples, 95%."
    )
    lines.append("")

    meta = _table(
        ["", ""],
        [
            ["source", f"`{source}`"],
            ["MEDS", f"`{results.get('meds_dir', '')}`"],
            ["cache", f"`{results.get('cache_dir', '')}`"],
            ["tasks", f"`{results.get('task_dir', '')}`"],
            ["anchor seed", str(results.get("anchor_seed", ""))],
            ["commit", f"`{results.get('commit', '')}`"],
            ["created", str(results.get("created", ""))],
            ["runtime (s)", _fmt(results.get("runtime_seconds"), 1)],
        ],
    )
    lines += meta + [""]

    lines.append("## Models")
    lines.append("")
    rows = []
    for name, spec in results.get("models", {}).items():
        rows.append(
            [
                f"`{name}`",
                spec.get("kind", ""),
                f"`{spec.get('checkpoint', '')}`" if spec.get("checkpoint") else "--",
                spec.get("features", ""),
            ]
        )
    lines += _table(["model", "kind", "checkpoint", "features"], rows) + [""]

    lines.append("## Cohorts")
    lines.append("")
    rows = []
    for task, entry in results.get("tasks", {}).items():
        counts = entry.get("counts", {})
        prevalence = entry.get("prevalence", {})
        row = [f"`{task}`"]
        for name in ("train", "tuning", split):
            n = counts.get(name)
            rate = prevalence.get(name)
            row.append("--" if n is None else f"{n} ({_fmt(rate, 4)})")
        rows.append(row)
    lines += _table(["task", "train n (rate)", "tuning n (rate)", f"{split} n (rate)"], rows) + [""]

    for metric, digits in (("auroc", 3), ("auprc", 3), ("brier", 4), ("calibration_slope", 3)):
        lines.append(f"## {metric.replace('_', ' ').upper()}")
        lines.append("")
        names = list(results.get("models", {}))
        rows = []
        for task, entry in results.get("tasks", {}).items():
            row = [f"`{task}`"]
            for name in names:
                model = entry.get("models", {}).get(name, {})
                row.append(_ci(model.get("metrics", {}).get(metric), digits))
            rows.append(row)
        lines += _table(["task", *names], rows) + [""]

    paired = [
        (task, entry.get("paired", []))
        for task, entry in results.get("tasks", {}).items()
        if entry.get("paired")
    ]
    if paired:
        lines.append("## Paired bootstrap (AUROC difference, identical subjects)")
        lines.append("")
        rows = []
        for task, comparisons in paired:
            for cmp in comparisons:
                rows.append(
                    [
                        f"`{task}`",
                        f"`{cmp['a']}` - `{cmp['b']}`",
                        _fmt(cmp.get("diff")),
                        f"[{_fmt(cmp.get('lo'))}, {_fmt(cmp.get('hi'))}]",
                        _fmt(cmp.get("p_value"), 3),
                    ]
                )
        lines += _table(["task", "comparison", "diff", "95% CI", "boot p"], rows) + [""]

    few = {
        task: entry
        for task, entry in results.get("tasks", {}).items()
        if any(m.get("few_shot") for m in entry.get("models", {}).values())
    }
    if few:
        lines.append("## Few-shot (k positives + k negatives from train, 5 seeds)")
        lines.append("")
        rows = []
        for task, entry in few.items():
            for name, model in entry.get("models", {}).items():
                for row in model.get("few_shot", []):
                    rows.append(
                        [
                            f"`{task}`",
                            f"`{name}`",
                            "all" if row["k"] is None else str(row["k"]),
                            str(row.get("n_train", "")),
                            f"{_fmt(row.get('auroc_mean'))} ± {_fmt(row.get('auroc_std'))}",
                            f"{_fmt(row.get('auprc_mean'))} ± {_fmt(row.get('auprc_std'))}",
                        ]
                    )
        lines += _table(
            ["task", "model", "k", "n train", "AUROC mean ± sd", "AUPRC mean ± sd"], rows
        ) + [""]

    skipped = results.get("skipped", {})
    if skipped:
        lines.append("## Skipped")
        lines.append("")
        lines += _table(
            ["task", "reason"], [[f"`{k}`", str(v)] for k, v in skipped.items()]
        ) + [""]
    return "\n".join(lines)


def write(results: Mapping, out_dir: Path | str, stem: str = "results") -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md``; returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    md_path.write_text(render(results))
    return md_path, json_path
