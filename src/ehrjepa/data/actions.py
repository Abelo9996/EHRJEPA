"""Clinical actions vs observations, and time windows over a token stream.

The action-conditioned state model (:mod:`ehrjepa.models.state`) needs two things
the cache does not store as features, both derived here from what it does store:

**Which tokens are actions.** A per-code boolean table over the vocabulary, built
from MEDS code strings by prefix, by regular expression, and optionally by an
explicit list of codes. The defaults are the ``meds_etl.mimic`` code families
that record something a clinician *did* rather than *saw*:

``NDC/``, ``MIMIC_IV_Drug/``       ``hosp/prescriptions`` (medications prescribed)
``ICD9Proc/``, ``ICD10PCS/``       ``hosp/procedures_icd`` (billed procedures)
``HCPCS/``                         ``hosp/hcpcsevents`` (billed procedures/services)
``MIMIC_IV_ITEM/<id>/<category>``  ``icu/inputevents`` (infusions, pushes, boluses):
                                   the only table whose code carries a third part

``icu/procedureevents`` and ``icu/ingredientevents`` share the bare
``MIMIC_IV_ITEM/<itemid>`` form with ``chartevents``/``outputevents``, so no
prefix can separate them; ``action_codes_file`` lists those itemids, and
``configs/actions/mimic_icu_action_items.txt`` is that list as built from
``d_items`` by ``python -m ehrjepa.data.actions items``. Everything else --
labs, vitals, diagnoses, transfers, demographics -- is an observation.

**Which window each token is in.** ``calendar`` windows are
``floor(time_min / window_minutes)`` -- crop-independent, so the same event lands
in the same window whichever 512-event crop it is drawn into. ``admission``
windows count ``window_minutes`` from the most recent admission event at or
before the token (``window_minutes = 1440`` is "admission day"), with tokens
before the first admission in view falling back to calendar windows. Ties are
grouped by *time*, not position: every event at one timestamp is in one window,
so no window boundary can fall between two simultaneous events.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "DEFAULT_ACTION_PATTERNS",
    "DEFAULT_ACTION_PREFIXES",
    "DEFAULT_ADMISSION_PREFIXES",
    "WINDOW_ANCHORS",
    "action_items_from_d_items",
    "assign_windows",
    "code_flags",
    "read_codes_file",
    "window_stats",
]

DEFAULT_ACTION_PREFIXES: tuple[str, ...] = (
    "NDC/",
    "MIMIC_IV_Drug/",
    "ICD9Proc/",
    "ICD10PCS/",
    "HCPCS/",
)
#: ``icu/inputevents``: ``MIMIC_IV_ITEM/<itemid>/<ordercategorydescription>``.
DEFAULT_ACTION_PATTERNS: tuple[str, ...] = (r"^MIMIC_IV_ITEM/\d+/.+",)
#: ``hosp/admissions`` ``admission_type`` at ``admittime``: one per admission.
#: The trailing slash keeps ``MIMIC_IV_Admission_Location/`` out.
DEFAULT_ADMISSION_PREFIXES: tuple[str, ...] = ("MIMIC_IV_Admission/",)
#: ``d_items.linksto`` values whose bare ``MIMIC_IV_ITEM/<id>`` codes are actions.
ACTION_LINKSTO: tuple[str, ...] = ("procedureevents", "ingredientevents")

WINDOW_ANCHORS = ("calendar", "admission")

_INT64_MAX = torch.iinfo(torch.int64).max


def read_codes_file(path: str | Path) -> set[str]:
    """One code per line; blank lines and ``#`` comments ignored."""
    out: set[str] = set()
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def code_flags(
    codes: Sequence[str],
    prefixes: Iterable[str] = (),
    patterns: Iterable[str] = (),
    codes_file: str | Path | None = None,
) -> np.ndarray:
    """``(len(codes),)`` bool: does each vocabulary code match any rule.

    Specials (``[PAD]``, ``[UNK]``, ...) never match unless a rule names them.
    """
    prefixes = tuple(prefixes)
    compiled = [re.compile(p) for p in patterns]
    listed = read_codes_file(codes_file) if codes_file else set()
    flags = np.zeros(len(codes), dtype=bool)
    for i, code in enumerate(codes):
        flags[i] = (
            (bool(prefixes) and code.startswith(prefixes))
            or any(p.match(code) for p in compiled)
            or code in listed
        )
    return flags


def action_items_from_d_items(
    d_items: str | Path, linksto: Iterable[str] = ACTION_LINKSTO
) -> list[str]:
    """``MIMIC_IV_ITEM/<itemid>`` for every ``d_items`` row linked to an action table."""
    import polars as pl

    table = pl.read_csv(d_items, infer_schema_length=0)
    keep = table.filter(pl.col("linksto").is_in(list(linksto)))
    return sorted(f"MIMIC_IV_ITEM/{i}" for i in keep["itemid"].unique().to_list())


@torch.no_grad()
def assign_windows(
    time_min: Tensor,
    valid: Tensor,
    window_minutes: int,
    admission: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """``(window, n_windows)``: a 0-based window index per token and the count per row.

    ``time_min`` is ``(B, L)`` int64 minutes, non-decreasing along each row's valid
    prefix (the cache's canonical order). Invalid positions get window ``-1``.
    With ``admission`` (``(B, L)`` bool, true on admission events) windows are
    counted from the latest admission at or before each token's *time*.
    """
    if window_minutes < 1:
        raise ValueError(f"window_minutes must be >= 1, got {window_minutes}")
    valid = valid.bool()
    times = time_min.to(torch.int64)
    calendar = torch.div(times, window_minutes, rounding_mode="floor")
    if admission is None:
        epoch = torch.zeros_like(times)
        key = calendar
    else:
        adm = admission.bool() & valid
        adm_times = torch.where(adm, times, torch.full_like(times, _INT64_MAX))
        adm_sorted = adm_times.sort(dim=1).values
        # Admissions at or before this token's time -- ties included, so an
        # event sharing the admission's timestamp is in the admission's window
        # whatever its position among the simultaneous events.
        count = torch.searchsorted(adm_sorted, times.contiguous(), right=True)
        last = adm_sorted.gather(1, (count - 1).clamp(min=0))
        since = torch.div(times - last, window_minutes, rounding_mode="floor")
        epoch = count
        key = torch.where(count > 0, since, calendar)
    prev_key = torch.roll(key, 1, dims=1)
    prev_epoch = torch.roll(epoch, 1, dims=1)
    prev_valid = torch.roll(valid, 1, dims=1)
    prev_valid[:, 0] = False
    boundary = valid & (~prev_valid | (key != prev_key) | (epoch != prev_epoch))
    window = boundary.long().cumsum(dim=1) - 1
    window = torch.where(valid, window, torch.full_like(window, -1))
    return window, boundary.sum(dim=1)


# --------------------------------------------------------------------------- #
# Stats CLI: action share and windows per training crop, on a built cache
# --------------------------------------------------------------------------- #


def window_stats(
    cache_dir: str | Path,
    split: str = "train",
    window_hours: Sequence[float] = (24.0,),
    anchor: str = "calendar",
    max_len: int = 512,
    min_len: int = 16,
    horizons: Sequence[int] = (1, 2, 4),
    samples: int = 4000,
    seed: int = 0,
    prefixes: Iterable[str] = DEFAULT_ACTION_PREFIXES,
    patterns: Iterable[str] = DEFAULT_ACTION_PATTERNS,
    codes_file: str | Path | None = None,
    share_events: int = 50_000_000,
) -> dict[str, object]:
    """Action token share and, per window size, state windows and pairs per crop.

    The share is over the first ``share_events`` events of the split (exact when
    the split is smaller). Crops are drawn the way the trainer draws them: one
    uniformly random subject, one uniformly random ``max_len`` window of it.
    """
    import polars as pl

    from ehrjepa.data.tokenize import Vocabulary

    root = Path(cache_dir)
    codes = Vocabulary.read(root / "vocab.parquet").codes
    action = code_flags(codes, prefixes, patterns, codes_file)
    admission = code_flags(codes, DEFAULT_ADMISSION_PREFIXES)
    code_id = np.load(root / split / "code_id.npy", mmap_mode="r")
    time_min = np.load(root / split / "time_min.npy", mmap_mode="r")
    head = np.asarray(code_id[: min(len(code_id), share_events)])
    share = float(action[head].mean()) if head.size else 0.0
    counts = np.bincount(head[action[head]], minlength=len(codes))
    top = [(codes[i], int(counts[i])) for i in np.argsort(-counts)[:15] if counts[i] > 0]

    index = pl.read_parquet(root / split / "subjects.parquet").filter(pl.col("length") >= min_len)
    offsets = index["offset"].to_numpy()
    lengths = index["length"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(offsets), size=samples)
    crops = []
    for r in rows:
        w = int(min(lengths[r], max_len))
        start = int(offsets[r] + rng.integers(0, lengths[r] - w + 1))
        crops.append((start, w))
    out: dict[str, object] = {
        "cache": str(root),
        "split": split,
        "events_for_share": int(head.size),
        "action_vocab": int(action.sum()),
        "action_token_share": share,
        "observation_token_share": 1.0 - share,
        "top_action_codes": top,
        "crops": samples,
        "max_len": max_len,
        "anchor": anchor,
        "windows": {},
    }
    for hours in window_hours:
        minutes = max(1, round(hours * 60))
        n_win, n_state, pairs = [], [], {k: [] for k in horizons}
        for start, w in crops:
            ids = torch.from_numpy(np.asarray(code_id[start : start + w]).astype(np.int64))[None]
            t = torch.from_numpy(np.asarray(time_min[start : start + w]).astype(np.int64))[None]
            valid = torch.ones_like(ids, dtype=torch.bool)
            adm = torch.from_numpy(admission)[ids] if anchor == "admission" else None
            win, n = assign_windows(t, valid, minutes, adm)
            obs = ~torch.from_numpy(action)[ids]
            has = torch.zeros(int(n), dtype=torch.bool)
            has[win[obs]] = True
            n_win.append(int(n))
            n_state.append(int(has.sum()))
            # Pairs (t, t+k) with a state at both ends, the last window excluded
            # as a target (it may be cut by the crop) -- the trainer's rule.
            usable = has.clone()
            if len(usable):
                usable[-1] = False
            for k in horizons:
                pairs[k].append(int((has[:-k] & usable[k:]).sum()) if len(has) > k else 0)
        arr = np.asarray(n_win)
        out["windows"][f"{hours:g}h"] = {  # type: ignore[index]
            "windows_per_crop_median": float(np.median(arr)),
            "windows_per_crop_mean": float(arr.mean()),
            "state_windows_per_crop_mean": float(np.mean(n_state)),
            "frac_crops_with_a_k1_pair": float(np.mean(np.asarray(pairs[horizons[0]]) > 0)),
            **{f"pairs_per_crop_k{k}": float(np.mean(v)) for k, v in pairs.items()},
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ehrjepa.data.actions")
    sub = parser.add_subparsers(dest="command", required=True)
    stats = sub.add_parser("stats", help="action share and windows per crop on a cache")
    stats.add_argument("--cache", required=True)
    stats.add_argument("--split", default="train")
    stats.add_argument("--window-hours", type=float, nargs="+", default=[24.0])
    stats.add_argument("--anchor", choices=WINDOW_ANCHORS, default="calendar")
    stats.add_argument("--max-len", type=int, default=512)
    stats.add_argument("--samples", type=int, default=4000)
    stats.add_argument("--codes-file", default=None)
    items = sub.add_parser("items", help="write the ICU action itemids from d_items")
    items.add_argument("--d-items", required=True)
    items.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "items":
        found = action_items_from_d_items(args.d_items)
        header = (
            "# MIMIC_IV_ITEM codes of icu/procedureevents and icu/ingredientevents "
            "(d_items.linksto), one per line.\n"
            f"# Built by: python -m ehrjepa.data.actions items --d-items {args.d_items}\n"
        )
        Path(args.out).write_text(header + "\n".join(found) + "\n")
        print(f"wrote {len(found)} codes to {args.out}")
        return 0
    result = window_stats(
        args.cache,
        split=args.split,
        window_hours=args.window_hours,
        anchor=args.anchor,
        max_len=args.max_len,
        samples=args.samples,
        codes_file=args.codes_file,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
