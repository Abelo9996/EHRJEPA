"""Event tokenizer: vocabulary with ontology rollup, plus a per-code value quantizer.

One MEDS event becomes one token. A token carries three things: *what* happened
(a code id), *how much* (a decile bin plus a z-score), and *when* (age at the
event and the log-gap since the previous one). This module owns the two objects
that have to be **fit on the train split only** before any of that can be
computed -- the :class:`Vocabulary` and the :class:`Quantizer` -- and the CLI that
drives fitting, cache building (:mod:`ehrjepa.data.cache`) and inspection.

Vocabulary
----------
Ids 0-3 are reserved: ``PAD``, ``UNK``, ``CLS`` (the subject-summary token) and
``MASK``. Every code observed at least ``min_count`` times in train gets its own
id. Rarer codes are *not* thrown at ``UNK`` immediately: medical codes are
hierarchical strings, so a rare code is first rolled up to a truncated ancestor
(``ICD9CM/250.01`` -> ``ICD9CM/250.0`` -> ``ICD9CM/250`` -> ... -> ``ICD9CM``),
and that ancestor is admitted to the vocabulary if the mass rolled into it clears
``min_count``. Rollup is a single bottom-up sweep over the implied forest, so an
ancestor only ever absorbs the mass of descendants that were themselves rejected;
mass never gets counted at two levels at once.

Quantizer
---------
Values are per-code, so a raw float means nothing without its code. For every
vocabulary entry with at least ``min_obs`` numeric observations in train we store
the nine interior decile edges (ten bins), a mean and a standard deviation.
Entries below that threshold share their ``PREFIX``'s quantizer, and failing that
a global one. Heavily right-skewed non-negative codes are ``log1p``-ed before the
mean/std are taken, which is most lab values and every count-like code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import meds
import polars as pl

from ehrjepa.data.canonical import SPLITS

__all__ = [
    "CLS_ID",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_MIN_VALUE_OBS",
    "MASK_ID",
    "N_VALUE_BINS",
    "PAD_ID",
    "SPECIAL_TOKENS",
    "TOKENIZER_VERSION",
    "UNK_ID",
    "Vocabulary",
    "ancestors",
    "fit_quantizer",
    "fit_tokenizer",
    "fit_vocabulary",
    "load_quantizer",
    "main",
    "split_code",
]

#: Reserved ids. ``CLS`` is the per-subject summary token; ``MASK`` is for the
#: masked-prediction objectives in phase 4.
PAD_ID, UNK_ID, CLS_ID, MASK_ID = 0, 1, 2, 3
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[MASK]")

#: Bumped whenever the on-disk cache semantics change.
TOKENIZER_VERSION = 1

#: A code needs this many train occurrences to get its own id.
DEFAULT_MIN_COUNT = 5

#: A vocabulary entry needs this many train numeric observations for its own quantizer.
DEFAULT_MIN_VALUE_OBS = 20

#: ``value_bin`` is 0 for "no numeric value" and 1..10 for the deciles.
N_VALUE_BINS = 10

#: ``|skew| > SKEW_THRESHOLD`` on non-negative train values triggers log1p.
SKEW_THRESHOLD = 2.0

#: z-scores are clipped to this range before they reach the model.
Z_CLIP = 5.0

_EDGE_COLUMNS: tuple[str, ...] = tuple(f"edge_{i}" for i in range(N_VALUE_BINS - 1))
_QUANTILES: tuple[float, ...] = tuple(i / N_VALUE_BINS for i in range(1, N_VALUE_BINS))


# --------------------------------------------------------------------------- #
# Code surgery: prefixes and hierarchical ancestors
# --------------------------------------------------------------------------- #


def split_code(code: str) -> tuple[str, str, str] | None:
    """Split ``code`` into ``(prefix, separator, value)``.

    MEDS codes are ``PREFIX//value`` by convention, but ``meds_etl``'s MIMIC
    output uses a single slash, so both are accepted; ``//`` wins wherever a code
    contains one. Returns ``None`` for codes with no separator or an empty side
    (``MEDS_BIRTH``, ``NDC//``).
    """
    sep = "//"
    head, found, tail = code.partition(sep)
    if not found:
        sep = "/"
        head, found, tail = code.partition(sep)
    return (head, sep, tail) if found and head and tail else None


def _icd_ancestors(value: str) -> list[str]:
    """``250.01`` -> ``['250.0', '250', '25', '2']``: strip one character at a time.

    A trailing ``.`` is never a code on its own, so it is dropped with the
    character that exposed it.
    """
    out: list[str] = []
    while len(value) > 1:
        value = value[:-1]
        if value.endswith("."):
            value = value[:-1]
        if not value:
            break
        out.append(value)
    return out


def ancestors(code: str) -> list[str]:
    """The rollup chain for ``code``, most specific first, ending at ``PREFIX``.

    ``UNK`` is the implicit last resort and is not part of the returned chain.
    The chain is *consistent*: ``ancestors(a)[0]`` is the parent of ``a`` for
    every ``a`` in ``ancestors(code)``, which is what makes the bottom-up rollup
    in :func:`fit_vocabulary` a well-defined forest walk.
    """
    parts = split_code(code)
    if parts is None:
        return []
    prefix, sep, value = parts
    family = prefix.upper()
    if family.startswith(("ICD9", "ICD10")):
        truncations = _icd_ancestors(value)
    elif family == "NDC" and value.isdigit():
        # 11-digit NDC -> 9-digit product code -> 5-digit labeler code.
        truncations = [value[:n] for n in (9, 5) if len(value) > n]
    elif family == "HCPCS" and len(value) > 3:
        truncations = [value[:3]]
    else:
        truncations = []
    return [f"{prefix}{sep}{t}" for t in truncations] + [prefix]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Vocabulary:
    """Code <-> id mapping with hierarchical fallback.

    ``codes[i]`` is the string for id ``i``; ids 0-3 are :data:`SPECIAL_TOKENS`.
    ``train_count`` is the event mass routed to an entry (its own occurrences
    plus those of every descendant that rolled up into it), ``direct_count`` its
    own literal occurrences, and ``is_ancestor`` marks entries that only exist
    because of rollup (``direct_count < min_count``).
    """

    codes: tuple[str, ...]
    train_count: tuple[int, ...]
    direct_count: tuple[int, ...]
    is_ancestor: tuple[bool, ...]
    min_count: int = DEFAULT_MIN_COUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {code: i for i, code in enumerate(self.codes)})

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def index(self) -> Mapping[str, int]:
        return self._index  # type: ignore[attr-defined]

    def resolve(self, code: str) -> tuple[int, str]:
        """Map ``code`` to ``(id, kind)`` with ``kind`` in ``direct``/``ancestor``/``unk``."""
        hit = self.index.get(code)
        if hit is not None:
            return hit, "direct"
        for parent in ancestors(code):
            hit = self.index.get(parent)
            if hit is not None:
                return hit, "ancestor"
        return UNK_ID, "unk"

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "code_id": pl.Series(range(len(self)), dtype=pl.Int32),
                "code": pl.Series(self.codes, dtype=pl.String),
                "train_count": pl.Series(self.train_count, dtype=pl.Int64),
                "direct_count": pl.Series(self.direct_count, dtype=pl.Int64),
                "is_ancestor": pl.Series(self.is_ancestor, dtype=pl.Boolean),
                "is_special": pl.Series(
                    [i < len(SPECIAL_TOKENS) for i in range(len(self))], dtype=pl.Boolean
                ),
            }
        )

    def write(self, path: str | Path) -> None:
        self.to_frame().write_parquet(path)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, min_count: int = DEFAULT_MIN_COUNT) -> Vocabulary:
        frame = frame.sort("code_id")
        return cls(
            codes=tuple(frame["code"].to_list()),
            train_count=tuple(frame["train_count"].to_list()),
            direct_count=tuple(frame["direct_count"].to_list()),
            is_ancestor=tuple(frame["is_ancestor"].to_list()),
            min_count=min_count,
        )

    @classmethod
    def read(cls, path: str | Path, min_count: int = DEFAULT_MIN_COUNT) -> Vocabulary:
        return cls.from_frame(pl.read_parquet(path), min_count=min_count)

    def resolve_many(self, codes: Iterable[str]) -> pl.DataFrame:
        """Vectorised :meth:`resolve` over ``codes``, as a joinable frame.

        ``kind`` is encoded as 0 direct / 1 ancestor / 2 unk so it can be counted
        with a group-by during tensorization.
        """
        kinds = {"direct": 0, "ancestor": 1, "unk": 2}
        rows = [(code, *self.resolve(code)) for code in codes]
        return pl.DataFrame(
            {
                "code": pl.Series([r[0] for r in rows], dtype=pl.String),
                "code_id": pl.Series([r[1] for r in rows], dtype=pl.Int32),
                "kind": pl.Series([kinds[r[2]] for r in rows], dtype=pl.Int8),
            }
        )


def fit_vocabulary(
    counts: Mapping[str, int], min_count: int = DEFAULT_MIN_COUNT
) -> tuple[Vocabulary, dict[str, float]]:
    """Fit a vocabulary from train code counts. Returns the vocabulary and coverage stats.

    The rollup is one bottom-up sweep over the forest induced by :func:`ancestors`.
    Every node holds its own count plus the mass of its rejected children; if that
    total clears ``min_count`` the node is admitted and *keeps* the mass, otherwise
    the mass moves to its parent. Mass that falls off a root becomes ``UNK``.
    """
    parent: dict[str, str | None] = {}
    pending: dict[str, int] = dict(counts)

    frontier = list(counts)
    while frontier:
        nxt = []
        for node in frontier:
            if node in parent:
                continue
            chain = ancestors(node)
            parent[node] = chain[0] if chain else None
            if chain and chain[0] not in parent:
                pending.setdefault(chain[0], 0)
                nxt.append(chain[0])
        frontier = nxt

    # Depth = distance to the root of the chain; deeper nodes settle first.
    depth = {node: len(ancestors(node)) for node in parent}
    admitted: dict[str, int] = {}
    unk_mass = 0
    for node in sorted(parent, key=lambda n: (-depth[n], n)):
        total = pending.get(node, 0)
        if total >= min_count:
            admitted[node] = total
        elif parent[node] is not None:
            pending[parent[node]] = pending.get(parent[node], 0) + total
        else:
            unk_mass += total

    order = sorted(admitted, key=lambda c: (-admitted[c], c))
    codes = list(SPECIAL_TOKENS) + order
    train_count = [0] * len(SPECIAL_TOKENS) + [admitted[c] for c in order]
    direct_count = [0] * len(SPECIAL_TOKENS) + [counts.get(c, 0) for c in order]
    is_ancestor = [False] * len(SPECIAL_TOKENS) + [counts.get(c, 0) < min_count for c in order]
    train_count[UNK_ID] = unk_mass

    vocab = Vocabulary(
        codes=tuple(codes),
        train_count=tuple(train_count),
        direct_count=tuple(direct_count),
        is_ancestor=tuple(is_ancestor),
        min_count=min_count,
    )

    total_events = sum(counts.values())
    routed = dict.fromkeys(("direct", "ancestor", "unk"), 0)
    for code, n in counts.items():
        routed[vocab.resolve(code)[1]] += n
    stats = {
        "train_events": total_events,
        "train_distinct_codes": len(counts),
        "vocab_size": len(vocab),
        "n_ancestor_entries": sum(is_ancestor),
        "direct_rate": routed["direct"] / total_events if total_events else 0.0,
        "ancestor_rate": routed["ancestor"] / total_events if total_events else 0.0,
        "unk_rate": routed["unk"] / total_events if total_events else 0.0,
    }
    return vocab, stats


# --------------------------------------------------------------------------- #
# Value quantizer
# --------------------------------------------------------------------------- #


def _value_stats(frame: pl.DataFrame, by: str | None) -> pl.DataFrame:
    """Decile edges plus raw and log1p moments, per group (or globally)."""
    aggs = [
        pl.len().alias("n_obs"),
        pl.col("value").min().alias("value_min"),
        pl.col("value").mean().alias("raw_mean"),
        pl.col("value").std().alias("raw_std"),
        pl.col("value").skew().alias("skew"),
        pl.col("value").clip(lower_bound=0.0).log1p().mean().alias("log_mean"),
        pl.col("value").clip(lower_bound=0.0).log1p().std().alias("log_std"),
        *[
            pl.col("value").quantile(q, interpolation="linear").alias(name)
            for q, name in zip(_QUANTILES, _EDGE_COLUMNS, strict=True)
        ],
    ]
    grouped = frame.group_by(by).agg(aggs) if by else frame.select(aggs)
    return grouped.with_columns(
        use_log1p=(
            (pl.col("value_min") >= 0.0)
            & (pl.col("skew").abs() > SKEW_THRESHOLD)
            & pl.col("skew").is_not_null()
        )
    ).with_columns(
        mean=pl.when(pl.col("use_log1p")).then(pl.col("log_mean")).otherwise(pl.col("raw_mean")),
        std=pl.when(pl.col("use_log1p")).then(pl.col("log_std")).otherwise(pl.col("raw_std")),
    )


_STAT_COLUMNS: tuple[str, ...] = ("n_obs", "use_log1p", "mean", "std", *_EDGE_COLUMNS)

_EMPTY_STATS: dict[str, object] = {
    "n_obs": 0,
    "use_log1p": False,
    "mean": 0.0,
    "std": 0.0,
    **dict.fromkeys(_EDGE_COLUMNS, float("nan")),
}


def fit_quantizer(
    values: pl.DataFrame,
    vocab: Vocabulary,
    min_obs: int = DEFAULT_MIN_VALUE_OBS,
) -> pl.DataFrame:
    """Fit the value quantizer and resolve it to one row per vocabulary id.

    ``values`` has columns ``code_id`` (already routed through the vocabulary)
    and ``value``. The returned frame covers *every* vocabulary id -- including
    ids that carry no numeric value in train, which inherit their prefix's or the
    global quantizer -- so tensorization is a single join with no null handling.
    """
    values = values.select(pl.col("code_id").cast(pl.Int32), pl.col("value").cast(pl.Float64))
    ids = pl.DataFrame(
        {
            "code_id": pl.Series(range(len(vocab)), dtype=pl.Int32),
            "code": pl.Series(vocab.codes, dtype=pl.String),
        }
    ).with_columns(
        prefix=pl.col("code").str.extract(r"^([^/]+)", 1).fill_null(pl.col("code")),
    )

    per_code = _value_stats(values, "code_id")
    per_prefix = _value_stats(values.join(ids.select("code_id", "prefix"), on="code_id"), "prefix")
    glob = _value_stats(values, None)
    global_row = glob.row(0, named=True) if glob.height and glob["n_obs"][0] else dict(_EMPTY_STATS)

    table = ids.join(
        per_code.select("code_id", *_STAT_COLUMNS).rename({c: f"{c}_c" for c in _STAT_COLUMNS}),
        on="code_id",
        how="left",
    ).join(
        per_prefix.select("prefix", *_STAT_COLUMNS).rename({c: f"{c}_p" for c in _STAT_COLUMNS}),
        on="prefix",
        how="left",
    )

    use_code = pl.col("n_obs_c").fill_null(0) >= min_obs
    use_prefix = ~use_code & (pl.col("n_obs_p").fill_null(0) >= min_obs)

    def pick(column: str) -> pl.Expr:
        return (
            pl.when(use_code)
            .then(pl.col(f"{column}_c"))
            .when(use_prefix)
            .then(pl.col(f"{column}_p"))
            .otherwise(pl.lit(global_row[column]))
            .alias(column)
        )

    return (
        table.with_columns(
            scope=pl.when(use_code)
            .then(pl.lit("code"))
            .when(use_prefix)
            .then(pl.lit("prefix"))
            .otherwise(pl.lit("global")),
            source_key=pl.when(use_code)
            .then(pl.col("code"))
            .when(use_prefix)
            .then(pl.col("prefix"))
            .otherwise(pl.lit("[GLOBAL]")),
            code_n_obs=pl.col("n_obs_c").fill_null(0).cast(pl.Int64),
            **{column: pick(column) for column in _STAT_COLUMNS},
        )
        .select(
            "code_id",
            "code",
            "prefix",
            "scope",
            "source_key",
            "code_n_obs",
            pl.col("n_obs").cast(pl.Int64),
            pl.col("use_log1p").fill_null(False),
            pl.col("mean").cast(pl.Float64).fill_null(0.0),
            pl.col("std").cast(pl.Float64).fill_null(0.0),
            *[pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in _EDGE_COLUMNS],
        )
        .sort("code_id")
    )


def load_quantizer(path: str | Path) -> pl.DataFrame:
    """Read a quantizer table written by :func:`fit_quantizer`."""
    return pl.read_parquet(path).sort("code_id")


# --------------------------------------------------------------------------- #
# Fitting from a MEDS directory
# --------------------------------------------------------------------------- #


def split_shards(meds_dir: str | Path, split: str) -> list[Path]:
    """The parquet shards of one split, in the order the cache will concatenate them."""
    return sorted((Path(meds_dir) / meds.data_subdirectory / split).glob("*.parquet"))


def _train_code_counts(meds_dir: Path) -> dict[str, int]:
    shards = split_shards(meds_dir, meds.train_split)
    if not shards:
        raise ValueError(f"no train shards under {meds_dir}")
    counted = (
        pl.scan_parquet(shards)
        .group_by("code")
        .agg(pl.len().alias("count"))
        .collect(engine="streaming")
    )
    return dict(counted.iter_rows())


def _train_values(meds_dir: Path, vocab: Vocabulary) -> pl.DataFrame:
    shards = split_shards(meds_dir, meds.train_split)
    numeric = (
        pl.scan_parquet(shards)
        .filter(pl.col("numeric_value").is_not_null() & pl.col("numeric_value").is_not_nan())
        .select("code", pl.col("numeric_value").cast(pl.Float64).alias("value"))
        .collect(engine="streaming")
    )
    mapping = vocab.resolve_many(numeric["code"].unique().to_list()).select("code", "code_id")
    return numeric.join(mapping, on="code", how="left").select("code_id", "value")


def fit_tokenizer(
    meds_dir: str | Path,
    min_count: int = DEFAULT_MIN_COUNT,
    min_value_obs: int = DEFAULT_MIN_VALUE_OBS,
) -> tuple[Vocabulary, pl.DataFrame, dict[str, object]]:
    """Fit the vocabulary and quantizer on the train split of ``meds_dir``."""
    root = Path(meds_dir)
    counts = _train_code_counts(root)
    vocab, vocab_stats = fit_vocabulary(counts, min_count=min_count)
    values = _train_values(root, vocab)
    quantizer = fit_quantizer(values, vocab, min_obs=min_value_obs)
    stats: dict[str, object] = {
        **vocab_stats,
        "min_count": min_count,
        "min_value_obs": min_value_obs,
        "train_numeric_events": values.height,
        "quantizer_scopes": dict(
            quantizer.group_by("scope").len().sort("scope").iter_rows()  # type: ignore[arg-type]
        ),
        "n_log1p_codes": int(quantizer["use_log1p"].sum()),
    }
    return vocab, quantizer, stats


def write_fit(
    cache_dir: str | Path,
    vocab: Vocabulary,
    quantizer: pl.DataFrame,
    stats: Mapping[str, object],
) -> None:
    """Write ``vocab.parquet``, ``vocab.json`` and ``quantizer.parquet``."""
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab.write(out / "vocab.parquet")
    quantizer.write_parquet(out / "quantizer.parquet")
    summary = {
        "tokenizer_version": TOKENIZER_VERSION,
        "specials": dict(zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS)), strict=True)),
        **dict(stats),
        "top_codes": vocab.to_frame()
        .filter(~pl.col("is_special"))
        .head(20)
        .select("code_id", "code", "train_count", "is_ancestor")
        .to_dicts(),
    }
    (out / "vocab.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _decode_window(cache_dir: Path, split: str, n_events: int) -> str:
    from ehrjepa.data.dataset import EventSequenceDataset

    dataset = EventSequenceDataset(cache_dir, split, max_len=n_events, min_len=1, sampling="full")
    if not len(dataset):
        return f"({split}: no subjects)"
    item = dataset[0]
    codes = dataset.decode(item["code_id"])
    lines = [
        f"subject {int(item['subject_id'])} ({split}), last {len(codes)} events",
        f"{'code':<38} {'bin':>3} {'z':>7} {'age':>7} {'log_dt':>7}",
    ]
    lines += [
        f"{code:<38} {int(b):>3} {float(z):>7.2f} {float(a):>7.2f} {float(d):>7.2f}"
        for code, b, z, a, d in zip(
            codes,
            item["value_bin"].tolist(),
            item["value_z"].tolist(),
            item["age"].tolist(),
            item["log_delta"].tolist(),
            strict=True,
        )
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ehrjepa.data.tokenize",
        description="Fit the event tokenizer and build the tensor cache.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit the vocabulary and quantizer on the train split")
    fit.add_argument("meds_dir", type=Path)
    fit.add_argument("--out", type=Path, required=True, help="cache directory to write into")
    fit.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    fit.add_argument("--min-value-obs", type=int, default=DEFAULT_MIN_VALUE_OBS)

    build = sub.add_parser("build", help="fit, then tensorize every split")
    build.add_argument("meds_dir", type=Path)
    build.add_argument("--cache", type=Path, required=True)
    build.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    build.add_argument("--min-value-obs", type=int, default=DEFAULT_MIN_VALUE_OBS)
    build.add_argument("--splits", nargs="+", default=list(SPLITS))

    inspect = sub.add_parser("inspect", help="print cache metadata and decoded example windows")
    inspect.add_argument("cache_dir", type=Path)
    inspect.add_argument("--events", type=int, default=12)

    args = parser.parse_args(argv)

    if args.command == "fit":
        vocab, quantizer, stats = fit_tokenizer(
            args.meds_dir, min_count=args.min_count, min_value_obs=args.min_value_obs
        )
        write_fit(args.out, vocab, quantizer, stats)
        print(json.dumps({"cache_dir": str(args.out), **stats}, indent=2, default=str))
        return 0

    if args.command == "build":
        from ehrjepa.data.cache import build_cache

        meta = build_cache(
            args.meds_dir,
            args.cache,
            min_count=args.min_count,
            min_value_obs=args.min_value_obs,
            splits=args.splits,
        )
        print(json.dumps(meta, indent=2, default=str))
        return 0

    meta = json.loads((args.cache_dir / "meta.json").read_text())
    json.dump(meta, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n\n")
    for split in meta["splits"]:
        print(_decode_window(args.cache_dir, split, args.events))
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
