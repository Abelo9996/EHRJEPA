"""Tests for the event tokenizer, the tensor cache and the torch dataset.

Everything runs against a hand-built MEDS directory a few hundred events wide,
constructed so that every hierarchical-fallback rule, every quantizer scope and
every awkward time case (tied timestamps, a subject with no birth event, an event
older than the 120-year age clip) is hit by at least one row. Nothing here reads
``data/``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from ehrjepa.data.cache import FEATURES, build_cache, shard_features
from ehrjepa.data.dataset import EventSequenceDataset, collate_events
from ehrjepa.data.tokenize import (
    CLS_ID,
    MASK_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    Vocabulary,
    ancestors,
    fit_quantizer,
    fit_tokenizer,
    fit_vocabulary,
    split_code,
)

BIRTH = "MEDS_BIRTH"
_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
    "text_value": pl.String,
}


def _frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame([(s, t, c, v, None) for s, t, c, v in rows], schema=_SCHEMA, orient="row")


def _no_values() -> pl.DataFrame:
    """An empty ``(code_id, value)`` frame: a dataset with no numeric values at all."""
    return pl.DataFrame(
        {"code_id": pl.Series([], dtype=pl.Int32), "value": pl.Series([], dtype=pl.Float64)}
    )


def _write_meds(root: Path, splits: dict[str, pl.DataFrame]) -> Path:
    """Write a minimal but canonical MEDS directory."""
    for split, frame in splits.items():
        out = root / "data" / split
        out.mkdir(parents=True, exist_ok=True)
        frame.sort("subject_id", "time", "code", nulls_last=False).write_parquet(out / "0.parquet")
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "dataset.json").write_text(json.dumps({"dataset_name": "fixture"}))
    return root


# --------------------------------------------------------------------------- #
# Code surgery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("LOINC//1234-5//mg", ("LOINC", "//", "1234-5//mg")),
        ("MIMIC_IV_LABITEM/51087", ("MIMIC_IV_LABITEM", "/", "51087")),
        ("MEDS_BIRTH", None),
        ("//x", None),
        ("NDC//", None),
    ],
)
def test_split_code(code: str, expected: tuple[str, str, str] | None) -> None:
    assert split_code(code) == expected


def test_icd_truncation_chain_matches_the_spec_example() -> None:
    assert ancestors("ICD9CM/250.01") == [
        "ICD9CM/250.0",
        "ICD9CM/250",
        "ICD9CM/25",
        "ICD9CM/2",
        "ICD9CM",
    ]
    # The double-slash convention is handled identically.
    assert ancestors("ICD10CM//E11.9")[0] == "ICD10CM//E11"
    assert ancestors("ICD9Proc//8154")[:2] == ["ICD9Proc//815", "ICD9Proc//81"]


def test_ndc_and_hcpcs_truncation_chains() -> None:
    assert ancestors("NDC//00003517887") == ["NDC//000035178", "NDC//00003", "NDC"]
    assert ancestors("HCPCS//99213") == ["HCPCS//992", "HCPCS"]
    # A non-numeric NDC has no digit-slice interpretation, so it falls straight
    # to its prefix.
    assert ancestors("NDC//abcdefghijk") == ["NDC"]


def test_generic_and_rootless_chains() -> None:
    assert ancestors("LOINC//1234-5//mg") == ["LOINC"]
    assert ancestors("SEX//F") == ["SEX"]
    assert ancestors("MEDS_BIRTH") == []


def test_ancestor_chain_is_a_consistent_forest() -> None:
    """``ancestors(a)[0]`` is the parent of ``a``: the rollup relies on this."""
    for code in ("ICD9CM/250.01", "NDC//00003517887", "HCPCS//99213", "LOINC//1-2//mg"):
        chain = ancestors(code)
        for node, parent in zip(chain, chain[1:], strict=False):
            assert ancestors(node)[0] == parent


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rollup_fit() -> tuple[Vocabulary, dict]:
    counts = {
        "LAB//A": 40,
        "ICD9CM//250": 6,
        "ICD9CM//250.01": 2,  # rolls up into the frequent ICD9CM//250
        "ICD9CM//250.02": 2,
        "NDC//00003517887": 2,  # three rare NDCs share a 5-digit labeler code
        "NDC//00003512345": 2,
        "NDC//00003599999": 2,
        "HCPCS//99213": 2,  # three rare HCPCS share a 3-character root
        "HCPCS//99214": 2,
        "HCPCS//99215": 2,
        "LOINC//1234-5//mg": 3,  # generic rule: only the bare prefix is available
        "LOINC//9999-9//mg": 3,
        "WEIRD//zzz": 1,  # nothing to roll into: this is the UNK case
        BIRTH: 9,
    }
    return fit_vocabulary(counts, min_count=5)


@pytest.fixture(scope="module")
def rollup_vocab(rollup_fit: tuple[Vocabulary, dict]) -> Vocabulary:
    return rollup_fit[0]


def test_special_ids_are_reserved(rollup_vocab: Vocabulary) -> None:
    assert rollup_vocab.codes[:4] == SPECIAL_TOKENS
    assert (PAD_ID, UNK_ID, CLS_ID, MASK_ID) == (0, 1, 2, 3)


def test_frequent_codes_are_direct(rollup_vocab: Vocabulary) -> None:
    assert rollup_vocab.resolve("LAB//A")[1] == "direct"
    assert rollup_vocab.resolve("ICD9CM//250")[1] == "direct"
    assert rollup_vocab.resolve(BIRTH)[1] == "direct"


@pytest.mark.parametrize(
    ("code", "parent"),
    [
        ("ICD9CM//250.01", "ICD9CM//250"),
        ("ICD9CM//250.02", "ICD9CM//250"),
        ("NDC//00003517887", "NDC//00003"),
        ("NDC//00003599999", "NDC//00003"),
        ("HCPCS//99213", "HCPCS//992"),
        ("LOINC//1234-5//mg", "LOINC"),
    ],
)
def test_rare_codes_roll_up_to_the_nearest_frequent_ancestor(
    rollup_vocab: Vocabulary, code: str, parent: str
) -> None:
    code_id, kind = rollup_vocab.resolve(code)
    assert kind == "ancestor"
    assert rollup_vocab.codes[code_id] == parent


def test_rollup_admits_ancestors_and_marks_them(rollup_vocab: Vocabulary) -> None:
    frame = rollup_vocab.to_frame()
    admitted = dict(zip(frame["code"], frame["is_ancestor"], strict=True))
    # Intermediate levels that never cleared min_count are absent.
    assert "ICD9CM//250.0" not in admitted
    assert "NDC//000035178" not in admitted
    for code in ("NDC//00003", "HCPCS//992", "LOINC"):
        assert admitted[code] is True
    # A code frequent in its own right is not an "ancestor" entry even though
    # rare descendants rolled into it.
    assert admitted["ICD9CM//250"] is False


def test_rollup_mass_is_counted_once_per_level(rollup_vocab: Vocabulary) -> None:
    counts = dict(zip(*rollup_vocab.to_frame().select("code", "train_count"), strict=True))
    assert counts["ICD9CM//250"] == 6 + 2 + 2
    assert counts["NDC//00003"] == 6
    # NDC's own root never sees the mass its admitted child kept.
    assert "NDC" not in counts


def test_unrollable_rare_codes_become_unk(rollup_fit: tuple[Vocabulary, dict]) -> None:
    vocab, stats = rollup_fit
    assert vocab.resolve("WEIRD//zzz") == (UNK_ID, "unk")
    assert vocab.resolve("NEVER_SEEN") == (UNK_ID, "unk")
    assert vocab.train_count[UNK_ID] == 1
    assert stats["unk_rate"] == pytest.approx(1 / stats["train_events"])
    assert stats["direct_rate"] + stats["ancestor_rate"] + stats["unk_rate"] == pytest.approx(1.0)


def test_min_count_controls_vocabulary_size(rollup_vocab: Vocabulary) -> None:
    counts = {"A//1": 4, "A//2": 4, "B": 10}
    small, _ = fit_vocabulary(counts, min_count=5)
    assert small.resolve("A//1")[1] == "ancestor"  # 4 + 4 = 8 rolls into "A"
    tiny, _ = fit_vocabulary(counts, min_count=3)
    assert tiny.resolve("A//1")[1] == "direct"


def test_vocabulary_survives_a_parquet_round_trip(rollup_vocab: Vocabulary, tmp_path: Path) -> None:
    rollup_vocab.write(tmp_path / "vocab.parquet")
    reloaded = Vocabulary.read(tmp_path / "vocab.parquet")
    assert reloaded.codes == rollup_vocab.codes
    assert reloaded.resolve("NDC//00003517887") == rollup_vocab.resolve("NDC//00003517887")


# --------------------------------------------------------------------------- #
# Quantizer
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def quantizer_fixture() -> tuple[Vocabulary, pl.DataFrame]:
    counts = {
        "LAB//A": 40,  # own quantizer
        "LAB//B": 10,  # too few values: falls back to the LAB prefix
        "LAB//SKEW": 40,  # own quantizer, log1p-transformed
        "OTHER//C": 10,  # no numeric values anywhere in its prefix: global
        BIRTH: 10,
    }
    vocab, _ = fit_vocabulary(counts, min_count=5)
    ids = {code: vocab.resolve(code)[0] for code in counts}
    values = pl.DataFrame(
        {
            "code_id": pl.Series(
                [ids["LAB//A"]] * 40 + [ids["LAB//B"]] * 10 + [ids["LAB//SKEW"]] * 40,
                dtype=pl.Int32,
            ),
            "value": pl.Series(
                [float(v) for v in range(1, 41)]
                + [100.0] * 10
                + [0.0] * 36
                + [1000.0, 2000.0, 3000.0, 4000.0],
                dtype=pl.Float64,
            ),
        }
    )
    return vocab, fit_quantizer(values, vocab, min_obs=20)


def _row(quantizer: pl.DataFrame, vocab: Vocabulary, code: str) -> dict:
    return quantizer.filter(pl.col("code_id") == vocab.resolve(code)[0]).row(0, named=True)


def test_quantizer_scopes(quantizer_fixture: tuple[Vocabulary, pl.DataFrame]) -> None:
    vocab, quantizer = quantizer_fixture
    assert _row(quantizer, vocab, "LAB//A")["scope"] == "code"
    # 10 observations is below min_obs, so LAB//B borrows the LAB//-wide fit.
    assert _row(quantizer, vocab, "LAB//B")["scope"] == "prefix"
    assert _row(quantizer, vocab, "LAB//B")["source_key"] == "LAB"
    assert _row(quantizer, vocab, "OTHER//C")["scope"] == "global"
    # Every vocabulary id gets a row, including the specials.
    assert quantizer.height == len(vocab)
    assert set(quantizer["code_id"].to_list()) == set(range(len(vocab)))


def test_quantizer_edges_are_the_deciles(
    quantizer_fixture: tuple[Vocabulary, pl.DataFrame],
) -> None:
    vocab, quantizer = quantizer_fixture
    row = _row(quantizer, vocab, "LAB//A")
    edges = [row[f"edge_{i}"] for i in range(9)]
    assert len(edges) == 9  # nine interior edges == ten bins
    assert edges == sorted(edges)
    expected = np.quantile(np.arange(1.0, 41.0), [i / 10 for i in range(1, 10)])
    assert edges == pytest.approx(list(expected))
    assert row["n_obs"] == 40


def test_quantizer_moments_and_log1p_flag(
    quantizer_fixture: tuple[Vocabulary, pl.DataFrame],
) -> None:
    vocab, quantizer = quantizer_fixture
    symmetric = _row(quantizer, vocab, "LAB//A")
    assert symmetric["use_log1p"] is False
    assert symmetric["mean"] == pytest.approx(20.5)
    assert symmetric["std"] == pytest.approx(float(np.std(np.arange(1.0, 41.0), ddof=1)))

    skewed = _row(quantizer, vocab, "LAB//SKEW")
    assert skewed["use_log1p"] is True
    # The moments are taken on the transformed values, not the raw ones.
    raw = np.array([0.0] * 36 + [1000.0, 2000.0, 3000.0, 4000.0])
    assert skewed["mean"] == pytest.approx(float(np.log1p(raw).mean()))


def test_binning_and_z_scores(
    quantizer_fixture: tuple[Vocabulary, pl.DataFrame], tmp_path: Path
) -> None:
    vocab, quantizer = quantizer_fixture
    t0 = dt.datetime(2020, 1, 1)
    events = _frame(
        [
            (1, t0, BIRTH, None),
            (1, t0, "LAB//A", 1.0),  # below every edge -> bin 1
            (1, t0, "LAB//A", 40.0),  # above every edge -> bin 10
            (1, t0, "LAB//A", 20.5),  # the mean -> z == 0, and sits on edge 5
            (1, t0, "LAB//A", 1e9),  # z clipped at +5
            (1, t0, "LAB//A", None),  # no value -> bin 0, z 0
        ]
    )
    features, _ = shard_features(events, vocab, quantizer)
    bins = features["value_bin"].to_list()
    z = features["value_z"].to_list()
    # A value exactly on an edge falls in the lower of the two bins.
    assert bins == [0, 1, 10, 5, 10, 0]
    assert z[1] < 0 < z[2]
    assert z[3] == pytest.approx(0.0, abs=1e-6)
    assert z[4] == pytest.approx(5.0)
    assert z[5] == 0.0


# --------------------------------------------------------------------------- #
# Time features
# --------------------------------------------------------------------------- #


def test_time_features_ages_deltas_and_ties() -> None:
    vocab, _ = fit_vocabulary({"E": 10, BIRTH: 10}, min_count=1)
    quantizer = fit_quantizer(_no_values(), vocab)
    birth = dt.datetime(2000, 1, 1)
    events = _frame(
        [
            (1, birth, BIRTH, None),
            (1, birth + dt.timedelta(days=365.25), "E", None),
            (1, birth + dt.timedelta(days=365.25, hours=1), "E", None),
            (1, birth + dt.timedelta(days=365.25, hours=1), "E", None),  # a tie
            (1, birth + dt.timedelta(days=365.25, hours=4), "E", None),
        ]
    )
    features, index = shard_features(events, vocab, quantizer)
    ages = features["age"].to_list()
    deltas = features["log_delta"].to_list()

    assert ages[0] == pytest.approx(0.0)
    assert ages[1] == pytest.approx(1.0, abs=1e-5)
    assert deltas[0] == 0.0  # the first event has no predecessor
    assert deltas[1] == pytest.approx(math.log1p(365.25 * 24), rel=1e-5)
    assert deltas[2] == pytest.approx(math.log1p(1.0), rel=1e-5)
    assert deltas[3] == 0.0  # identical timestamps
    assert deltas[4] == pytest.approx(math.log1p(3.0), rel=1e-5)
    assert bool(index["has_birth"][0]) is True

    # time_min is a monotone int64 in minutes, usable for label alignment.
    times = features["time_min"].to_list()
    assert times == sorted(times)
    assert times[1] - times[0] == int(365.25 * 24 * 60)


def test_age_is_clipped_and_missing_birth_falls_back_to_the_first_event() -> None:
    vocab, _ = fit_vocabulary({"E": 10, BIRTH: 10}, min_count=1)
    quantizer = fit_quantizer(_no_values(), vocab)
    events = _frame(
        [
            (1, dt.datetime(1800, 1, 1), BIRTH, None),
            (1, dt.datetime(2000, 1, 1), "E", None),  # 200 years later
            (2, dt.datetime(2010, 1, 1), "E", None),  # no birth event at all
            (2, dt.datetime(2011, 1, 1), "E", None),
        ]
    )
    features, index = shard_features(events, vocab, quantizer)
    ages = features["age"].to_list()
    assert ages[1] == pytest.approx(120.0)  # clipped
    assert ages[2] == pytest.approx(0.0)  # anchored at the first event
    assert ages[3] == pytest.approx(1.0, abs=1e-3)
    has_birth = dict(zip(index["subject_id"], index["has_birth"], strict=True))
    assert has_birth == {1: True, 2: False}


# --------------------------------------------------------------------------- #
# Cache and dataset
# --------------------------------------------------------------------------- #


def _fixture_events(subject: int, n: int, start: dt.datetime) -> list[tuple]:
    rows: list[tuple] = [(subject, start, BIRTH, None)]
    for i in range(n):
        when = start + dt.timedelta(days=365 + i)
        rows.append((subject, when, "LAB//A", float(i % 40 + 1)))
        rows.append((subject, when, "ICD9CM//250", None))
    return rows


@pytest.fixture(scope="module")
def built_cache(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("meds")
    cache = tmp_path_factory.mktemp("cache")
    start = dt.datetime(1980, 1, 1)
    train = _frame(
        [row for s in range(1, 6) for row in _fixture_events(s, 40, start)]
        + [
            (6, start, BIRTH, None),
            (6, start + dt.timedelta(days=1), "NDC//00003517887", 30.0),
            (6, start + dt.timedelta(days=2), "NDC//00003512345", 30.0),
            (6, start + dt.timedelta(days=3), "NDC//00003599999", 30.0),
            (6, start + dt.timedelta(days=4), "NDC//00003517887", 30.0),
            (6, start + dt.timedelta(days=5), "NDC//00003512345", 30.0),
            (6, start + dt.timedelta(days=6), "NDC//00003599999", 30.0),
        ]
    )
    tuning = _frame(
        _fixture_events(7, 30, start)
        + [
            (8, start, "E_SHORT", None),  # a two-event subject, dropped by min_len
            (8, start + dt.timedelta(days=1), "WEIRD//zzz", None),  # -> UNK
        ]
    )
    held_out = _frame(_fixture_events(9, 25, start))
    _write_meds(root, {"train": train, "tuning": tuning, "held_out": held_out})
    meta = build_cache(root, cache, min_count=5, min_value_obs=20)
    return cache, meta


def test_cache_layout_and_meta(built_cache: tuple[Path, dict]) -> None:
    cache, meta = built_cache
    for name in ("vocab.parquet", "quantizer.parquet", "vocab.json", "meta.json"):
        assert (cache / name).exists()
    for split in ("train", "tuning", "held_out"):
        for feature in FEATURES:
            assert (cache / split / f"{feature}.npy").exists()
        assert (cache / split / "subjects.parquet").exists()
    assert meta["vocab_size"] == len(Vocabulary.read(cache / "vocab.parquet"))
    assert meta["source_dataset"]["dataset_name"] == "fixture"
    assert set(meta["features"]) == set(FEATURES)
    assert meta["per_split"]["tuning"]["unk_rate"] > 0  # WEIRD//zzz had nowhere to go
    assert meta["per_split"]["train"]["subjects"] == 6


def test_cache_round_trip_matches_the_source(built_cache: tuple[Path, dict]) -> None:
    cache, _ = built_cache
    index = pl.read_parquet(cache / "train" / "subjects.parquet")
    codes = np.load(cache / "train" / "code_id.npy")
    times = np.load(cache / "train" / "time_min.npy")
    assert list(index.columns) == ["subject_id", "offset", "length", "split", "has_birth"]
    assert int(index["length"].sum()) == codes.shape[0]
    # Slices are contiguous, ordered and time-sorted.
    assert index["offset"].to_list() == np.cumsum([0, *index["length"].to_list()[:-1]]).tolist()
    for offset, length in zip(index["offset"], index["length"], strict=True):
        window = times[offset : offset + length]
        assert (np.diff(window) >= 0).all()

    vocab = Vocabulary.read(cache / "vocab.parquet")
    row = index.filter(pl.col("subject_id") == 6).row(0, named=True)
    decoded = [vocab.codes[i] for i in codes[row["offset"] : row["offset"] + row["length"]]]
    assert decoded[0] == BIRTH
    # The three rare NDCs all rolled up into the shared 5-digit labeler code.
    assert set(decoded[1:]) == {"NDC//00003"}


def test_dataset_windows_and_dropping(built_cache: tuple[Path, dict]) -> None:
    cache, _ = built_cache
    dataset = EventSequenceDataset(cache, "tuning", max_len=16, min_len=16)
    assert dataset.n_dropped == 1  # subject 8 has two events
    assert len(dataset) == 1
    item = dataset[0]
    assert set(item) >= set(FEATURES)
    assert item["code_id"].shape == (16,)
    assert item["code_id"].dtype == torch.long
    assert item["value_z"].dtype == torch.float32
    assert int(item["length"]) == 16

    starts = {int(dataset[0]["start"]) for _ in range(60)}
    assert len(starts) > 1, "random_window should not return the same crop every time"

    full = EventSequenceDataset(cache, "tuning", max_len=16, min_len=16, sampling="full")
    assert {int(full[0]["start"]) for _ in range(5)} == {int(full[0]["start"])}
    # "full" is the most recent window, so it ends at that subject's last event.
    row = full.index.row(0, named=True)
    times = np.load(cache / "tuning" / "time_min.npy")
    assert int(full[0]["time_min"][-1]) == int(times[row["offset"] + row["length"] - 1])


def test_short_subject_is_kept_when_min_len_allows(built_cache: tuple[Path, dict]) -> None:
    cache, _ = built_cache
    dataset = EventSequenceDataset(cache, "tuning", max_len=512, min_len=1)
    assert dataset.n_dropped == 0
    assert len(dataset) == 2
    lengths = sorted(int(dataset[i]["length"]) for i in range(len(dataset)))
    assert lengths[0] == 2


def test_collate_pads_and_masks(built_cache: tuple[Path, dict]) -> None:
    cache, _ = built_cache
    dataset = EventSequenceDataset(cache, "tuning", max_len=512, min_len=1, sampling="full")
    batch = collate_events([dataset[0], dataset[1]])
    lengths = batch["length"].tolist()
    width = max(lengths)
    for name in FEATURES:
        assert batch[name].shape == (2, width)
    assert batch["attention_mask"].shape == (2, width)
    assert batch["attention_mask"].sum(dim=1).tolist() == lengths
    short = int(np.argmin(lengths))
    assert (batch["code_id"][short, lengths[short] :] == PAD_ID).all()
    assert (batch["value_z"][short, lengths[short] :] == 0).all()
    assert batch["subject_id"].shape == (2,)


def test_collate_of_equal_length_windows_needs_no_padding(built_cache: tuple[Path, dict]) -> None:
    cache, _ = built_cache
    dataset = EventSequenceDataset(cache, "train", max_len=8, min_len=8)
    batch = collate_events([dataset[i] for i in range(4)])
    assert batch["code_id"].shape == (4, 8)
    assert batch["attention_mask"].all()


def test_windows_at_excludes_an_event_exactly_at_the_boundary(
    built_cache: tuple[Path, dict],
) -> None:
    cache, _ = built_cache
    dataset = EventSequenceDataset(cache, "held_out", max_len=512, min_len=1)
    subject = int(dataset.index["subject_id"][0])
    times = np.load(cache / "held_out" / "time_min.npy")
    offset = int(dataset.index["offset"][0])
    length = int(dataset.index["length"][0])
    subject_times = times[offset : offset + length]

    boundary = int(subject_times[10])
    window = dataset.windows_at(subject, boundary)
    assert int(window["length"]) == int((subject_times < boundary).sum())
    assert int(window["time_min"][-1]) < boundary

    # One minute later, every event at that timestamp is included.
    later = dataset.windows_at(subject, boundary + 1)
    assert int(later["length"]) > int(window["length"])
    assert int(later["time_min"][-1]) == boundary

    # A datetime cutoff and its minute-integer form agree.
    as_datetime = dt.datetime(1970, 1, 1) + dt.timedelta(minutes=boundary)
    assert int(dataset.windows_at(subject, as_datetime)["length"]) == int(window["length"])

    # Nothing before the first event, and the window respects max_len.
    assert int(dataset.windows_at(subject, int(subject_times[0]))["length"]) == 0
    capped = EventSequenceDataset(cache, "held_out", max_len=4, min_len=1)
    assert int(capped.windows_at(subject, int(subject_times[-1]))["length"]) == 4

    with pytest.raises(KeyError):
        dataset.windows_at(-1, boundary)


def test_dataset_rejects_bad_sampling(built_cache: tuple[Path, dict]) -> None:
    cache, _ = built_cache
    with pytest.raises(ValueError, match="sampling"):
        EventSequenceDataset(cache, "train", sampling="sliding")


def test_fit_is_train_only(tmp_path: Path) -> None:
    """A code that only ever appears outside train never enters the vocabulary."""
    start = dt.datetime(1990, 1, 1)
    train = _frame([(1, start + dt.timedelta(days=i), "LAB//A", 1.0) for i in range(10)])
    tuning = _frame([(2, start + dt.timedelta(days=i), "TUNING_ONLY//x", 1.0) for i in range(10)])
    _write_meds(tmp_path, {"train": train, "tuning": tuning, "held_out": train.clear()})
    vocab, quantizer, stats = fit_tokenizer(tmp_path, min_count=5)
    assert "TUNING_ONLY//x" not in vocab.index
    assert vocab.resolve("TUNING_ONLY//x") == (UNK_ID, "unk")
    assert stats["train_events"] == 10
    assert quantizer.height == len(vocab)
