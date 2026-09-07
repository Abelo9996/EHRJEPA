"""Code descriptions, the PCA projection, and ``model.code_init: text``.

Nothing here touches the network or ``sentence-transformers``: the description
side is pure lookup, and :func:`~ehrjepa.data.code_text.build_table` takes the
encoder as an argument precisely so a test can hand it a deterministic one.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ehrjepa.data import code_text
from ehrjepa.data.code_text import (
    DescriptionTables,
    build_table,
    coverage,
    describe,
    describe_vocab,
    project,
)
from ehrjepa.data.tokenize import SPECIAL_TOKENS, Vocabulary
from ehrjepa.models import EHRJEPA, EHRJEPAConfig, EventEmbedding
from ehrjepa.train.config import load_config

REPO = Path(__file__).resolve().parents[1]

CODES = (
    *SPECIAL_TOKENS,
    "VISIT//OUTPATIENT",
    "SP_ALZHDMTA//1",
    "SP_RA_OA//2",
    "SEX//F",
    "RACE//HISPANIC",
    "MEDS_BIRTH",
    "ICD9CM//25000",
    "ICD9CM//250",
    "ICD9CM//9999",
    "ICD9PROC//9904",
    "ICD9PROC//4444",
    "HCPCS//J2501",
    "HCPCS//36415",
    "HCPCS//Z9999",
    "NDC//000020152",
    "NDC//999990001",
    "NDC//00002",
    "DRG//177",
    "DRG//999",
    "NDC",
    "WEIRD//thing",
)


def _tables(**overrides) -> DescriptionTables:
    values = dict(
        icd9_dx={"25000": "Diabetes mellitus, type II"},
        icd9_sg={"9904": "Transfusion of packed cells"},
        icd9_category=lambda code: "Diabetes mellitus" if code == "250" else None,
        hcpcs={"J2501": "Injection, paricalcitol, 1 mcg"},
        hcpcs_section={"36415": "Cardiovascular system"},
        drg={"177": "RESPIRATORY INFECTIONS"},
        ndc_product={"000020152": "Zepbound (tirzepatide), injection, solution"},
        ndc_labeler={"00002": "Eli Lilly and Company", "99999": "Some Labs"},
    )
    values.update(overrides)
    return DescriptionTables(**values)


def _vocab(codes=CODES) -> Vocabulary:
    n = len(codes)
    return Vocabulary(
        codes=tuple(codes),
        train_count=tuple(range(n)),
        direct_count=tuple(range(n)),
        is_ancestor=tuple(False for _ in range(n)),
    )


# --------------------------------------------------------------------------- #
# Descriptions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("code", "tier", "source", "needle"),
    [
        ("[PAD]", "exact", "handwritten", "padding"),
        ("[CLS]", "exact", "handwritten", "summary"),
        ("MEDS_BIRTH", "exact", "handwritten", "date of birth"),
        ("VISIT//OUTPATIENT", "exact", "handwritten", "outpatient visit"),
        ("SEX//F", "exact", "handwritten", "female"),
        ("RACE//HISPANIC", "exact", "handwritten", "hispanic"),
        ("SP_ALZHDMTA//1", "exact", "desynpuf-codebook", "Alzheimer"),
        ("SP_RA_OA//2", "exact", "desynpuf-codebook", "no chronic condition"),
        ("ICD9CM//25000", "exact", "cms-icd9-dx", "Diabetes mellitus, type II"),
        ("ICD9CM//250", "ancestor", "icd9cms", "Diabetes mellitus"),
        ("ICD9CM//9999", "fallback", "fallback", "ICD-9-CM code 9999"),
        ("ICD9PROC//9904", "exact", "cms-icd9-sg", "packed cells"),
        ("ICD9PROC//4444", "fallback", "fallback", "inpatient procedure"),
        ("HCPCS//J2501", "exact", "mimic-d-hcpcs", "paricalcitol"),
        ("HCPCS//36415", "ancestor", "mimic-cpt-section", "cardiovascular system"),
        ("HCPCS//Z9999", "fallback", "fallback", "HCPCS code Z9999"),
        ("NDC//000020152", "exact", "fda-ndc", "Zepbound"),
        ("NDC//999990001", "ancestor", "fda-ndc-labeler", "Some Labs"),
        ("DRG//177", "exact", "mimic-drgcodes", "respiratory infections"),
        ("DRG//999", "fallback", "fallback", "diagnosis-related group 999"),
        ("NDC", "ancestor", "handwritten", "prescribed drug product"),
        ("WEIRD//thing", "fallback", "fallback", "WEIRD//thing"),
    ],
)
def test_each_code_family_resolves_to_its_documented_tier(
    code: str, tier: str, source: str, needle: str
) -> None:
    item = describe(code, _tables())
    assert item.tier == tier, item
    assert item.source == source, item
    assert needle in item.text, item


def test_the_hcpcs_letter_group_catches_a_level_two_code_the_table_lacks() -> None:
    item = describe("HCPCS//A9999", _tables())
    assert item.tier == "ancestor" and item.source == "hcpcs-group"
    assert "supply" in item.text


def test_a_missing_source_table_degrades_to_fallback_rather_than_failing() -> None:
    """The GPU machine may have no MIMIC extract; that costs coverage, not the build."""
    empty = _tables(icd9_dx={}, icd9_category=None, hcpcs={}, hcpcs_section={}, drg={})
    item = describe("ICD9CM//25000", empty)
    assert item.tier == "fallback" and "25000" in item.text
    assert describe("DRG//177", empty).tier == "fallback"


def test_an_ndc_labeler_is_read_from_the_five_digit_prefix() -> None:
    item = describe("NDC//00002", _tables())
    assert item.source == "fda-ndc-labeler" and "Eli Lilly" in item.text


def test_every_description_is_non_empty_and_distinct_enough_to_embed() -> None:
    descriptions = describe_vocab(_vocab(), _tables())
    assert len(descriptions) == len(CODES)
    assert all(item.text.strip() for item in descriptions)
    assert len({item.text for item in descriptions}) == len(descriptions)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_coverage_tiers_partition_the_vocabulary_and_its_event_mass() -> None:
    vocab = _vocab()
    stats = coverage(vocab, describe_vocab(vocab, _tables()))
    assert sum(b["entries"] for b in stats["by_tier"].values()) == len(vocab)
    assert sum(b["events"] for b in stats["by_tier"].values()) == sum(vocab.train_count)
    assert sum(b["entries"] for b in stats["by_source"].values()) == len(vocab)
    assert stats["described_entries"] == len(vocab) - stats["by_tier"]["fallback"]["entries"]
    assert 0.0 < stats["described_entry_rate"] <= 1.0


def test_coverage_splits_by_family_with_a_tier_breakdown() -> None:
    vocab = _vocab()
    stats = coverage(vocab, describe_vocab(vocab, _tables()))
    families = stats["by_family"]
    assert families["[SPECIAL]"]["entries"] == len(SPECIAL_TOKENS)
    assert families["SP_*"]["entries"] == 2 and families["SP_*"]["exact"] == 2
    icd = families["ICD9CM"]
    assert icd["exact"] + icd["ancestor"] + icd["fallback"] == icd["entries"] == 3


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def test_project_reaches_the_requested_width_and_the_requested_scale() -> None:
    vectors = np.random.default_rng(0).normal(size=(200, 384)).astype(np.float32)
    out = project(vectors, 256, std=0.02)
    assert out.shape == (200, 256) and out.dtype == np.float32
    assert out.std() == pytest.approx(0.02, rel=1e-4)


def test_project_is_deterministic_given_a_seed() -> None:
    vectors = np.random.default_rng(1).normal(size=(64, 384)).astype(np.float32)
    assert np.array_equal(project(vectors, 32, seed=3), project(vectors, 32, seed=3))


def test_project_fills_the_columns_pca_cannot_reach_rather_than_leaving_them_dead() -> None:
    """``width > sentence_dim``: the extra columns are noise, not a zero block."""
    vectors = np.random.default_rng(2).normal(size=(50, 8)).astype(np.float32)
    out = project(vectors, 24, seed=0)
    assert out.shape == (50, 24)
    assert (out[:, 8:].std(axis=0) > 0).all()
    assert out.std() == pytest.approx(0.02, rel=1e-4)


def test_project_preserves_the_geometry_it_can() -> None:
    """A rank-``width`` PCA of rank-``width`` data is exact up to the global rescale."""
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(120, 6))
    vectors = (latent @ rng.normal(size=(6, 64))).astype(np.float32)
    out = project(vectors, 6)
    before = np.corrcoef(
        np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1).ravel(),
        np.linalg.norm(out[:, None] - out[None, :], axis=-1).ravel(),
    )[0, 1]
    assert before > 0.999


def test_project_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="sentence vectors"):
        project(np.zeros(10), 4)


# --------------------------------------------------------------------------- #
# build_table
# --------------------------------------------------------------------------- #


def _fake_encoder(dim: int = 16):
    """Deterministic per-string vectors -- no model, no download, no network."""

    def encode(texts):
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.normal(size=dim))
        return np.asarray(rows, dtype=np.float32)

    return encode


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    _vocab().write(cache / "vocab.parquet")
    return cache


def test_build_table_writes_a_table_of_the_right_shape_and_a_coverage_json(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    stats = build_table(
        cache,
        8,
        source_dir=tmp_path / "empty",
        encoder=_fake_encoder(),
        stats_out=tmp_path / "s.json",
    )
    table = np.load(cache / "code_init_text_8.npy")
    assert table.shape == (len(CODES), 8) and table.dtype == np.float32
    assert not np.isnan(table).any()
    assert np.array_equal(table[0], np.zeros(8)), "PAD must stay at zero"
    assert table.std() == pytest.approx(0.02, rel=1e-3)

    written = json.loads((cache / "code_init_text_8.json").read_text())
    assert written == stats
    assert json.loads((tmp_path / "s.json").read_text()) == stats
    assert stats["width"] == 8 and stats["vocab_size"] == len(CODES)
    assert set(stats["by_tier"]) <= {"exact", "ancestor", "fallback"}
    assert len(stats["table_dim_std"]) == 8


def test_build_table_refuses_an_encoder_that_returns_the_wrong_number_of_rows(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)

    def short(texts):
        return np.zeros((len(texts) - 1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="rows for"):
        build_table(cache, 4, source_dir=tmp_path / "empty", encoder=short)


def test_build_table_reads_source_tables_it_finds(tmp_path: Path) -> None:
    """A ``d_hcpcs.csv.gz`` dropped into ``--sources`` is picked up without MIMIC."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "CMS32_DESC_LONG_DX.txt").write_text("25000  Diabetes mellitus type II\n")
    with gzip.open(sources / "d_hcpcs.csv.gz", "wt", newline="") as handle:
        handle.write("code,category,long_description,short_description\n")
        handle.write("J2501,,Injection paricalcitol,Inj\n")
    tables = code_text.load_tables(sources, repo=tmp_path)
    assert tables.icd9_dx["25000"].startswith("Diabetes")
    assert tables.hcpcs["J2501"] == "Injection paricalcitol"
    assert describe("ICD9CM//25000", tables).tier == "exact"


# --------------------------------------------------------------------------- #
# The model side
# --------------------------------------------------------------------------- #


def _write_table(path: Path, rows: int, dim: int, scale: float = 0.02) -> np.ndarray:
    table = (np.random.default_rng(0).normal(size=(rows, dim)) * scale).astype(np.float32)
    np.save(path, table)
    return table


def test_random_is_the_default_and_leaves_the_table_untouched(tmp_path: Path) -> None:
    torch.manual_seed(0)
    plain = EventEmbedding(32, 8)
    torch.manual_seed(0)
    explicit = EventEmbedding(32, 8, code_init="random")
    assert torch.equal(plain.code_emb.weight, explicit.code_emb.weight)
    assert plain.code_emb.weight.requires_grad


def test_text_init_copies_the_table_and_keeps_pad_at_zero(tmp_path: Path) -> None:
    path = tmp_path / "init.npy"
    table = _write_table(path, 32, 8)
    embed = EventEmbedding(32, 8, code_init="text", code_init_path=path)
    assert torch.allclose(embed.code_emb.weight[1:], torch.from_numpy(table[1:]))
    assert torch.equal(embed.code_emb.weight[0], torch.zeros(8))
    assert embed.code_emb.weight.requires_grad


def test_text_init_does_not_change_the_rng_stream(tmp_path: Path) -> None:
    """A ``text`` run and a ``random`` run must draw identically, or every checksum moves."""
    path = tmp_path / "init.npy"
    _write_table(path, 32, 8)
    torch.manual_seed(0)
    EventEmbedding(32, 8)
    after_random = torch.randn(4)
    torch.manual_seed(0)
    EventEmbedding(32, 8, code_init="text", code_init_path=path)
    assert torch.equal(after_random, torch.randn(4))


def test_freezing_the_code_table_removes_it_from_the_gradient(tmp_path: Path) -> None:
    embed = EventEmbedding(32, 8, freeze_code_embeddings=True)
    assert not embed.code_emb.weight.requires_grad
    assert embed.value_bin_emb.weight.requires_grad
    trainable = sum(p.numel() for p in embed.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in embed.parameters()) - 32 * 8


def test_a_frozen_code_table_never_receives_a_gradient(tmp_path: Path) -> None:
    embed = EventEmbedding(32, 8, freeze_code_embeddings=True)
    code = torch.randint(0, 32, (2, 5))
    zeros = torch.zeros(2, 5)
    out = embed(code, torch.ones(2, 5, dtype=torch.long), zeros, zeros, zeros)
    out.sum().backward()
    assert embed.code_emb.weight.grad is None
    assert embed.value_bin_emb.weight.grad is not None


def test_a_mismatched_table_is_refused_by_shape(tmp_path: Path) -> None:
    path = tmp_path / "init.npy"
    _write_table(path, 32, 16)
    with pytest.raises(ValueError, match=r"\(32, 8\)"):
        EventEmbedding(32, 8, code_init="text", code_init_path=path)


def test_a_missing_table_names_the_cli_that_builds_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="code_text build"):
        EventEmbedding(32, 8, code_init="text", code_init_path=tmp_path / "nope.npy")


def test_an_unknown_code_init_is_rejected() -> None:
    with pytest.raises(ValueError, match="code_init"):
        EventEmbedding(32, 8, code_init="glove")
    with pytest.raises(ValueError, match="code_init"):
        EHRJEPAConfig(vocab_size=32, code_init="glove")


def test_the_model_passes_code_init_through_to_its_embedding(tmp_path: Path) -> None:
    path = tmp_path / "init.npy"
    table = _write_table(path, 32, 8)
    model = EHRJEPA(
        EHRJEPAConfig(
            vocab_size=32,
            dim=8,
            depth=1,
            heads=2,
            pred_dim=8,
            pred_depth=1,
            pred_heads=2,
            n_freq=4,
            code_init="text",
            code_init_path=str(path),
            freeze_code_embeddings=True,
        )
    )
    assert torch.allclose(model.embed.code_emb.weight[1:], torch.from_numpy(table[1:]))
    assert not model.embed.code_emb.weight.requires_grad


def test_config_fills_the_table_path_from_the_cache_and_the_width() -> None:
    config = load_config(
        REPO / "configs" / "pretrain_debug.yaml", ["model.code_init=text", "model.dim=64"]
    )
    model = config.model_config(vocab_size=101)
    assert model.code_init_path == "data/cache/mimic-demo/code_init_text_64.npy"


def test_an_explicit_table_path_wins_over_the_derived_one() -> None:
    config = load_config(
        REPO / "configs" / "pretrain_debug.yaml",
        ["model.code_init=text", "model.code_init_path=/tmp/elsewhere.npy"],
    )
    assert config.model_config(vocab_size=101).code_init_path == "/tmp/elsewhere.npy"


def test_text_init_needs_a_path_at_the_config_level() -> None:
    with pytest.raises(ValueError, match="code_init_path"):
        EHRJEPAConfig(vocab_size=32, code_init="text")
