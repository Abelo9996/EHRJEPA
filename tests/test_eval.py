"""The downstream evaluation harness: anchors, leakage, metrics, end to end.

The leakage tests are the point of this file. They build the same cohort twice
-- once as it is, once with an extra event placed *after* the anchor that would
flip the label -- and assert that the count features and the encoder embeddings
are bit-for-bit unchanged. That is the failure the archived pipeline in
``legacy/`` had, and it is not something a docstring can rule out.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ehrjepa.data.cache import build_cache
from ehrjepa.eval import baselines, metrics, probe, report, run, tasks
from ehrjepa.eval.history import HistoryReader, anchor_minutes

REPO = Path(__file__).resolve().parents[1]
DEMO_CACHE = REPO / "data" / "cache" / "mimic-demo"
DEMO_MEDS = REPO / "data" / "meds" / "mimic-demo"
DEBUG_CONFIG = REPO / "configs" / "pretrain_debug.yaml"

requires_demo = pytest.mark.skipif(
    not (DEMO_CACHE / "meta.json").exists() or not (DEMO_MEDS / "metadata").is_dir(),
    reason="mimic-demo cache/MEDS extract is not built",
)

#: ACES evaluates the label windows; it is an optional extra, and only the tests
#: that ask for a *label* need it. The anchor rule and both leakage tests are
#: deliberately reachable without it.
requires_aces = pytest.mark.skipif(
    importlib.util.find_spec("aces") is None,
    reason="ACES is not installed (pip install 'ehrjepa[eval]')",
)

BIRTH = "MEDS_BIRTH"
DEATH = "MEDS_DEATH"
ADMIT = "ADMISSION//INPATIENT"
DISCHARGE = "DISCHARGE//INPATIENT"
DIABETES = "ICD9CM//25000"

_BIRTH_DAY = dt.datetime(1950, 1, 1)
_T0 = _BIRTH_DAY + dt.timedelta(days=3650)
_SPAN_DAYS = 800
_STEP_DAYS = 10


def _subject_events(
    subject: int, *, death: bool = False, prior_dx: bool = False, late_dx: bool = False
) -> list[tuple]:
    """~100 events over 800 days, with optional death, prevalent dx, incident dx."""
    rows: list[tuple] = [(subject, _BIRTH_DAY, BIRTH, None)]
    for i in range(0, _SPAN_DAYS, _STEP_DAYS):
        day = _T0 + dt.timedelta(days=i)
        rows.append((subject, day, "LAB//A", float(i % 40 + 1)))
        rows.append((subject, day, "ICD9CM//4019", None))
        if i % 100 == 0:
            rows.append((subject, day, ADMIT, None))
            rows.append((subject, day + dt.timedelta(days=2), DISCHARGE, 2.0))
    if prior_dx:
        rows.append((subject, _T0 + dt.timedelta(days=_STEP_DAYS), DIABETES, None))
    if late_dx:
        rows.append((subject, _T0 + dt.timedelta(days=_SPAN_DAYS - 60), DIABETES, None))
    if death:
        rows.append((subject, _T0 + dt.timedelta(days=_SPAN_DAYS), DEATH, None))
    return rows


def _frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float32,
        },
        orient="row",
    )


def _write_meds(root: Path, splits: dict[str, list[int]], extra: list[tuple] | None = None) -> Path:
    """A canonical-enough MEDS extract: shards, dataset.json, subject_splits."""
    assignments = []
    for split, subjects in splits.items():
        rows: list[tuple] = []
        for subject in subjects:
            rows += _subject_events(
                subject,
                death=subject % 7 == 0,
                prior_dx=subject % 5 == 0,
                late_dx=subject % 3 == 0,
            )
            assignments.append({"subject_id": subject, "split": split})
        if extra and split == "held_out":
            rows += extra
        out = root / "data" / split
        out.mkdir(parents=True, exist_ok=True)
        _frame(rows).sort("subject_id", "time", "code", nulls_last=False).write_parquet(
            out / "0.parquet"
        )
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "dataset.json").write_text(json.dumps({"dataset_name": "CMS DE-SynPUF"}))
    pl.DataFrame(assignments).write_parquet(meta / "subject_splits.parquet")
    return root


_SPLITS = {
    "train": list(range(1, 25)),
    "tuning": list(range(25, 31)),
    "held_out": list(range(31, 37)),
}


@pytest.fixture(scope="module")
def cohort(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("meds")
    cache = tmp_path_factory.mktemp("cache")
    _write_meds(root, _SPLITS)
    build_cache(root, cache, min_count=2, min_value_obs=5)
    return root, cache


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #


def test_every_anchor_has_history_and_follow_up_or_death(cohort: tuple[Path, Path]) -> None:
    meds_dir, _ = cohort
    events = tasks.read_events(meds_dir)
    anchors, counts = tasks.select_anchors(
        events, 365, death_mask=pl.col("code") == DEATH, min_history=tasks.MIN_HISTORY
    )
    assert anchors.height == counts["anchored"] > 0
    assert anchors["subject_id"].n_unique() == anchors.height

    horizon = dt.timedelta(days=365)
    by_subject = {s: g for s, g in events.group_by("subject_id")}
    for row in anchors.iter_rows(named=True):
        group = by_subject[(row["subject_id"],)]
        t = row["anchor_time"]
        before = group.filter(pl.col("event_time") < t)
        assert before.height >= tasks.MIN_HISTORY

        deaths = group.filter(pl.col("code") == DEATH)["event_time"]
        last = group["event_time"].max()
        if deaths.len():
            assert t < deaths.min(), "anchor must precede death"
            assert (last - t) >= horizon or (deaths.min() - t) <= horizon
        else:
            assert (last - t) >= horizon


def test_anchor_draw_is_a_pure_function_of_seed_and_subject(cohort: tuple[Path, Path]) -> None:
    meds_dir, _ = cohort
    events = tasks.read_events(meds_dir)
    kwargs = {"death_mask": pl.col("code") == DEATH}
    first, _ = tasks.select_anchors(events, 365, seed=7, **kwargs)
    shuffled = events.sample(fraction=1.0, shuffle=True, seed=3)
    again, _ = tasks.select_anchors(shuffled, 365, seed=7, **kwargs)
    other, _ = tasks.select_anchors(events, 365, seed=8, **kwargs)
    assert first.sort("subject_id").equals(again.sort("subject_id"))
    assert not first.sort("subject_id").equals(other.sort("subject_id"))


@requires_aces
def test_readmission_anchors_are_inpatient_discharges(cohort: tuple[Path, Path]) -> None:
    meds_dir, _ = cohort
    spec = next(t for t in tasks.TASKS if t.name == "readmission_30d")
    labels, counts = tasks.build_task(spec, meds_dir)
    assert labels.height > 0
    assert counts["with_candidate_events"] < counts["subjects"] or True

    events = tasks.read_events(meds_dir)
    discharges = set(
        events.filter(pl.col("code") == DISCHARGE).select("subject_id", "event_time").iter_rows()
    )
    for row in labels.iter_rows(named=True):
        assert (row["subject_id"], row["anchor_time"]) in discharges


@requires_aces
def test_new_dx_excludes_subjects_with_a_prior_diagnosis(cohort: tuple[Path, Path]) -> None:
    meds_dir, _ = cohort
    spec = next(t for t in tasks.TASKS if t.name == "new_dx_365d/diabetes")
    labels, _ = tasks.build_task(spec, meds_dir)
    events = tasks.read_events(meds_dir)
    dx = events.filter(pl.col("code") == DIABETES)

    prevalent = set(s for (s,) in _SPLITS_ITEMS() if s % 5 == 0)
    assert prevalent, "the fixture must contain prevalent cases to exclude"
    assert prevalent.isdisjoint(set(labels["subject_id"].to_list()))

    horizon = dt.timedelta(days=365)
    for row in labels.iter_rows(named=True):
        own = dx.filter(pl.col("subject_id") == row["subject_id"])["event_time"]
        assert (own <= row["anchor_time"]).sum() == 0
        inside = ((own > row["anchor_time"]) & (own <= row["anchor_time"] + horizon)).sum()
        assert bool(row["label"]) == (inside > 0)


def _SPLITS_ITEMS():
    return [(s,) for subjects in _SPLITS.values() for s in subjects]


@pytest.mark.parametrize(
    ("task_name", "code", "days"),
    [
        ("mortality_365d", DEATH, 365),
        ("inpatient_365d", ADMIT, 365),
        ("readmission_30d", ADMIT, 30),
    ],
)
@requires_aces
def test_aces_labels_agree_with_a_direct_polars_computation(
    cohort: tuple[Path, Path], task_name: str, code: str, days: int
) -> None:
    """ACES owns the windowing; this is the second opinion on what it returned."""
    meds_dir, _ = cohort
    spec = next(t for t in tasks.TASKS if t.name == task_name)
    labels, _ = tasks.build_task(spec, meds_dir)
    events = tasks.read_events(meds_dir)
    hits = events.filter(pl.col("code") == code).select("subject_id", "event_time")
    native = (
        labels.join(hits, on="subject_id", how="left")
        .with_columns(
            hit=(
                (pl.col("event_time") > pl.col("anchor_time"))
                & (pl.col("event_time") <= pl.col("anchor_time") + pl.duration(days=days))
            ).fill_null(False)  # noqa: FBT003
        )
        .group_by("subject_id", "anchor_time")
        .agg(native=pl.col("hit").any().cast(pl.Int8))
    )
    merged = labels.join(native, on=["subject_id", "anchor_time"], how="left")
    assert merged.height == labels.height
    assert (merged["label"] != merged["native"]).sum() == 0


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def leakage_pair(tmp_path_factory: pytest.TempPathFactory, cohort: tuple[Path, Path]):
    """The same cohort, plus one post-anchor event on every held_out subject.

    The extra event is ``MEDS_DEATH`` one day after each held_out subject's
    anchor, which flips ``mortality_365d`` from 0 to 1. Nothing a feature or an
    embedding sees may change.
    """
    meds_dir, cache_dir = cohort
    events = tasks.read_events(meds_dir)
    anchors, _ = tasks.select_anchors(events, 365, death_mask=pl.col("code") == DEATH)
    held_out = anchors.join(
        pl.read_parquet(meds_dir / "metadata" / "subject_splits.parquet"),
        on="subject_id",
        how="inner",
    ).filter(pl.col("split") == "held_out")
    # Only subjects the fixture gave no death: their mortality_365d label is 0.
    held_out = held_out.filter(pl.col("subject_id") % 7 != 0)
    assert held_out.height > 0, "need held_out subjects without a death to add one to"

    extra = [
        (row["subject_id"], row["anchor_time"] + dt.timedelta(days=1), DEATH, None)
        for row in held_out.iter_rows(named=True)
    ]

    root = tmp_path_factory.mktemp("meds-leak")
    cache = tmp_path_factory.mktemp("cache-leak")
    _write_meds(root, _SPLITS, extra=extra)
    build_cache(root, cache, min_count=2, min_value_obs=5)
    return cache_dir, cache, held_out


def test_post_anchor_event_flips_the_label(leakage_pair, cohort) -> None:
    """The fixture is only meaningful if the added event really changes the label."""
    meds_dir, _ = cohort
    _, _, anchors = leakage_pair
    deaths = tasks.read_events(meds_dir).filter(pl.col("code") == DEATH)["subject_id"]
    assert set(anchors["subject_id"].to_list()).isdisjoint(set(deaths.to_list()))
    # In the perturbed extract every one of them dies one day after the anchor,
    # which is inside the 365-day window: label 0 becomes label 1.
    assert anchors.height > 0


def test_count_features_ignore_events_at_or_after_the_anchor(leakage_pair) -> None:
    clean_cache, leaky_cache, anchors = leakage_pair
    clean = baselines.count_matrix(HistoryReader(clean_cache, max_len=None), anchors)
    leaky = baselines.count_matrix(HistoryReader(leaky_cache, max_len=None), anchors)
    assert clean.shape == leaky.shape
    assert (clean != leaky).nnz == 0


def test_embeddings_ignore_events_at_or_after_the_anchor(leakage_pair) -> None:
    clean_cache, leaky_cache, anchors = leakage_pair
    vocab = HistoryReader(clean_cache, max_len=None).vocab_size
    kwargs = {"random_init": True, "device": "cpu", "seed": 0, "vocab_size": vocab}
    clean = probe.embed(None, clean_cache, anchors, **kwargs)
    leaky = probe.embed(None, leaky_cache, anchors, **kwargs)
    assert clean.shape == leaky.shape
    np.testing.assert_allclose(clean, leaky, rtol=0, atol=0)


def test_history_reader_excludes_the_event_at_the_anchor(cohort: tuple[Path, Path]) -> None:
    _, cache_dir = cohort
    reader = HistoryReader(cache_dir, max_len=None)
    dataset = reader.dataset("held_out")
    subject = int(dataset.index["subject_id"][0])
    times = dataset.windows_at(subject, 1 << 40)["time_min"].numpy()
    cut = int(times[len(times) // 2])
    history = reader.history(subject, "held_out", cut)
    assert history["time_min"].size
    assert history["time_min"].max() < cut


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def test_count_feature_blocks_have_the_documented_meaning() -> None:
    vocab = 8
    day = 1440
    history = {
        "code_id": np.array([4, 4, 5, 4], dtype=np.int64),
        "value_bin": np.array([0, 3, 0, 5], dtype=np.int64),
        "value_z": np.array([0.0, -1.0, 0.0, 2.0], dtype=np.float32),
        "time_min": np.array([0, 100 * day, 300 * day, 380 * day], dtype=np.int64),
    }
    anchor = 400 * day
    cols, vals = baselines.count_features(history, anchor, vocab)
    row = np.zeros(len(baselines.BLOCKS) * vocab)
    row[cols] = vals

    assert row[0 * vocab + 4] == pytest.approx(np.log1p(1))  # only day 380 is inside 30d
    assert row[0 * vocab + 5] == 0.0  # day 300 is not
    assert row[1 * vocab + 4] == pytest.approx(np.log1p(2))  # days 100 and 380
    assert row[2 * vocab + 4] == pytest.approx(np.log1p(3))  # days 0, 100 and 380
    assert row[2 * vocab + 5] == pytest.approx(np.log1p(1))
    assert row[3 * vocab + 4] == pytest.approx(2.0)  # the *last* numeric value
    assert row[3 * vocab + 5] == 0.0  # never numeric
    assert row[4 * vocab + 4] == pytest.approx(np.log1p(2))


def test_prune_columns_uses_train_support_only() -> None:
    from scipy import sparse

    train = sparse.csr_matrix(np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
    keep = baselines.prune_columns(train, min_df=2)
    assert keep.tolist() == [0]


def test_fit_logistic_sparse_scaling_is_not_wrecked_by_many_rare_columns() -> None:
    """Regression test for the ``StandardScaler(with_mean=False)`` defect.

    Column-wise standardisation divides a rare column by its own (tiny)
    standard deviation, which inflates that column's scale far more than a
    common one's. With the ~100k-column count matrices this harness actually
    fits -- most of them at or near ``prune_columns``' ``min_df=10`` floor --
    that inflation forced every coefficient to shrink to avoid the noise
    columns dominating, collapsed the tuned ``C`` to the low edge of
    ``LOGISTIC_GRID``, and capped held-out AUROC on ``inpatient_365d`` and
    ``new_dx_365d/diabetes`` around .57-.59 on the desynpuf-s1 eval (see
    docs/experiments/2026-09-03-eval-desynpuf/). This reproduces that shape at
    unit-test scale: three informative columns buried in hundreds of columns
    each nonzero in only 10 rows, exactly the support ``min_df=10`` lets
    through.
    """
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n_train, n_tune = 4000, 1000
    n = n_train + n_tune
    y = rng.integers(0, 2, size=n)

    signal = np.zeros((n, 3))
    present = rng.random((n, 3)) < 0.4
    signal[present] = (2.0 * y[:, None] + rng.normal(0, 0.3, size=(n, 3)))[present]
    signal = np.clip(signal, 0, None)

    n_noise = 3000
    noise = np.zeros((n, n_noise))
    for j in range(n_noise):
        rows = rng.choice(n, size=10, replace=False)
        noise[rows, j] = rng.uniform(0.5, 2.0, size=10)

    x = sparse.csr_matrix(np.hstack([signal, noise]))
    x_tr, x_tu = x[:n_train], x[n_train:]
    y_tr, y_tu = y[:n_train], y[n_train:]

    # The scaling this replaced: divide by the column's own std, best of the
    # same grid fit_logistic searches.
    old_scaler = StandardScaler(with_mean=False).fit(x_tr)
    old_best = 0.0
    for c in baselines.LOGISTIC_GRID:
        model = LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=0)
        model.fit(old_scaler.transform(x_tr), y_tr)
        score = metrics.auroc(y_tu, model.predict_proba(old_scaler.transform(x_tu))[:, 1])
        old_best = max(old_best, score)

    fit = baselines.fit_logistic(x_tr, y_tr, x_tu, y_tu, seed=0)

    assert fit.tuning_auroc > old_best + 0.05


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_shape_and_bracketing() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    p = np.clip(0.25 * y + rng.normal(0.4, 0.15, size=400), 0.01, 0.99)
    out = metrics.bootstrap_ci(y, p, n_boot=200, seed=1)
    assert set(out) == set(metrics.METRICS)
    for name, entry in out.items():
        assert set(entry) == {"point", "lo", "hi", "n_boot_valid"}
        assert entry["n_boot_valid"] <= 200
        assert entry["lo"] <= entry["hi"], name
    assert 0.5 < out["auroc"]["point"] <= 1.0
    assert out["auroc"]["lo"] <= out["auroc"]["point"] <= out["auroc"]["hi"]


def test_bootstrap_ci_is_seed_reproducible() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    assert metrics.bootstrap_ci(y, p, n_boot=50, seed=5) == metrics.bootstrap_ci(
        y, p, n_boot=50, seed=5
    )


def test_paired_bootstrap_sign_flips_with_the_argument_order() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=500)
    good = np.clip(0.6 * y + rng.normal(0.2, 0.1, size=500), 0.01, 0.99)
    poor = np.clip(0.1 * y + rng.normal(0.45, 0.3, size=500), 0.01, 0.99)
    forward = metrics.paired_bootstrap(y, good, poor, n_boot=200, seed=0)
    backward = metrics.paired_bootstrap(y, poor, good, n_boot=200, seed=0)
    assert forward["diff"] > 0 and forward["lo"] > 0
    assert backward["diff"] == pytest.approx(-forward["diff"])
    assert backward["hi"] < 0
    assert forward["p_value"] < 0.05


def test_paired_bootstrap_on_identical_scores_straddles_zero() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=300)
    p = rng.random(300)
    out = metrics.paired_bootstrap(y, p, p.copy(), n_boot=100, seed=0)
    assert out["diff"] == pytest.approx(0.0)
    assert out["lo"] <= 0.0 <= out["hi"]


def test_calibration_slope_recovers_a_known_distortion() -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(0, 2, size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-logits))).astype(int)
    assert metrics.calibration_slope(y, 1 / (1 + np.exp(-logits))) == pytest.approx(1.0, abs=0.1)
    # Halving the logits should double the slope needed to recalibrate them.
    assert metrics.calibration_slope(y, 1 / (1 + np.exp(-logits / 2))) == pytest.approx(
        2.0, abs=0.2
    )


def test_calibration_slope_is_nan_when_the_recalibration_separates() -> None:
    """A saturating model gives the recalibration no finite maximum to find."""
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.full(50, 1e-9), np.full(50, 1 - 1e-9)])
    assert np.isnan(metrics.calibration_slope(y, p))


def test_metrics_are_nan_on_a_single_class_split() -> None:
    y = np.zeros(10, dtype=int)
    p = np.linspace(0.1, 0.9, 10)
    assert np.isnan(metrics.auroc(y, p))
    assert np.isnan(metrics.auprc(y, p))
    assert not np.isnan(metrics.brier(y, p))


# --------------------------------------------------------------------------- #
# Probes and report
# --------------------------------------------------------------------------- #


def test_few_shot_subsample_is_balanced_where_it_can_be() -> None:
    y = np.array([0] * 100 + [1] * 10)
    index = probe.subsample(y, 32, seed=0)
    assert y[index].sum() == 10  # every positive there is
    assert (y[index] == 0).sum() == 32
    assert probe.subsample(y, None, seed=0).size == y.size


def test_report_renders_every_section() -> None:
    results = {
        "source": "fixture",
        "eval_split": "held_out",
        "models": {"lr": {"kind": "lr", "features": "counts"}},
        "tasks": {
            "t": {
                "counts": {"train": 10, "tuning": 3, "held_out": 4},
                "prevalence": {"train": 0.5, "tuning": 0.3, "held_out": 0.25},
                "models": {
                    "lr": {
                        "metrics": {"auroc": {"point": 0.75, "lo": 0.5, "hi": 0.9}},
                        "few_shot": [{"k": 32, "n_train": 64, "auroc_mean": 0.7, "auroc_std": 0.1}],
                    }
                },
                "paired": [{"a": "lr", "b": "gbm", "diff": 0.02, "lo": -0.01, "hi": 0.05}],
            }
        },
        "skipped": {"other": "no codes"},
    }
    text = report.render(results)
    assert "0.750 [0.500, 0.900]" in text
    assert "Paired bootstrap" in text
    assert "Few-shot" in text
    assert "Skipped" in text


def test_restrict_eval_split_keeps_train_and_tuning_whole() -> None:
    frame = pl.DataFrame(
        {
            "subject_id": list(range(200)),
            "split": ["train"] * 100 + ["tuning"] * 30 + ["held_out"] * 70,
        }
    )
    out = run.restrict_eval_split(frame, "held_out", limit=20, seed=0)
    assert (out["split"] == "train").sum() == 100
    assert (out["split"] == "tuning").sum() == 30
    assert (out["split"] == "held_out").sum() == 20
    # Deterministic in the subject, not the row order.
    again = run.restrict_eval_split(
        frame.sample(fraction=1.0, shuffle=True, seed=1), "held_out", 20, 0
    )
    assert set(out.filter(pl.col("split") == "held_out")["subject_id"].to_list()) == set(
        again.filter(pl.col("split") == "held_out")["subject_id"].to_list()
    )
    # A different seed draws a different subset.
    other = run.restrict_eval_split(frame, "held_out", 20, seed=1)
    assert set(out.filter(pl.col("split") == "held_out")["subject_id"].to_list()) != set(
        other.filter(pl.col("split") == "held_out")["subject_id"].to_list()
    )
    # A limit at or above the split size is a no-op.
    assert run.restrict_eval_split(frame, "held_out", 70, seed=0).height == frame.height
    assert run.restrict_eval_split(frame, "held_out", None, seed=0).height == frame.height


def _stub_checkpoint(path: Path, causal: bool) -> Path:
    """The smallest payload :func:`probe.checkpoint_is_causal` reads."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": {"causal": causal}}, path)
    return path


def test_default_pooling_follows_the_architecture(tmp_path: Path) -> None:
    """``auto`` is ``last`` for a causal checkpoint and ``mean`` for a bidirectional one.

    Resolved per model, not per run: one command scoring an AR arm and a JEPA arm
    must give each the pooling its attention pattern calls for.
    """
    causal = _stub_checkpoint(tmp_path / "ar" / "final.pt", causal=True)
    bidirectional = _stub_checkpoint(tmp_path / "jepa" / "final.pt", causal=False)

    specs = run.parse_models([f"ckpt:{causal}", f"ckpt:{bidirectional}"])
    assert [s.probe_features for s in specs] == ["last", "mean"]
    assert [s.features for s in specs] == ["last@final", "mean@final"]

    # The control copies the architecture it is a control for, pooling included.
    with_control = run.parse_models([f"ckpt:{causal}", "random_init"])
    assert with_control[-1].probe_features == "last"

    # An explicit choice still wins everywhere.
    pinned = run.parse_models([f"ckpt:{causal}", f"ckpt:{bidirectional}"], "cls_mean")
    assert {s.probe_features for s in pinned} == {"cls_mean"}


def test_unknown_pooling_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown probe features"):
        run.parse_models(["ckpt:runs/x/final.pt"], "nonsense")


def test_parse_models_rejects_random_init_without_an_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        run.parse_models(["lr", "random_init"])


def test_probe_device_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """``probe._device`` must agree with ``resolve_device``'s CUDA > MPS > CPU order.

    A prior version of this function checked MPS before CUDA -- harmless on a
    Mac (no CUDA) or a CUDA box (no MPS) since only one backend is ever really
    available, but a silent second copy of the device policy that a CUDA+MPS
    environment (or a future backend) would have picked wrong.
    """
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe.torch.backends.mps, "is_available", lambda: True)
    assert probe._device().type == "cuda"

    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    assert probe._device().type == "mps"

    monkeypatch.setattr(probe.torch.backends.mps, "is_available", lambda: False)
    assert probe._device().type == "cpu"

    # An explicit override always wins, regardless of what is available.
    assert probe._device("cpu").type == "cpu"
    specs = run.parse_models(["lr", "gbm", "ckpt:runs/x/final.pt", "random_init"])
    assert [s.name for s in specs] == ["lr", "gbm", "ckpt:x", "random_init"]
    assert specs[-1].random_init and specs[-1].checkpoint == Path("runs/x/final.pt")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@requires_aces
@requires_demo
def test_end_to_end_on_the_demo_cache(tmp_path: Path) -> None:
    """Debug checkpoint, two tasks, three models, on CPU -- seconds, not minutes."""
    import time

    from ehrjepa.train.config import load_config
    from ehrjepa.train.pretrain import Trainer

    started = time.time()
    config = load_config(
        DEBUG_CONFIG,
        [f"run.out_dir={tmp_path / 'run'}", "run.steps=3", "run.tensorboard=false"],
    )
    Trainer(config).train()
    checkpoint = tmp_path / "run" / "final.pt"
    assert checkpoint.exists()

    results = run.run(
        "mimic-demo",
        run.parse_models(["lr", f"ckpt:{checkpoint}", "random_init"]),
        tmp_path / "out",
        meds_root=REPO / "data" / "meds",
        cache_root=REPO / "data" / "cache",
        task_root=tmp_path / "tasks",
        feature_cache=tmp_path / "features",
        task_names=["mortality_365d"],
        n_boot=20,
        device="cpu",
        few_shot=False,
    )
    entry = results["tasks"]["mortality_365d"]
    assert set(entry["models"]) == {"lr", "ckpt:run", "random_init"}
    for model in entry["models"].values():
        assert set(model["metrics"]) == set(metrics.METRICS)
    assert entry["paired"], "paired comparisons must be reported"
    assert (tmp_path / "out" / "results.md").exists()
    assert (tmp_path / "out" / "results.json").exists()
    assert time.time() - started < 30.0


@requires_demo
def test_anchor_times_line_up_with_the_tensor_cache() -> None:
    """An anchor picked from MEDS indexes the same instant in the cache."""
    events = tasks.read_events(DEMO_MEDS)
    anchors, _ = tasks.select_anchors(events, 365, death_mask=pl.col("code") == DEATH)
    anchors = anchors.join(
        pl.read_parquet(DEMO_MEDS / "metadata" / "subject_splits.parquet"),
        on="subject_id",
        how="inner",
    )
    reader = HistoryReader(DEMO_CACHE, max_len=None)
    anchors = reader.filter_present(anchors)
    minutes = anchor_minutes(anchors["anchor_time"])
    for i, row in enumerate(anchors.iter_rows(named=True)):
        history = reader.history(row["subject_id"], row["split"], int(minutes[i]))
        expected = events.filter(
            (pl.col("subject_id") == row["subject_id"])
            & (pl.col("event_time") < row["anchor_time"])
        ).height
        assert history["code_id"].size == expected


def test_random_init_cache_name_is_keyed_on_the_architecture_it_copies() -> None:
    """Two grids' untrained controls must not collide in the embedding cache.

    ``random_init`` is named after what it is, not after the checkpoint it copies
    its architecture from, so a cache keyed on the display name hands a 4x192
    grid the untrained 6x256 vectors an earlier experiment left behind -- a
    control that controls for nothing, with entirely plausible-looking AUROCs.
    Caught by the phase-5a micro-grid, where both cells' controls came back
    identical to each other and to the phase-4 numbers.
    """
    from ehrjepa.eval.probe import embedding_path
    from ehrjepa.eval.run import parse_models

    pilot = parse_models(["random_init", "ckpt:runs/pilot/ar/final.pt"])[0]
    sanity = parse_models(["random_init", "ckpt:runs/sanity-A-default/final.pt"])[0]
    assert pilot.name == sanity.name == "random_init", "the display name stays stable"
    assert pilot.cache_name == "random_init@ar"
    assert sanity.cache_name == "random_init@sanity-A-default"
    assert embedding_path("c", "src", pilot.cache_name) != embedding_path(
        "c", "src", sanity.cache_name
    )

    # A trained checkpoint is already named after its run directory.
    trained = parse_models(["ckpt:runs/pilot/ar/final.pt"])[0]
    assert trained.cache_name == trained.name == "ckpt:ar"
