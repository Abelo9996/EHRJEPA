"""The two PhysioNet ICU sources: ETL shape, and the hourly task labels.

Everything here runs against the hand-written fixtures in
``tests/fixtures/physionet2019`` and ``tests/fixtures/physionet2012`` -- a few
stays each, chosen for the awkward cases: an hour whose every measurement is
missing, a stay too short for either sepsis task, a stay whose sepsis flag is
already up on hour 1, a ``-1`` height sentinel, a weight re-recorded mid-stay,
and a record with too few events to anchor. Nothing here touches ``data/``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from ehrjepa.data import etl
from ehrjepa.data.etl import physionet2012, physionet2019
from ehrjepa.eval import icu_tasks, tasks
from test_data_etl import assert_canonical

FIXTURES = Path(__file__).parent / "fixtures"

ORIGIN_2019 = physionet2019.ORIGIN
ORIGIN_2012 = physionet2012.ORIGIN


def _hour_2019(hours: int) -> dt.datetime:
    return ORIGIN_2019 + dt.timedelta(hours=hours)


def _hour_2012(hours: int) -> dt.datetime:
    return ORIGIN_2012 + dt.timedelta(hours=hours)


def _events(out: Path) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(out.rglob("data/*/*.parquet"))])


# --------------------------------------------------------------------------- #
# PhysioNet 2019
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def p2019_out(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("physionet2019") / "meds"
    etl.run("physionet2019", FIXTURES / "physionet2019", out, shard_size=2)
    return out


def test_p2019_end_to_end(p2019_out: Path) -> None:
    events = assert_canonical(p2019_out)
    assert set(events["subject_id"]) == {1, 2, 3, 100001}

    by_code = dict(events.group_by("code").len().iter_rows())
    assert by_code["MEDS_BIRTH"] == 4
    assert by_code["AGE"] == 4
    assert by_code["SEX//M"] == 2
    assert by_code["SEX//F"] == 2
    assert by_code["HOSP_ADM_TIME"] == 4
    # Unit1/Unit2 are read with max() over the stay, so p000002's null first row
    # does not lose its SICU flag.
    assert by_code["UNIT//MICU"] == 2
    assert by_code["UNIT//SICU"] == 1
    # One marker per recorded hour: 20 + 30 + 4 + 14.
    assert by_code["ICU_HOUR"] == 68
    # Two septic stays, one onset event each, and no hourly flag events at all.
    assert by_code["SEPSIS_ONSET"] == 2
    assert not [code for code in by_code if code.startswith("SEPSIS_LABEL")]


def test_p2019_iculos_hour_one_lands_on_the_origin(p2019_out: Path) -> None:
    events = _events(p2019_out).filter(pl.col("subject_id") == 1)
    first_hour = events.filter(pl.col("code") == "ICU_HOUR").sort("time")
    assert first_hour["time"][0] == ORIGIN_2019
    assert first_hour["numeric_value"][0] == pytest.approx(1.0)
    # Hour 5 (ICULOS 5) has no measurements at all; the marker keeps it in the
    # stream, so the hour grid has no holes.
    hour_five = events.filter(pl.col("time") == _hour_2019(4))
    assert hour_five["code"].to_list() == ["ICU_HOUR"]
    assert hour_five["numeric_value"][0] == pytest.approx(5.0)


def test_p2019_onset_is_the_first_flagged_hour_plus_six(p2019_out: Path) -> None:
    events = _events(p2019_out).filter(pl.col("code") == "SEPSIS_ONSET")
    onset = dict(zip(events["subject_id"].to_list(), events["time"].to_list(), strict=True))
    # p000002 first flags SepsisLabel at ICULOS 12, i.e. hour 11 on the grid.
    assert onset[2] == _hour_2019(11 + physionet2019.SEPSIS_SHIFT_HOURS)
    # p100001 flags from ICULOS 1, i.e. hour 0.
    assert onset[100001] == _hour_2019(physionet2019.SEPSIS_SHIFT_HOURS)


def test_p2019_birth_encodes_the_reported_age(p2019_out: Path) -> None:
    events = _events(p2019_out)
    birth = events.filter((pl.col("subject_id") == 1) & (pl.col("code") == "MEDS_BIRTH"))
    age = events.filter((pl.col("subject_id") == 1) & (pl.col("code") == "AGE"))
    assert age["numeric_value"][0] == pytest.approx(61.5)
    years = (ORIGIN_2019 - birth["time"][0]).days / 365.25
    assert years == pytest.approx(61.5, abs=0.01)


def test_p2019_measurements_carry_their_value_and_skip_nulls(p2019_out: Path) -> None:
    events = _events(p2019_out).filter(pl.col("subject_id") == 1)
    hr = events.filter(pl.col("code") == "VAR//HR").sort("time")
    # 20 hours minus the one all-missing hour.
    assert hr.height == 19
    assert hr["numeric_value"][0] == pytest.approx(80.0)
    # Lactate is only recorded on hours 1, 7 and 13 of the fixture.
    assert events.filter(pl.col("code") == "VAR//Lactate").height == 4


def test_p2019_find_files_accepts_both_layouts(tmp_path: Path) -> None:
    nested = tmp_path / "training" / "training_setA"
    nested.mkdir(parents=True)
    source = FIXTURES / "physionet2019" / "training_setA" / "p000001.psv"
    (nested / "p000001.psv").write_text(source.read_text())
    assert physionet2019.find_files(tmp_path) == {"training_setA": [nested / "p000001.psv"]}
    with pytest.raises(ValueError, match="no PhysioNet 2019"):
        physionet2019.find_files(tmp_path / "training" / "empty")


# --------------------------------------------------------------------------- #
# PhysioNet 2012
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def p2012_out(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("physionet2012") / "meds"
    etl.run("physionet2012", FIXTURES / "physionet2012", out, shard_size=2)
    return out


def test_p2012_end_to_end(p2012_out: Path) -> None:
    events = assert_canonical(p2012_out)
    assert set(events["subject_id"]) == {132539, 132540, 132541, 132542}

    by_code = dict(events.group_by("code").len().iter_rows())
    assert by_code["MEDS_BIRTH"] == 4
    assert by_code["AGE"] == 4
    assert by_code["MEDS_DEATH"] == 2
    assert by_code["SEX//M"] == 2
    assert by_code["SEX//F"] == 2
    assert by_code["UNIT//CCU"] == 1
    assert by_code["UNIT//CSRU"] == 1
    assert by_code["UNIT//MICU"] == 1
    assert by_code["UNIT//SICU"] == 1
    assert by_code["LOS"] == by_code["SAPS_I"] == by_code["SOFA"] == 4
    # Survival is -1 for the two survivors, which is a sentinel, not a duration.
    assert by_code["SURVIVAL"] == 2
    # RecordID, Age, Gender and ICUType are statics, not VAR// channels.
    assert "VAR//RecordID" not in by_code
    assert "VAR//ICUType" not in by_code


def test_p2012_outcome_events_sit_past_the_observation_window(p2012_out: Path) -> None:
    events = _events(p2012_out)
    outcome_time = _hour_2012(physionet2012.OUTCOME_HOUR)
    outcomes = events.filter(pl.col("code").is_in(["MEDS_DEATH", "LOS", "SAPS_I", "SOFA"]))
    assert set(outcomes["time"]) == {outcome_time}
    # Nothing else reaches that far, so an hour-48 history cannot contain one.
    others = events.filter(
        ~pl.col("code").is_in(["MEDS_DEATH", "LOS", "SAPS_I", "SOFA", "SURVIVAL"])
    )
    assert others["time"].max() < outcome_time


def test_p2012_height_and_weight_sentinels_are_dropped(p2012_out: Path) -> None:
    events = _events(p2012_out)
    # 132539 has Height -1; 132540 has Weight -1 at 00:00 but a real one at 24:00.
    assert events.filter(
        (pl.col("subject_id") == 132539) & (pl.col("code") == "VAR//Height")
    ).is_empty()
    weight = events.filter((pl.col("subject_id") == 132540) & (pl.col("code") == "VAR//Weight"))
    assert weight.height == 1
    assert weight["numeric_value"][0] == pytest.approx(88.0)


def test_p2012_time_parses_hours_past_twenty_four(p2012_out: Path) -> None:
    events = _events(p2012_out)
    lactate = events.filter(pl.col("code") == "VAR//Lactate")
    assert lactate.height == 1
    assert lactate["time"][0] == ORIGIN_2012 + dt.timedelta(hours=47, minutes=59)


def test_p2012_read_outcomes_is_keyed_and_typed() -> None:
    outcomes = physionet2012.read_outcomes([FIXTURES / "physionet2012" / "Outcomes-a.txt"])
    assert outcomes["subject_id"].to_list() == [132539, 132540, 132541, 132542]
    assert outcomes["death"].to_list() == [0, 1, 1, 0]
    assert outcomes["LOS"].to_list() == [5.0, 8.0, 3.0, 12.0]


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


def test_task_specs_are_gated_on_the_source_family(p2019_out: Path, p2012_out: Path) -> None:
    supported, skipped = tasks.task_specs_for(p2019_out)
    assert [spec.name for spec in supported] == ["sepsis_6h", "sepsis_stay"]
    assert "mortality_inhospital/24h" in skipped
    assert "mortality_365d" in skipped

    supported, skipped = tasks.task_specs_for(p2012_out)
    assert [spec.name for spec in supported] == [
        "mortality_inhospital/24h",
        "mortality_inhospital/48h",
    ]
    assert "sepsis_6h" in skipped


def _build(name: str, out: Path) -> tuple[pl.DataFrame, dict]:
    spec = next(spec for spec in tasks.TASKS if spec.name == name)
    return tasks.build_task(spec, out)


def test_sepsis_6h_labels_the_onset_window_and_drops_post_onset_anchors(
    p2019_out: Path,
) -> None:
    labels, counts = _build("sepsis_6h", p2019_out)
    assert counts["multi_anchor"] is True
    hours = labels.with_columns(
        hour=((pl.col("anchor_time") - pl.lit(ORIGIN_2019)).dt.total_hours())
    )
    per_subject = {
        key[0]: dict(zip(frame["hour"].to_list(), frame["label"].to_list(), strict=True))
        for key, frame in hours.group_by("subject_id")
    }

    # p000003 has four hours: no anchor is even a candidate.
    assert 3 not in per_subject
    # p000001 is never septic; every anchor it gets is a negative.
    assert set(per_subject[1].values()) == {0}
    # p100001's onset is hour 6, which is the earliest hour the task would
    # anchor at, so it has no candidate strictly before onset and drops out
    # entirely rather than contributing an hour-6 row labelled 0.
    assert 100001 not in per_subject
    # p000002's onset is hour 17. Anchors at hours 11..16 are positives (onset
    # within six hours), earlier ones negatives, and nothing at or after 17.
    p2 = per_subject[2]
    assert max(p2) < 17
    for hour, label in p2.items():
        assert label == int(11 <= hour <= 16), (hour, label)
    # At most eight anchors per stay, from a seeded draw.
    assert all(len(v) <= icu_tasks.MAX_ANCHORS_PER_STAY for v in per_subject.values())


def test_sepsis_6h_anchor_thinning_is_a_pure_function_of_the_seed(p2019_out: Path) -> None:
    spec = next(spec for spec in tasks.TASKS if spec.name == "sepsis_6h")
    first, _ = tasks.build_task(spec, p2019_out, seed=7)
    again, _ = tasks.build_task(spec, p2019_out, seed=7)
    other, _ = tasks.build_task(spec, p2019_out, seed=8)
    assert first.equals(again)
    assert not first.equals(other)


def test_sepsis_stay_anchors_at_hour_twelve_and_needs_onset_after_it(
    p2019_out: Path,
) -> None:
    labels, counts = _build("sepsis_stay", p2019_out)
    assert set(labels["anchor_time"]) == {_hour_2019(icu_tasks.SEPSIS_STAY_HOUR)}
    assert labels["subject_id"].n_unique() == labels.height
    got = dict(zip(labels["subject_id"].to_list(), labels["label"].to_list(), strict=True))
    # p000001 (20 h, never septic) is a negative; p000002 (30 h, onset hour 17)
    # is a positive; p000003 is four hours long and p100001's onset is hour 6,
    # which is before the anchor, so neither is admitted.
    assert got == {1: 0, 2: 1}
    assert counts["long_enough"] == 3
    assert counts["onset_after_anchor"] == 2


def test_mortality_anchors_at_both_hours_and_labels_in_hospital_death(
    p2012_out: Path,
) -> None:
    for hour in (24, 48):
        labels, counts = _build(f"mortality_inhospital/{hour}h", p2012_out)
        assert set(labels["anchor_time"]) == {_hour_2012(hour)}
        assert labels["subject_id"].n_unique() == labels.height
        got = dict(zip(labels["subject_id"].to_list(), labels["label"].to_list(), strict=True))
        # 132542 has three events, below the minimum, so it never anchors.
        assert got == {132539: 0, 132540: 1, 132541: 1}
        assert counts["deaths"] == 2
        assert counts["anchor_hour"] == hour


def test_mortality_history_at_the_anchor_excludes_every_outcome_event(
    p2012_out: Path,
) -> None:
    """The label event must be strictly after the anchor, for both anchor hours."""
    events = tasks.read_events(p2012_out)
    outcomes = events.filter(
        pl.col("code").is_in(["MEDS_DEATH", "LOS", "SAPS_I", "SOFA", "SURVIVAL"])
    )
    for hour in (24, 48):
        labels, _ = _build(f"mortality_inhospital/{hour}h", p2012_out)
        anchor = labels["anchor_time"][0]
        leaked = outcomes.filter(pl.col("event_time") < anchor)
        assert leaked.is_empty(), leaked


def test_sepsis_onset_is_never_inside_a_sepsis_anchor_history(p2019_out: Path) -> None:
    events = tasks.read_events(p2019_out)
    onsets = events.filter(pl.col("code") == "SEPSIS_ONSET").select(
        "subject_id", onset=pl.col("event_time")
    )
    for name in ("sepsis_6h", "sepsis_stay"):
        labels, _ = _build(name, p2019_out)
        joined = labels.join(onsets, on="subject_id", how="inner")
        assert joined.filter(pl.col("onset") <= pl.col("anchor_time")).is_empty()


def test_windows_at_cuts_the_synthetic_clock_at_the_anchor(p2019_out: Path, tmp_path: Path) -> None:
    """The anchor timestamps are ordinary datetimes to the tensor cache."""
    from ehrjepa.data.tokenize import main as tokenize_main
    from ehrjepa.eval.history import HistoryReader, anchor_minutes

    cache = tmp_path / "cache"
    assert tokenize_main(["build", str(p2019_out), "--cache", str(cache), "--min-count", "1"]) == 0
    reader = HistoryReader(cache, max_len=None)
    labels, _ = _build("sepsis_stay", p2019_out)
    minutes = anchor_minutes(labels["anchor_time"])
    for row, minute in zip(labels.iter_rows(named=True), minutes.tolist(), strict=True):
        if not reader.has_subject(row["subject_id"], row["split"]):
            continue
        history = reader.history(row["subject_id"], row["split"], minute)
        assert history["time_min"].size > 0
        assert history["time_min"].max() < minute
