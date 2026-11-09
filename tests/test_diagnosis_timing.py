"""Boundary and integrity checks for the model-free diagnosis timing audit."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ehrjepa.eval.diagnosis_timing import classify_timing, first_diagnoses, summarize

T0 = datetime(2020, 1, 1)


def labels(subjects, anchors, values):
    return pl.DataFrame(
        {"subject_id": subjects, "anchor_time": anchors, "label": values, "split": "held_out"}
    )


def diagnoses(subjects, times):
    return pl.DataFrame(
        {"subject_id": subjects, "first_diagnosis": times},
        schema={"subject_id": pl.Int64, "first_diagnosis": pl.Datetime("us")},
    )


def admissions(subjects, starts, ends):
    return pl.DataFrame(
        {
            "subject_id": subjects,
            "hadm_id": list(range(len(subjects))),
            "admittime": starts,
            "dischtime": ends,
        }
    )


def test_current_stay_discharge_vs_later_stay_and_no_diagnosis():
    end = T0 + timedelta(days=5)
    later = T0 + timedelta(days=50)
    rows = classify_timing(
        labels([1, 2, 3], [T0] * 3, [1, 1, 0]),
        diagnoses([1, 2], [end, later]),
        admissions(
            [1, 2, 2, 3],
            [T0 - timedelta(days=1), T0 - timedelta(days=1), later - timedelta(days=2), T0],
            [end, end, later, end],
        ),
    )
    counts = summarize(rows)
    assert counts["positives"] == 2
    assert counts["positives_at_current_stay_discharge"] == 1
    assert counts["anchors_in_stay"] == 3
    assert counts["anchors_strictly_in_stay"] == 2
    assert counts["label_mismatches"] == 0


@pytest.mark.parametrize(
    ("offset", "label", "mismatches"),
    [(-1, 0, 1), (0, 0, 1), (1, 1, 0), (365, 1, 0), (366, 0, 0), (365, 0, 1)],
)
def test_aces_boundaries_and_prior_diagnosis_exclusion(offset, label, mismatches):
    rows = classify_timing(
        labels([1], [T0], [label]),
        diagnoses([1], [T0 + timedelta(days=offset)]),
        admissions([1], [T0 - timedelta(days=5)], [T0]),
    )
    counts = summarize(rows)
    assert counts["label_mismatches"] == mismatches
    # At discharge the old stay no longer contains the anchor.
    assert counts["anchors_in_stay"] == 0
    assert counts["positives_at_current_stay_discharge"] == 0


def test_overlapping_stays_do_not_double_count_labels():
    end = T0 + timedelta(days=5)
    rows = classify_timing(
        labels([1], [T0], [1]),
        diagnoses([1], [end]),
        admissions([1, 1], [T0 - timedelta(days=2)] * 2, [end] * 2),
    )
    counts = summarize(rows)
    assert counts["label_rows"] == 1
    assert counts["positives_at_current_stay_discharge"] == 1
    assert counts["anchors_with_overlapping_stays"] == 1


def test_positive_without_matching_diagnosis_is_inconsistent():
    rows = classify_timing(
        labels([1], [T0], [1]),
        diagnoses([1], [None]),
        admissions([1], [T0], [T0 + timedelta(days=2)]),
    )
    assert summarize(rows)["label_mismatches"] == 1


def test_duplicate_labels_rejected():
    with pytest.raises(ValueError, match="duplicate subject/anchor"):
        classify_timing(
            labels([1, 1], [T0, T0], [0, 0]),
            diagnoses([1], [None]),
            admissions([1], [T0], [T0 + timedelta(days=2)]),
        )


def test_outpatient_subject_without_admission_is_counted_not_discarded():
    rows = classify_timing(
        labels([1], [T0], [0]),
        diagnoses([1], [None]),
        admissions([2], [T0], [T0 + timedelta(days=2)]),
    )
    counts = summarize(rows)
    assert counts["label_rows"] == 1
    assert counts["subjects_without_admission_rows"] == 1
    assert counts["anchors_in_stay"] == 0
    assert counts["label_mismatches"] == 0


def test_empty_split_has_no_rate():
    rows = classify_timing(
        labels([1], [T0], [0]),
        diagnoses([1], [None]),
        admissions([1], [T0], [T0 + timedelta(days=2)]),
    ).head(0)
    counts = summarize(rows)
    assert counts["label_rows"] == 0
    assert counts["fraction_of_positives_at_current_stay_discharge"] is None


def test_first_diagnoses_scans_all_shards_with_exact_task_predicates(tmp_path):
    import json

    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "dataset.json").write_text(json.dumps({"dataset_name": "MIMIC-IV"}))
    for split, time in [("train", T0), ("held_out", T0 - timedelta(days=1))]:
        folder = tmp_path / "data" / split
        folder.mkdir(parents=True)
        pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "time": [time] * 3,
                "code": ["ICD10CM/N18.3", "LAB/A", "ICD9CM/250.00"],
            }
        ).write_parquet(folder / "0.parquet")
    result = first_diagnoses(tmp_path).sort("subject_id")
    assert result["new_dx_365d/ckd"].to_list() == [T0 - timedelta(days=1), None]
    assert result["new_dx_365d/diabetes"].to_list() == [None, T0 - timedelta(days=1)]
