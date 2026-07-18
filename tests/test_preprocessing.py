import pandas as pd
import pytest

from src.preprocessing import (
    build_monthly_series,
    deduplicate,
    detect_missing_months,
    run_pipeline,
    validate_timestamps,
    validate_units,
)


def _raw_row(month: str, cost: float, kwh: float, source: str) -> dict:
    return {
        "month_start": pd.Timestamp(month),
        "cost_gbp": cost,
        "consumption_kwh": kwh,
        "source_file": source,
    }


def test_deduplicate_removes_exact_duplicates():
    df = pd.DataFrame(
        [
            _raw_row("2024-01-01", 100.0, 900.0, "a.csv"),
            _raw_row("2024-01-01", 100.0, 900.0, "a.csv"),
        ]
    )

    result, n_exact, conflicts = deduplicate(df)

    assert len(result) == 1
    assert n_exact == 1
    assert conflicts == []


def test_deduplicate_resolves_conflict_by_keeping_last():
    df = pd.DataFrame(
        [
            _raw_row("2024-01-01", 100.0, 900.0, "old_export.csv"),
            _raw_row("2024-01-01", 110.0, 950.0, "new_export.csv"),
        ]
    )

    result, n_exact, conflicts = deduplicate(df)

    assert len(result) == 1
    assert result.iloc[0]["cost_gbp"] == pytest.approx(110.0)
    assert result.iloc[0]["source_file"] == "new_export.csv"
    assert len(conflicts) == 1
    assert "new_export.csv" in conflicts[0]


def test_validate_timestamps_sorts_ascending():
    df = pd.DataFrame(
        [
            _raw_row("2024-03-01", 10.0, 10.0, "a.csv"),
            _raw_row("2024-01-01", 10.0, 10.0, "a.csv"),
        ]
    )

    result = validate_timestamps(df)

    assert list(result["month_start"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")]


def test_validate_timestamps_raises_on_duplicate_month():
    df = pd.DataFrame(
        [
            _raw_row("2024-01-01", 10.0, 10.0, "a.csv"),
            _raw_row("2024-01-01", 20.0, 20.0, "b.csv"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate month"):
        validate_timestamps(df)


def test_validate_units_flags_without_dropping():
    df = pd.DataFrame(
        [
            _raw_row("2024-01-01", -5.0, 900.0, "a.csv"),
            _raw_row("2024-02-01", 100.0, 999999.0, "a.csv"),
            _raw_row("2024-03-01", 50.0, 300.0, "a.csv"),
        ]
    )

    result, warnings = validate_units(df)

    assert len(result) == 3
    assert any("negative cost" in w for w in warnings)
    assert any("unusually high consumption" in w for w in warnings)
    assert len(warnings) == 2


def test_detect_missing_months_finds_synthetic_gap():
    df = pd.DataFrame(
        [
            _raw_row("2024-01-01", 10.0, 10.0, "a.csv"),
            _raw_row("2024-03-01", 10.0, 10.0, "a.csv"),
        ]
    )

    missing = detect_missing_months(df)

    assert missing == [pd.Timestamp("2024-02-01")]


def test_detect_missing_months_empty_when_contiguous():
    df = pd.DataFrame(
        [
            _raw_row("2024-01-01", 10.0, 10.0, "a.csv"),
            _raw_row("2024-02-01", 10.0, 10.0, "a.csv"),
        ]
    )

    assert detect_missing_months(df) == []


def test_build_monthly_series_derived_columns():
    df = pd.DataFrame(
        [
            _raw_row("2024-02-01", 58.0, 290.0, "a.csv"),  # Feb 2024 = 29 days (leap year)
        ]
    )

    result = build_monthly_series(df)
    row = result.iloc[0]

    assert row["days_in_month"] == 29
    assert row["avg_daily_kwh"] == pytest.approx(290.0 / 29)
    assert row["avg_daily_cost_gbp"] == pytest.approx(58.0 / 29)
    assert row["unit_rate_gbp_per_kwh"] == pytest.approx(58.0 / 290.0)
    assert row["year"] == 2024
    assert row["month_name"] == "February"


def test_build_monthly_series_flags_partial_year():
    df = pd.DataFrame(
        [_raw_row(f"2024-{m:02d}-01", 10.0, 10.0, "a.csv") for m in range(1, 6)]
    )  # only 5 of 12 months

    result = build_monthly_series(df)

    assert result["is_partial_year"].all()


def test_run_pipeline_end_to_end_produces_report():
    df = pd.DataFrame(
        [
            _raw_row("2024-01-01", 100.0, 900.0, "a.csv"),
            _raw_row("2024-01-01", 100.0, 900.0, "a.csv"),  # exact dup
            _raw_row("2024-03-01", 50.0, 300.0, "a.csv"),  # gap at Feb
        ]
    )

    clean, report = run_pipeline(df)

    assert report.duplicates_removed == 1
    assert report.missing_months == [pd.Timestamp("2024-02-01")]
    assert report.n_months == 2
    assert len(clean) == 2
