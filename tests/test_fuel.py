import numpy as np
import pandas as pd
import pytest

from src.fuel import (
    combine_fuel_frames,
    cross_check_fuel_totals,
    finding_fuel_mix,
    infer_total_from_electricity_and_gas,
)


def _fuel_df(n, consumption, cost, start="2023-09-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame({"month_start": idx, "consumption_kwh": consumption, "cost_gbp": cost})


def test_cross_check_fuel_totals_no_warnings_when_matching():
    elec = _fuel_df(3, [100.0, 110.0, 120.0], [30.0, 33.0, 36.0])
    gas = _fuel_df(3, [500.0, 300.0, 100.0], [50.0, 30.0, 10.0])
    total = _fuel_df(3, [600.0, 410.0, 220.0], [80.0, 63.0, 46.0])

    assert cross_check_fuel_totals(total, elec, gas) == []


def test_cross_check_fuel_totals_flags_mismatch():
    elec = _fuel_df(2, [100.0, 110.0], [30.0, 33.0])
    gas = _fuel_df(2, [500.0, 300.0], [50.0, 30.0])
    total = _fuel_df(2, [600.0, 999.0], [80.0, 63.0])  # 2nd month deliberately wrong

    warnings = cross_check_fuel_totals(total, elec, gas)

    assert len(warnings) == 1
    assert "October 2023" in warnings[0]


def test_cross_check_fuel_totals_tolerates_small_rounding():
    elec = _fuel_df(1, [100.0], [30.0])
    gas = _fuel_df(1, [500.0], [50.0])
    total = _fuel_df(1, [600.3], [80.02])  # within default tolerances

    assert cross_check_fuel_totals(total, elec, gas) == []


def test_combine_fuel_frames_joins_and_computes_share():
    elec = _fuel_df(2, [100.0, 200.0], [30.0, 60.0])
    gas = _fuel_df(2, [300.0, 200.0], [30.0, 20.0])

    combined = combine_fuel_frames(elec, gas)

    assert list(combined["electricity_kwh"]) == [100.0, 200.0]
    assert list(combined["gas_kwh"]) == [300.0, 200.0]
    assert combined["electricity_share_pct"].tolist() == pytest.approx([25.0, 50.0])


def test_combine_fuel_frames_only_keeps_overlapping_months():
    elec = _fuel_df(3, [100.0, 100.0, 100.0], [30.0, 30.0, 30.0], start="2023-09-01")
    gas = _fuel_df(2, [300.0, 300.0], [30.0, 30.0], start="2023-10-01")  # starts one month later

    combined = combine_fuel_frames(elec, gas)

    assert len(combined) == 2
    assert combined["month_start"].min() == pd.Timestamp("2023-10-01")


def test_finding_fuel_mix_none_with_too_little_history():
    elec = _fuel_df(3, [100.0] * 3, [30.0] * 3)
    gas = _fuel_df(3, [300.0] * 3, [30.0] * 3)
    combined = combine_fuel_frames(elec, gas)

    assert finding_fuel_mix(combined) is None


def test_finding_fuel_mix_reports_gas_dominant_and_seasonal():
    idx = pd.date_range("2023-01-01", periods=12, freq="MS")
    month = idx.month
    # Gas swings hard with the seasons (heating); electricity stays comparatively flat.
    gas_kwh = np.where(np.isin(month, [12, 1, 2]), 800.0, np.where(np.isin(month, [6, 7, 8]), 30.0, 200.0))
    elec_kwh = np.where(np.isin(month, [12, 1, 2]), 160.0, np.where(np.isin(month, [6, 7, 8]), 100.0, 130.0))
    elec = pd.DataFrame({"month_start": idx, "consumption_kwh": elec_kwh, "cost_gbp": elec_kwh * 0.3})
    gas = pd.DataFrame({"month_start": idx, "consumption_kwh": gas_kwh, "cost_gbp": gas_kwh * 0.1})

    combined = combine_fuel_frames(elec, gas)
    finding = finding_fuel_mix(combined)

    assert finding is not None
    assert finding.category == "fuel"
    assert finding.confidence == "High"
    assert "gas is the more weather-driven fuel" in finding.narrative
    assert "gas central heating" in finding.narrative
    # Gas dominates consumption in this synthetic example.
    assert "%" in finding.narrative


def test_finding_fuel_mix_none_when_no_consumption_at_all():
    elec = _fuel_df(6, [0.0] * 6, [0.0] * 6)
    gas = _fuel_df(6, [0.0] * 6, [0.0] * 6)
    combined = combine_fuel_frames(elec, gas)

    assert finding_fuel_mix(combined) is None


def test_infer_total_sums_electricity_and_gas():
    elec = _fuel_df(3, [100.0, 110.0, 120.0], [30.0, 33.0, 36.0])
    gas = _fuel_df(3, [500.0, 300.0, 100.0], [50.0, 30.0, 10.0])

    total, report = infer_total_from_electricity_and_gas(elec, gas)

    assert list(total["consumption_kwh"]) == [600.0, 410.0, 220.0]
    assert list(total["cost_gbp"]) == pytest.approx([80.0, 63.0, 46.0])
    assert report.n_files_loaded == 0
    assert report.source_files == ["inferred: Electricity Use + Gas Use"]
    assert report.n_months == 3
    assert report.missing_months == []


def test_infer_total_has_same_derived_columns_as_a_real_clean_frame():
    """The inferred total must be a drop-in substitute for a real Total Use file --
    every column downstream analysis expects (year, days_in_month, etc.) must be present."""
    elec = _fuel_df(3, [100.0, 110.0, 120.0], [30.0, 33.0, 36.0])
    gas = _fuel_df(3, [500.0, 300.0, 100.0], [50.0, 30.0, 10.0])

    total, _ = infer_total_from_electricity_and_gas(elec, gas)

    for col in ("year", "month_num", "month_name", "days_in_month", "avg_daily_kwh", "is_partial_year"):
        assert col in total.columns


def test_infer_total_only_keeps_overlapping_months():
    elec = _fuel_df(3, [100.0, 100.0, 100.0], [30.0, 30.0, 30.0], start="2023-09-01")
    gas = _fuel_df(2, [300.0, 300.0], [30.0, 30.0], start="2023-10-01")  # starts one month later

    total, report = infer_total_from_electricity_and_gas(elec, gas)

    assert len(total) == 2
    assert total["month_start"].min() == pd.Timestamp("2023-10-01")
    assert report.n_months == 2


def test_infer_total_detects_a_gap_within_the_overlapping_range():
    idx = pd.to_datetime(["2023-09-01", "2023-11-01"])  # October missing from electricity
    elec = pd.DataFrame({"month_start": idx, "consumption_kwh": [100.0, 120.0], "cost_gbp": [30.0, 36.0]})
    gas = _fuel_df(3, [500.0, 300.0, 100.0], [50.0, 30.0, 10.0], start="2023-09-01")

    total, report = infer_total_from_electricity_and_gas(elec, gas)

    assert len(total) == 2  # October dropped by the inner join
    assert report.missing_months == [pd.Timestamp("2023-10-01")]


def test_infer_total_flags_implausibly_high_combined_consumption():
    from config import SETTINGS

    threshold = SETTINGS.validation.max_monthly_kwh
    elec = _fuel_df(1, [threshold], [500.0])
    gas = _fuel_df(1, [threshold], [500.0])  # each individually plausible; summed, not

    total, report = infer_total_from_electricity_and_gas(elec, gas)

    assert total["consumption_kwh"].iloc[0] == pytest.approx(threshold * 2)
    assert report.outlier_warnings
    assert "unusually high consumption" in report.outlier_warnings[0]


def test_infer_total_empty_when_either_fuel_is_empty():
    elec = _fuel_df(3, [100.0, 110.0, 120.0], [30.0, 33.0, 36.0])
    empty_gas = _fuel_df(0, [], [])

    total, report = infer_total_from_electricity_and_gas(elec, empty_gas)

    assert total.empty
    assert report.n_files_loaded == 0
    assert report.n_months == 0
    assert report.date_range is None
