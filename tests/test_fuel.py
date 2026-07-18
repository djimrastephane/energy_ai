import numpy as np
import pandas as pd
import pytest

from src.fuel import combine_fuel_frames, cross_check_fuel_totals, finding_fuel_mix


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
